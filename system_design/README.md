# System Design

Architecture evolution notes — how a system grows from a single server to a cloud-native platform, and why each step exists.

The core pattern repeats: a bottleneck appears, a targeted fix is added, the fix creates a new bottleneck. Every component here exists to solve a specific, real problem.

---

## Concepts

### Why app and DB fight on the same machine

A single server's resources — CPU, memory, disk IO — are zero-sum. App processes and the database have directly conflicting resource profiles.

**Memory** is the sharpest conflict. MySQL InnoDB wants its Buffer Pool as large as possible: the more table data it can keep in RAM, the fewer disk reads it needs (memory is ~1000× faster than disk). Meanwhile the app also needs memory for threads, JVM heap, or worker processes. As traffic rises, both sides grow. When the app eats memory the DB can't grow its Buffer Pool → more queries fall through to disk → DB slows → app waits longer → more threads pile up → more memory pressure. A death spiral.

**Disk IO** is the most structurally incompatible conflict. The DB's IO pattern is heavy, continuous, and random (flushing dirty pages, writing WAL/redo log, fsync on every commit). The app's IO pattern is sporadic and sequential (writing logs). The DB's fsync calls are blocking — the disk queue is shared, so app log writes can directly delay transaction commits, inflating every write latency.

**CPU** contention is more bursty but equally real. A traffic spike drives the app to saturate CPU; the DB's query threads can't get scheduled; query latency rises; the app accumulates blocked threads; the CPU contention worsens.

**Fault domain coupling** is the hidden cost. An app memory leak can trigger the OOM killer to shoot the DB process. A DB slow query can max out CPU and make the app unresponsive. Restarting the app drops all DB connections. They share a blast radius.

Separation gives each process its own machine, so each can be sized, tuned, and failed independently.

---

### WAL — why the DB fsyncs the log, not the data

When a transaction commits, the DB must guarantee durability (the D in ACID): even if the machine loses power the next millisecond, committed data cannot be lost.

The naive approach — fsync the actual data pages on every commit — is too slow. Data pages are scattered randomly across disk; flushing them requires random IO, which is expensive.

The solution is **Write-Ahead Logging (WAL)**:

```
COMMIT
  → append operation to redo log (sequential write)
  → fsync redo log          ← only this must hit disk before returning
  → return "commit OK"

actual data pages
  → stay in Buffer Pool     ← flushed to disk in batches later (checkpoint)
```

The redo log records *what changed* ("set id=1 balance to 90"), not the full page. On restart after a crash, the DB replays the redo log to reconstruct any changes that never made it to disk.

This works because:
- **Sequential IO is fast** — the redo log is append-only, so one fsync flushes a small contiguous chunk.
- **Random IO is slow** — data pages are scattered; flushing them on every commit would kill write throughput.

WAL trades one fast sequential write (redo log fsync) for many slow random writes (data page fsyncs). That's why high-performance SSDs (NVMe) matter so much for DB servers — they shrink the fsync latency from milliseconds to microseconds, directly multiplying transaction throughput.

---

### How a slow SQL cascades into a full outage

A single slow query can take down an entire system through a chain of positive feedback loops.

**Step 1 — connection pool exhaustion.** The app maintains a connection pool (e.g. 20 connections per app server instance), shared across all users. A slow query holds its connection open for the duration. Under load, slow queries accumulate and fill all 20 slots.

**Step 2 — app thread pile-up.** New requests need a DB connection to proceed. With the pool full, they block and wait — each holding an app thread. App servers also have a thread limit (Tomcat defaults to 200). All threads stall waiting for a connection; new requests are rejected.

**Step 3 — retry amplification.** Users see timeouts and retry. Upstream services retry. More requests arrive into an already-saturated system, making it worse. This is a positive feedback loop.

**Step 4 — lock contention spreads.** If the slow query holds row or table locks (e.g. a long-running UPDATE), unrelated queries waiting on the same rows also pile up — consuming even more connections.

**Step 5 — memory and CPU collapse.** Accumulated threads and connections consume memory. The OS starts swapping. The DB reads large amounts of data for the slow query, evicting hot pages from the Buffer Pool — cache hit rate drops, more queries hit disk.

The reason the DB is more dangerous than the app: the app is stateless — one request failing doesn't affect others. The DB is a **shared stateful resource** — connection pool, locks, and Buffer Pool are shared by every request. One slow query's blast radius spans the entire system.

#### Prevention: three layers of defence

**DB layer — kill slow queries fast.**

```sql
-- MySQL: enforce a per-connection query timeout
SET SESSION MAX_EXECUTION_TIME = 3000;  -- 3 seconds

-- PostgreSQL:
SET statement_timeout = '3s';

-- Monitor and kill manually if needed
SELECT id, time, info FROM information_schema.processlist
WHERE command != 'Sleep' AND time > 10;
KILL QUERY <id>;
```

**Connection pool layer — fail fast, don't let threads pile up.**

Set a short wait timeout on pool acquisition. A request that can't get a connection within 500ms returns an error immediately instead of holding a thread indefinitely.

```
max connections: 20
connection wait timeout: 500ms   ← fail fast, don't block the thread
idle timeout: 10min              ← recycle unused connections
max lifetime: 30min              ← rotate connections periodically
```

Pool size is not "bigger is better" — each connection costs a DB thread and memory. Too many connections cause DB-side context-switching overhead that outweighs the parallelism. A common heuristic (HikariCP): `connections = CPU cores × 2 + disk count`.

**Architecture layer — isolate failures.**

Circuit breaker: monitor error rate to a downstream (DB or service). When it crosses a threshold, stop sending requests and return a degraded response (cached data, default value). After a recovery window, probe with a small amount of traffic before fully reopening.

```
normal          → requests pass through
error rate > 50% → circuit opens, requests return degraded response
after 30s       → probe with small traffic
recovered       → circuit closes
```

Rate limiting: cap requests at the entry point. Excess requests are rejected with 429 before they reach the DB at all.

---

### Connection pool and database concurrency model

**A connection pool is shared per app process, not per user.** All requests on a given app server instance borrow from the same pool, use the connection for one query or transaction, then return it. The pool lives for the lifetime of the process.

```
User A ──┐
User B ──┤                         ┌─ connection 1 ──┐
User C ──┼──→ app process (pool) ──┼─ connection 2 ──┼──→ DB
User D ──┤                         └─ connection 3 ──┘
User E ──┘
```

**A single connection is serial.** Only one SQL can run on a connection at a time — the connection carries session state (open transaction, session variables) that would be corrupted by interleaved queries. Parallel execution requires multiple connections.

**Database concurrency = number of connections.** Each connection maps to a DB thread (MySQL's model). Real parallelism is bounded by both the connection count and the number of CPU cores — more threads than cores causes context-switching overhead that erodes throughput.

```
20 connections = at most 20 SQL statements executing simultaneously
```

This is why connection pool exhaustion is fatal: even if the DB has spare capacity, a 21st request simply cannot enter.

**Connection assignment is availability-based, not random.** The pool hands out whichever connection is currently idle. The application doesn't know or care which connection it gets — they're interchangeable.

The exception is explicit transactions. Once a `BEGIN` is issued, the pool locks that connection to the request for the entire transaction. The session state (open transaction, locks held) is tied to that specific connection and cannot be transferred mid-flight.

```
outside a transaction:   request may get any idle connection each time
inside a transaction:    same connection held from BEGIN to COMMIT/ROLLBACK
```

---

### Multi-level cache

Even with app scaling, all reads still hit the database. Cache intercepts them at multiple layers:

```
request → browser cache → local cache → distributed cache (Redis) → DB
```

Each layer is checked in order. Only a miss falls through to the next. In practice the vast majority of reads are for the same hot data — those never reach the DB.

**Browser cache**: the request never leaves the client. Zero network cost. Controlled via HTTP headers (`Cache-Control: max-age=3600`). Good for static assets and infrequently-changing API responses.

**Local cache** (in-process, e.g. Guava Cache): data lives in each app server's own process memory. No network hop — nanosecond access. The downside is that every server holds its own copy. When data changes, each server's copy stays stale until its TTL expires. Only suitable for data where short-term inconsistency is acceptable: system config, category trees, rate-limit rules. Not suitable for inventory, balances, or anything requiring real-time accuracy.

**Distributed cache (Redis)**: a separate process that all app servers connect to over the network. Any server that writes to it, every other server can immediately read. One shared copy — consistent across the fleet. Microsecond-range latency (one network round-trip). This is the main workhorse layer.

```
server A  [local cache] ──┐
server B  [local cache] ──┼──→ Redis ──→ DB
server C  [local cache] ──┘
```

| | local cache | distributed cache |
|---|---|---|
| lives in | each server's process memory | independent Redis process |
| consistency | each server may differ | all servers see the same copy |
| speed | nanoseconds | microseconds |
| capacity | limited by one server's RAM | scales independently |

#### Cache vs DB consistency

On a write, the standard pattern is: update the DB, then **delete** the cache key (not update it). The next read misses the cache and reloads from the DB, rebuilding the cache entry with fresh data.

```
write:  UPDATE db → DELETE cache key
read:   GET cache → miss → SELECT db → SET cache
```

Deleting is safer than updating. If two writes arrive concurrently and both try to update the cache, the cache write order may not match the DB write order — leaving a stale value. Deleting forces the next read to rebuild from the authoritative source.

The window between deletion and rebuild is a stampede risk — see [caching/README.md](../caching/README.md) for how to handle it.

#### Distributed cache vs Redis cluster

These are two separate concepts.

**Distributed cache** is an architectural role: a cache layer shared by all app servers, regardless of how it is deployed underneath.

**Redis cluster** is a deployment topology: multiple Redis nodes that shard data across them, used when a single Redis instance runs out of capacity or needs fault tolerance.

A single-node Redis shared by multiple app servers is still a distributed cache. "Distributed" describes who uses it (multiple servers sharing one cache), not how the cache itself is deployed.

---

### Read/write split

After caching intercepts hot reads, write traffic and non-hot reads still hit the DB. The problem with writes: concurrent writes acquire row or table locks, causing queuing. Reads and writes also block each other on the same instance.

The fix is to separate concerns: **primary handles writes, replicas handle reads**.

```
app ──write──→ primary
                │
                └──(async)──→ replica 1 ←──read──┐
                            → replica 2 ←──read──┼── app
                            → replica 3 ←──read──┘
```

Internet workloads are typically read-heavy. One primary plus multiple replicas distributes read load across the fleet while the primary focuses on writes. Read/write split is the database layer's answer to horizontal scaling — the same idea as adding app servers, but applied to read capacity.

#### Why reads and writes block each other

Databases use locks to prevent concurrent modifications from corrupting data. There are two lock types:

- **Shared lock (read lock)**: multiple readers can hold simultaneously
- **Exclusive lock (write lock)**: held by writes; blocks all other reads and writes until released

Without locks, two concurrent writes reading the same value and each decrementing it would produce the wrong result — both read 10, both write 9, when the correct answer is 8.

The read/write blocking problem: a write holds an exclusive lock while the row is in an intermediate state. Reads must wait for the lock to release before they can proceed.

#### MVCC largely solves this

Modern databases (PostgreSQL, MySQL InnoDB) implement **MVCC (Multi-Version Concurrency Control)**: writes preserve the old version of a row alongside the new one. Reads see a historical snapshot and never need to acquire a lock — they simply read the version that existed at their transaction's start time.

```
write: update row, keep old version
read:  read old snapshot directly, no lock needed
```

With MVCC, reads and writes rarely block each other. The primary motivation for read/write split today is **capacity**, not lock contention: more replicas means more connection pools, more parallel read throughput, and better user experience under load.

#### How replication works

The primary writes to its binlog (similar to WAL — a record of what changed). Each replica runs a background thread that asynchronously pulls the binlog and replays it locally. The primary does not wait for replicas to confirm before returning success to the app.

```
primary: write → record to binlog → return success
                      │
                      └──(async)──→ replica background thread pulls and replays
replica: read requests are served at any time, independent of sync progress
```

#### Replication lag

Because replication is async, replicas can trail the primary by milliseconds to hundreds of milliseconds. The classic symptom: a user updates their profile, refreshes immediately, and sees the old value — their read hit a replica that hadn't received the change yet.

The standard handling: route reads that must see the latest write (post-payment balance check, post-order status check) directly to the primary. Route everything else to replicas. Routing is usually handled by middleware — application code doesn't need to know which node it's talking to.

#### Primary failover

When the primary goes down, the system needs to promote a new primary — this is called **failover**.

**Option 1 — elect from replicas**: a sentinel or cluster manager detects the primary is down, picks the replica with the most up-to-date data, and promotes it. The remaining replicas switch to following the new primary.

```
primary down
    ↓
sentinel picks most up-to-date replica
    ↓
replica 1 promoted to primary → accepts writes
replica 2, 3 resync from new primary
```

Tools: Redis Sentinel, MySQL MHA.

**Option 2 — dedicated standby**: a standby instance runs alongside the primary, receives all replication but serves no traffic. On failure, the standby is promoted immediately. Fastest failover, but the standby is idle under normal conditions — a resource trade-off.

Both approaches face the same risk: **split-brain**.

#### Split-brain

The primary hasn't actually crashed — a network glitch makes the sentinel temporarily unable to reach it. The sentinel misjudges and promotes a replica. Now two primaries are accepting writes simultaneously and data diverges.

```
sentinel loses contact with primary → promotes replica 1
primary is still alive → still accepting writes

primary writes A
replica 1 (new primary) writes B
→ two diverging copies, no automatic way to merge
```

The fix is **majority voting**: failover only triggers when more than half of the nodes agree the primary is unreachable. A single node's network glitch won't reach majority. This is why high-availability clusters use an odd number of nodes (3 or 5) — so a clear majority always exists.

#### Replication lag vs read uncommitted

These look similar but operate at different levels:

| | read uncommitted | replication lag |
|---|---|---|
| scope | within a single DB instance | across two separate machines |
| cause | isolation level setting | async network replication |
| data seen | another transaction's uncommitted data | already-committed data not yet synced |

Replication lag exposes **committed but not-yet-propagated** data — the write succeeded on the primary, the replica just hasn't received it yet.

---

### CAP theorem

Primary-replica replication is a concrete example of the trade-off CAP describes.

**CAP states that a distributed system can satisfy at most two of three properties:**

- **C (Consistency)**: every read returns the most recent write
- **A (Availability)**: every request receives a response (no errors or timeouts)
- **P (Partition tolerance)**: the system continues operating when the network between nodes breaks

#### Why only two

If the network between primary and replica breaks (a partition):

- **To preserve C**: the replica isn't sure its data is current, so it must refuse read requests — sacrificing A.
- **To preserve A**: the replica continues serving reads, but the data may be stale — sacrificing C.

There is no third option.

#### Primary-replica replication chooses AP

Async replication trades consistency for availability: replicas always respond (A), but data may lag behind the primary (weaker C). This is a deliberate choice — most internet workloads can tolerate brief inconsistency but cannot tolerate request failures.

#### The real choice is CP vs AP

Partition tolerance (P) is non-negotiable in distributed systems — network interruptions are normal, not exceptional. The practical question is: **when a partition occurs, do you sacrifice C or A?**

- **CP**: preserve consistency, reject requests when nodes can't agree. Example: ZooKeeper.
- **AP**: preserve availability, allow temporary inconsistency. Example: Cassandra, primary-replica read/write split.

CAP is not a design-time menu of options. It describes the forced trade-off when a partition actually happens.

---

### Sharding

Read/write split solves concurrency but not data volume. When a single table grows to hundreds of millions of rows, queries slow down and a single machine's disk starts to fill up. Sharding splits the data itself.

#### Vertical sharding (split by business domain)

Move different business tables into separate databases:

```
before: one large DB
  users, products, orders, payments all in one place

after:
  user DB    → users table
  product DB → products table
  order DB   → orders, payments tables
```

This isn't primarily about disk space — adding disks is easier than resharding. The real value is isolation:

- **Connection isolation**: each DB has its own connection pool. Total connection capacity multiplies across databases.
- **Fault isolation**: order DB getting hammered no longer affects user login or product browsing.
- **Independent scaling**: high-write DBs can get NVMe SSDs; read-heavy DBs can use cheaper hardware.

Vertical sharding is also the foundation of microservices — each service owns its own DB.

#### Horizontal sharding (split by data volume)

After vertical sharding, a single table (e.g. orders) may still have hundreds of millions of rows. Split it by a shard key:

```
orders table (300M rows)
    ↓  shard by user_id % 3
orders_0  (user_id % 3 = 0)
orders_1  (user_id % 3 = 1)
orders_2  (user_id % 3 = 2)
```

Queries include the shard key, so routing is direct — no full table scan. A database middleware layer handles routing transparently so application code doesn't change.

#### Problem 1: cross-DB JOIN

Before sharding, joining orders and users is a single SQL query. After sharding, the two tables live in different databases — the DB engine can't reach across them.

The only option is to split it into two queries and join in the application layer:

```
step 1: query order DB   → SELECT * FROM orders WHERE user_id = 42
step 2: query user DB    → SELECT name FROM users WHERE id = 42
step 3: application code joins the two results
```

Downsides:
- Two network round-trips instead of one in-memory join
- N+1 problem: joining 1000 orders means 1000 separate user lookups
- Aggregation queries (GROUP BY across both tables) must be assembled in application code

The implication for design: tables that are frequently joined together should stay in the same database. Sharding tables that join heavily carries a high ongoing cost.

#### Problem 2: distributed transactions

Before sharding, placing an order (insert order + decrement stock) is one ACID transaction. After sharding, these live in different databases. A local transaction can't span machines.

**Two-phase commit (2PC)**: a coordinator asks all participants to prepare (lock resources, don't commit yet), then issues a final commit or rollback once all confirm ready.

```
coordinator → order DB:  "ready?" → "yes, row locked"
coordinator → stock DB:  "ready?" → "yes, row locked"
coordinator → both:      "commit"
```

Problem: locks are held across two network round-trips. Under high concurrency, these held locks queue up large numbers of requests. If the coordinator crashes between phases, participants hold locks indefinitely — the system stalls.

**Saga**: each step has a corresponding compensating operation. Execute steps forward; on failure, run compensations in reverse to undo completed steps.

```
step 1: insert order       → compensation: delete order
step 2: decrement stock    → compensation: increment stock
```

No long-held locks — each step commits immediately. But compensation logic must be written manually, and the system passes through intermediate inconsistent states during execution.

**Message queue (eventual consistency)**: don't require both operations to succeed simultaneously. Insert the order and publish a message to the queue; the stock service consumes the message and decrements stock asynchronously.

```
user places order
  → order DB: INSERT ✅
  → queue: publish "decrement stock for order X" ✅
  → return success to user

(moments later)
  → stock service consumes message
  → UPDATE stock -1 ✅
```

The queue guarantees the message is eventually consumed even if the stock service is temporarily down. This is **eventual consistency** — the two operations are not atomic, but both are guaranteed to complete.

| | message queue | Saga | 2PC |
|---|---|---|---|
| consistency | eventual | eventual | strong |
| failure handling | retry until success | run compensations | coordinator rollback |
| complexity | low | high (write compensations) | medium |
| performance | high | high | low (held locks) |

**The architect's priority order**: first, design strongly-related data to live in the same database and avoid cross-DB transactions entirely. If unavoidable, prefer message queues for eventual consistency. Use 2PC only when strong consistency is a hard requirement (e.g. financial transfers where partial completion is unacceptable).

#### Problem 3: choosing the shard key

The shard key is effectively irreversible. Once data is distributed by a given key, changing it requires migrating the entire dataset — a massive operation that's nearly impossible to do without downtime.

**Rule: shard by the dimension you query most.** If 90% of queries are "get orders for user X", shard by `user_id`. That query routes directly to one shard. Queries that don't include the shard key must scatter across all shards.

**Conflicting query patterns**: if both "orders by user" and "orders by merchant" are high-frequency, no single shard key satisfies both. The common solution is to store two copies — one sharded by `user_id`, one by `merchant_id` — and write to both. Trade storage for query performance.

**Data skew**: shard keys must distribute data evenly. Sharding by city gives some shards 100× more data than others. Sharding by `user_id` hash distributes uniformly.

**Growth direction**: time-based sharding (by `order_date`) makes archiving easy but concentrates all new writes in the current period's shard. Key-based sharding distributes writes evenly but complicates archival. Neither is universally better — pick based on the actual query and write patterns.

---

### Horizontal scaling and load balancing

When a single app server hits its ceiling, the solution is to run multiple identical instances and distribute traffic across them. This is horizontal scaling.

**The prerequisite: stateless application.** If session data lives in one server's local memory, a user whose next request lands on a different server loses their session. Three ways to fix this:

- **Sticky sessions**: the load balancer remembers which server each user last hit and always routes them there. Simple, but if that server dies the session is lost, and load distribution becomes uneven.
- **Centralised session store (Redis)**: move session out of local memory into Redis. Every server reads from the same store — any server can handle any request. This is the standard approach.
- **JWT (stateless token)**: encode user identity into a signed token sent to the client. Servers verify the signature on each request with no shared store needed. Trade-off: a token cannot be invalidated before it expires (logout just means the client discards it).

Once the app is stateless, any instance can handle any request and instances can be added or removed freely.

#### Load balancing strategies

**Round robin**: requests are distributed in turn — A, B, C, A, B, C... Simple and works well when request processing times are uniform.

**Least connections**: the next request goes to whichever server currently has the fewest active connections. Better than round robin when request durations vary widely (e.g. a mix of fast and slow queries).

**Consistent hashing**: hash a request attribute (user ID, IP) to determine which server handles it. The same user always lands on the same server — useful when servers maintain local caches, since cache hits are only possible if the request returns to the same machine.

#### Why consistent hashing, not modulo

Naive modulo (`user_id % N`) breaks on resize. With 3 servers, `42 % 3 = 0`. Add a fourth: `42 % 4 = 2`. Nearly every user remaps to a different server — all local caches invalidate simultaneously and the DB gets hammered.

Consistent hashing places both servers and requests on a ring spanning 0 to 2³²:

```
        A
   ┌────┘────┐
   │         │
   C         B
   └─────────┘
```

Each request hashes onto the ring, then routes clockwise to the first server it encounters. Adding server D between C and B affects only the requests that previously mapped to the arc from C to D — everything else is unchanged.

```
before:  ... C ──────────────── B ...   (entire arc hits B)
after:   ... C ───── D ───────── B ...  (C→D arc hits D, D→B arc still hits B)
```

Only 1/N of requests remap on a resize, versus nearly all with modulo.

**Virtual nodes** handle uneven ring distribution. Each physical server is placed at multiple points on the ring (A1, A2, A3...). With enough virtual nodes the arc lengths equalise and load stays balanced even with few real servers.

**Connection assignment is availability-based, not random.** The pool hands out whichever connection is currently idle. The application doesn't know or care which connection it gets — they're interchangeable.

The exception is explicit transactions. Once a `BEGIN` is issued, the pool locks that connection to the request for the entire transaction. The session state (open transaction, locks held) is tied to that specific connection and cannot be transferred mid-flight.

```
outside a transaction:   request may get any idle connection each time
inside a transaction:    same connection held from BEGIN to COMMIT/ROLLBACK
```
