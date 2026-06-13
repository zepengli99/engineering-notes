# System Design

Architecture evolution notes — how a system grows from a single server to a cloud-native platform, and why each step exists.

The core pattern repeats: a bottleneck appears, a targeted fix is added, the fix creates a new bottleneck. Every component here exists to solve a specific, real problem.

```
single server
    └─ app + DB on one machine → resource contention
app / DB separation
    └─ dedicate one machine per role → independent scaling
multi-level cache
    └─ DB hammered by reads → intercept hot reads in memory
DB read/write split
    └─ read capacity ceiling → replicas handle reads, primary handles writes
sharding
    └─ single table too large → distribute data across machines
CDN + reverse proxy
    └─ distant users slow, servers exposed → edge caching + shielded entry point
search engine + NoSQL
    └─ relational DB struggles with full-text search and flexible schemas
distributed architecture + microservices
    └─ monolith too big to change, scale, or own → split by business domain
message queue
    └─ synchronous service chains cascade failures → async decoupling
containers + K8s
    └─ hundreds of services, environment hell → package once, run anywhere, auto-manage
cloud-native
    └─ on-prem capacity planning is waste → elastic resources, pay per use
```

---

## Architect's rules

1. **No best architecture — only the most appropriate for the current business.** A startup should start with a single machine. Architecture follows business growth, not the other way around.

2. **Architecture evolution trades complexity for performance.** Every layer added — cache, load balancer, message queue — is a new moving part. Add it only when the pain it solves outweighs the complexity it introduces.

3. **Always optimise for five goals: high performance, high availability, scalability, extensibility, security.**

---

## Concepts

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

### CDN and reverse proxy

#### CDN (Content Delivery Network)

Users far from the data centre experience high latency — a round trip from Guangzhou to a Beijing server takes tens of milliseconds just from physics. Adding more servers doesn't help; the data still has to travel the same distance.

CDN is a solution (not a specific technology) sold by providers (Cloudflare, AWS CloudFront, Alibaba Cloud CDN). You buy access to their global network of edge nodes and point your domain at it. The provider handles everything else.

**How it works — pull-on-demand caching:**

```
first user in Guangzhou requests logo.png
    ↓
DNS resolves to nearest Guangzhou edge node
    ↓
node has no cache (cold) → fetches from origin data centre
    ↓
node caches the file, returns to user

every subsequent Guangzhou user
    ↓
edge node returns cached file directly — origin never contacted
```

CDN is only for **static assets** (JS, CSS, images, video) — content that rarely changes and can be cached. Dynamic requests (place order, query balance) must reach the origin server. When static assets are updated, you explicitly purge the CDN cache so nodes re-fetch from origin.

CDN is effectively the geographic dimension of caching — the same core idea as Redis or browser cache, applied to network distance rather than data access speed.

```
browser cache     → zero network cost
CDN edge node     → milliseconds (geographically close)
local/Redis cache → microseconds (within data centre)
database          → milliseconds (disk)
```

Every layer asks the same question: can we avoid sending this request further than necessary?

#### Reverse proxy

A reverse proxy sits between the public internet and the internal app servers. All traffic hits the proxy first; app servers live on a private network and are never directly exposed.

```
user → reverse proxy (public) → app servers (private network)
```

**What it handles centrally:**

- **Hides real server IPs**: attackers can only see the proxy's IP. DDoS hits the proxy, not app servers.
- **SSL termination**: HTTPS decryption happens once at the proxy. App servers receive plain HTTP — no certificates to configure per server, no CPU spent on crypto. When a certificate expires, update it in one place.
- **Load balancing**: all traffic passes through here anyway, so request distribution is a natural fit. Nginx handles both roles simultaneously.
- **Rate limiting and auth**: enforced at the single entry point rather than duplicated across every app server.

**SSL/TLS in brief**: certificates prove a server is who it claims to be, issued by a trusted Certificate Authority (CA). The browser verifies the certificate, negotiates a session key using asymmetric encryption, then encrypts all further communication with that key. Without a certificate, a man-in-the-middle can impersonate the server and intercept credentials.

With a reverse proxy, only the proxy needs a certificate. New app servers can be added without any SSL configuration.

**Does the reverse proxy become a bottleneck?** Rarely. Nginx is event-driven and async — a single instance handles tens of thousands of concurrent connections while doing only lightweight work (forwarding, SSL, auth). App servers saturate long before Nginx does. At extreme scale, Nginx itself can be clustered with DNS round-robin or Anycast routing in front of it.

```
user → CDN (static assets returned immediately)
          ↓ dynamic requests only
       reverse proxy (SSL termination, auth, rate limiting, load balancing)
          ↓
       app server cluster
          ↓
       database
```

---

### Search engines and NoSQL

#### Why relational databases struggle in certain scenarios

MySQL excels at precise queries with known structure:

```sql
SELECT * FROM orders WHERE user_id = 42
SELECT * FROM users WHERE email = 'foo@bar.com'
```

Two structural limitations drive the need for specialised databases:

**1. Full-text and fuzzy search**

MySQL's B+ tree index is sorted by complete field values. It can answer "starts with X" but not "contains X" — there's no anchor point to start from, so the engine must scan every row.

```sql
LIKE 'iPhone%'    -- can use index (prefix match, has an anchor)
LIKE '%phone%'    -- full table scan (no anchor, index useless)
```

Other common index-killing patterns:
- Functions on indexed columns: `WHERE YEAR(created_at) = 2024` — B+ tree stores raw values, not computed results
- Implicit type casting: `WHERE phone = 13800138000` on a VARCHAR column — triggers implicit conversion
- Skipping a composite index prefix: index on `(a, b, c)`, querying `WHERE b = 2` — index is ordered by `a` first, skipping it makes it unusable

**2. Rigid schema**

MySQL requires every row to have the same structure defined at table creation. Storing entities with varying attributes (normal users vs merchants vs overseas users) means either many NULL columns or complex multi-table JOINs. Schema changes on large tables are expensive operations.

#### Search engine (Elasticsearch)

ES builds an **inverted index** — instead of "what words does this document contain", it stores "which documents contain this word":

```
documents:
  doc 1001: "Samsung Galaxy S24 phone"
  doc 1002: "Apple iPhone 15 phone"

inverted index:
  samsung → [1001]
  phone   → [1001, 1002]
  apple   → [1002]
```

Searching "phone" looks up the inverted index directly — no table scan. Searching "apple phone" tokenises to ["apple", "phone"], intersects the results, and returns 1002.

**Typical architecture**: MySQL stores the source of truth. ES stores a subset of fields needed for search and listing pages. A CDC pipeline (Canal for MySQL, Debezium for PostgreSQL) monitors the binlog/WAL and syncs changes to ES via Kafka. This sync is async — eventual consistency (AP).

```
write → MySQL
           ↓ binlog/WAL
        Canal/Debezium → Kafka → ES

search request  → ES (returns id + display fields)
detail request  → MySQL (full row by id)
```

#### NoSQL

NoSQL ("Not Only SQL") is a family of databases that trade relational structure for flexibility or scale. Each type solves a specific problem MySQL handles poorly.

**Document DB (MongoDB)**: stores JSON documents with no fixed schema. Each document can have different fields. Good for rapidly-changing data structures or entities with varying attributes.

```json
{ "id": 1, "name": "Alice", "email": "alice@example.com" }
{ "id": 2, "name": "Bob", "shop_name": "Bob's Store", "license_no": "xxx" }
```

Trade-off: no joins, no strong transactions — consistency is the application's responsibility.

**Key-value DB (Redis)**: the familiar cache. Simplest possible structure, nanosecond access.

**Wide-column DB (HBase, Cassandra)**: built for massive write throughput and time-series queries — user behaviour logs, monitoring metrics, message history. Natively distributed, data auto-sharded across nodes. Trade-off: no complex queries, only key lookups and range scans.

**Graph DB (Neo4j)**: stores nodes and edges natively. Traversing relationship networks ("friends of friends of friends") is a native operation — in MySQL it requires N self-joins that get exponentially slower with depth. Used for social graphs, recommendation engines, fraud detection.

**One-line selection guide:**

```
precise queries, transactions          → MySQL
full-text search, fuzzy search         → Elasticsearch
cache, simple key-value                → Redis
flexible schema, document storage      → MongoDB
high-volume writes, time-series data   → HBase / Cassandra
relationship networks                  → Neo4j
```

No single database handles everything well. Large systems typically run several simultaneously, each doing what it's best at.

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

### Distributed architecture

#### Why monoliths break down at scale

A monolith bundles all business logic into one deployable unit. Three pain points emerge as the system grows:

**Deployment coupling**: fixing a bug in the user module requires redeploying the entire application — all modules go down together, and the blast radius of a bad deploy is the whole system.

**Team interference**: multiple teams editing the same codebase cause constant merge conflicts. One team's change can silently break another team's feature.

**Coarse-grained scaling**: the order module is under load during a sale, but scaling requires adding capacity for the entire monolith — user and product modules get extra machines they don't need.

The fix: split along business boundaries. Each service owns its domain, deploys independently, and scales independently.

```
user service    ← owns everything user-related
product service ← owns everything product-related
order service   ← owns everything order-related
payment service ← owns everything payment-related
```

#### RPC (Remote Procedure Call)

Before splitting, cross-module calls are in-process function calls. After splitting, they become network calls across machines. RPC frameworks make remote calls look like local function calls — serialisation, connection management, retries, and deserialisation are all handled by the framework.

```python
# without RPC: manual HTTP wiring
response = requests.post("http://order-service/createOrder",
                         json={"user_id": 42, "product_id": 1001})
result = response.json()

# with RPC: feels like a local call
order = order_service.createOrder(user_id=42, product_id=1001)
```

#### Service registry and discovery

RPC solves *how* to call. Service discovery solves *where* to call. Hardcoding IP addresses breaks as soon as a service scales or restarts on a new machine.

A **registry** is a shared directory of running services:

```
order service starts → registers: { "order-service": "10.0.0.1:8080" }
user service calls order service:
  1. ask registry: "where is order-service?"
  2. registry returns: ["10.0.0.1:8080", "10.0.0.2:8080"]
  3. user service picks one and calls it
```

Services register on startup and send periodic heartbeats. If a heartbeat stops, the registry removes that instance from the list automatically. New instances appear in the list as soon as they register. Application code never needs to change.

Common registries: Consul, Nacos, Etcd, Zookeeper.

#### Microservice governance

With many services, the problems shift from "how to build" to "how to keep stable and observable."

**Circuit breaker**

Synchronous service calls create dependency chains. One slow downstream service holds threads in every upstream caller — the slowness propagates up the chain until the whole system stalls.

A circuit breaker mimics an electrical fuse. Three states:

```
closed (normal)  → requests pass through, error rate tracked
    ↓ error rate exceeds threshold
open (tripped)   → requests immediately take fallback path, downstream not called
    ↓ after recovery window
half-open (probe) → small fraction of requests let through to test downstream
    ↓ downstream healthy
closed (normal)
```

**Degradation** is the fallback when the circuit is open — return a reasonable default rather than an error:

```
recommendation service down → return default popular items (not blank)
reviews service down        → return "reviews temporarily unavailable" (not 500)
```

Users see reduced functionality, not a crashed page.

**Rate limiting**

Circuit breaking is reactive (trips after failure). Rate limiting is proactive (caps requests before they overwhelm a service).

- **Token bucket**: tokens are added to a bucket at a fixed rate. Each request consumes one token. Empty bucket = request rejected. Allows short bursts (accumulated tokens can be spent at once).
- **Sliding window**: count requests in the last N seconds. Smoother than a fixed window — no spike at window boundaries.

**Distributed tracing**

A single user request fans out across many services. When something is slow or broken, tracing identifies exactly where.

Every request is assigned a globally unique **Trace ID** that propagates through every service call:

```
request arrives → generate trace_id = "xyz789"
gateway logs:         [xyz789] request received
order service logs:   [xyz789] processing started
stock service logs:   [xyz789] inventory query took 2000ms  ← bottleneck found
logistics logs:       [xyz789] shipment created
```

Aggregating all logs with the same trace ID reconstructs the full call chain — latency at each hop, which service errored, where time was spent. Common tools: Jaeger, Zipkin.

```
rate limiting    → proactive: cap traffic at entry point before damage
circuit breaker  → reactive: isolate failing downstream, protect upstream
distributed tracing → diagnostic: find where in the chain the problem is
```

#### Message queue

When services call each other synchronously, they form a chain where every step waits for the previous one. One slow or failing service stalls the entire chain.

A message queue breaks this: producers drop a message and move on immediately. Consumers read at their own pace.

```
order service → drop message → return "order placed"
                     ↓ (async)
              stock service consumes → deduct stock
              logistics service consumes → create shipment
              notification service consumes → send SMS
```

**Three benefits:**

- **Async decoupling**: order service doesn't wait for stock, logistics, or notifications. User sees a response immediately.
- **Fault isolation**: notification service crashes → messages accumulate in the queue → order flow unaffected. Service recovers → resumes consuming.
- **Peak shaving**: traffic spike dumps messages into the queue; downstream services drain at their sustainable rate. The queue absorbs the burst so backends never see it directly.

**Not everything can be async.** Steps that require an immediate answer must stay synchronous:

```
must be sync:   check if stock is available, validate coupon
can be async:   deduct stock, create shipment, send notification
```

**Three problems message queues introduce:**

**1. Message loss** — can happen at three stages:

```
producer → queue → consumer
   ↑          ↑        ↑
stage 1    stage 2   stage 3
```

- Stage 1: producer sends, network drops it. Fix: queue sends ACK; producer retries until ACK received.
- Stage 2: message in queue memory, not yet on disk, queue crashes. Fix: persist messages to disk on arrival (Kafka does this by default).
- Stage 3: consumer receives message, crashes before finishing. Fix: consumer only sends ACK after processing completes; queue redelivers if no ACK.

**2. Duplicate messages** — consumer processes a message, crashes before sending ACK, queue redelivers, consumer processes it again.

Fix: **idempotency via unique message ID**. Before processing, check whether this ID has already been handled:

```
receive message (message_id = "abc123")
    ↓
check Redis: processed:abc123 exists? → yes → skip, send ACK
                                      → no  → process business logic
                                               write processed:abc123 = 1
                                               send ACK
```

Critical detail: the business operation and the "mark as processed" write must be **atomic** — otherwise a crash between the two leaves no record, and the next delivery re-executes the business logic.

```sql
BEGIN
  UPDATE stock SET count = count - 1 WHERE id = 1;
  INSERT INTO processed_messages (message_id) VALUES ('abc123');  -- unique index
COMMIT
```

If the INSERT fails (duplicate key), the transaction rolls back — the message was already handled. This is ACID atomicity applied at the application layer.

**3. Message ordering** — Kafka spreads messages across partitions for throughput. Messages from the same user can land in different partitions and be consumed out of order.

Fix: route all messages for the same business entity to the same partition using the entity's ID as the partition key:

```
user_id=42 messages → partition 2  (always)
user_id=99 messages → partition 5  (always)
```

Within a partition, order is guaranteed. Across different users, order doesn't matter.

Common queues: Kafka (high throughput, log/stream processing), RabbitMQ (feature-rich, general business messaging), RocketMQ (Alibaba, mature for e-commerce).

#### Zookeeper as a registry

Zookeeper is a distributed tree of nodes (znodes), similar to a filesystem:

```
/services/
    order-service/
        10.0.0.1:8080   ← ephemeral node
        10.0.0.2:8080   ← ephemeral node
    user-service/
        10.0.0.3:8080   ← ephemeral node
```

Services create **ephemeral nodes** on startup. Ephemeral nodes are tied to the connection that created them — when the connection drops (service crashes), the node disappears automatically. No manual cleanup needed.

Callers read the child nodes under a service path to get the address list, then do their own load balancing (round-robin or random).

**Watch mechanism**: instead of polling Zookeeper on every call, callers register a Watch — "notify me when this path changes." Zookeeper pushes a notification when a node is added or removed; the caller updates its local cache. Watches are one-shot: after firing they must be re-registered.

```
register Watch on /services/order-service/
    ↓
(nothing, waiting)
    ↓
order service instance crashes → node deleted
    ↓
Zookeeper pushes notification
    ↓
caller updates local cache + re-registers Watch
```

Watch is publish-subscribe applied to distributed state — the same pattern as frontend event listeners or message queue subscriptions. It's also used for distributed config (config changes propagate instantly), distributed locks (waiters watch the lock node), and leader election (followers watch the leader node).

**Zookeeper is CP**: writes require majority-node confirmation (ZAB protocol). Under a network partition, the minority partition refuses requests rather than returning potentially stale data. For service discovery, Nacos (configurable AP) is now more common — brief inconsistency in the address list is acceptable, but an unavailable registry is not.

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

---

### Containers and Kubernetes

#### Docker

Microservices mean hundreds of independent services, each needing its own environment. Configuring dependencies and runtime settings per machine is slow, error-prone, and produces the classic "works on my machine" problem.

The fix: package the service and everything it needs to run into a single **image**:

```
code + dependencies + runtime + config → image
```

An image runs identically on any machine — dev laptop, test server, production node. No environment configuration needed on the target machine.

Containers are much lighter than virtual machines. A VM emulates an entire OS; a container shares the host kernel and starts in seconds rather than minutes.

#### Kubernetes (K8s)

Docker solves packaging. K8s solves managing hundreds of containers across a fleet of machines:

- **Scheduling**: decides which machine each container runs on based on available resources
- **Auto-scaling**: detects traffic increase → adds containers; traffic drops → removes them
- **Self-healing**: container crashes → automatically restarted; machine dies → containers migrated to healthy nodes
- **Rolling deploys**: new version gradually replaces old, zero downtime

```
before sale:        10 order-service containers
traffic spikes    → K8s scales to 100 containers automatically
sale ends         → K8s scales back to 10
container crashes → K8s restarts it within seconds
```

```
Docker → solves "how to package and run a single service"
K8s    → solves "how to manage hundreds of containers"
```

---

### Cloud-native

Running services on rented cloud VMs is not cloud-native — it's just a traditional architecture in someone else's data centre, still managed manually.

**The core problem with on-prem**: hardware must be provisioned for peak load but sits mostly idle. Buying 10× capacity for a one-day sale wastes 9× the cost year-round.

Cloud platforms turn compute into a utility:

```
normal day: use 10 machines, pay for 10
sale day:   scale to 100 machines, pay for 100
sale ends:  scale back to 10, stop paying for the rest
```

**Cloud-native** means designing the system from the start to exploit this elasticity — with containers, K8s auto-scaling, and microservices fine-grained enough that individual services scale independently. The engineering team never thinks about physical machines.

Four pillars:
- **Containerisation**: every service is a portable image
- **Dynamic orchestration (K8s)**: auto-schedule, auto-scale, auto-heal
- **Microservices**: fine-grained enough for per-service independent scaling
- **DevOps + CI/CD**: code commit triggers automated build, test, deploy — minutes from change to production

Cloud-native is where the architecture evolution ends: a system that automatically handles any traffic level, pays only for what it uses, and requires minimal operational intervention.

---

### AI-integrated write flows

Modern systems increasingly involve AI API calls as part of a multi-step write flow — e.g. a user fills a form across three steps: basic info, AI-generated content, then vector DB ingestion. This introduces problems that classic CRUD never had.

#### Why this is different

**External cost per step**: calling an AI API costs money. Unlike a pure DB operation, there is no free rollback — if the AI call succeeds but a later step fails, the money is already spent.

**Wildly uneven step latency**: step 1 is milliseconds; step 2 (AI API) may take several seconds; step 3 (vector ingestion) is also slow. Wrapping all three in a single DB transaction holds a connection and locks for the entire AI call duration — catastrophic under concurrency.

A large transaction is not viable. Saga alone is insufficient — compensating SQL can delete DB records but cannot reclaim API spend.

#### The core goal: detect conflicts before spending money

The right framing is not "how do we roll back after a conflict" but **"how do we prevent entering step 2 if step 1 would ultimately fail."**

**Pre-emptive reservation with optimistic locking**

Step 1 derives a `resource_id` from the form's key business fields — only requests that are genuinely duplicates share the same key:

```python
# only blocks requests with identical username + email; all others proceed independently
resource_id = sha256(f"{username}:{email}".encode()).hexdigest()
```

Step 1 then writes basic info and creates a `reservation` record that claims this resource:

```sql
INSERT INTO reservations (id, user_id, resource_id, status, expires_at)
VALUES (uuid, 42, '<hash>', 'pending', now() + interval '5 minutes');
```

The `reservation` table has a partial unique index on `resource_id` scoped to active statuses:

```sql
CREATE UNIQUE INDEX idx_active_reservation
  ON reservations(resource_id)
  WHERE status IN ('pending', 'completed');
```

If another request with the same key fields already holds an active reservation, step 1 hits the index and fails immediately — before any AI call is made. Requests with different key fields hash to different `resource_id` values and proceed in parallel without any interference.

**Soft-delete over hard-delete for compensation**

Rather than deleting rows on failure, mark them with a terminal status and let a background job handle cleanup. This preserves an audit trail and keeps the compensation path to a cheap UPDATE:

```
reservation.status:  pending -> completed   (happy path)
                             -> failed       (step 2 or 3 failed)
                             -> expired      (background job, user abandoned mid-flow)
```

The background job runs periodically:

```sql
-- mark abandoned flows as expired
UPDATE reservations SET status = 'expired'
WHERE status = 'pending' AND expires_at < now();

-- hard-delete after retention window (e.g. 7 days for audit)
DELETE FROM reservations
WHERE status IN ('failed', 'expired') AND updated_at < now() - interval '7 days';
```

The partial unique index only covers `pending` and `completed` rows, so a `failed` or `expired` reservation does not block a new attempt for the same resource.

**Idempotency key on the AI call**

Network timeouts on the AI API call are ambiguous — did the call execute or not? Retrying without protection may bill twice.

Most AI APIs (OpenAI, Anthropic) accept an `idempotency_key`. Retrying with the same key returns the cached result without re-executing:

```python
idempotency_key = hash(user_id + session_id + "step_2")
response = ai_client.generate(prompt, idempotency_key=idempotency_key)
```

**Saga compensation for genuine failures**

If step 2 or step 3 genuinely fails after exhausting retries, run compensations in reverse — marking rows as `failed` rather than deleting them:

```
step 3 fails → mark vector entry failed (or delete if not yet committed)
step 2 fails → mark AI result failed
step 1 compensate → mark reservation failed, background job cleans up later
```

Compensation handles cleanup, not cost recovery. The pre-emptive reservation is what prevents unnecessary spend.

#### Full flow

```
step 1: derive resource_id = hash(username + email)
        write basic info + reservation (status=pending, expires_at=now+5min)
        → same hash already active? reject immediately, no AI call made
        → different hash? proceeds independently, no interference
        → success: continue

step 2: call AI API with idempotency_key
        → timeout? retry with same key, no double billing
        → hard failure after retries? mark reservation failed, stop
        → success: write AI result to DB, continue

step 3: ingest into vector DB
        → failure? mark AI result + reservation failed, background job cleans up
        → success: mark reservation completed

background job (runs periodically):
        expired pending reservations → mark expired
        failed/expired rows past retention window → hard delete
```

See [ai_write_flow.py](ai_write_flow.py) for a runnable example.

#### Workflow engines as an alternative

Hand-writing the state machine and Saga compensation is tedious. Managed workflow engines — **AWS Step Functions**, **Temporal** — solve the orchestration plumbing: they persist execution state after every step, handle retries with configurable backoff, and trigger compensation branches automatically on failure. Your code becomes independent functions (Lambdas); the engine wires them together and resumes from the exact step after a crash.

What the engine does not solve: the business logic inside each step — conflict detection, idempotency key generation, the reservation data model — is still your responsibility. The engine provides a durable `run_flow`; you still write the steps.
