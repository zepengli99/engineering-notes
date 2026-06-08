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
