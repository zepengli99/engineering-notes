# Caching

Personal notes from working through these concepts with code. The idea is simple — put data in memory so you don't have to hit the database every time — but the failure modes are surprisingly sharp.

Scripts are meant to be run in order. Each one builds on the last.

---

## Prerequisites

```bash
# Start Redis (Docker)
docker run --rm --name redis-cache -p 6379:6379 -d redis

pip install redis

# Clear Redis before each demo
python 00_setup.py
```

---

## The 30-second decision guide

```
Read-heavy, data doesn't change often                   →  Cache-Aside + TTL + active invalidation
Hot key is about to expire, data must be fresh          →  Mutex lock (SET NX), waiters block
Hot key is about to expire, stale data is acceptable    →  Logical TTL, no blocking
Many keys expire at the same time                       →  Add jitter to TTL
Service restart flushes all cache                       →  Warm-up before opening traffic
Querying data that might not exist                      →  Cache null results + mutex
DB is already failing under load                        →  Circuit breaker + degraded response
```

---

## Scripts

| File | Concept |
|---|---|
| [00_setup.py](00_setup.py) | Connect to Redis, clear all keys |
| [01_no_cache.py](01_no_cache.py) | Baseline: 50 concurrent requests, 50 DB queries |
| [02_cache_aside.py](02_cache_aside.py) | Cache-Aside: cold then warm batch, DB queries drop to 0 |
| [02b_cache_aside_warmed.py](02b_cache_aside_warmed.py) | Pre-warmed cache: all 50 requests hit cache |
| [03_invalidation.py](03_invalidation.py) | Delete cache on write, next read refetches fresh |
| [04_stampede.py](04_stampede.py) | Hot key expires, 20 processes all miss, all hit DB |
| [04b_stampede_fixed.py](04b_stampede_fixed.py) | Mutex lock: only 1 process queries DB, others wait |
| [04c_logical_ttl.py](04c_logical_ttl.py) | Logical TTL: serve stale data, 1 background refresh |
| [05_avalanche.py](05_avalanche.py) | 100 keys expire at once, DB gets hammered |
| [05b_avalanche_jitter.py](05b_avalanche_jitter.py) | Jitter spreads expiry, DB load stays smooth |
| [06_penetration.py](06_penetration.py) | Querying missing keys — cache never helps |
| [06b_penetration_fixed.py](06b_penetration_fixed.py) | Cache null + mutex: DB queried once per missing key |
| [io_models.py](io_models.py) | Threading vs event loop: same behavior, different mechanism |

---

## Concepts

### Why cache exists

Every database query has two costs: **latency** (waiting for the response) and **connection pressure** (a database has a finite connection pool).

```
Without cache:
  50 concurrent requests for user:42
  → 50 DB queries for identical data
  → 50 connections consumed simultaneously

With cache (warm):
  Request 1 → cache miss → DB query → write to cache
  Requests 2-50 → cache hit (~0.1ms each)
  → 1 DB query total
```

> **Q I had: I added cache but avg latency didn't improve much. Why?**
> Cache doesn't always reduce per-request latency — it reduces DB load. Adding Redis introduces a network round-trip on every request. Redis is single-threaded, so 50 concurrent GETs are serialized inside Redis. This can keep wall time similar to the no-cache case. But DB queries dropped from 50 to 1. At 50,000 QPS, the DB sees 50,000 queries without cache and maybe 500 with cache — the difference between a healthy system and a crashed one.

> **Q I had: the DB can handle queries in parallel too, so why does query count matter?**
> Each query holds a connection. PostgreSQL's default connection limit is 100. Past that, new connections are rejected or queued. The DB also burns CPU for parsing, planning, and IO on every query. Multiplying queries multiplies all of this.

---

### Cache-Aside

The application owns the cache logic: check cache first, miss → query DB, write back.

```
Read path:
  1. GET key from cache
  2a. Hit  → return cached value (~0.1ms)
  2b. Miss → query DB → SET key in cache with TTL → return

Write path:
  UPDATE DB → DELETE cache key
  (next read will miss and refetch fresh data)
```

```python
async def get_user(key):
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)        # hit

    data = await db_query(key)           # miss
    await redis.setex(key, TTL, json.dumps(data))
    return data

async def update_user(key, new_data):
    await db.update(key, new_data)       # write to DB first
    await redis.delete(key)              # then invalidate cache
```

> **Q I had: on write, why delete the cache instead of updating it?**
> Deleting is safer. Updating requires the new value to be ready immediately — not always true if the write is async or involves joins. Delete forces the next read to fetch fresh data. Cost: one cache miss after each write. For most apps that's acceptable.

**TTL is a safety net, not the primary invalidation mechanism:**

Active deletion on write keeps cache fresh immediately. TTL is for cases where the deletion was missed (crash, bug, async failure).

```
TTL = 60s:   data stale for up to 60s if active deletion fails
TTL = 1s:    data always fresh, but nearly no caching benefit
```

---

### Redis as cache vs Redis as store

Same Redis, different responsibility — and different consistency guarantees.

```
Redis as cache (this chapter):
  DB is source of truth. Redis is a disposable copy.
  Write consistency is handled by DB (transactions, MVCC, locks).
  Redis only handles invalidation and read acceleration.
  Cache can be wiped and rebuilt from DB at any time.

Redis as store (rate limiting, sessions, leaderboards):
  Redis IS source of truth. No DB backup.
  Write consistency must be handled in Redis.
  Multi-step operations (GET + SET) have a race gap — same as DB lost update.
  Fix: use atomic commands (INCR, SETNX) or Lua scripts.
```

> **Q I had: Redis is single-threaded, so doesn't it prevent all race conditions?**
> Single-threaded means each *command* is atomic. But the gap between commands belongs to the application, not Redis. `GET counter` → compute in app → `SET counter new_value` has a gap where another process can read and write. This is the same lost update from the transactions chapter, at the Redis layer. Fix: `INCR` (read + increment + write as one command) or Lua scripts.

---

### Redis is single-threaded

Redis processes commands on a single thread using an event loop (epoll):

```
Main thread
  └── event loop (ae.c)
        ├── epoll_wait() monitors all client file descriptors (fds)
        ├── fd ready → read command → execute → write response
        └── back to epoll_wait
```

A **file descriptor (fd)** is the integer the OS assigns to every open "thing" — files, sockets, pipes. Each Redis client connection is a fd. `epoll` monitors all fds at once, notifying the event loop when any has data ready.

This is not coroutines. Redis is C, using OS-level IO multiplexing. Python's `asyncio` is a higher-level abstraction of the same concept (`await` = yield until fd is ready).

**Three consequences of single-threaded execution:**

**1. Every command is atomic.**
No two commands execute simultaneously. `INCR`, `SETNX`, `LPUSH` are all atomic for free. This is the foundation for distributed locks and rate limiters.

**2. Slow commands block everyone.**
One event loop, one thread. A `KEYS *` scanning 1M keys blocks all other clients for the entire duration. In production: ban `KEYS *`, use `SCAN` instead (iterates in batches, yields between each).

**3. Concurrency is at the IO level, not the execution level.**
10,000 connections can exist simultaneously (epoll manages all their fds). But commands execute one at a time — 10,000 commands at ~0.1μs each = 1ms. Users never notice.

> **Q I had: Redis 6.0 added multi-threading, doesn't that change things?**
> IO threads only — reading bytes from sockets and writing bytes back can now use multiple threads. Command execution is still single-threaded. The bottleneck was always the network IO, not the command execution.

---

### Cache Stampede (Thundering Herd)

A hot key expires while traffic is high. Every concurrent request checks cache, gets a miss, and hits the DB simultaneously.

```
t=0    hot key expires
t=0    20 servers each receive a request for that key
t=0    all 20 check cache → miss
t=0    all 20 query DB simultaneously
       DB under high concurrency → each query: 50ms + (20-10)×2ms = 70ms
       20 DB queries that should have been 1
```

**Why asyncio can't demonstrate this:**
Stampede is a multi-process problem. A single asyncio event loop serializes Redis GETs — the first miss writes back before most GETs are even sent. Real stampede happens when 20 independent server processes (each with their own event loop) all check the cache at once. The demo uses `multiprocessing` to model this accurately.

**Fix 1: Mutex lock (SET NX)**

```python
acquired = redis.set(f"lock:{key}", "1", nx=True, px=10_000)

if acquired:
    # winner: query DB, write to cache, release lock
    data = db_query(key)
    redis.setex(key, TTL, json.dumps(data))
    redis.delete(f"lock:{key}")
else:
    # waiters: poll until cache is populated
    while not (cached := redis.get(key)):
        time.sleep(0.002)
```

`SET NX` (Set if Not eXists) is atomic — exactly one process wins regardless of how many try simultaneously.

**Fix 2: Logical TTL**

Store the expiry time inside the value. Physical TTL is very long — the key never actually expires.

```python
# write
entry = {"data": data, "expire_at": time.time() + LOGICAL_TTL}
redis.set(key, json.dumps(entry), ex=86400)   # physical TTL: 1 day

# read
entry = json.loads(redis.get(key))
if time.time() > entry["expire_at"]:
    if redis.set(f"refresh:{key}", "1", nx=True, px=30_000):
        background_refresh(key)   # only one process refreshes
return entry["data"]              # all processes return immediately
```

**Comparison:**

```
Mutex lock:      data must be fresh → waiters block until refresh done
Logical TTL:     stale data is acceptable → all return immediately, one refreshes

Underlying mechanism: both use SET NX. The difference is what losers do —
  mutex: block and wait
  logical TTL: return old data immediately
```

This is the CAP theorem trade-off at the cache layer: consistency (mutex) vs availability (logical TTL).

---

### Distributed lock vs SELECT FOR UPDATE

Both prevent the same lost-update pattern. The choice depends on where the source of truth lives:

```
All operations go through the same DB  →  SELECT FOR UPDATE (DB handles it)
  (doesn't matter how many servers)

Operations span multiple services/DBs,
or the resource being protected isn't in a DB  →  Distributed lock (Redis)
```

The distributed lock is also used beyond caching: preventing duplicate cron jobs, ensuring idempotent payment processing, leader election.

> **Q I had: is a distributed lock just a Redis version of threading.Lock()?**
> Same concept, different failure modes. A thread mutex is managed by the OS — if the thread dies, the OS can clean up. A Redis lock is just a key — if the holder crashes, the key stays until TTL expires. This is why `px=10_000` (auto-expiry) is non-negotiable: it's the safety net for crashed holders. Production locks also use a unique value per holder and check before deleting, to avoid releasing someone else's lock.

---

### Cache Avalanche

Many keys expire at the same time. All corresponding requests miss simultaneously, DB gets hammered all at once.

```
Common causes:
  - Service restart flushes cache → all keys cold simultaneously
  - Batch cache population with same TTL → all expire together
  - A cache server fails → large fraction of keyspace suddenly cold
```

**Fix: TTL jitter**

```python
# dangerous: all keys expire at the same second
redis.setex(key, 60, value)

# safe: expiry spread over a window
import random
redis.setex(key, 60 + random.randint(0, 30), value)
```

Jitter converts a synchronized expiry spike into a Poisson distribution. DB sees smooth traffic instead of a pulse.

**Other defenses:**

```
Cache warming:    before opening traffic (after restart), proactively load
                  top-N hot keys. k8s readiness probe buys time for this.

Circuit breaker:  when DB error rate exceeds threshold, stop sending requests.
                  Return degraded response (stale data, empty result, retry message).
                  Three states: Closed (normal) → Open (cut off) → Half-Open (probing).
                  Jitter prevents the avalanche. Circuit breaker contains it if it starts.

Message queue:    cache misses publish to a queue, worker queries DB at controlled rate.
                  Trades latency (async) for DB protection. Better for write paths.
```

**Warm-up is space-time trade-off:**
Keeping connections open (connection pool) and data pre-loaded (cache warm-up) both pay a resource cost upfront to avoid time cost per request. More precisely: they move the initialization cost from "distributed across requests" to "paid once at startup." Latency distribution becomes predictable instead of spikey.

---

### Cache Penetration

Querying data that doesn't exist. Cache can never help — every request misses and hits the DB, which also returns null. Nothing gets written to cache, so the cycle repeats forever.

```
Common causes:
  - Malicious scan: for id in range(1_000_000): GET /user/{id}
  - Deleted data still being requested by clients
  - Application bugs referencing non-existent IDs
```

**Fix: cache the null result**

```python
data = await db_query(key)

if data:
    await redis.setex(key, TTL, json.dumps(data))
else:
    await redis.setex(key, NULL_TTL, "__null__")   # cache the absence
```

On read, check for the sentinel before treating as a miss:

```python
cached = await redis.get(key)
if cached == "__null__":
    return None    # cached null — no DB query
if cached:
    return json.loads(cached)
# else: real miss, proceed to DB
```

**NULL_TTL should be shorter than TTL**: the missing data might be created later (a new user registers). A long null TTL means legitimate new data isn't visible for a long time.

**Penetration + stampede compound:** before any null is cached, many concurrent requests can all miss simultaneously and all hit DB. Fix requires both null caching and a mutex (as in 06b).

> **Q I had: what about Bloom filters?**
> A Bloom filter answers "does this key definitely NOT exist?" with zero false negatives — if the filter says no, it's definitely not in DB. This prevents even the first DB query for non-existent keys, at the cost of some memory and the possibility of false positives (filter says "might exist" for a key that doesn't). Used when the set of valid keys is known upfront (user IDs, product IDs). For dynamic data, null caching is simpler.

---

## What this means for backend development

### The same pattern at every layer

The problems in this chapter are all variations of one pattern:

```
Concurrent access to shared state with a gap between read and write.
```

This appears identically across layers:

```
DB transactions:   read row → compute → write row         (lost update)
Redis multi-step:  GET key  → compute → SET key           (same gap)
Cache stampede:    check cache → miss → query DB → write  (same gap)
```

The fix is always the same: collapse the gap. Use atomic operations (`INCR`, `SELECT FOR UPDATE`, `SET NX`) or accept the gap and handle the consequences (logical TTL serving stale data, circuit breaker absorbing DB failures).

### Caching and MVCC

There's a structural parallel between caching and MVCC:

```
MVCC:        readers see a consistent snapshot; writers create new versions
Cache:       readers see cached data; writers invalidate or refresh

MVCC snapshot lifetime → how stale reads can be (isolation level)
Cache TTL             → how stale cached data can be

Long-lived MVCC transaction → table bloat (old versions can't be cleaned)
Long TTL                    → stale data served for longer

Read replicas are safe because MVCC guarantees readers never block writers.
Cache serves the same role at the application layer.
```

### Practical rules

**Know your source of truth.**
Redis as cache: consistency is the DB's job. Redis as store: consistency is your job. The same key can be both in different parts of the same system — be explicit about which role it plays.

**Jitter is free.**
Adding `random.randint(0, 30)` to every `setex` call costs nothing and prevents an entire failure mode. Do it by default.

**Instrument cache hit rate.**
A drop in hit rate is often the first sign of a problem (bad invalidation logic, TTL too short, memory pressure evicting keys). P99 latency rising while hit rate is fine usually points to Redis connection pool saturation, not DB load.

**Warm up before opening traffic.**
After any restart or deploy, let the cache warm before routing real users. A 30-second burn-in under synthetic traffic prevents the cold-start avalanche.

---

## Summary

| Problem | Cause | Fix |
|---|---|---|
| Stampede | Hot key expires under high traffic | Mutex (SET NX) or Logical TTL |
| Avalanche | Many keys expire simultaneously | TTL jitter + warm-up |
| Penetration | Querying non-existent data | Cache null + mutex |

| Approach | Blocks waiters | Serves stale data | Complexity |
|---|---|---|---|
| Mutex lock | Yes | No | Low |
| Logical TTL | No | Yes | Medium |
| Circuit breaker | No (degrades) | Sometimes | High |

| Tool | Use when |
|---|---|
| `SET NX` | Distributed mutex, exactly-once execution |
| `INCR` / `INCRBY` | Atomic counter (rate limiting, inventory) |
| `TTL + jitter` | Preventing synchronized expiry |
| `px=N` on locks | Safety net for crashed lock holders |
| Null caching | Preventing penetration on missing keys |
