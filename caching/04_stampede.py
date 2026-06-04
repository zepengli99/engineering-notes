# 04_stampede.py
# Cache Stampede: multiple server processes all check a cold cache simultaneously.
# All miss, all hit the DB — stampede.
#
# Why multiprocessing, not asyncio:
#   asyncio is single-threaded. Redis GETs are serialized through the event loop.
#   The first miss writes back before most GETs are even sent, so only 1 misses.
#   Stampede is a *multi-process* problem — 20 servers hitting Redis at once.
#   Each process here = one server worker (gunicorn, uvicorn, etc.)

import json
import multiprocessing as mp
import time

import redis

FAKE_DB = {
    f"user:{i}": {"id": i, "name": f"User_{i}", "score": i * 10}
    for i in range(1, 101)
}

# 50ms models a DB already under moderate load — realistic for production.
# Needs to be > NUM_SERVERS × Redis_GET_RTT (~0.5ms) so all GETs complete
# before the first write-back, ensuring every process sees a miss.
DB_BASE_LATENCY_S = 0.050
DB_OVERLOAD_THRESHOLD = 10
DB_OVERLOAD_PENALTY_S = 0.002

NUM_SERVERS = 20
KEY = "user:42"
TTL_SECONDS = 60


def worker(key, ttl, barrier, active_queries, aq_lock, result_queue):
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    barrier.wait()  # all processes check cache at the same moment

    t0 = time.perf_counter()
    cached = r.get(key)
    if cached:
        result_queue.put(("hit", (time.perf_counter() - t0) * 1000))
        r.close()
        return

    # cache miss — query DB with load-dependent latency
    with aq_lock:
        active_queries.value += 1
        concurrent = active_queries.value

    overhead = max(0, concurrent - DB_OVERLOAD_THRESHOLD) * DB_OVERLOAD_PENALTY_S
    time.sleep(DB_BASE_LATENCY_S + overhead)
    data = FAKE_DB.get(key)
    if data:
        r.setex(key, ttl, json.dumps(data))

    with aq_lock:
        active_queries.value -= 1

    result_queue.put(("miss", (time.perf_counter() - t0) * 1000))
    r.close()


if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.delete(KEY)
    r.close()

    barrier = mp.Barrier(NUM_SERVERS)
    active_queries = mp.Value("i", 0)
    aq_lock = mp.Lock()
    result_queue = mp.Queue()

    processes = [
        mp.Process(
            target=worker,
            args=(KEY, TTL_SECONDS, barrier, active_queries, aq_lock, result_queue),
        )
        for _ in range(NUM_SERVERS)
    ]

    print(f"Hot key '{KEY}' just expired.")
    print(f"{NUM_SERVERS} server processes incoming...")
    print()

    wall_start = time.perf_counter()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    hits = sum(1 for s, _ in results if s == "hit")
    misses = sum(1 for s, _ in results if s == "miss")
    latencies = [ms for _, ms in results]

    print(f"Cache hits  : {hits}")
    print(f"Cache misses: {misses}  <- each miss -> 1 DB query")
    print(f"DB queries  : {misses}  (should have been 1)")
    print(f"Wall time   : {wall_ms:.0f}ms")
    if latencies:
        print(f"Avg latency : {sum(latencies) / len(latencies):.1f}ms")
    print()
    print("Fix: only let 1 process query the DB. Others wait for the result.")
    print("See 04b_stampede_fixed.py")
