# 04b_stampede_fixed.py
# Fix for Cache Stampede: distributed mutex via Redis SET NX.
#
# When multiple processes miss simultaneously, only one acquires the lock
# and queries the DB. Others wait (poll) until the cache is populated.
#
# SET NX (Set if Not eXists) is atomic in Redis — exactly one process
# wins the race. The rest become waiters.

import json
import multiprocessing as mp
import time

import redis

FAKE_DB = {
    f"user:{i}": {"id": i, "name": f"User_{i}", "score": i * 10}
    for i in range(1, 101)
}

DB_BASE_LATENCY_S = 0.050
DB_OVERLOAD_THRESHOLD = 10
DB_OVERLOAD_PENALTY_S = 0.002

NUM_SERVERS = 20
KEY = "user:42"
LOCK_KEY = f"lock:{KEY}"
TTL_SECONDS = 60
LOCK_TIMEOUT_MS = 10_000  # safety net: lock auto-expires after 10s


def worker(key, lock_key, ttl, barrier, active_queries, aq_lock, result_queue):
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    barrier.wait()  # all processes start simultaneously

    t0 = time.perf_counter()

    # Step 1: check cache
    cached = r.get(key)
    if cached:
        result_queue.put(("hit", (time.perf_counter() - t0) * 1000))
        r.close()
        return

    # Step 2: miss — try to acquire lock (atomic, exactly one wins)
    acquired = r.set(lock_key, "1", nx=True, px=LOCK_TIMEOUT_MS)

    if acquired:
        # Winner: query DB, write back, release lock
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

        r.delete(lock_key)
        result_queue.put(("db_query", (time.perf_counter() - t0) * 1000))

    else:
        # Waiters: poll until cache is populated
        while True:
            time.sleep(0.002)  # 2ms poll interval
            cached = r.get(key)
            if cached:
                break
            if not r.exists(lock_key):  # lock gone but cache empty (winner crashed)
                break
        result_queue.put(("waited", (time.perf_counter() - t0) * 1000))

    r.close()


if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.delete(KEY)
    r.delete(LOCK_KEY)
    r.close()

    barrier = mp.Barrier(NUM_SERVERS)
    active_queries = mp.Value("i", 0)
    aq_lock = mp.Lock()
    result_queue = mp.Queue()

    processes = [
        mp.Process(
            target=worker,
            args=(KEY, LOCK_KEY, TTL_SECONDS, barrier, active_queries, aq_lock, result_queue),
        )
        for _ in range(NUM_SERVERS)
    ]

    print(f"Hot key '{KEY}' just expired.")
    print(f"{NUM_SERVERS} server processes incoming... (mutex fix)")
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

    hits     = sum(1 for s, _ in results if s == "hit")
    db_queries = sum(1 for s, _ in results if s == "db_query")
    waited   = sum(1 for s, _ in results if s == "waited")
    latencies = [ms for _, ms in results]

    print(f"Cache hits     : {hits}")
    print(f"DB queries     : {db_queries}  (was {NUM_SERVERS} in 04_stampede.py)")
    print(f"Waited (mutex) : {waited}  <- missed, locked out, polled, then hit cache")
    print(f"Wall time      : {wall_ms:.0f}ms")
    if latencies:
        print(f"Avg latency    : {sum(latencies) / len(latencies):.1f}ms")
