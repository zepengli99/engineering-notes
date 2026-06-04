# 04c_logical_ttl.py
# Alternative stampede fix: Logical TTL.
#
# Instead of letting the key expire (which causes a cold miss),
# store the expiry time INSIDE the value. Physical TTL is very long.
# When logically expired: serve stale data immediately + one process refreshes.
#
# Mutex approach (04b):  miss → lock → 1 queries DB → 19 wait (blocked)
# Logical TTL (this):    hit  → stale data served → 1 refreshes → 19 skip

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
LOGICAL_TTL_S = 60      # data is "fresh" for 60s
PHYSICAL_TTL_S = 86400  # key lives for 1 day — never actually expires


def make_entry(data: dict) -> str:
    return json.dumps({"data": data, "expire_at": time.time() + LOGICAL_TTL_S})


def do_db_query(key, active_queries, aq_lock) -> dict | None:
    with aq_lock:
        active_queries.value += 1
        concurrent = active_queries.value

    overhead = max(0, concurrent - DB_OVERLOAD_THRESHOLD) * DB_OVERLOAD_PENALTY_S
    time.sleep(DB_BASE_LATENCY_S + overhead)

    with aq_lock:
        active_queries.value -= 1

    return FAKE_DB.get(key)


def worker(key, barrier, active_queries, aq_lock, result_queue):
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    barrier.wait()  # all processes start simultaneously

    t0 = time.perf_counter()
    raw = r.get(key)

    if not raw:
        # cold start: key truly missing (rare after warm-up)
        data = do_db_query(key, active_queries, aq_lock)
        r.set(key, make_entry(data), ex=PHYSICAL_TTL_S)
        result_queue.put(("cold", (time.perf_counter() - t0) * 1000))
        r.close()
        return

    entry = json.loads(raw)

    if time.time() < entry["expire_at"]:
        # logically fresh — return immediately
        result_queue.put(("fresh", (time.perf_counter() - t0) * 1000))
        r.close()
        return

    # logically expired — try to be the one to refresh
    refresh_lock = f"refresh:{key}"
    acquired = r.set(refresh_lock, "1", nx=True, px=30_000)

    if acquired:
        # winner: refresh DB, update cache, release lock
        data = do_db_query(key, active_queries, aq_lock)
        r.set(key, make_entry(data), ex=PHYSICAL_TTL_S)
        r.delete(refresh_lock)
        result_queue.put(("refreshed", (time.perf_counter() - t0) * 1000))
    else:
        # others: return stale data immediately, don't wait
        result_queue.put(("stale", (time.perf_counter() - t0) * 1000))

    r.close()


if __name__ == "__main__":
    # seed cache with logically expired data
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    expired_entry = json.dumps({
        "data": FAKE_DB[KEY],
        "expire_at": time.time() - 1,   # already expired 1s ago
    })
    r.set(KEY, expired_entry, ex=PHYSICAL_TTL_S)
    r.delete(f"refresh:{KEY}")
    r.close()

    barrier = mp.Barrier(NUM_SERVERS)
    active_queries = mp.Value("i", 0)
    aq_lock = mp.Lock()
    result_queue = mp.Queue()

    processes = [
        mp.Process(
            target=worker,
            args=(KEY, barrier, active_queries, aq_lock, result_queue),
        )
        for _ in range(NUM_SERVERS)
    ]

    print(f"Key '{KEY}' is logically expired (but still in cache).")
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

    by_type = {"cold": [], "fresh": [], "stale": [], "refreshed": []}
    for status, ms in results:
        by_type[status].append(ms)

    print(f"Stale served immediately : {len(by_type['stale'])}  "
          f"avg {sum(by_type['stale'])/len(by_type['stale']):.1f}ms"
          if by_type['stale'] else f"Stale served immediately : 0")
    print(f"Background refresh       : {len(by_type['refreshed'])}  "
          f"avg {sum(by_type['refreshed'])/len(by_type['refreshed']):.1f}ms  <- DB queried"
          if by_type['refreshed'] else f"Background refresh       : 0")
    print(f"DB queries               : {len(by_type['refreshed'])}  "
          f"(was {NUM_SERVERS} in 04_stampede.py)")
    print(f"Wall time                : {wall_ms:.0f}ms")

    all_latencies = [ms for _, ms in results]
    print(f"Avg latency              : {sum(all_latencies)/len(all_latencies):.1f}ms")
    print()
    print("19 processes returned stale data in ~1ms.")
    print("1 process refreshed the cache in ~55ms — without blocking anyone.")
