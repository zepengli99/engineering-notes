# 06b_penetration_fixed.py
# Fix for Cache Penetration: cache the null result.
#
# When DB returns null, write a sentinel value to cache with a short TTL.
# Subsequent requests hit the cached null instead of going to DB.
#
# Short TTL for null: the data might be created later (new user registers).
# Long TTL would mean legitimate new data isn't visible for a long time.

import asyncio
import json
import time

from common import db_query, get_redis

CONCURRENT_REQUESTS = 50
MISSING_KEY = "user:99999"
TTL_SECONDS = 60
NULL_TTL_SECONDS = 30    # short TTL: null data might become real soon

NULL_SENTINEL = "__null__"

r = get_redis()
stats = {"hits": 0, "null_hits": 0, "db_queries": 0, "waited": 0}
latencies = []


async def get_user(key: str):
    cached = await r.get(key)

    if cached == NULL_SENTINEL:
        stats["null_hits"] += 1
        return None

    if cached:
        stats["hits"] += 1
        return json.loads(cached)

    # cache miss — try to acquire lock
    lock_key = f"lock:{key}"
    acquired = await r.set(lock_key, "1", nx=True, px=10_000)

    if acquired:
        # winner: query DB, write result (null or real), release lock
        data = await db_query(key)
        stats["db_queries"] += 1
        if data:
            await r.setex(key, TTL_SECONDS, json.dumps(data))
        else:
            await r.setex(key, NULL_TTL_SECONDS, NULL_SENTINEL)
        await r.delete(lock_key)
        return data
    else:
        # waiters: poll until winner writes result
        stats["waited"] += 1
        while True:
            await asyncio.sleep(0.002)
            cached = await r.get(key)
            if cached is not None:
                return None if cached == NULL_SENTINEL else json.loads(cached)
            if not await r.exists(lock_key):
                break
        return None


async def handle_request():
    t0 = time.perf_counter()
    await get_user(MISSING_KEY)
    latencies.append((time.perf_counter() - t0) * 1000)


async def main():
    await r.delete(MISSING_KEY)

    print(f"Querying '{MISSING_KEY}' — does not exist in DB")
    print(f"{CONCURRENT_REQUESTS} concurrent requests...\n")

    wall_start = time.perf_counter()
    await asyncio.gather(*[handle_request() for _ in range(CONCURRENT_REQUESTS)])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    print(f"Cache hits (null) : {stats['null_hits']}  <- cached null, no DB query")
    print(f"Waited (mutex)    : {stats['waited']}  <- locked out, polled, got null from cache")
    print(f"DB queries        : {stats['db_queries']}  (was {CONCURRENT_REQUESTS} without fix)")
    print(f"Wall time         : {wall_ms:.0f}ms")
    print(f"Avg latency       : {sum(latencies)/len(latencies):.1f}ms")
    print()
    print(f"Trade-off: null cached for {NULL_TTL_SECONDS}s.")
    print(f"If user:99999 is created in DB within {NULL_TTL_SECONDS}s,")
    print(f"requests will still see null until the cached null expires.")

    await r.aclose()


asyncio.run(main())
