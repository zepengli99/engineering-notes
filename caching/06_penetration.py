# 06_penetration.py
# Cache Penetration: querying data that doesn't exist.
# Cache never helps — every request goes to DB and returns null.
# Nothing is written to cache, so the next request misses again.
#
# Common causes:
#   - Malicious scans: for i in range(999999): GET /user/{i}
#   - Deleted data still being requested by clients
#   - Application bugs referencing non-existent IDs

import asyncio
import json
import time

from common import db_query, get_redis

CONCURRENT_REQUESTS = 50
MISSING_KEY = "user:99999"   # not in FAKE_DB
TTL_SECONDS = 60

r = get_redis()
stats = {"hits": 0, "db_queries": 0}
latencies = []


async def get_user(key: str):
    cached = await r.get(key)
    if cached:
        stats["hits"] += 1
        return json.loads(cached)

    data = await db_query(key)   # returns None — key doesn't exist
    stats["db_queries"] += 1

    if data:
        await r.setex(key, TTL_SECONDS, json.dumps(data))
    # data is None: nothing written to cache
    # next request will miss again, hit DB again

    return data


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

    print(f"Cache hits  : {stats['hits']}")
    print(f"DB queries  : {stats['db_queries']}  <- all {CONCURRENT_REQUESTS} hit DB for null data")
    print(f"Wall time   : {wall_ms:.0f}ms")
    print(f"Avg latency : {sum(latencies)/len(latencies):.1f}ms")
    print()
    print("Problem: malicious scan of 1M non-existent IDs = 1M DB queries.")
    print("Fix: cache the null result. See 06b_penetration_fixed.py")

    await r.aclose()


asyncio.run(main())
