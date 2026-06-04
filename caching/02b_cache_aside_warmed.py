import asyncio
import json
import time

from common import db_query, get_redis

CONCURRENT_REQUESTS = 50
TARGET_KEY = "user:42"
TTL_SECONDS = 60

r = get_redis()
stats = {"hits": 0, "misses": 0}
latencies = []


async def get_user(key: str) -> dict | None:
    cached = await r.get(key)
    if cached:
        stats["hits"] += 1
        return json.loads(cached)

    data = await db_query(key)
    if data:
        await r.setex(key, TTL_SECONDS, json.dumps(data))
    stats["misses"] += 1
    return data


async def handle_request():
    t0 = time.perf_counter()
    await get_user(TARGET_KEY)
    latencies.append((time.perf_counter() - t0) * 1000)


async def main():
    await r.delete(TARGET_KEY)
    await get_user(TARGET_KEY)  # warm up: one miss to populate cache
    stats["hits"] = 0
    stats["misses"] = 0

    wall_start = time.perf_counter()
    await asyncio.gather(*[handle_request() for _ in range(CONCURRENT_REQUESTS)])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    print(f"Requests   : {CONCURRENT_REQUESTS} (same key: {TARGET_KEY})")
    print(f"Cache hits : {stats['hits']}")
    print(f"Cache miss : {stats['misses']}")
    print(f"DB queries : {stats['misses']}  (was {CONCURRENT_REQUESTS} without cache)")
    print(f"Wall time  : {wall_ms:.0f}ms")
    print(f"Avg req    : {sum(latencies) / len(latencies):.1f}ms per request")

    await r.aclose()


asyncio.run(main())
