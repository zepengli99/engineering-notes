import asyncio
import json
import time

from common import db_query, get_redis

CONCURRENT_REQUESTS = 50
TARGET_KEY = "user:42"
TTL_SECONDS = 60

r = get_redis()


async def get_user(key: str, stats: dict) -> dict | None:
    cached = await r.get(key)
    if cached:
        stats["hits"] += 1
        return json.loads(cached)

    data = await db_query(key)
    if data:
        await r.setex(key, TTL_SECONDS, json.dumps(data))
    stats["misses"] += 1
    return data


async def run_batch(label: str):
    stats = {"hits": 0, "misses": 0}
    latencies = []

    async def handle_request():
        t0 = time.perf_counter()
        await get_user(TARGET_KEY, stats)
        latencies.append((time.perf_counter() - t0) * 1000)

    wall_start = time.perf_counter()
    await asyncio.gather(*[handle_request() for _ in range(CONCURRENT_REQUESTS)])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    print(f"[{label}]")
    print(f"  Cache hits : {stats['hits']}")
    print(f"  Cache miss : {stats['misses']}")
    print(f"  DB queries : {stats['misses']}")
    print(f"  Wall time  : {wall_ms:.0f}ms")
    print(f"  Avg req    : {sum(latencies) / len(latencies):.1f}ms")
    print()


async def main():
    await r.delete(TARGET_KEY)

    await run_batch("cold cache — first batch")   # some misses, DB gets queried
    await run_batch("warm cache — second batch")  # all hits, DB not touched

    await r.aclose()


asyncio.run(main())
