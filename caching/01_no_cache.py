import asyncio
import time

from common import db_query

CONCURRENT_REQUESTS = 50
TARGET_KEY = "user:42"

latencies = []


async def handle_request():
    t0 = time.perf_counter()
    await db_query(TARGET_KEY)
    latencies.append((time.perf_counter() - t0) * 1000)


async def main():
    wall_start = time.perf_counter()
    await asyncio.gather(*[handle_request() for _ in range(CONCURRENT_REQUESTS)])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    print(f"Requests   : {CONCURRENT_REQUESTS} (same key: {TARGET_KEY})")
    print(f"DB queries : {CONCURRENT_REQUESTS}  <- every request hit DB")
    print(f"Wall time  : {wall_ms:.0f}ms")
    print(f"Avg req    : {sum(latencies) / len(latencies):.1f}ms per request")


asyncio.run(main())
