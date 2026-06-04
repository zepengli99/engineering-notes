# 05_avalanche.py
# Cache Avalanche: many keys expire at the same time.
# All requests miss simultaneously, DB gets hammered all at once.
#
# Common cause: service restart flushes all cache,
# or batch-set keys all given the same TTL.

import asyncio
import json
import time

from common import db_query, get_redis

NUM_KEYS = 100
BASE_TTL = 2   # all keys expire at the same moment

r = get_redis()

peak_concurrent = 0
active = 0
db_queries = 0
latencies = []


async def get_product(key: str):
    global peak_concurrent, active, db_queries

    cached = await r.get(key)
    if cached:
        return

    # cache miss
    active += 1
    peak_concurrent = max(peak_concurrent, active)
    db_queries += 1

    t0 = time.perf_counter()
    data = await db_query(key)
    latencies.append((time.perf_counter() - t0) * 1000)

    active -= 1

    if data:
        await r.setex(key, BASE_TTL, json.dumps(data))


async def main():
    global peak_concurrent, active, db_queries

    # populate: all keys with the same TTL
    await asyncio.gather(*[
        r.setex(f"product:{i}", BASE_TTL, json.dumps({"id": i, "price": i * 10}))
        for i in range(NUM_KEYS)
    ])
    print(f"{NUM_KEYS} keys cached with TTL={BASE_TTL}s (no jitter)")
    print(f"Waiting {BASE_TTL}s for all keys to expire simultaneously...")
    await asyncio.sleep(BASE_TTL + 0.2)

    # burst: all requests fire at once, all keys already expired
    wall_start = time.perf_counter()
    await asyncio.gather(*[get_product(f"product:{i}") for i in range(NUM_KEYS)])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    print()
    print(f"DB queries      : {db_queries}  <- all {NUM_KEYS} keys expired at once")
    print(f"Peak concurrent : {peak_concurrent}")
    print(f"Avg DB latency  : {sum(latencies)/len(latencies):.1f}ms  "
          f"(load penalty: +{max(0, peak_concurrent - 10)}ms)")
    print(f"Wall time       : {wall_ms:.0f}ms")
    print()
    print("Fix: spread expiry with jitter. See 05b_avalanche_jitter.py")

    await r.aclose()


asyncio.run(main())
