# 05b_avalanche_jitter.py
# Fix for Cache Avalanche: add random jitter to TTL.
# Keys expire at different times, spreading DB load over a window.
#
# No jitter: 100 keys expire at t=2s → 100 simultaneous DB queries
# With jitter: 100 keys expire between t=2s and t=6s → ~20 per second

import asyncio
import json
import random
import time

from common import db_query, get_redis

NUM_KEYS = 100
BASE_TTL = 2
JITTER = 4    # random extra 0-4s → keys expire between t=2s and t=6s

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

    active += 1
    peak_concurrent = max(peak_concurrent, active)
    db_queries += 1

    t0 = time.perf_counter()
    data = await db_query(key)
    latencies.append((time.perf_counter() - t0) * 1000)

    active -= 1

    if data:
        ttl = BASE_TTL + random.randint(0, JITTER)
        await r.setex(key, ttl, json.dumps(data))


async def main():
    # populate: each key gets a different TTL
    ttls = {f"product:{i}": BASE_TTL + random.randint(0, JITTER) for i in range(NUM_KEYS)}
    await asyncio.gather(*[
        r.setex(key, ttl, json.dumps({"id": key, "price": 0}))
        for key, ttl in ttls.items()
    ])

    expired_count = sum(1 for ttl in ttls.values() if ttl == BASE_TTL)
    print(f"{NUM_KEYS} keys cached with TTL={BASE_TTL}~{BASE_TTL+JITTER}s (jitter={JITTER}s)")
    print(f"Keys expiring at t={BASE_TTL}s: ~{expired_count} out of {NUM_KEYS}")
    print(f"Waiting {BASE_TTL}s...")
    await asyncio.sleep(BASE_TTL + 0.2)

    # same burst at the same moment as 05_avalanche.py
    # but only keys with TTL == BASE_TTL have expired — the rest are still cached
    wall_start = time.perf_counter()
    await asyncio.gather(*[get_product(f"product:{i}") for i in range(NUM_KEYS)])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    print()
    print(f"DB queries      : {db_queries}  <- only {expired_count} keys expired so far")
    print(f"Peak concurrent : {peak_concurrent}")
    if latencies:
        print(f"Avg DB latency  : {sum(latencies)/len(latencies):.1f}ms  "
              f"(load penalty: +{max(0, peak_concurrent - 10)}ms)")
    print(f"Wall time       : {wall_ms:.0f}ms")
    print()
    print(f"Remaining {NUM_KEYS - db_queries} keys still cached,")
    print(f"expiring gradually over the next {JITTER}s — DB never spikes.")

    await r.aclose()


asyncio.run(main())
