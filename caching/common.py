import asyncio

import redis.asyncio as aioredis

FAKE_DB = {
    f"user:{i}": {"id": i, "name": f"User_{i}", "score": i * 10}
    for i in range(1, 101)
}

DB_BASE_LATENCY_S = 0.050
DB_OVERLOAD_THRESHOLD = 10
DB_OVERLOAD_PENALTY_S = 0.001

_active_queries = 0


async def db_query(key: str) -> dict | None:
    """Simulates a DB query with load-dependent latency.

    asyncio is single-threaded: no locks needed.
    += 1 and -= 1 are each atomic between await points.
    """
    global _active_queries
    _active_queries += 1
    concurrent = _active_queries

    overhead = max(0, concurrent - DB_OVERLOAD_THRESHOLD) * DB_OVERLOAD_PENALTY_S
    await asyncio.sleep(DB_BASE_LATENCY_S + overhead)

    _active_queries -= 1
    return FAKE_DB.get(key)


def get_redis() -> aioredis.Redis:
    return aioredis.Redis(host="localhost", port=6379, decode_responses=True)
