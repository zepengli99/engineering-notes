import asyncio

from common import FAKE_DB, db_query, get_redis


async def main():
    r = get_redis()
    await r.ping()
    await r.flushall()
    await r.aclose()

    print("Redis connected — all keys cleared")
    print(f"Fake DB: {len(FAKE_DB)} records, 5ms latency per query")

    result = await db_query("user:1")
    print(f"DB sanity check: user:1 -> {result}")


asyncio.run(main())
