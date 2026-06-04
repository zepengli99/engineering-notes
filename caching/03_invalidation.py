import asyncio
import json

from common import FAKE_DB, db_query, get_redis

TTL_SECONDS = 60
KEY = "user:42"


async def main():
    r = get_redis()
    await r.delete(KEY)

    async def get_user(key: str) -> tuple[dict | None, str]:
        cached = await r.get(key)
        if cached:
            return json.loads(cached), "cache"
        data = await db_query(key)
        if data:
            await r.setex(key, TTL_SECONDS, json.dumps(data))
        return data, "db"

    async def update_user_score(key: str, new_score: int):
        FAKE_DB[key]["score"] = new_score
        await r.delete(key)

    data, source = await get_user(KEY)
    print(f"read  [{source}]  score={data['score']}")

    old_score = data["score"]
    await update_user_score(KEY, 9999)
    print(f"write  DB updated: score {old_score} -> 9999, cache key deleted")

    data, source = await get_user(KEY)
    print(f"read  [{source}]  score={data['score']}  <- fresh immediately")

    data, source = await get_user(KEY)
    print(f"read  [{source}]  score={data['score']}")

    print()
    print("Pattern: write to DB -> delete cache key -> next read refetches")
    print("TTL=60s is a safety net only, not the primary invalidation mechanism.")

    await r.aclose()


asyncio.run(main())
