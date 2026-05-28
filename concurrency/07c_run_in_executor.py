"""
07c - run_in_executor: running blocking code inside async

Problem: calling a blocking/CPU-bound function directly in async code
         freezes the entire event loop — all other coroutines stall.

Solution: run_in_executor offloads it to a thread pool,
          keeping the event loop free to run other coroutines.
"""

import asyncio
import time


START = time.time()


def ts():
    return f"[+{time.time() - START:.2f}s]"


def blocking_task(name: str, duration: float):
    """A plain synchronous function — no async, just blocks."""
    print(f"  {ts()} {name} started  (sync, blocking {duration}s)")
    time.sleep(duration)   # blocks the thread it runs on
    print(f"  {ts()} {name} finished")
    return f"{name}_result"


async def io_task(name: str, duration: float):
    """A normal async task to show whether the event loop stays free."""
    print(f"  {ts()} {name} started  (async)")
    await asyncio.sleep(duration)
    print(f"  {ts()} {name} finished")


# ── Problem: blocking call inside async ──────────────────────────────────────

async def demo_blocking():
    print("[ Problem: blocking_task called directly — freezes event loop ]")
    global START; START = time.time()

    # Schedule an async task first — we expect it to run concurrently.
    asyncio.create_task(io_task("async-A", 0.5))

    # This blocks the thread for 2s — event loop can't run anything else.
    blocking_task("sync-B", 2.0)

    # async-A should have finished at 0.5s, but it was stuck waiting.
    await asyncio.sleep(0)   # yield once to let async-A finish
    print(f"  total: {time.time()-START:.2f}s\n")
    print("  Notice: async-A couldn't run until sync-B finished.\n")


# ── Solution: run_in_executor ─────────────────────────────────────────────────

async def demo_executor():
    print("[ Solution: blocking_task via run_in_executor ]")
    global START; START = time.time()

    loop = asyncio.get_event_loop()

    # Schedule async-A as before.
    asyncio.create_task(io_task("async-A", 0.5))
    # c = asyncio.create_task(io_task("async-C", 3.0))

    # Offload blocking_task to a thread pool.
    # The event loop is free — async-A can run while sync-B blocks its thread.
    result = await loop.run_in_executor(None, blocking_task, "sync-B", 2.0)
    # await c  # wait for async-C to finish too

    # print(f"  result from executor: {result}")
    print(f"  total: {time.time()-START:.2f}s\n")
    print("  Notice: async-A finished at ~0.5s even though sync-B took 2s.\n")


# ── Custom thread pool ────────────────────────────────────────────────────────

async def demo_custom_pool():
    print("[ Custom ThreadPoolExecutor — control max workers ]")
    global START; START = time.time()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as pool:
    # with ThreadPoolExecutor(max_workers=2) as pool:
        loop = asyncio.get_event_loop()

        # Run multiple blocking tasks concurrently in the thread pool.
        results = await asyncio.gather(
            loop.run_in_executor(pool, blocking_task, "job-1", 1.0),
            loop.run_in_executor(pool, blocking_task, "job-2", 1.0),
            loop.run_in_executor(pool, blocking_task, "job-3", 1.0),
        )

    print(f"  results: {results}")
    print(f"  total: {time.time()-START:.2f}s  (expected ~1.0s, not ~3.0s)\n")


async def main():
    print("=" * 55)
    await demo_blocking()
    await demo_executor()
    await demo_custom_pool()

    print("Summary:")
    print("  blocking call in async  → freezes event loop")
    print("  run_in_executor(None, fn, args)")
    print("    → offload to default thread pool")
    print("    → event loop stays free")
    print("    → returns awaitable, result via await")
    print("  run_in_executor(pool, fn, args)")
    print("    → use custom pool, control max_workers")


asyncio.run(main())
