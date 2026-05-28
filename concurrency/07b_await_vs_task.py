"""
07b - await vs create_task vs gather

Three patterns, same tasks, completely different timing.
Watch the timestamps to understand what each one actually does.
"""

import asyncio
import time

START = time.time()


def ts():
    return f"[+{time.time() - START:.2f}s]"


async def job(name: str, duration: float):
    print(f"  {ts()} {name} started")
    await asyncio.sleep(duration)
    print(f"  {ts()} {name} finished")
    return f"{name}_result"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: plain await — sequential
# ─────────────────────────────────────────────────────────────────────────────
async def pattern_sequential():
    print("[ Pattern 1: plain await — sequential ]")
    global START; START = time.time()

    await job("A", 1.0)   # stop here, wait for A to finish
    await job("B", 1.0)   # then stop here, wait for B to finish

    print(f"  total: {time.time()-START:.2f}s  (expected ~2.0s)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: create_task — concurrent, collect results later
# ─────────────────────────────────────────────────────────────────────────────
async def pattern_create_task():
    print("[ Pattern 2: create_task — concurrent ]")
    global START; START = time.time()

    # create_task: hands job to the event loop and returns IMMEDIATELY.
    # The coroutine is now scheduled — it will start at the next await point.
    task_a = asyncio.create_task(job("A", 1.0))
    task_b = asyncio.create_task(job("B", 1.0))
    # At this point neither A nor B has started yet — we haven't yielded control.

    print(f"  {ts()} both tasks created (not started yet)")

    # The moment we hit any await, the event loop runs the scheduled tasks.
    result_a = await task_a   # yield → event loop starts A and B → wait for A
    result_b = await task_b   # A already done, B likely done too
    print(f"  results: {result_a}, {result_b}")
    print(f"  total: {time.time()-START:.2f}s  (expected ~1.0s)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: gather — concurrent, same as create_task but shorter syntax
# ─────────────────────────────────────────────────────────────────────────────
async def pattern_gather():
    print("[ Pattern 3: gather — concurrent ]")
    global START; START = time.time()

    # gather internally does create_task for each coroutine,
    # then awaits all of them together.
    results = await asyncio.gather(job("A", 1.0), job("B", 1.0))
    print(f"  results: {results}")
    print(f"  total: {time.time()-START:.2f}s  (expected ~1.0s)\n")


# ─────────────────────────────────────────────────────────────────────────────
# create_task advantage: you can do other work while tasks run
# ─────────────────────────────────────────────────────────────────────────────
async def pattern_task_while_working():
    print("[ Bonus: create_task while doing other work ]")
    global START; START = time.time()

    task_a = asyncio.create_task(job("A", 1.0))  # start A in background

    # do other work while A is running
    print(f"  {ts()} doing other work...")
    await asyncio.sleep(0.3)                      # simulate other work
    print(f"  {ts()} other work done")

    await task_a                                  # now wait for A
    print(f"  total: {time.time()-START:.2f}s  (A ran while we worked)\n")


async def main():
    print("=" * 55)
    await pattern_sequential()
    await pattern_create_task()
    await pattern_gather()
    await pattern_task_while_working()

    print("Summary:")
    print("  await job()           → sequential, blocks until done")
    print("  create_task(job())    → schedules in event loop, returns Task")
    print("  await task            → blocks until that specific task done")
    print("  gather(job(), job())  → create_task × N + await all")


asyncio.run(main())

# import asyncio

# async def foo():
#     print("foo start")

# async def main():
#     task = asyncio.create_task(foo())
#     # await asyncio.sleep(0)
#     print("after create_task")

# asyncio.run(main())

# import asyncio

# async def foo():
#     asyncio.create_task(B())
#     print("foo start")
#     await asyncio.sleep(1)
#     print("foo end")

# async def A():
#     print("A start")
#     await foo()
#     print("A end")

# async def B():
#     print("B start")
#     print("B end")

# asyncio.run(A())