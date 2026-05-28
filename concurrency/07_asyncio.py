"""
07 - Async / Coroutines (asyncio)

Same three IO tasks as script 02 (threading), but now with a single thread.
No threads, no processes — just one thread and an event loop.

Key question: how can one thread handle multiple tasks "at the same time"?
"""

import asyncio
import time

START = time.time()


def ts():
    return f"[+{time.time() - START:.2f}s]"


# ── coroutines ────────────────────────────────────────────────────────────────

async def io_task(name: str, duration: float):
    print(f"{ts()} {name:12} started")
    await asyncio.sleep(duration)   # "I'm waiting — event loop, run someone else"
    print(f"{ts()} {name:12} finished")


# ── what is a coroutine? ──────────────────────────────────────────────────────
#
# A regular function runs to completion and returns.
# A coroutine (async def) can PAUSE at an "await" point and let other
# coroutines run. When its awaited thing is ready, it resumes.
#
# It's cooperative: the coroutine voluntarily yields control.
# (Threads are preemptive: the OS can pause them at any time.)
#
# Think of it like a single chef managing multiple dishes:
#   - put dish A in oven (waiting)     ← await
#   - start chopping for dish B        ← running
#   - oven beeps for A                 ← A resumes
#   - plate dish A                     ← running
# One person, multiple things "in progress" at once.


async def main():
    print("=" * 55)
    print("  asyncio — single thread, cooperative multitasking")
    print(f"  thread count: 1 (always)")
    print("=" * 55)

    # asyncio.gather() schedules all coroutines and runs them concurrently.
    # They all start immediately, and the event loop switches between them
    # whenever one hits an "await".
    await asyncio.gather(
        io_task("Download-A", duration=1.0),
        io_task("Download-B", duration=1.5),
        io_task("Download-C", duration=0.8),
    )

    print()
    print(f"Total time: {time.time() - START:.2f}s")
    print("=" * 55)
    print()
    print("Same result as threading (script 02): ~1.5s, not ~3.3s.")
    print("But achieved with a single thread — no GIL fights, no race conditions.")
    print()
    print("How the event loop works:")
    print()
    print("  t=0.00  start A, B, C — all three are 'in progress'")
    print("  t=0.00  event loop: nobody ready, wait...")
    print("  t=0.80  C's sleep done → resume C, C prints finished")
    print("  t=1.00  A's sleep done → resume A, A prints finished")
    print("  t=1.50  B's sleep done → resume B, B prints finished")
    print()
    print("Threads vs Async (both good for IO-bound):")
    print("  Threads: OS switches between them (preemptive)")
    print("           each thread has its own stack (~8MB)")
    print("           race conditions possible")
    print()
    print("  Async:   coroutine yields voluntarily (cooperative)")
    print("           all share one stack, one thread")
    print("           no race conditions on simple variables")
    print("           can handle 10,000s of concurrent tasks cheaply")


asyncio.run(main())
