"""
05 - GIL: why threads don't help CPU-bound tasks

Script 02 showed threads making IO tasks ~3x faster.
Here we try the same with CPU-bound tasks.

Expectation (if threads were truly parallel): 2x faster with 2 threads
Reality (GIL):                                same speed or slower
"""

import threading
import time


def cpu_task(n: int):
    """Pure computation — no sleep, no IO. Burns real CPU cycles."""
    return sum(i * i for i in range(n))


N = 5_000_000  # total work units


# ── baseline: one thread does all the work ────────────────────────────────────

print("=" * 55)
print("  GIL demo — CPU-bound tasks")
print("=" * 55)

t0 = time.time()
cpu_task(N)
single_time = time.time() - t0
print(f"  1 thread  (all {N:,} in one go):     {single_time:.2f}s")


# ── 2 threads: split the work in half ─────────────────────────────────────────
# If truly parallel, each thread does N/2 → total time should be ~half.
# With GIL: both threads take turns on the same core → still ~same total time.

t0 = time.time()
threads = [
    threading.Thread(target=cpu_task, args=(N // 2,)),
    threading.Thread(target=cpu_task, args=(N // 2,)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
two_thread_time = time.time() - t0
print(f"  2 threads (each does {N//2:,}): {two_thread_time:.2f}s")


# ── 4 threads ─────────────────────────────────────────────────────────────────

t0 = time.time()
threads = [
    threading.Thread(target=cpu_task, args=(N // 4,))
    for _ in range(4)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
four_thread_time = time.time() - t0
print(f"  4 threads (each does {N//4:,}):   {four_thread_time:.2f}s")

print()
print(f"  Expected speedup with 2 threads: ~{single_time:.2f}s / 2 = ~{single_time/2:.2f}s")
print(f"  Actual   speedup with 2 threads: {two_thread_time:.2f}s")
print()
print("Why?")
print("  GIL = Global Interpreter Lock.")
print("  CPython allows only ONE thread to execute Python")
print("  bytecode at any moment — even on a multi-core CPU.")
print()
print("  IO tasks:  thread spends most time WAITING (not using CPU)")
print("             → GIL is released during wait → other threads run")
print("             → real concurrency benefit")
print()
print("  CPU tasks: thread is ALWAYS using CPU, never releases GIL")
print("             → other threads are stuck waiting for the lock")
print("             → no benefit, sometimes slower (lock overhead)")
print()
print("  Visual:")
print("  IO:  T1[=wait=======][run][=wait=====][run]")
print("       T2      [run][=wait=========][run]")
print("       (gaps let other threads in)")
print()
print("  CPU: T1[run][run][run][run][run][run][run]")
print("       T2                                   [run][run]...")
print("       (T1 never yields → T2 can't get in)")
print()
print("  Solution: multiprocessing — next script.")
