"""
08 - Final comparison: threads vs multiprocessing vs asyncio

Same two workloads, three solutions.
Watch the numbers — they tell you which tool fits which job.
"""

import asyncio
import threading
import multiprocessing
import time
import httpx  # pip install httpx — real HTTP client for IO demo


# ─────────────────────────────────────────────────────────────────────────────
# Workload A: IO-bound (simulated with sleep)
# ─────────────────────────────────────────────────────────────────────────────

URLS = [0.3, 0.5, 0.8, 0.4, 0.6]   # pretend these are request durations
N_IO = len(URLS)


def io_sync(duration):
    time.sleep(duration)


async def io_async(duration):
    await asyncio.sleep(duration)


def bench_io_sequential():
    t0 = time.time()
    for d in URLS:
        io_sync(d)
    return time.time() - t0


def bench_io_threads():
    t0 = time.time()
    threads = [threading.Thread(target=io_sync, args=(d,)) for d in URLS]
    for t in threads: t.start()
    for t in threads: t.join()
    return time.time() - t0


async def bench_io_async():
    t0 = time.time()
    await asyncio.gather(*[io_async(d) for d in URLS])
    return time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Workload B: CPU-bound
# ─────────────────────────────────────────────────────────────────────────────

N_CPU = 4
CHUNK = 30_000_000


def cpu_work(n):
    return sum(i * i for i in range(n))


def bench_cpu_sequential():
    t0 = time.time()
    for _ in range(N_CPU):
        cpu_work(CHUNK)
    return time.time() - t0


def bench_cpu_threads():
    t0 = time.time()
    threads = [threading.Thread(target=cpu_work, args=(CHUNK,)) for _ in range(N_CPU)]
    for t in threads: t.start()
    for t in threads: t.join()
    return time.time() - t0


def bench_cpu_multiprocessing():
    t0 = time.time()
    with multiprocessing.Pool(processes=N_CPU) as pool:
        pool.map(cpu_work, [CHUNK] * N_CPU)
    return time.time() - t0


async def bench_cpu_async():
    """CPU work inside async — runs in executor to avoid blocking event loop."""
    t0 = time.time()
    loop = asyncio.get_event_loop()
    await asyncio.gather(*[
        loop.run_in_executor(None, cpu_work, CHUNK)
        for _ in range(N_CPU)
    ])
    return time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print(f"  IO-bound: {N_IO} tasks, durations {URLS}")
    print(f"  Sequential baseline (sum): ~{sum(URLS):.1f}s")
    print("=" * 55)
    t1 = bench_io_sequential()
    t2 = bench_io_threads()
    t3 = await bench_io_async()
    print(f"  sequential:      {t1:.2f}s")
    print(f"  threads:         {t2:.2f}s   ({t1/t2:.1f}x faster)")
    print(f"  asyncio:         {t3:.2f}s   ({t1/t3:.1f}x faster)")

    print()
    print("=" * 55)
    print(f"  CPU-bound: {N_CPU} tasks × {CHUNK:,} iterations")
    print("=" * 55)
    t1 = bench_cpu_sequential()
    t2 = bench_cpu_threads()
    t3 = bench_cpu_multiprocessing()
    t4 = await bench_cpu_async()
    print(f"  sequential:      {t1:.2f}s")
    print(f"  threads:         {t2:.2f}s   ({t1/t2:.1f}x)  ← GIL, no real gain")
    print(f"  multiprocessing: {t3:.2f}s   ({t1/t3:.1f}x)  ← true parallel")
    print(f"  async+executor:  {t4:.2f}s   ({t1/t4:.1f}x)  ← threads under the hood")

    print()
    print("=" * 55)
    print("  Decision guide")
    print("=" * 55)
    print()
    print("  IO-bound, high concurrency  → asyncio")
    print("  IO-bound, moderate          → threads")
    print("  CPU-bound                   → multiprocessing")
    print("  blocking lib inside async   → run_in_executor")
    print()
    print("  Threads:          shared memory, lightweight, GIL-limited on CPU")
    print("  Multiprocessing:  isolated memory, heavy spawn, truly parallel")
    print("  Asyncio:          single thread, cooperative, cheap at scale")


if __name__ == "__main__":
    asyncio.run(main())
