"""
06 - Multiprocessing: true parallelism for CPU-bound tasks

Script 05 showed threads can't speed up CPU-bound work (GIL).
Here we use processes instead — each has its own Python interpreter
and its own GIL, so they run truly in parallel on separate CPU cores.

Cost: processes don't share memory. Spawning is heavier than threads.
"""

import multiprocessing
import time


def cpu_task(n: int):
    return sum(i * i for i in range(n))


N = 100_000_000


def run(label: str, worker_count: int):
    t0 = time.time()

    if worker_count == 1:
        cpu_task(N)
    else:
        with multiprocessing.Pool(processes=worker_count) as pool:
            # Split N into equal chunks, one per process.
            chunk = N // worker_count
            pool.map(cpu_task, [chunk] * worker_count)

    elapsed = time.time() - t0
    print(f"  {label:<35} {elapsed:.2f}s")
    return elapsed


if __name__ == "__main__":
    # multiprocessing on Windows requires this guard —
    # otherwise each spawned process re-runs the whole script.

    cpu_count = multiprocessing.cpu_count()

    print("=" * 55)
    print(f"  Multiprocessing — CPU cores available: {cpu_count}")
    print("=" * 55)

    t1 = run("1 process  (baseline)",          worker_count=1)
    t2 = run("2 processes",                    worker_count=2)
    t4 = run("4 processes",                    worker_count=4)
    run(     f"{cpu_count} processes (all cores)", worker_count=cpu_count)

    print()
    print(f"  2 processes speedup: {t1/t2:.1f}x  (ideal: 2.0x)")
    print(f"  4 processes speedup: {t1/t4:.1f}x  (ideal: 4.0x)")
    print()
    print("Why not perfectly 2x / 4x?")
    print("  - Process spawn overhead (each starts a fresh Python)")
    print("  - Data serialization (args/results go through pickle)")
    print("  - OS scheduling isn't perfectly even")
    print()
    print("Threads vs Processes summary:")
    print("  Threads:   shared memory, lightweight, GIL-limited")
    print("  Processes: isolated memory, heavier, truly parallel")
    print()
    print("Rule of thumb:")
    print("  IO-bound  → threads (or async, next script)")
    print("  CPU-bound → processes")
