"""
01 - Sequential Execution (baseline)

Run this first. All other scripts will be compared against this.

We simulate two types of tasks:
  - IO-bound:  mostly waiting (network, disk, database)
  - CPU-bound: mostly computing (hashing, encoding, math)
"""

import time


# ── helpers ──────────────────────────────────────────────────────────────────

def ts():
    """Return elapsed seconds since START, formatted as [+1.23s]"""
    return f"[+{time.time() - START:.2f}s]"


def io_task(name: str, duration: float):
    """Simulates an IO-bound task (e.g. HTTP request, DB query)."""
    print(f"{ts()} {name:12} started  (IO, will wait {duration}s)")
    time.sleep(duration)   # <- the program is blocked here, doing nothing
    print(f"{ts()} {name:12} finished")


def cpu_task(name: str, n: int):
    """Simulates a CPU-bound task (actual computation, not sleeping)."""
    print(f"{ts()} {name:12} started  (CPU, counting to {n:,})")
    total = sum(i * i for i in range(n))  # burns real CPU cycles
    print(f"{ts()} {name:12} finished  (result={total})")


# ── main ─────────────────────────────────────────────────────────────────────

START = time.time()

print("=" * 55)
print("  Sequential execution — one task at a time")
print("=" * 55)

# Three IO tasks, one after another.
# Each one blocks the program until it's done before the next starts.
io_task("Download-A", duration=1.0)
io_task("Download-B", duration=1.5)
io_task("Download-C", duration=0.8)

print()

# Two CPU tasks, one after another.
cpu_task("Compute-X", n=5_000_000)
cpu_task("Compute-Y", n=5_000_000)

print()
print(f"Total time: {time.time() - START:.2f}s")
print("=" * 55)
print()
print("Notice:")
print("  IO total should be ~3.3s (1.0 + 1.5 + 0.8)")
print("  While Download-A waits, the program does NOTHING.")
print("  This is the waste we'll fix with concurrency.")
