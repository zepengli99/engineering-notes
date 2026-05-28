"""
03 - Race Condition

Multiple threads share one counter and increment it simultaneously.
Expected result: 1,000,000
Actual result:   ??? (less, and different every run)

This is a race condition — threads "race" to read/write the same memory.
"""

import threading
import time

START = time.time()


def ts():
    return f"[+{time.time() - START:.2f}s]"


# ── shared state ──────────────────────────────────────────────────────────────

counter = 0   # every thread reads and writes this same variable


def increment(n: int, thread_name: str):
    global counter
    for _ in range(n):
        # This looks like one operation, but it's actually THREE steps:
        #   1. READ  — load counter's current value into a register
        #   2. ADD   — compute value + 1
        #   3. WRITE — store the result back to counter
        #
        # The OS can pause this thread between any two steps
        # and switch to another thread. If that thread also does
        # step 1 before we finish step 3, both threads read the
        # same old value and write the same new value — one increment is lost.
        counter += 1


# ── run ───────────────────────────────────────────────────────────────────────

THREADS = 5
OPS_PER_THREAD = 200_000
EXPECTED = THREADS * OPS_PER_THREAD

print("=" * 55)
print("  Race condition demo")
print(f"  {THREADS} threads × {OPS_PER_THREAD:,} increments = {EXPECTED:,} expected")
print("=" * 55)

# Run 3 times to show the result changes every time
for run in range(1, 4):
    counter = 0

    threads = [
        threading.Thread(target=increment, args=(OPS_PER_THREAD, f"T-{i}"))
        for i in range(THREADS)
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    lost = EXPECTED - counter
    print(f"  Run {run}: counter = {counter:>9,}  "
          f"lost = {lost:>7,}  ({elapsed:.2f}s)")

print()
print(f"  Expected: {EXPECTED:,}")
print()
print("Notice:")
print("  Each run gives a DIFFERENT wrong answer.")
print("  The lost increments vary — this is non-deterministic.")
print("  Bugs like this are extremely hard to reproduce and debug.")
