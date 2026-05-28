"""
03b - Race condition (forced)

counter += 1 is too fast for the GIL to release mid-operation.
We manually split READ and WRITE with a time.sleep(0), which
forces the OS to switch threads right in the danger window.

This makes the race condition visible and reproducible.
"""

import threading
import time

counter = 0


def increment_slow(n: int):
    """Explicitly broken into READ / sleep / WRITE to expose the race."""
    global counter
    for _ in range(n):
        temp = counter       # step 1: READ into local variable
        time.sleep(0)        # step 2: yield — let OS switch to another thread NOW
        counter = temp + 1   # step 3: WRITE back (using the stale temp)


THREADS = 5
OPS_PER_THREAD = 100
EXPECTED = THREADS * OPS_PER_THREAD

print("=" * 55)
print("  Forced race condition (sleep between read and write)")
print(f"  {THREADS} threads × {OPS_PER_THREAD} increments = {EXPECTED} expected")
print("=" * 55)

for run in range(1, 4):
    counter = 0
    threads = [
        threading.Thread(target=increment_slow, args=(OPS_PER_THREAD,))
        for _ in range(THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lost = EXPECTED - counter
    print(f"  Run {run}: counter = {counter:>5}   lost = {lost:>5}")

print()
print(f"  Expected: {EXPECTED}")
print()
print("Why sleep(0)?")
print("  sleep(0) doesn't actually sleep — it just tells the OS")
print("  'I'm done for now, give CPU to someone else.'")
print("  This forces a context switch right between READ and WRITE,")
print("  which is exactly the danger window.")
