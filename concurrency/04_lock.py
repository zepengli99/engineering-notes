"""
04 - Lock (fixing the race condition)

Same broken increment from 03b, but now protected by a Lock.
A Lock guarantees only one thread can be inside the "with lock" block at a time.
Everyone else waits outside until the lock is released.
"""

import threading
import time

counter = 0
lock = threading.Lock()   # one lock shared by all threads


def increment_safe(n: int):
    global counter
    for _ in range(n):
        with lock:           # only one thread enters here at a time
            temp = counter   # READ  }
            time.sleep(0)    # yield } -- danger window, but now protected
            counter = temp + 1  # WRITE }
        # lock is released here automatically, next thread can enter


THREADS = 5
OPS_PER_THREAD = 100
EXPECTED = THREADS * OPS_PER_THREAD

print("=" * 55)
print("  Lock demo — race condition fixed")
print(f"  {THREADS} threads × {OPS_PER_THREAD} increments = {EXPECTED} expected")
print("=" * 55)

for run in range(1, 4):
    counter = 0
    threads = [
        threading.Thread(target=increment_safe, args=(OPS_PER_THREAD,))
        for _ in range(THREADS)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    print(f"  Run {run}: counter = {counter:>5}   correct={counter == EXPECTED}   ({elapsed:.2f}s)")

print()
print(f"  Expected: {EXPECTED}")
print()
print("How lock works:")
print()
print("  Thread A        Thread B        Thread C")
print("  with lock: ✓    with lock: …    with lock: …")
print("  (enters)        (blocked,       (blocked,")
print("                   waiting)        waiting)")
print("  ...runs...")
print("  releases lock")
print("                  with lock: ✓")
print("                  (enters)")
print("                  ...runs...")
print("                  releases lock")
print("                                  with lock: ✓")
print("                                  (enters)")
print()
print("Cost:")
print("  Threads now take TURNS — less concurrent.")
print("  Notice the time is much longer than 03b.")
print("  Correctness and speed are often in tension.")
