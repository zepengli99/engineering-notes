"""
02 - Threading for IO-bound tasks

Same three downloads as script 01, but now run in parallel threads.

Key question to watch:
  - Script 01 total IO time: ~3.3s (tasks ran one by one)
  - This script total IO time: ~?  (tasks run at the same time)
"""

import threading
import time


START = time.time()


def ts():
    return f"[+{time.time() - START:.2f}s]"


def io_task(name: str, duration: float):
    # Exact same function as script 01 — nothing changed here.
    print(f"{ts()} {name:12} started   (thread: {threading.current_thread().name})")
    time.sleep(duration)
    print(f"{ts()} {name:12} finished")


# ── run with threads ──────────────────────────────────────────────────────────

print("=" * 55)
print("  Threaded IO — tasks run concurrently")
print("=" * 55)

# Create one thread per task.
# Each thread will call io_task() independently.
threads = [
    threading.Thread(target=io_task, args=("Download-A", 1.0), name="T-A"),
    threading.Thread(target=io_task, args=("Download-B", 1.5), name="T-B"),
    threading.Thread(target=io_task, args=("Download-C", 0.8), name="T-C"),
]

# .start() launches the thread — it does NOT wait for it to finish.
for t in threads:
    t.start()

print(f"{ts()} {'main':12} all threads started, now waiting...")

# .join() blocks the main thread until that thread finishes.
# Without this, the program would exit before threads complete.
for t in threads:
    t.join()

print()
print(f"Total time: {time.time() - START:.2f}s")
print("=" * 55)
print()
print("Notice:")
print("  All three threads started almost simultaneously.")
print("  Total time ≈ slowest task (1.5s), not sum (3.3s).")
print()
print("Why does this work?")
print("  time.sleep() releases the thread — it just waits.")
print("  While T-A waits, Python switches to T-B, then T-C.")
print("  The CPU is free to run other threads during IO waits.")
print()
print("What is a thread?")
print("  A thread is a unit of execution inside a process.")
print("  All threads here share the same memory (same process).")
print("  The OS switches between them rapidly — 'concurrent'.")
