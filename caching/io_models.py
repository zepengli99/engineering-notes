# io_models.py
# Two ways to handle concurrent IO: thread-per-connection vs event loop.
# Same behavior, completely different mechanism.
#
# Context: why Redis uses a single-threaded event loop instead of threads.

import asyncio
import threading
import time

NUM_CLIENTS = 5
IO_DELAY_PER_CLIENT = 0.3  # each client's data arrives 0.3s after the previous


# --- Model 1: Thread-per-connection -------------------------------------------
#
# Each client gets a dedicated thread.
# The thread blocks on recv() while waiting for data — doing nothing.
# 5 clients = 5 threads sitting idle most of the time.

def handle_client_threaded(client_id: int, io_delay: float):
    print(f"  [thread-{client_id}] blocking... waiting {io_delay:.1f}s for data")
    time.sleep(io_delay)      # thread is stuck here — OS parks it, can't do other work
    print(f"  [thread-{client_id}] data arrived -> process -> respond")
    time.sleep(0.0001)        # tiny CPU work (0.1ms)

def run_threaded():
    print(f"--- Thread-per-connection: {NUM_CLIENTS} clients ---")
    clients = [(i, i * IO_DELAY_PER_CLIENT) for i in range(1, NUM_CLIENTS + 1)]

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=handle_client_threaded, args=(cid, delay))
        for cid, delay in clients
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  wall time : {wall_ms:.0f}ms")
    print(f"  threads   : {NUM_CLIENTS}  (each blocked on IO, doing nothing while waiting)")
    print()


# --- Model 2: Event loop ------------------------------------------------------
#
# Single thread, never blocks.
# When a client's data isn't ready yet, the coroutine yields control.
# The loop immediately moves to another client.
# When data arrives (epoll fires), the loop resumes that coroutine.

async def handle_client_async(client_id: int, io_delay: float):
    print(f"  [event-loop] client-{client_id}: waiting {io_delay:.1f}s for data")
    await asyncio.sleep(io_delay)   # yields — loop handles other clients in the meantime
    print(f"  [event-loop] client-{client_id}: data arrived -> process -> respond")
    await asyncio.sleep(0.0001)

async def run_async():
    print(f"--- Event loop: {NUM_CLIENTS} clients ---")
    t0 = time.perf_counter()
    await asyncio.gather(*[
        handle_client_async(i, i * IO_DELAY_PER_CLIENT)
        for i in range(1, NUM_CLIENTS + 1)
    ])
    wall_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  wall time : {wall_ms:.0f}ms")
    print(f"  threads   : 1  (never blocked, processes each client the moment data is ready)")
    print()


# --- Run both -----------------------------------------------------------------

run_threaded()
asyncio.run(run_async())

print("-" * 60)
print(f"Same wall time. Scale to 10,000 clients:")
print(f"  threading:  10,000 threads × 8MB stack = 80GB memory")
print(f"              + context switch overhead between 10,000 threads")
print(f"  event loop: 1 thread, memory nearly flat")
print(f"              OS epoll tells the loop which socket has data")
print()
print("Redis uses the event loop model.")
print("Command execution stays single-threaded -> every command is atomic for free.")
