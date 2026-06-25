# Concurrency, Parallelism, Async & Threads

Personal notes from working through these concepts with code. Not a survey — just things I actually ran, got confused by, and eventually understood.

Scripts are meant to be run in order. Each one builds on the last.

---

## The 30-second decision guide

```
IO-bound, high concurrency (web servers, APIs)  →  asyncio
IO-bound, moderate concurrency                  →  threads
CPU-bound (image processing, computation)       →  multiprocessing
Blocking library call inside async code         →  run_in_executor
```

---

## Scripts

| File | Concept |
|---|---|
| [01_sequential.py](01_sequential.py) | Baseline: one task at a time |
| [02_threads_io.py](02_threads_io.py) | Threads for IO: concurrent, ~max(durations) |
| [03_race_condition.py](03_race_condition.py) | Shared memory goes wrong |
| [03b_race_forced.py](03b_race_forced.py) | Force the race to be visible |
| [04_lock.py](04_lock.py) | Lock fixes correctness, costs speed |
| [05_cpu_bound_gil.py](05_cpu_bound_gil.py) | GIL: threads don't help CPU work |
| [06_multiprocessing.py](06_multiprocessing.py) | True parallelism, separate memory |
| [07_asyncio.py](07_asyncio.py) | Single thread, event loop, cooperative |
| [07b_await_vs_task.py](07b_await_vs_task.py) | await vs create_task vs gather |
| [07c_run_in_executor.py](07c_run_in_executor.py) | Bridging sync code into async |
| [08_comparison.py](08_comparison.py) | All three, same workloads, timed |

---

## Concepts

### Process vs Thread

A **process** is an isolated unit with its own memory space. A **thread** lives inside a process and shares memory with all other threads in the same process.

```
OS
├── Process A (your Python script)
│   ├── shared memory (heap, globals)
│   ├── Thread 1 (main)    ← own call stack
│   ├── Thread 2 (T-A)     ← own call stack
│   └── Thread 3 (T-B)     ← own call stack
│
└── Process B (your browser)
    └── completely isolated memory
```

Threads don't literally "have their own registers" — there's only one set of physical registers per CPU core. What each thread has is a **saved snapshot** (Thread Control Block / TCB) that gets swapped into the registers when the OS schedules it. This is the context switch.

---

### Why race conditions happen

`counter += 1` compiles to three bytecode instructions:

```
LOAD   reg ← counter    # copy from memory into register
ADD    reg ← reg + 1    # compute in register
STORE  counter ← reg    # write back to memory
```

The OS can pause a thread between any two instructions. If Thread B reads `counter` after Thread A's LOAD but before A's STORE, B gets a stale value. Both write back the same result — one increment is lost.

> **Q I had: isn't it just because B read the value before A finished writing?**
> Yes, exactly. There's a window between READ and WRITE. B slips in and reads an outdated snapshot. When B writes back, it overwrites A's result.

The deeper layer: each CPU core has its own L1/L2 cache. A write doesn't immediately reach other cores — it propagates through the cache hierarchy. This is the **visibility problem**, the root cause below even the non-atomicity issue.

---

### Lock

A lock closes the race window by making READ → WRITE atomic:

```python
with lock:
    temp = counter   # READ
    counter = temp + 1  # WRITE
# lock released here — next thread can enter
```

Only one thread can be inside `with lock` at a time. Others wait outside.

> **Q I had: this is basically the same as database ACID atomicity.**
> Right. Lock at the thread level, transaction at the database level — same idea: bundle multiple steps into one indivisible unit. The pattern appears at every layer: CPU LOCK prefix, language mutexes, DB transactions, distributed locks (Redis/ZooKeeper). Bigger scope, higher cost.

**Trade-off**: correctness costs concurrency. Threads now take turns inside the lock — less parallel.

---

### GIL (Global Interpreter Lock)

CPython allows only **one thread to execute Python bytecode at a time**, even on a multi-core CPU.

**Why does GIL exist?** Because threads share memory, and CPython manages memory with reference counting — every object tracks how many variables point to it. Without the GIL, two threads could simultaneously modify a reference count (the same race condition as `counter += 1`), corrupting memory.

The GIL is a pragmatic choice: one big lock is simpler and faster than fine-grained locks on every object.

**Consequences:**
- IO-bound threads: thread spends most time *waiting* (sleep, network, disk). The GIL is released during waits → other threads can run → real concurrency benefit.
- CPU-bound threads: thread is *always computing*, never releases GIL voluntarily. After 5ms (`sys.getswitchinterval()`), Python forces a switch — but that just means threads take turns. Total work is the same. Sometimes slower due to switch overhead.

```
IO:  T1 [==wait==][run][==wait==][run]
     T2      [run][==wait=========][run]
     (gaps let other threads in)

CPU: T1 [run 5ms][run 5ms][run 5ms]...
     T2                              [run 5ms]...
     (T1 never voluntarily yields → T2 starves until 5ms timer fires)
```

---

### Multiprocessing

Each process gets its own Python interpreter and its own GIL. Truly parallel on separate CPU cores.

**Cost**: processes don't share memory. To pass data between them, it must be **serialized** (converted to bytes), sent through a pipe (two kernel syscalls), then **deserialized** on the other side.

> *Note to self: syscall = crossing from user space to kernel space, involves a context switch. Deep dive belongs in the OS chapter.*

Thread data transfer by contrast is just `MOV [address], reg` — a direct memory write with no kernel involvement.

**Rule of thumb**: multiprocessing pays off when tasks are independent and heavy. If the task is lighter than the process spawn cost (~100–300ms on Windows), you lose.

> **Q I had: isn't multiprocessing basically for tasks that don't share state — like batch image processing or parallel data transformations?**
> Exactly. The defining property is **independence** — each task takes its own input, produces its own output, never needs to coordinate. The moment tasks need to share and synchronize state, the IPC cost eats your parallelism gain.

---

### Thread pool size

Threads are not free. Each OS thread needs ~8MB of stack (pre-allocated for worst-case call depth). 10,000 threads = 80GB.

More threads also means more **context switching** — the OS saves/restores register snapshots constantly. Beyond a point, the CPU spends more time switching than working.

```
Too few threads:  CPU sits idle while all threads wait
Too many threads: CPU wastes time context-switching

Sweet spot (rule of thumb):
  IO-bound:  threads ≈ CPU cores × 4–10  (threads spend most time waiting)
  CPU-bound: threads ≈ CPU cores          (more just adds overhead)
```

> **Q I had: isn't this the same idea as message queues — bounded consumers protect the server from being overwhelmed?**
> Yes, exactly the same idea at different scales. Thread pool = bounded workers for a single process. Message queue (Kafka, RabbitMQ) = bounded workers across a distributed system. Both use the "bounded capacity + queue" pattern to absorb traffic spikes without collapsing. The same thinking shows up in [DB connection pools](../system_design/README.md#connection-pool-and-database-concurrency-model), OS network packet queues, and checkout lines.

---

### Asyncio

Single-threaded. One event loop drives all coroutines cooperatively.

**Coroutine** (`async def`): a function that can pause at `await` points and let others run. Calling it returns a coroutine object — inert, not yet running.

**Key distinction — who decides when to switch:**
- Threads: OS decides, can interrupt at any point (preemptive)
- Coroutines: you decide, only at `await` (cooperative)

This makes coroutines **predictable** — you know exactly where context switches can happen. No `await` between two operations = no one can interrupt = no race condition on simple variables.

**The three await patterns:**

```python
# 1. await coroutine() — sequential, inline execution
#    calling coroutine drives it directly, waits for it to finish
await job("A", 1.0)
await job("B", 1.0)   # B doesn't start until A finishes. Total = 2s.

# 2. create_task + await — concurrent
#    task is scheduled in the event loop, runs independently
task_a = asyncio.create_task(job("A", 1.0))  # registered in queue, not running yet
task_b = asyncio.create_task(job("B", 1.0))  # same
# tasks only start when we hit an await and yield control
result_a = await task_a   # yield → event loop starts A and B → wait for A
result_b = await task_b   # B is likely already done
# Total = ~1s

# 3. gather — syntactic sugar for create_task × N + await all
results = await asyncio.gather(job("A", 1.0), job("B", 1.0))
# Internally: create_task for each, then await all of them
# Total = ~1s
```

> **Q I had: so `await` alone is what makes things concurrent?**
> No — this was a confusion point. `await` alone is **sequential**. It just means "pause here and wait for this to finish." `create_task` is the concurrency switch — it registers the task with the event loop so it can run alongside others. `gather` is `create_task × N + await all` in one line.

**Why asyncio over threads for IO?**

Memory. Each thread needs ~8MB of stack, pre-allocated. A coroutine only stores its local variables at the current pause point — a few KB. 10,000 threads = 80GB. 10,000 coroutines = tens of MB. This is why modern web servers (FastAPI, aiohttp) use async.

---

### run_in_executor

The event loop is single-threaded. Calling a blocking function inside async code freezes everything:

```python
async def bad():
    time.sleep(2)  # blocks the entire event loop for 2s
                   # all other coroutines are stuck
```

`run_in_executor` offloads it to a thread pool, keeping the event loop free:

```python
result = await asyncio.get_event_loop().run_in_executor(None, blocking_fn, args)
# or in Python 3.9+:
result = await asyncio.to_thread(blocking_fn, args)
```

What actually happens:
1. `blocking_fn` starts running in a thread pool thread — immediately
2. A Future (empty result slot) is returned
3. `await` suspends the current coroutine, event loop keeps spinning
4. Thread finishes → fills the Future → event loop wakes up the coroutine

> **Q I had: is Future like a thread waiting to start?**
> No. The thread starts immediately when `run_in_executor` is called. Future is just the **result container** — it has nothing to do with when the thread runs. Better analogy: Future is a locker box. `run_in_executor` dispatches the courier (thread) and hands you the box number. `await future` means standing at the locker waiting for the package to arrive.

> **Insight: this is the same pattern as async web APIs.**
> Submit a long task → get a `job_id` back immediately (don't wait) → webhook fires when done. `run_in_executor` returns a Future immediately (the "job_id"), the thread runs in the background, `set_result` is the webhook, `await` is waiting at the door. `gather` = `Promise.all()` = fire multiple requests and wait for all of them.

---

### Why utilization plateaus

A counterintuitive but common signal: load is high, requests are queuing, yet CPU utilization sits flat at some fraction — say 50% — and never climbs. The instinct is "add more cores." Almost always the real cause is one of five, and none of them is "not enough compute":

```
parallelism not filled  → one CPU-bound thread on a 2-core box maxes one core → 50%
serial bottleneck       → threads exist but serialize on a lock, the GIL, or a single DB connection
waiting for data        → threads block on disk / network / DB; the core idles with nothing to run (IO-bound)
artificial cap          → a thread pool size or cgroup CPU quota holds concurrency below the core count
misleading metric       → iowait counted as "busy", or hyperthread siblings give diminishing returns
```

The fix follows from the cause, not the symptom: parallelize to fill the cores (not "buy more cores"), remove the serial point, feed data faster, or raise the cap. Adding compute when the bottleneck is a serial point or IO wait changes nothing — the new cores idle exactly like the old ones did.

> **Insight: "shared finite resource + head-of-line blocking" recurs across the whole stack.** The same shape that plateaus a CPU shows up everywhere utilization stalls under load. A single [slow SQL exhausting the connection pool](../system_design/README.md#how-a-slow-sql-cascades-into-a-full-outage) is the distributed version — one query holds a shared slot the way one thread holds a core. A GPU stuck at "50% useful work" under [static batching](../llm_architecture/README.md#continuous-batching) is the scheduling version — finished requests leave batch slots idle the way a thread pool too small to fill the cores does. And [decode being memory-bandwidth-bound](../llm_architecture/README.md#quantisation-why-its-fast) — compute units idle waiting for weights to arrive from VRAM — is the exact GPU analogue of a CPU idling on IO wait: same "waiting for data", different data.

---

## Summary table

| | Sequential | Threads | Multiprocessing | Asyncio |
|---|---|---|---|---|
| Memory model | — | shared | isolated | shared (single thread) |
| True parallel | no | no (GIL) | yes | no |
| Good for IO | baseline | yes | overkill | yes, best at scale |
| Good for CPU | baseline | no (GIL) | yes | no (use executor) |
| Race conditions | no | yes | no | no (within one loop) |
| Spawn cost | — | light | heavy | none |
| Max practical scale | — | hundreds | dozens | tens of thousands |
