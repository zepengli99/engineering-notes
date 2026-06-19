# vLLM

Deep dive into vLLM's internals — how it manages memory and schedules requests to maximise GPU throughput.

The concepts here build on [LLM Architecture](../llm_architecture/README.md), which covers PagedAttention and continuous batching at a high level. This document goes deeper into the mechanics.

Simulation code lives in [simulation/](simulation/).

---

## The problem vLLM solves

Naive KV cache management pre-allocates a contiguous VRAM block for each request at maximum sequence length. Two things go wrong:

**Internal waste.** A request that generates 50 tokens still holds a 4096-token reservation. 98% of its VRAM sits empty.

**External fragmentation.** As requests finish at different times, the free space becomes non-contiguous. A new request needing 400 MB may find 600 MB free in total but no single contiguous 400 MB block.

vLLM solves both with PagedAttention — the same idea as OS virtual memory paging, applied to KV cache.

---

## PagedAttention: block allocator

VRAM available for KV cache is divided into fixed-size **physical blocks** (default 16 tokens each). Each request gets a **block table** mapping its logical block indices to physical blocks scattered anywhere in VRAM.

```
request A's logical view:  [block 0][block 1][block 2]  (appears contiguous)
physical locations:
  logical 0 -> physical block 7   (VRAM addr 0x1000)
  logical 1 -> physical block 2   (VRAM addr 0x5000)  <- not adjacent
  logical 2 -> physical block 15  (VRAM addr 0x0200)
```

A central `BlockAllocator` maintains a **free list**. Allocation and free are both O(1). When a request finishes, its blocks return to the free list immediately — no fragmentation, no waiting.

**On-demand allocation.** Blocks are not reserved upfront. A new block is allocated only when the current tail block fills up. This means VRAM is consumed exactly proportional to tokens actually generated, not tokens potentially generated.

### ref_count: the key to sharing

Every physical block carries a `ref_count`. The invariant:

```
ref_count == 1  ->  private, safe to write
ref_count  > 1  ->  shared, must copy-on-write before writing
ref_count == 0  ->  returned to free list
```

Three operations drive everything:

```python
allocate()   # pop from free list, ref_count = 1
free(block)  # ref_count -= 1; if 0, push to free list
fork(block)  # ref_count += 1, no data copy, no new allocation
```

### Copy-on-write

`fork()` is used whenever blocks are shared — prefix caching, beam search. No data is copied at fork time. Copying is deferred until the moment a write actually happens:

```python
def copy_on_write(block):
    if block.ref_count == 1:
        return block          # already private, write directly
    new_block = allocate()    # get a private copy
    free(block)               # release this owner's share of the original
    return new_block          # caller replaces its reference with new_block
```

This is identical to how Linux implements fork() for process memory — copy is deferred until a write, so processes that never diverge pay nothing.

### Prefix caching

Multiple requests sharing the same system prompt can share the same physical blocks for that prefix. The prefix blocks are computed once and `fork()`ed — each new request increments ref_count with zero memory overhead.

```
system prompt -> B5, B6  (ref_count = 1, held by cache)

request A arrives: fork(B5), fork(B6)  -> ref_count = 2
request B arrives: fork(B5), fork(B6)  -> ref_count = 3

no new blocks allocated — 3 requests share 2 blocks

request A generates its own tokens -> gets private new block B9
  B5, B6 still shared between B and cache
```

When a request writes its first new token past the prefix, copy-on-write triggers and it gets a private block. The shared prefix blocks are unaffected.

---

## Continuous batching

Continuous batching is an **inference-only** concern. Training batches are fixed-size with known lengths; inference requests have unknown output lengths and arrive at arbitrary times.

The naive approach: wait for all requests in a batch to finish before admitting new ones. A 10-token request finishes early but its GPU slot sits idle while the 100-token request in the same batch keeps running.

**Continuous batching schedules at the iteration level.** After every single decode step, the scheduler checks for finished requests and immediately replaces them with waiting ones:

```
step 10: [A, B, C] -> A generates EOS, done
step 11: [B, C, D] -> A removed, D inserted immediately
step 12: [B, C, D]
```

The GPU runs at maximum batch size at all times. PagedAttention makes this possible — a new request can claim free blocks without needing a contiguous pre-reserved slot.

---

## Preemption

When the block pool fills mid-decode, the scheduler must evict a running sequence to free blocks for others.

### Who gets evicted: FCFS reverse

The **most recently admitted** sequence is preempted first. It has generated the fewest tokens so far, so redoing its work costs the least.

### Two strategies after eviction

**Swap.** Move the KV cache from GPU VRAM to CPU RAM. The sequence pauses; when VRAM frees up, the blocks are transferred back and generation resumes from where it stopped.
- Pro: no wasted compute, generation continues exactly where it left off
- Con: PCIe bandwidth (~64 GB/s) is a bottleneck; transferring a 640 MB KV cache takes ~10ms each way

**Recompute.** Drop the KV cache entirely. The sequence is reset and re-queued. When admitted again, it re-prefills from scratch.
- Pro: no memory transfer, simpler implementation
- Con: compute already spent on prior decode steps is wasted

vLLM defaults to **recompute**. The rationale: preempted sequences have typically just been admitted and generated few tokens, so the recompute cost is small. Swap's transfer overhead often costs more than the saved compute.

### Proactive vs reactive

**Reactive** preemption waits for an allocation to actually fail. At that point the system is already in a crisis and handling it is messy.

**Proactive** preemption runs a check before each decode step: "does any running sequence need a new block this step, and do we have enough?" If not, evict before starting — not after failing.

vLLM uses proactive preemption. The check runs at the start of every scheduler iteration.

---

## Real vLLM: observed numbers

Tested with vLLM 0.23.0, Qwen2.5-1.5B-Instruct, RTX 5060 8GB, Docker on WSL2.

**Startup — KV cache allocation**

```
GPU VRAM total:          7.96 GiB
Model weights (bf16):   ~3.0 GiB
Available for KV cache:  2.47 GiB   (after weights + CUDA graph overhead)
KV cache block size:     16 tokens
GPU KV cache capacity:   92,640 tokens
```

`gpu_memory_utilization=0.8` means vLLM claims 80% of VRAM for weights + KV cache combined.
The remaining 20% is left for CUDA kernels, activations, and the display driver.

**Prefix caching**

8 requests sharing the same ~120-token system prompt, sent sequentially:

```
after request 1:  hit rate  69%
after request 4:  hit rate  76%
after request 8:  hit rate  80%
```

Hit rate converges toward ~83% (120 shared tokens / 145 total tokens per request).
The remaining 17% is the user question — unique per request, never cached.

**Throughput: serial vs concurrent**

8 requests, 120 tokens each, no shared system prompt:

```
serial     (one at a time):  87.8 tok/s   wall time: 14.3s
concurrent (all at once):   730.6 tok/s   wall time:  1.7s
speedup:  8.3x
```

The token count is identical — the difference is pure scheduling efficiency.
Concurrent requests let vLLM fill its batch; serial requests leave the GPU waiting.

**Streaming latency (single request)**

```
TTFT (time to first token):   67 ms    <- prefill completes, first token arrives
TBT  (avg time between tokens): 12 ms  <- decode speed, one token per step
single-request throughput:    81.8 tok/s
```

TTFT and TBT are independent axes of latency. A long prompt increases TTFT (more prefill work)
without affecting TBT. A loaded server increases TBT (decode steps are shared across requests)
without necessarily affecting TTFT.

---

## Process architecture: APIServer + EngineCore

vLLM V1 (0.4+) runs the API layer and the inference engine as two separate OS processes.

```
APIServer  (pid=1)                     EngineCore  (pid=178)
─────────────────────────────────      ──────────────────────────────────
handles HTTP connections (asyncio)     owns the GPU
manages request queue                  runs the scheduler
streams SSE tokens to clients          manages KV cache blocks
pure Python, never touches GPU         blocks for the duration of each
                                       forward pass — that's fine,
        ZeroMQ socket (IPC)            it's its own process
        send request / recv result
```

**Why GPU inference is blocking**

A PyTorch forward pass submits CUDA kernels to the GPU, then calls `cudaDeviceSynchronize()` — it blocks the calling thread until the GPU finishes. asyncio cannot help here: `await` only yields on I/O (network, disk). A CUDA kernel is not I/O, so it holds the Python thread for the full duration of the forward pass (tens to hundreds of milliseconds).

In a single-process design, that blocked thread is also the one running the HTTP event loop — so no new requests can be accepted while the GPU is computing.

The two-process fix: EngineCore blocks as much as it wants inside its own process. APIServer has no GPU code at all, so its event loop is never blocked. Communication between them goes over a ZeroMQ socket, which is real I/O that asyncio can `await`.

**Three concrete benefits**

1. **GPU OOM does not kill the service.** If EngineCore is killed by the OOM killer, APIServer survives and returns 503 to clients instead of disappearing entirely.

2. **Truly async API layer.** APIServer handles hundreds of concurrent HTTP connections without any GPU stall. Before V1 this was not possible in a single process.

3. **Data parallel scaling.** One APIServer can connect to multiple EngineCores and route requests across them. This is the foundation for multi-replica deployments.

**Visible in the startup logs**

```
APIServer  pid=1    INFO  Waiting for application startup.
EngineCore pid=178  INFO  Initializing a V1 LLM engine...
EngineCore pid=178  INFO  Starting to load model...
           ↑ all GPU work happens in the subprocess; API server just waits
APIServer  pid=1    INFO  Application startup complete.
           ↑ EngineCore signals ready; only then does the API server open
```

The analogy to web infrastructure: APIServer + EngineCore is the same separation as Nginx (accepts connections) + Gunicorn worker (runs application code). The worker can block; the frontend never does.

---

## Running vLLM locally

```bash
# pull image (~10 GB)
docker pull vllm/vllm-openai:latest

# start server
docker run --runtime nvidia --gpus all \
  -v C:/Users/<you>/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 --ipc=host \
  --name vllm-server \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --enable-prefix-caching
```

Key flags:
- `--gpu-memory-utilization` — fraction of VRAM for weights + KV cache; lower if OOM on startup
- `--max-model-len` — caps KV cache size; reduces memory pressure for long-context models
- `--enable-prefix-caching` — turns on automatic block sharing for repeated prefixes

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1/`.
Metrics (Prometheus format) at `http://localhost:8000/metrics`.

---

## Simulation code

```
simulation/
  block_allocator.py   BlockAllocator, Sequence, Scheduler — core allocation logic
                       demos: continuous batching, copy-on-write, prefix caching
  preemption_demo.py   ProactiveScheduler — OOM prediction and FCFS eviction

prefix_cache_test.py   measures prefix cache hit rate across sequential requests
throughput_test.py     serial vs concurrent throughput comparison
streaming_test.py      TTFT and TBT measurement via SSE streaming
```

Run any script directly with `python <file>.py`. No dependencies beyond the standard library.
