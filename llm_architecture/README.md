# LLM System Design

Architecture evolution notes — how a system changes when the core compute unit shifts from CPU to GPU, and the core product shifts from data retrieval to generative inference.

The same pattern from internet architecture repeats: a bottleneck appears, a targeted fix is added, the fix creates a new bottleneck. But the bottlenecks are different. VRAM instead of RAM. Token throughput instead of QPS. KV cache instead of query cache. GPU cluster topology instead of network topology.

```
call external API
    └─ third-party dependency, cost per token, data leaves your system
self-host the model
    └─ GPU required; VRAM is the scarce resource that everything else is designed around
quantisation
    └─ model too large for one GPU; reduce precision to compress weights
model parallelism
    └─ still doesn't fit; split the model across GPUs or machines
streaming inference
    └─ autoregressive generation is slow; tokens must flow to the user immediately
continuous batching
    └─ GPU idles between requests; pack multiple requests in-flight
KV cache
    └─ recomputing attention over all history is wasteful; cache key-value pairs
RAG + vector database
    └─ model hallucinates and has a knowledge cutoff; retrieval grounds it
agents
    └─ single inference isn't enough; tools + loops + persistent state
```

---

## Architect's rules (LLM edition)

1. **VRAM is the new disk.** In traditional architecture, disk IO is the slowest resource and everything is designed around avoiding it. In LLM systems, GPU VRAM is the hard constraint. Model weights, KV cache, and activations all compete for the same small, fixed pool.

2. **Latency has two dimensions.** Traditional latency is one number: time to response. LLM latency splits into TTFT (time to first token) and TBT (time between tokens). Users experience them differently — a fast TTFT feels responsive even if total generation takes 20 seconds.

3. **Decode is memory-bandwidth-bound, not compute-bound.** During token generation, the GPU spends most of its time reading weights from VRAM, not computing. The compute units are often idle, waiting for data. Every optimisation that reduces the amount of data to read (quantisation, KV cache, batching) directly reduces latency and cost.

---

## Concepts

---

### The naive start: just call the API

Your existing backend serves users. You want to add an AI feature. The fastest path is obvious: call OpenAI's or Anthropic's API. No GPU setup, no model downloads, no infrastructure. In production in a day.

Three problems emerge as the product grows.

**Cost scales with every token.** GPT-4 charges ~$30 per million input tokens. A chatbot with a long system prompt and multi-turn history can consume 2000 tokens of context for every user message. At scale, the API bill becomes the dominant engineering constraint.

**Data leaves your system.** Every user message is transmitted to a third-party server. For enterprise customers handling legal, medical, or financial data, this is a dealbreaker. The API call is a data-sharing agreement, not just a network request.

**You have no control over availability or behaviour.** API outages take your feature down. Silent model version upgrades change your product's behaviour overnight. Rate limits constrain your traffic patterns.

The natural response: host the model yourself.

---

### GPU and VRAM: the new physical constraint

You download LLaMA-3-8B and run inference on a CPU server. Twenty-two seconds later, a response arrives. This is not a bug — it is physics.

A transformer model is a sequence of matrix multiplications. A CPU has ~64 cores, each powerful and general-purpose. A GPU has 10,000+ small cores designed for exactly one thing: parallel matrix math. Transformer inference on CPU is roughly 100× slower than on GPU.

You provision a GPU server. The same request takes 1-2 seconds.

You want a bigger model. LLaMA-3-70B is meaningfully better than 8B, so you try to load it.

```
LLaMA-3-70B parameters: 70 billion
Storage per parameter at fp16: 2 bytes
Total weight size: 140 GB

A100 GPU VRAM: 80 GB

140 GB > 80 GB → model does not fit
```

**VRAM is the hard constraint.** Unlike CPU RAM, you cannot add more VRAM to a GPU — it is fixed at manufacture. If the model does not fit, it does not run.

---

### Precision and numerical formats

Model weights are floating-point numbers. The choice of precision determines both memory usage and numerical stability.

```
fp32 (full precision):  32 bits per value = 4 bytes  → 280 GB for 70B model
fp16 (half precision):  16 bits per value = 2 bytes  → 140 GB for 70B model
bf16 (bfloat16):        16 bits per value = 2 bytes  → 140 GB for 70B model
int8 (8-bit integer):    8 bits per value = 1 byte   →  70 GB for 70B model
int4 (4-bit integer):    4 bits per value = 0.5 bytes →  35 GB for 70B model
```

**fp16 vs bf16** — same memory footprint, different bit layout:

```
fp16:  [1 sign][5 exponent][10 mantissa]  → high precision, small range (max ~65504)
bf16:  [1 sign][8 exponent][ 7 mantissa]  → lower precision, same range as fp32 (max ~3.4×10³⁸)
```

fp16's small range causes overflow in operations like softmax where intermediate values like `exp(120)` far exceed 65504. bf16's exponent range matches fp32, so it never overflows. Most modern LLM training and inference uses bf16 rather than fp16.

**Mixed precision** — not all operations use the same precision:

Even when weights are fp16/bf16, some operations run in fp32 for numerical stability, then convert back. The critical cases are softmax and LayerNorm.

Softmax computes `exp(x) / Σexp(x)`. With fp16, `exp(120.5) ≈ 1.7×10⁵²` overflows immediately. In fp32 the range is sufficient, and paired with the standard max-subtraction trick the computation is stable.

LayerNorm computes variance over a set of values. If activations are very small — e.g. `[0.001, 0.0012, ...]` — fp16 has only 3 significant digits. The difference between two close values becomes zero (catastrophic cancellation). fp32 has 7 significant digits and preserves the small differences correctly.

Modern frameworks handle this automatically. PyTorch's `autocast` selects fp16 for matrix multiplications (where GPU Tensor Cores are fastest) and fp32 for numerically sensitive ops, without requiring manual casts in user code.

---

### LayerNorm and RMSNorm

Each transformer layer performs matrix multiplications whose outputs can drift wildly — some activation dimensions explode, others collapse toward zero. Normalisation after each layer brings activations back to a stable distribution, preventing gradient explosion and collapse during training.

**Why not BatchNorm?**

BatchNorm normalises across the batch dimension: it computes mean and variance over all samples in a batch for each feature. This has two problems for LLMs.

First, sequence positions have different semantics. The 3rd token in "today's weather" and the 3rd token in "write me code" are completely unrelated — averaging their statistics produces meaningless normalisation.

Second, LLM inference commonly runs with batch size 1. BatchNorm on a single sample is undefined.

**LayerNorm** normalises within a single token, across all its feature dimensions:

```
token activation: x = [120.3, 0.002, -88.5, 45.1]

step 1 — compute mean and variance across dimensions:
    mean = 19.23,  var ≈ 5832

step 2 — normalise:
    x_norm = (x - mean) / sqrt(var + ε)
           ≈ [-0.08, -0.27, ..., 0.34]   ← pulled to ~zero mean, unit variance

step 3 — learned scale and shift (γ, β):
    output = γ × x_norm + β   ← model learns the optimal distribution per layer
```

Each token is normalised independently, regardless of batch size or what other tokens are in the sequence. Training and inference use identical code — no running statistics, no train/inference inconsistency.

**RMSNorm** — the modern simplification used by LLaMA, Mistral, and most recent models:

```
LayerNorm:  x_norm = (x - mean) / sqrt(var + ε)
RMSNorm:    x_norm = x / sqrt(mean(x²) + ε)       ← drop the mean subtraction
```

Empirically, subtracting the mean contributes almost nothing to training stability. Removing it makes the computation faster. At the scale of trillions of training tokens, every small operation that can be eliminated compounds into significant wall-clock savings.

---

### Memory hierarchy and data flow

Model weights travel through a hierarchy before inference can begin:

```
Disk (SSD)
    ↓ one-time load at startup (~minutes for large models)
CPU RAM
    ↓ one-time transfer at startup
GPU VRAM
    ↓ stays here permanently during serving
GPU compute units
    ↑ weights are read from VRAM on every forward pass
```

Once weights are in VRAM, the CPU and RAM are largely uninvolved in inference. The GPU operates entirely within its own memory.

**Why weights must live in VRAM:**

```
VRAM → GPU compute units (internal bandwidth): ~2,000 GB/s  (A100)
CPU RAM → GPU VRAM (PCIe bandwidth):              ~64 GB/s

If weights lived in RAM:
  140 GB ÷ 64 GB/s = 2.2 seconds just to transfer weights for each forward pass
  → unusable

Weights in VRAM:
  140 GB ÷ 2,000 GB/s = 70ms to read all weights once
  → this is the actual per-token budget
```

**Input data is tiny and not a bottleneck:**

User input — a few hundred token IDs — is tokenised on the CPU and transferred to the GPU over PCIe. A 2000-token prompt is ~8 KB. At 64 GB/s this takes microseconds and is never the bottleneck.

---

### Quantisation: why it's fast

During token generation (the decode phase), the GPU generates one token per step. Each step reads the entire model's weights from VRAM to the compute units, performs a matrix-vector multiplication, and produces a single output vector. The computation itself is fast — the bottleneck is reading the weights.

```
A100 VRAM bandwidth: 2,000 GB/s
A100 compute:        312 TFLOPS (fp16)

70B model, fp16 weights (140 GB):
  Time to read all weights: 140 GB ÷ 2,000 GB/s = 70 ms/token
  Theoretical max speed:    ~14 tokens/second

The compute units are mostly idle, waiting for data to arrive from VRAM.
This phase is memory-bandwidth-bound, not compute-bound.
```

Quantisation reduces the weight size. Less data to read means less waiting:

```
fp16  → 140 GB → 70 ms/token
int8  →  70 GB → 35 ms/token  (2× faster)
int4  →  35 GB → 17 ms/token  (4× faster)
```

**Quantisation is about storage, not computation.** int4 weights are stored as 4-bit integers. Before the matrix multiply, they are dequantised back to fp16. The dequantisation step is computationally negligible — the compute units perform it during the time they would otherwise be idle waiting for more data.

```
int4 weight (stored): 5   (4 bits)
         ↓ dequantise (nearly free, done while waiting for next chunk)
fp16 value: 0.342
         ↓ matrix multiply (fast, always was)
```

**The exception: the prefill phase.** Processing the input prompt processes all tokens simultaneously as a matrix-matrix multiplication. The GPU's compute units are fully utilised — the prefill phase is compute-bound, not bandwidth-bound. Quantisation helps less here, because the bottleneck is compute capacity, not data transfer rate.

```
Decode (generate 1 token):   bandwidth-bound → quantisation helps a lot
Prefill (process N tokens):  compute-bound   → quantisation helps less
```

---

### Model parallelism: when VRAM is still not enough

Quantisation can stretch VRAM further but has a floor — int4 quality loss is noticeable, and some applications cannot accept it. When a model simply does not fit on one GPU, the model must be split across multiple GPUs.

**Tensor Parallelism** — split each weight matrix horizontally across GPUs:

```
Weight matrix: [4096 × 4096]
Split across 4 GPUs:
  GPU 0: [4096 × 1024]
  GPU 1: [4096 × 1024]
  GPU 2: [4096 × 1024]
  GPU 3: [4096 × 1024]

Forward pass: all 4 GPUs compute in parallel, then AllReduce to sum results.
```

Each GPU holds 1/N of each layer's weights. Every layer requires synchronisation across all GPUs — the AllReduce at the end of each layer is a blocking collective communication. This requires extremely high bandwidth between GPUs.

```
NVLink (within one machine): ~600 GB/s  ← fast enough for tensor parallelism
PCIe (within one machine):    ~64 GB/s  ← too slow, becomes the bottleneck
InfiniBand (across machines): ~25 GB/s  ← far too slow for layer-level sync
```

Tensor parallelism only works within a single machine where NVLink connects the GPUs.

**Pipeline Parallelism** — split the model by layers across machines:

```
80-layer model split across 4 machines:
  Machine 1: layers  1–20
  Machine 2: layers 21–40
  Machine 3: layers 41–60
  Machine 4: layers 61–80

A token's activations flow through each machine in sequence.
Machines only communicate activations (a few MB), not weights.
```

Communication is small enough to run over InfiniBand between machines. The trade-off is the pipeline bubble: machine 1 finishes its layers and sits idle while the token completes its journey through machines 2, 3, and 4.

**In practice, large models combine both:**

```
Within each machine (NVLink):   tensor parallelism
Across machines (InfiniBand):   pipeline parallelism
Multiple full replicas:         data parallelism (for higher throughput)
```

A 70B model might run across 2 machines × 4 GPUs each — tensor parallel within each machine, pipeline parallel across the two machines.
