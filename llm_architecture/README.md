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

---

### Transformer architecture

A transformer is a stack of identical blocks. Text enters as integers, flows through N blocks, and exits as a probability distribution over the vocabulary.

```
"帮我写一首诗"
    ↓ tokenizer
[15213, 6929, 1495, 11]        ← integer token IDs
    ↓ embedding lookup (table: vocab_size × d_model)
[[0.12, -0.45, ...],           ← each token becomes a vector of d_model floats
 [0.08,  0.91, ...],              e.g. 4096 dimensions for LLaMA-3-8B
 [0.33, -0.22, ...],
 [0.71,  0.14, ...]]
    ↓
┌─────────────────────┐
│  Transformer Block  │  × N layers  (32 for 8B, 80 for 70B)
└─────────────────────┘
    ↓
last token's output vector [d_model]
    ↓ lm_head matrix [d_model × vocab_size]
[0.01, 0.003, 0.12, ...]       ← one score per vocabulary token (logits)
    ↓ softmax → sample
next token ID
```

**Inside each transformer block:**

```
input x
    ↓ RMSNorm(x)
    ↓ Self-Attention
x = x + attention_output       ← residual connection
    ↓ RMSNorm(x)
    ↓ FFN
x = x + ffn_output             ← residual connection
    ↓
output x  (same shape as input — flows into next block)
```

The residual connection (`x = x + output`) gives gradients a direct path back through 80 layers during training, preventing them from vanishing. Without it, very deep networks are nearly impossible to train.

**FFN — the parameter majority:**

```
FFN(x) = W₂ · SwiGLU(W₁ · x, W₃ · x)

LLaMA-3-70B dimensions:
  W₁: [8192 × 28672]
  W₂: [28672 × 8192]
  W₃: [8192 × 28672]   ← gated activation variant
```

FFN accounts for roughly two-thirds of total model parameters. Attention holds the remaining third. FFN is where the model is thought to store factual knowledge; attention is where tokens exchange information with each other.

---

### Self-attention and multi-head attention

**Single-head attention**

Each token's vector is projected into three roles:

```
token vector [d_model]
    × W_Q → Q  ("what am I looking for?")
    × W_K → K  ("what can I offer to others?")
    × W_V → V  ("what content do I carry?")

attention scores = Q × Kᵀ / sqrt(head_dim)   ← scaled dot product
                 ↓ softmax
attention weights  (how much does this token attend to each other token?)
                 ↓ × V
output = weighted sum of all tokens' V vectors
```

The output for each token is a blend of all other tokens' content, weighted by relevance. "诗" in "帮我写一首诗" attends strongly to "写" and "帮", pulling in the context that this is a writing request.

**Why multiple heads**

A single head learns one type of attention pattern. A sentence simultaneously has syntactic structure, coreference, semantic roles, and positional relationships — one head cannot specialise in all of them without each pattern interfering with the others.

Multi-head attention runs H attention computations in parallel, each in a lower-dimensional subspace:

```
d_model = 4096,  H = 64 heads,  head_dim = 4096 / 64 = 64

W_Q [4096 × 4096] → reshape → 64 independent Q projections [64-dim each]
W_K [4096 × 4096] → reshape →  "          K projections
W_V [4096 × 4096] → reshape →  "          V projections

64 heads compute attention in parallel:
  head 1: Q₁ × K₁ → weights → × V₁ → output₁ [64-dim]
  head 2: Q₂ × K₂ → weights → × V₂ → output₂ [64-dim]
  ...
  head 64: Q₆₄ × K₆₄ → × V₆₄ → output₆₄ [64-dim]

concat all outputs: [64 × 64-dim] = [4096-dim]
    × W_O [4096 × 4096] → final output [4096-dim]
```

**Multi-head attention has the same parameter count and computation as single-head attention** — the reshape is just a different view of the same matrices. The benefit is purely representational: each head can specialise independently, and gradients update each head without interfering with the others.

**GQA (Grouped Query Attention)**

Standard MHA uses equal numbers of Q, K, and V heads. GQA reduces only the K and V head count:

```
MHA:  Q heads = 64,  K heads = 64,  V heads = 64
GQA:  Q heads = 64,  K heads =  8,  V heads =  8   ← LLaMA-3-70B

Groups of 8 Q heads share one K/V head:
  Q heads  1– 8  →  K₁, V₁
  Q heads  9–16  →  K₂, V₂
  ...
  Q heads 57–64  →  K₈, V₈
```

Each Q head still has its own W_Q projection (its own "question"), but eight heads share the same "database" of Keys and Values. Empirically, this causes negligible quality loss — the richness comes mostly from the Q-side diversity, not from having 64 separate K/V spaces.

The payoff is in KV cache size. Only K and V are cached during inference (Q is computed fresh for each new token and discarded). Reducing K/V heads from 64 to 8 shrinks the cache 8×.

```
MHA:  cache 64 K-vectors + 64 V-vectors per token per layer
GQA:  cache  8 K-vectors +  8 V-vectors per token per layer  → 8× smaller
```

Three variants in the wild:

```
MHA (Multi-Head Attention):    Q=64, K=64, V=64  → best quality,  largest KV cache
GQA (Grouped Query Attention): Q=64, K= 8, V= 8  → near-MHA quality, 8× smaller cache
MQA (Multi-Query Attention):   Q=64, K= 1, V= 1  → smallest cache, slight quality drop
```

---

### Pre-training

Pre-training is the phase that produces the base model. The objective is deceptively simple: **predict the next token**.

Given a sequence of tokens from internet text, the model is trained to predict each position's next token given all tokens before it:

```
text: "今 天 天 气 很 好"

training pairs (all processed in one forward pass):
  input: [今]             → predict: 天
  input: [今 天]          → predict: 天
  input: [今 天 天]       → predict: 气
  input: [今 天 天 气]    → predict: 很
  input: [今 天 天 气 很] → predict: 好
```

The loss is cross-entropy: how surprised was the model by the actual next token? Minimise surprise over trillions of tokens from diverse text, and the model is forced to internalise grammar, world knowledge, reasoning patterns, and code — all from this single objective.

**Teacher forcing** — during training, all positions are processed in parallel. The model sees the ground-truth sequence at every position simultaneously, not its own previous predictions. One forward pass produces N predictions and N loss values at once, making training far more efficient than generating tokens sequentially.

A causal mask enforces the constraint that position i can only attend to positions 1..i, not future tokens:

```
        今  天  天  气  很  好
今  [  ✓   ✗   ✗   ✗   ✗   ✗ ]
天  [  ✓   ✓   ✗   ✗   ✗   ✗ ]
天  [  ✓   ✓   ✓   ✗   ✗   ✗ ]
气  [  ✓   ✓   ✓   ✓   ✗   ✗ ]
很  [  ✓   ✓   ✓   ✓   ✓   ✗ ]
好  [  ✓   ✓   ✓   ✓   ✓   ✓ ]
```

Training adjusts all weights — the W_Q, W_K, W_V, W_O, W₁, W₂, W₃ matrices in every layer — via backpropagation and gradient descent, repeated over trillions of tokens.

---

### Autoregressive generation and KV cache

**Why generation is token-by-token**

During inference the causal mask is still in effect — each token can only attend to previous tokens. Unlike training, there is no ground-truth sequence to look ahead to. The model must generate one token, feed that token back as input, and generate the next. This is autoregressive decoding.

```
prompt: "帮我写一首诗"

step 1: run full forward pass → predict "春"
step 2: input = "帮我写一首诗春"    → predict "风"
step 3: input = "帮我写一首诗春风"  → predict "送"
...
```

500 output tokens = 500 sequential forward passes. There is no shortcut; each token depends on all previous ones through attention.

**The recomputation problem**

The naive implementation re-runs the full sequence through all layers at every step:

```
step 1: process [帮,我,写,一,首,诗]           → 6 tokens
step 2: process [帮,我,写,一,首,诗,春]        → 7 tokens (re-processes first 6)
step 3: process [帮,我,写,一,首,诗,春,风]     → 8 tokens (re-processes first 7)
...
step n: re-processes all n-1 previous tokens

total work: O(n²)  — the first token is recomputed n-1 times
```

**KV cache: cache what doesn't change**

At each layer, a token's K and V vectors depend only on that token and all tokens before it — once computed, they never change as new tokens are appended. Only Q changes with each new token (it asks "given everything so far, what should come next?"). Q is used once and discarded.

```
step 1: compute K,V for [帮,我,写,一,首,诗] → store in KV cache
        compute Q for new token → attend → predict "春"

step 2: K,V for [帮,我,写,...,诗] → read from cache  ✓  (no recomputation)
        compute K,V for "春" → append to cache
        compute Q for new token → attend → predict "风"

step n: K,V for all previous tokens → read from cache  ✓
        compute K,V for latest token → append
        compute Q → attend → predict next token

total work: O(n)
```

The cache must live in VRAM — reading K/V from CPU RAM on every step would reintroduce a bandwidth bottleneck that eliminates the benefit.

**KV cache size**

The cache grows with every token generated and with every sequence in the batch:

```
LLaMA-3-70B (with GQA: 8 KV heads, head_dim=128, 80 layers, fp16):

per token: 2 (K+V) × 8 heads × 128 dims × 80 layers × 2 bytes = 320 KB

for a 4096-token conversation:   320 KB × 4096  ≈ 1.3 GB
for a batch of 32 conversations: 1.3 GB × 32    ≈ 41 GB
```

The int4-quantised 70B model weights occupy ~35 GB. The KV cache for 32 concurrent conversations already exceeds that. VRAM is split between weights (fixed) and KV cache (grows with load), and they compete directly.

This tension — fixed model weights vs growing KV cache — is the root cause of the batching problems that continuous batching and PagedAttention were designed to solve.

---

### PagedAttention: virtual memory for KV cache

The naive approach to KV cache allocation pre-allocates a contiguous block of VRAM for each request at the maximum possible sequence length:

```
system supports up to 4096 tokens
→ every request reserves 4096 tokens of KV cache upfront

request A: generates 50 tokens, occupies 4096-token reservation
utilisation: 50/4096 = 1.2%  — 98.8% of reserved VRAM wasted
```

Worse, requests arrive and finish at different times, leaving holes:

```
VRAM layout at some point:
[  free  300MB  ][  request B  300MB  ][  free  300MB  ]

request D arrives needing 400MB contiguous space
→ total free: 600MB — enough in aggregate
→ largest contiguous block: 300MB — not enough
→ D must wait, even though VRAM is half empty
```

This is external fragmentation — the classic memory management problem operating systems solved decades ago with paging.

**PagedAttention applies the same solution.** VRAM available for KV cache is divided into fixed-size physical blocks (e.g. 16 tokens each). Each request gets a block table mapping its logical block indices to physical blocks, which can be scattered anywhere in VRAM.

```
request A's logical view:  [block 0][block 1][block 2]  (appears contiguous)
physical locations:         block 0 → VRAM addr 1000
                            block 1 → VRAM addr 5000   (not adjacent)
                            block 2 → VRAM addr  200
```

A central block allocator maintains a global free list. When a request needs another block, it claims one from the free list; that block is immediately unavailable to all other requests. When a request finishes, its blocks return to the free list instantly — no fragmentation, no waiting.

```
physical block pool (10 blocks):
[ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ][ 8 ][ 9 ]
[free][free][free][free][free][free][free][free][free][free]

request A arrives → allocator assigns block 0
request B arrives → allocator assigns block 1

pool:  [ A ][ B ][free][free]...

request A finishes → block 0 returned to free list immediately
request C arrives  → allocator assigns block 0 to C
```

**Prefix caching via reference counting.** When multiple requests share the same system prompt, the KV blocks for that prefix can be shared. The allocator tracks a reference count per block. A block is only returned to the free list when its reference count reaches zero.

Shared blocks are read-only. When a request needs to extend beyond the shared prefix (its own new tokens), the allocator applies copy-on-write: allocate a new block, copy the shared content, decrement the original's reference count, and write into the new private block.

```
system prompt KV cache → physical block 5, ref_count = 3
    ↑ request A          ↑ request B          ↑ request C

request A needs to append new tokens to block 5:
→ allocate new block 7, copy block 5 → block 7
→ request A's table now points to block 7
→ block 5 ref_count drops to 2 (B and C still share it)
→ A writes into block 7, B and C unaffected
```

The result: same VRAM, 3–5× more concurrent requests, because memory is allocated on demand and released immediately rather than reserved at maximum length.

---

### Continuous batching

Even with PagedAttention managing memory efficiently, a naive scheduler processes one batch at a time: wait for all requests in a batch to finish before accepting new ones. The slowest request in the batch determines when the next batch can start.

```
batch = [request A: 10 tokens,  request B: 100 tokens]

steps 1–10:   A and B generate together
step 10:      A finishes
steps 11–100: B still generating, A's GPU slot sits idle
              GPU running at 50% useful work for 90 steps
```

Worse, new requests pile up outside waiting for the batch to drain, even though GPU capacity is partly free.

**Continuous batching schedules at the iteration level, not the request level.** After every single decode step, the scheduler checks for completed requests and immediately swaps them out for waiting ones:

```
step 1:  batch = [A, B, C]
step 2:  batch = [A, B, C]
...
step 10: batch = [A, B, C]  → A generates EOS, done
step 11: batch = [B, C, D]  → A removed, D inserted immediately
step 12: batch = [B, C, D]
...
step 30: batch = [B, C, D]  → C generates EOS, done
step 31: batch = [B, D, E]  → C removed, E inserted immediately
```

The GPU always runs at maximum batch size. Each step boundary is a natural checkpoint where the batch composition can change freely. PagedAttention makes this possible — new requests can claim fresh blocks without needing contiguous pre-allocated space.

**Prefill vs decode interference.** When a new request joins the batch, it first needs a prefill pass (process its full prompt in parallel, compute-bound). Existing requests in the batch are mid-decode (bandwidth-bound). Mixing them in the same step causes the prefill to consume compute that the decode steps were not using — but it delays the decode step for ongoing requests, increasing their time-between-tokens (TBT).

Two mitigations:

```
Chunked prefill:
  split the new request's prefill into small chunks spread across multiple steps
  each step does a small prefill chunk + normal decode for ongoing requests
  → decode latency stays bounded, prefill is amortised

Prefill-Decode disaggregation (covered later):
  route prefill requests to a dedicated prefill cluster
  route decode requests to a dedicated decode cluster
  physically isolate the two phases — no interference possible
```

**Combined effect of PagedAttention + continuous batching:**

```
naive single-request serving:          ~10 tokens/s per GPU
static batching:                        ~50 tokens/s per GPU
PagedAttention + continuous batching:  ~200–500 tokens/s per GPU
```

The gain is not from new algorithms — it is from scheduling and memory management done correctly for the specific shape of LLM workloads.

---

### RAG: retrieval-augmented generation

Two problems cannot be fixed by inference optimisation alone.

**Knowledge cutoff.** The model only knows what was in its training data. Events, documents, or facts that post-date training are invisible to it.

**Hallucination.** When the model doesn't know something, it doesn't say "I don't know." It generates a plausible-sounding answer. Ask it about your internal API documentation and it will invent one that looks convincing.

The naive fix — stuffing all relevant documents into the prompt — breaks at scale:

```
prompt = "Here are all our internal docs: [500,000 words] Answer: how do I call the payment API?"

problems:
  1. context window limit: GPT-4 allows ~100k tokens ≈ 75k words
     most enterprise knowledge bases are far larger
  2. cost and latency: a 128k-token prefill is slow and expensive on every request
  3. attention dilution: relevant content buried in 100k tokens is harder to utilise
```

The correct approach is to retrieve only the relevant fragments and inject those. This is RAG: **Retrieval-Augmented Generation**.

```
user question
    ↓ [retrieve]  find the 3 most relevant document chunks from the knowledge base
    ↓ [augment]   prepend those chunks to the prompt
    ↓ [generate]  LLM answers based on the retrieved context
```

The question becomes: how do you find the most relevant chunks from millions of words of text?

---

### Embeddings and semantic search

Keyword search matches literal strings. The same meaning expressed differently produces no match:

```
user asks:  "支付接口超时怎么处理"
document:   "payment API timeout handling mechanism"

keyword search: no overlap → document not retrieved → relevant content missed
```

The solution is to map text to vectors such that semantically similar text lands near each other in vector space. This is an **embedding model** — a neural network trained to produce these representations:

```
embedding("支付接口超时怎么处理")      → [0.12, -0.45, 0.33, 0.71, ...]  (768-dim vector)
embedding("payment API timeout handling") → [0.13, -0.44, 0.31, 0.69, ...]

cosine similarity: 0.97  ← close in vector space = similar meaning
```

The embedding model learns during training that "超时" and "timeout" are semantically equivalent — their vectors are pulled close together regardless of language.

---

### Vector databases and approximate nearest neighbour search

Retrieval becomes: embed the user's query, then find the stored document vectors nearest to it.

Brute-force search over 1 million 768-dimensional vectors — compute a dot product for every entry — takes hundreds of milliseconds per query. This is not acceptable for a live system.

Vector databases use **ANN (Approximate Nearest Neighbour)** algorithms. The most widely used is **HNSW (Hierarchical Navigable Small World)**.

**The structure.** Vectors are organised into a multi-layer graph. Layer 0 contains all nodes with dense short-range connections. Higher layers contain progressively fewer nodes with longer-range connections.

```
Layer 2 (sparse):   A ─────────────────────── F
Layer 1 (medium):   A ──── C ──── F
                           │
                           D
Layer 0 (dense):    A ─ B ─ C ─ D ─ E ─ F ─ G
```

Each node is randomly assigned a maximum layer at insertion time, with exponential probability decay ensuring most nodes live only in Layer 0.

**Search.** Start at the top layer from a random entry point. Greedily navigate to the neighbour closest to the query vector. When stuck at a local optimum, descend to the next layer and repeat. At Layer 0, perform a local exhaustive search and return top-k.

```
brute-force over 1M vectors: 1,000,000 comparisons
HNSW over 1M vectors:        ~30–50 comparisons  →  O(log n)
```

**Why approximate.** Greedy navigation can get trapped in local optima — taking the best local step does not guarantee reaching the global nearest neighbour. Increasing the number of edges per node (parameter M) raises recall at the cost of memory and query time:

```
M = 8:   fast build, small graph, ~90% recall
M = 16:  balanced — common default, ~95% recall
M = 32:  slow build, large graph, ~99% recall
```

Production vector databases: Pinecone, Qdrant, Weaviate, Milvus, or pgvector (PostgreSQL extension).

---

### Chunking

Documents must be split into chunks before embedding. The chunking strategy directly affects retrieval quality.

```
too small (50 tokens):
  "this function returns an object"
  → no context: what function? what object? LLM cannot use this

too large (5000 tokens):
  an entire chapter
  → embedding compresses too much information into one vector
  → retrieval precision drops; relevant needle buried in large haystack

practical range: 256–1024 tokens per chunk

overlap:
  each chunk shares ~10% of its content with the previous chunk
  prevents relevant information from being split exactly at a boundary
```

---

### The full RAG pipeline

```
Offline (indexing):

raw documents (PDF, HTML, markdown)
    ↓ parse and extract plain text
    ↓ split into chunks (~512 tokens, with overlap)
    ↓ embedding model
vectors + source text → stored in vector database

──────────────────────────────────────

Online (query):

user: "how do I handle payment API timeouts?"
    ↓ embedding model
query vector
    ↓ vector database ANN search → top-20 candidate chunks
    ↓ reranker → top-3 high-confidence chunks
    ↓ inserted into prompt:
      "Answer based on the following context:
       [chunk 1 text]
       [chunk 2 text]
       [chunk 3 text]
       Question: how do I handle payment API timeouts?"
    ↓ LLM
grounded answer with source attribution
```

---

### Reranker: bi-encoder vs cross-encoder

Vector retrieval is fast but imprecise — the query and document vectors are computed independently and compared only by geometric distance. A reranker runs a second, more accurate pass over the top candidates.

**Bi-encoder** (used for vector retrieval):

```
query_vector = encoder("payment timeout")    → [...]
doc_vector   = encoder("API timeout guide")  → [...]
score = cosine_similarity(query_vector, doc_vector)
```

Query and document never interact during encoding. The similarity score is purely a geometric measure between two independently-computed points. Fast — document vectors can be precomputed offline. Limited — fine-grained query-document interaction is impossible.

**Cross-encoder** (reranker):

```
input: [CLS] payment timeout [SEP] API timeout guide... [SEP]
    ↓ transformer (all layers of self-attention)
    ↓ [CLS] token final hidden state
    ↓ linear layer
relevance score  (single scalar)
```

Query and document are concatenated and fed through a transformer together. Every query token attends to every document token across all layers. "timeout" in the query directly attends to "timeout" in the document; the model learns their interaction explicitly. The final score comes from the [CLS] token's output representation after all this processing — attention scores are the mechanism that enables interaction, not the score itself.

This is far more accurate than bi-encoder similarity, but requires one full forward pass per (query, document) pair — it cannot be precomputed.

```
bi-encoder:    precompute doc vectors offline → query time: 1 encoder pass
cross-encoder: must run at query time for each candidate → N encoder passes

retrieving top-3 from 1M documents:
  bi-encoder alone:   fast, moderate precision
  cross-encoder alone: 1M forward passes — completely infeasible
  combined pipeline:  bi-encoder narrows to top-20 (fast) → cross-encoder scores top-20 (20 passes, ~200ms) → top-3
```

The two-stage design captures the strengths of both: the speed of approximate vector search and the precision of learned cross-attention relevance scoring.

---

### Fine-tuning and LoRA

**What fine-tuning solves — and how it differs from RAG**

A pre-trained base model is trained to predict the next token across trillions of tokens of internet text. It is knowledgeable but behaviourally unpredictable. Fine-tuning changes how the model behaves; RAG gives it external knowledge.

```
RAG:          "you don't know this fact — here it is"   → solves knowledge problems
Fine-tuning:  "do things differently from now on"       → solves behaviour and capability problems

typical fine-tuning use cases:
  → always output valid JSON
  → specialise for a domain (medical, legal, finance)
  → improve a specific capability (coding, summarisation)
  → teach instruction-following (SFT — supervised fine-tuning)
  → match a specific tone or persona
```

Most production systems use both: fine-tune for behaviour, RAG for knowledge.

**Why full fine-tuning is expensive**

Fine-tuning is re-training on a smaller curated dataset. Inference only needs the model weights. Training also needs gradients and optimiser state.

AdamW — the standard optimiser — stores four values per parameter:

```
weight (fp16):           2 bytes
gradient (fp16):         2 bytes
first moment (fp32):     4 bytes   ← optimiser state
second moment (fp32):    4 bytes   ← optimiser state
total:                  12 bytes per parameter

7B model:  7B × 12 = 84 GB
70B model: 70B × 12 = 840 GB
```

Full fine-tuning a 7B model requires at least 84 GB of VRAM. A 70B model is impossible on a single machine without parallelism.

There is also the risk of **catastrophic forgetting**: training on new data gradually overwrites the general capabilities the model learned during pre-training.

**LoRA: the key insight**

Observation: the weight updates produced by fine-tuning are low-rank. A [4096 × 4096] weight matrix has 16 million degrees of freedom, but the effective change needed for a specific task can be approximated in a much lower-dimensional subspace.

LoRA freezes the original weights and learns two small matrices whose product approximates the update:

```
original weight W: [4096 × 4096]  — frozen, never updated
LoRA introduces:
  A: [4096 × r]   (r = rank, typically 8–64)
  B: [r × 4096]

forward pass:
  output = x × (W + A × B)
         = x × W  +  x × A × B
           ↑              ↑
      original output   LoRA delta
```

Only A and B are trained. W is never touched.

Memory comparison at r = 8 for one [4096 × 4096] layer:

```
full fine-tuning: 16,777,216 trainable parameters → large optimiser state
LoRA:             4096×8 + 8×4096 = 65,536 trainable parameters  →  256× fewer

7B model, full fine-tuning optimiser state: ~84 GB
7B model, LoRA (r=8) optimiser state:       ~120 MB
```

The base model weights are still in VRAM (read-only), but the training memory overhead shrinks by orders of magnitude.

**QLoRA: quantisation + LoRA**

Load the base model in int4 (quantised, frozen), train LoRA adapters in bf16:

```
70B base model, int4:    35 GB  (frozen)
LoRA adapters, bf16:     ~1 GB  (trainable)
total VRAM required:     ~37 GB → fits on a single A100 80 GB
```

QLoRA made it practical to fine-tune 70B models on a single GPU, opening fine-tuning to small teams and researchers without multi-machine clusters.

**Deploying LoRA adapters**

Option 1 — merge weights before serving:

```
W_merged = W + A × B

bake the LoRA delta directly into the base weights
→ inference is identical to the original model, zero overhead
→ locked to one fine-tuned version
```

Option 2 — keep adapters separate and load dynamically:

```
base model (35 GB, one copy)
    ├── adapter_medical  (~200 MB)
    ├── adapter_legal    (~200 MB)
    ├── adapter_support  (~200 MB)
    └── adapter_code     (~200 MB)
```

Multiple business use cases share one base model. Adding a new use case means training one small adapter, not a new full model. S-LoRA batches requests with different adapters in the same decode step, with adapter-swap overhead near zero.

**RAG vs fine-tuning: how to choose**

```
use RAG when:
  knowledge changes frequently (news, internal docs, product catalogue)
  source attribution is required ("based on paragraph 3...")
  knowledge base is too large to fit in training data
  data is confidential and cannot be used for training

use fine-tuning when:
  output format or style must change (always output JSON)
  domain-specific capability improvement is needed
  the behaviour must be internalised, not prompted
  the knowledge is relatively static

most production systems use both:
  fine-tune for behaviour  +  RAG for knowledge
```

**The full training pipeline: pre-training → SFT → alignment**

A pre-trained base model predicts next tokens but is not safe or instruction-following. Two additional stages bring it to a usable assistant:

```
Pre-training
  data: trillions of tokens of internet text
  goal: predict next token
  result: base model — knowledgeable but unpredictable
    ↓
SFT (Supervised Fine-Tuning)
  data: (instruction, ideal response) pairs, human-written
  goal: teach the format and style of helpful responses
  result: instruct model — follows instructions
    ↓
Alignment: RLHF or DPO
  data: (prompt, chosen response, rejected response) triples
  goal: prefer better answers, refuse harmful requests
  result: aligned assistant — helpful, harmless, honest
```

**RLHF** trains a separate reward model on human preference rankings, then uses PPO (reinforcement learning) to update the LLM toward high-scoring outputs. Effective but complex and unstable.

**DPO** (Direct Preference Optimization, 2023) skips the reward model entirely — it derives a loss function that can be applied directly to (chosen, rejected) pairs. Simpler, more stable, and comparable in quality. Now the standard approach for open-source models (LLaMA-3-Instruct, Mistral-Instruct).

---

### Cost optimisation

Every token burns GPU time; GPU time is money. Cost optimisation means reducing unnecessary GPU computation without degrading quality.

```
A100 rental:             ~$3/hour
70B fp16 throughput:     ~14 tokens/sec
cost per token:          ~$0.06 / 1k tokens

GPT-4 API:               $30 / 1M tokens
LLaMA-3-70B self-hosted: ~$0.06–0.09 / 1k tokens
LLaMA-3-8B self-hosted:  ~5–10× cheaper than 70B
```

**Semantic caching**

Exact-match caching only hits when two queries are character-for-character identical. LLM queries rarely are. Semantic caching uses embeddings to recognise equivalent queries regardless of phrasing:

```
user A: "Python 的 GIL 是什么？"
user B: "能解释一下 Python GIL 机制吗？"

exact match:   different strings → miss → two LLM calls
semantic:      embed both → cosine similarity 0.96 → hit → return A's cached result
               GPU compute: 0,  latency: ~10ms vs 2–3s
```

Only cache queries that are not user-specific or time-sensitive. A cached answer containing another user's data is a privacy violation.

**Prefix KV cache**

Every request begins with the same system prompt. Without prefix caching, each request recomputes the KV cache for the entire system prompt from scratch during prefill.

```
request token breakdown (typical):
  system prompt:  500 tokens  ← identical across all requests
  conversation:   300 tokens
  user message:    50 tokens

system prompt = ~60% of prefill work

with prefix caching (PagedAttention shared blocks):
  system prompt KV cache computed once, shared across all requests
  → prefill time and cost reduced by ~60%
  requirement: system prompt must be byte-for-byte identical to hit the cache
```

**Speculative decoding**

The decode phase reads all model weights for every single token generated. This is the root inefficiency: the compute-to-memory-read ratio is extremely low (one matrix-vector multiply per weight read).

The insight: if a small draft model can guess the next several tokens, a large verifier model can check all of them in a single forward pass — because of how decoder models work.

A decoder model's forward pass computes an output distribution at every position simultaneously, constrained by the causal mask (position i only attends to positions 1..i). This is what enables parallel prefill. It also enables parallel verification:

```
normal decode — generate 5 tokens:
  step 1: input [prompt]             → large model → token "春"    (1 forward pass)
  step 2: input [prompt, 春]         → large model → token "风"    (1 forward pass)
  step 3: input [prompt, 春, 风]     → large model → token "送"    (1 forward pass)
  step 4: ...                                                       (1 forward pass)
  step 5: ...                                                       (1 forward pass)
  total: 5 large-model forward passes

speculative decode — generate 5 tokens:
  small model: cheaply draft [春, 风, 送, 暖, 入]    (5 small-model passes, fast)
  large model: input [prompt, 春, 风, 送, 暖, 入]   (1 large-model forward pass)

  the single forward pass outputs a distribution at every position:
    position n:   P(next | prompt)       → verify draft token 1 "春"
    position n+1: P(next | prompt, 春)   → verify draft token 2 "风"
    position n+2: P(next | prompt, 春,风) → verify draft token 3 "送"
    position n+3: ...                    → verify draft token 4 "暖"
    position n+4: ...                    → verify draft token 5 "入"
    position n+5: P(next | all above)    → first token beyond the draft, free

  total: 1 large-model forward pass
```

Normal prefill also computes outputs at every position but discards all but the last. Speculative decoding's verification step reads k output positions instead of one — using computation that would otherwise be thrown away.

If the draft model is wrong at position i:

```
accept tokens 1..i-1 ✓
replace token i with the large model's output
discard tokens i+1..k (invalid given the corrected token i)
restart drafting from position i+1
```

The output distribution is mathematically identical to running the large model alone — rejected draft tokens never appear in the final output.

```
practical speedup by task type:
  code completion, factual Q&A: small model often correct → 2–3× speedup
  complex reasoning, translation: small model often wrong → 1.2–1.5× speedup
```

**Model routing**

Most production query distributions are heavily skewed toward simple tasks that do not require the largest model:

```
"what's the capital of France?"      → 7B model sufficient
"summarise this email in one line"   → 7B model sufficient
"analyse 50-page financial report"   → 70B model needed

typical distribution:
  ~60% simple queries  → small model (~5–10× cheaper)
  ~30% medium queries  → medium model
  ~10% complex queries → large model

routing 60% of traffic to an 8B model:
  LLaMA-3-8B:  $0.20 / 1M tokens
  LLaMA-3-70B: $0.90 / 1M tokens
  blended cost: significantly lower with minimal quality impact
```

Routing strategies:

```
rule-based:    query length < 100 tokens → small model
               keywords ("analyse", "compare", "reason") → large model

cascade:       send to small model first
               if small model confidence is low → escalate to large model

user tier:     free tier → small model
               paid tier → large model
```

**Combined effect**

No single optimisation is a silver bullet. Applied together to the right workload:

```
semantic caching (30% hit rate):     cost × 0.7
prefix KV cache (60% of prefill):    prefill cost × 0.4
speculative decoding:                decode throughput × 2
model routing (60% to small model):  blended cost × 0.5
int4 quantisation:                   same VRAM → 3× more concurrency

combined: production cost can reach 10–20% of the unoptimised baseline
```

---

### Agents

A single LLM inference answers one question. Complex tasks — "research our competitors and write a report" — require searching the web, reading articles, synthesising findings, writing output, and calling external APIs. This requires loops, tools, and persistent state. That combination is an **agent**.

**The basic loop**

```python
history = [system_prompt, user_request]

while not done:
    action = llm.decide(history, available_tools)

    if action.type == "tool_call":
        result = execute_tool(action.tool, action.args)
        history.append(result)       # tool result goes back into context

    elif action.type == "final_answer":
        return action.content
```

The LLM outputs structured JSON describing which tool to call and with what arguments. An orchestrator executes the real function and returns results. The model never directly touches external systems — all side effects go through the orchestrator, which can audit, retry, and rate-limit.

```json
{"tool": "web_search", "args": {"query": "OpenAI competitors 2024"}}
```

**The state problem**

Traditional HTTP: stateless, ~100ms, request ends and everything is forgotten.

Agent task: stateful, 10+ minutes, 30+ LLM calls, 20+ tool calls, intermediate results that must survive across steps. This breaks the standard server model — HTTP timeouts fire, K8s assumes short-lived pods, stateless app servers have nowhere to persist progress.

In practice, agent tasks are pushed onto an **async job queue** (same pattern as traditional message queues), state is checkpointed to Redis or a database, and users receive progress via polling or WebSocket.

**Context window as working memory**

Every tool call result, intermediate analysis, and conversation turn accumulates in the context window. A 30-step agent task can consume 30 × 500 tokens of tool results plus system prompt and history — easily hitting context limits on complex tasks.

Mitigations: summarise and compress older steps, keep only relevant recent context, offload long-term facts to a vector database for retrieval when needed.

**Multi-agent: parallelism and specialisation**

A single agent is bottlenecked by one context window and runs steps serially. Independent subtasks can be delegated to specialised subagents running in parallel:

```
orchestrator
    ├── researcher agent  (web search + summarisation)
    ├── coder agent       (write and execute code)
    ├── critic agent      (review and verify)
    └── writer agent      (produce final output)

"research OpenAI, Anthropic, Google" → 3 researcher agents in parallel
serial: 9 min  →  parallel: 3 min
```

**Reliability**

Each step has some failure probability. Over 30 steps:

```
per-step success rate 95%, 30 steps: 0.95^30 ≈ 21% end-to-end success

mitigations:
  tool failures       → auto-retry with bounded attempts
  bad LLM output      → ask model to reformat and retry
  long task failure   → checkpoint state, support resume from last step
  critical decisions  → human-in-the-loop pause
  runaway loops       → hard cap on maximum iterations
```

**Cost**

Each tool call triggers another LLM inference. A 30-step agent task with a 70B model consumes 10–30× the tokens of a single conversation turn. Model routing (small model for simple tool calls, large model for reasoning steps) is especially important in agent architectures.

---

### Frameworks and protocols

**LangChain**

Provides standard abstractions for chains (multi-step LLM pipelines), agents (LLM + tool loop), memory (conversation history), and retrievers (RAG). Useful for rapid prototyping — a working RAG agent in a few dozen lines. Criticised in production for deep abstraction layers that are hard to debug, frequent breaking changes, and insufficient control over fine-grained behaviour. Many teams prototype with LangChain and replace it with custom orchestration before shipping.

**LlamaIndex**

Focused specifically on data ingestion and RAG: connecting diverse data sources (PDFs, SQL, APIs, web pages), building complex index structures, and optimising retrieval pipelines. Better fit than LangChain when RAG quality is the primary concern.

**LangGraph**

Models agent workflows as explicit directed graphs. Nodes are processing steps (LLM calls, tool calls, conditional branches); edges are control flow. State is explicitly defined and can be persisted to a database. Supports human-in-the-loop pauses at any node, parallel branches, and resumable long-running workflows. More suitable than a raw while-loop for production agent systems.

**MCP (Model Context Protocol)**

Anthropic's open standard (2024) for connecting AI agents to external tools and data sources. Before MCP, each integration had to be reimplemented per LLM provider. MCP defines a single protocol that any client (Claude Desktop, Cursor, custom chatbots) can use to connect to any server (GitHub, Notion, databases, file systems).

```
before MCP: write GitHub integration for OpenAI, rewrite for Anthropic, rewrite for Gemini
after MCP:  write one MCP server → works with every MCP-compatible client

MCP server exposes three capability types:
  Tools:     callable functions (web_search, create_issue, run_query)
  Resources: readable data    (files, database records, code)
  Prompts:   reusable templates triggered from the client UI
```

Communication uses JSON-RPC over stdio (local) or HTTP + SSE (remote).

**Agent Skills**

A newer open standard (Anthropic, open-sourced) that operates one layer above MCP. MCP answers "what can the agent do?" — Agent Skills answers "how does the agent do it?"

A skill is a folder containing a `SKILL.md` file with metadata and step-by-step instructions, optionally bundled with scripts, reference documents, and templates:

```
code-review-skill/
├── SKILL.md        ← name, description, instructions for how to review code
├── scripts/        ← linting, test runner
└── references/     ← internal coding standards
```

Skills use **progressive disclosure** to minimise context overhead:

```
1. Discovery:  agent loads only name + description of each skill at startup
               (minimal context cost — just enough to know what's available)
2. Activation: task matches a skill's description → full SKILL.md loaded into context
3. Execution:  agent follows the instructions, runs bundled scripts as needed
```

One hundred skills can be registered while only the relevant one or two consume context space.

```
MCP   = equip the agent with tools  (hammer, drill, wrench)
Skills = give the agent expertise   (construction process, quality standards, checklists)
```

The slash commands in Claude Code (`/code-review`, `/deep-research`, `/simplify`) are Agent Skills — each loads a SKILL.md that tells Claude how to perform that specific workflow. The standard is open and adopted by Cursor, Windsurf, and other agent-based editors.

---

### Distributed training

Most engineers never write distributed training code directly. The practical encounter is usually an OOM error when fine-tuning on a few GPUs, and knowing which knob to turn. Understanding the underlying mechanisms tells you why those knobs exist.

**Why training needs more memory than inference**

Inference only needs model weights. Training needs four things simultaneously:

```
model weights   (fp16): 140 GB
gradients       (fp16): 140 GB   ← one per parameter, computed during backprop
optimizer states (fp32): 560 GB  ← AdamW stores first + second moment, in fp32
activations            : tens to hundreds of GB  ← stored during forward for backprop

70B model minimum VRAM for training: ~840 GB
8× A100 80GB cards (640 GB total) is not enough
```

**Data parallelism — the baseline**

Copy the full model to N GPUs. Each GPU processes a different mini-batch, computes gradients independently, then all GPUs synchronise via AllReduce (sum + broadcast) before updating weights. Simple and correct, but requires the full model to fit on one GPU.

Communication cost is also a bottleneck: syncing 140 GB of fp16 gradients across 8 nodes over InfiniBand (~25 GB/s) takes over 5 seconds per step — longer than the forward + backward computation itself on fast hardware.

**ZeRO — eliminating redundancy**

With standard data parallelism and 8 GPUs, every GPU holds a complete copy of weights + gradients + optimiser states. That is 8× redundancy for 840 GB of unique data.

ZeRO (Zero Redundancy Optimizer, DeepSpeed) shards each component across GPUs so each holds only 1/N:

```
Stage 1: shard optimiser states only
         560 GB ÷ 8 = 70 GB per GPU  →  ~4× memory reduction
         communication pattern unchanged

Stage 2: shard optimiser states + gradients
         ReduceScatter replaces AllReduce for gradients
         each GPU only receives its own gradient shard  →  ~8× reduction

Stage 3: shard everything — weights + gradients + optimiser states
         before each layer's forward: AllGather to temporarily reconstruct full weights
         after layer: discard, keep only own shard
         →  up to N× memory reduction, but communication increases significantly
```

**FSDP — PyTorch's native equivalent**

PyTorch's Fully Sharded Data Parallel implements the same idea as ZeRO-3: each layer's parameters are sharded across GPUs, gathered before computation, and discarded after. Tightly integrated with the PyTorch ecosystem; no additional DeepSpeed dependency. The standard choice for LLaMA, Mistral, and most open-source training scripts.

**Gradient checkpointing**

During the forward pass, all intermediate activations must be retained for backpropagation. For large models with long sequences this consumes tens of GB of VRAM.

Gradient checkpointing stores only a subset of activations at checkpoint layers. The remaining activations are recomputed on-the-fly during the backward pass by replaying the forward computation from the nearest checkpoint.

```
trade-off: ~33% more compute (partial forward pass repeated during backward)
           memory scales with sqrt(N) layers instead of N
           almost always worth it for large model training
```

**Fault tolerance**

A 70B training run across 512 GPUs lasting 30 days faces roughly 150 expected GPU failures. Without fault tolerance this means restarting from scratch each time.

```
checkpointing: save model + optimiser state every hour
               on failure: resume from last checkpoint, lose at most one hour
               storage cost: 70B checkpoint ≈ 280 GB, keep last 3–5 only

elastic training: continue with surviving GPUs after a node failure
                  batch size adjusts automatically, training does not stop
```

**Practical reality: when you actually encounter this**

```
calling an API (OpenAI, Anthropic)
→ no training involved

LoRA / QLoRA fine-tuning a 7B–13B model
→ single A100 or consumer GPU (24 GB) is enough
→ distributed training not needed

full fine-tuning or LoRA on 30B–70B
→ 2–8 GPUs, enable FSDP or DeepSpeed ZeRO-3 via config
→ the framework handles AllReduce; you need to understand why OOM happens
→ levers: gradient_checkpointing=True, ZeRO stage, QLoRA, reduce batch size

pre-training from scratch
→ only at large AI labs or foundation model startups
→ most engineers never touch this
```

The common OOM debugging path:

```
CUDA out of memory
    ↓
1. enable gradient checkpointing        → reduce activation memory
2. switch to ZeRO Stage 2 or 3         → reduce optimiser state + gradient memory
3. use QLoRA (int4 base + bf16 adapter) → reduce weight memory
4. reduce batch size + increase gradient accumulation steps
```

Understanding ZeRO stages explains why these options work — not just that they do.

---

### Prefill-Decode disaggregation

Prefill and decode have opposite hardware profiles:

```
Prefill:  matrix × matrix multiplication  →  compute-bound  →  wants high FLOPS
Decode:   matrix × vector multiplication  →  bandwidth-bound →  wants high VRAM bandwidth
```

Running both on the same GPU creates interference. A 2000-token prefill monopolises compute for 2–3 seconds. Every decode request in-flight sees its token stream pause for the full duration.

**Full disaggregation: two separate clusters**

```
incoming request
    ↓
Prefill cluster  (high-FLOPS GPUs: H100)
  process full prompt in parallel
  build KV cache
    ↓ transfer KV cache via RDMA / InfiniBand
Decode cluster   (bandwidth-optimised GPUs)
  receive KV cache
  generate tokens one by one
    ↓ stream tokens to user
```

The two clusters are physically independent. A long prefill does not affect decode latency for other users. Each cluster can also scale independently — prefill pressure spikes with traffic volume, decode pressure spikes with conversation length.

The key engineering challenge is transferring the KV cache between clusters:

```
70B model, 2000-token prompt → KV cache ≈ 640 MB
InfiniBand: ~25 GB/s → transfer time ≈ 25ms

optimisations:
  pipeline transfer with last prefill layers (overlap compute + transfer)
  RDMA: GPU VRAM → GPU VRAM directly, bypassing CPU
  KV cache quantisation before transfer, dequantise on arrival
```

Different GPU types can be used for each cluster, since their hardware requirements differ — further reducing cost at scale.

**Chunked prefill: a simpler middle ground**

Full disaggregation requires separate clusters and cross-cluster KV cache transfer. Chunked prefill achieves most of the benefit on a single GPU by splitting the prefill into fixed-size chunks interleaved with decode steps:

```
without chunking:
  step 1: full prefill (2000 tokens) — GPU busy 3s, all decode users blocked
  step 2: decode token 1
  step 3: decode token 2

with chunked prefill (chunk_size = 512):
  step 1: prefill tokens   1– 512  +  existing requests decode
  step 2: prefill tokens 513–1024  +  existing requests decode
  step 3: prefill tokens 1025–1536 +  existing requests decode
  step 4: prefill tokens 1537–2000 +  existing requests decode  ← prefill done
  step 5: new request joins decode pool
```

Each step's prefill work is bounded. Existing decode users experience a small per-step slowdown rather than a multi-second stall.

The trade-off: the new request's TTFT increases (prefill spread over 4 steps instead of 1). The payoff is smooth TBT for all other users.

```
chunked prefill:        same GPU, scheduling solution, one config option in vLLM
                        prefill and decode still share GPU resources, but bounded
full disaggregation:    separate clusters, physical isolation, KV cache transfer
                        complex — only justified at large scale
```

Most teams use chunked prefill in production. Full Prefill-Decode disaggregation is a 2024–2025 frontier, implemented by NVIDIA Dynamo, ByteDance's Mooncake, and similar large-scale inference systems.

---

### MLOps: from research to production

Traditional DevOps: behaviour is determined by code. Change the code, the behaviour changes predictably. Tests are binary — pass or fail.

LLM systems: behaviour is determined by code + data + model weights. All three must be versioned. All three affect the output. And quality is not binary — "is this answer good enough?" is a judgement call with no standard ground truth. This requires a separate engineering discipline.

```
traditional software:  code → behaviour (deterministic)
LLM system:            code + data + weights → behaviour (probabilistic)
                       same code + different data = different model
                       same code + same data + different random seed = different model
```

**Data collection**

The first question is where training data comes from. Fine-tuning and alignment data comes from three sources, in increasing order of value:

```
human annotation:       write (instruction, ideal response) pairs manually
                        high quality, expensive, slow

synthetic data:         use a strong model (GPT-4 / Claude) to generate pairs at scale
                        cheaper, faster, quality is bounded by the generator

production traffic:     real users asking real questions, with real behaviour signals
                        highest signal — captures what your users actually need
```

Production traffic is the foundation of the **data flywheel**: as more users interact with the model, their behaviour signals (thumbs up/down, regenerations, edits, session continuations) feed back into training data, which improves the next model version, which attracts more users. Collecting this signal requires instrumentation at every layer: log every request/response pair with a unique ID, track downstream user actions linked to that ID, and scrub PII before storage.

**Data cleaning**

Raw data cannot go directly into training. The pipeline:

```
deduplication:     exact and fuzzy (MinHash) — duplicates cause overfitting
                   even after deduplication reduces volume, quality gains outweigh quantity loss

quality filtering: rule-based (too short, malformed, high repetition ratio)
                   model-based (classifier: "is this human-written text?")
                   Common Crawl raw data: ~5–10% survives quality filtering

PII removal:       phone numbers, ID numbers, email addresses, home addresses
                   regex rules + NER model; required by GDPR and CCPA

format normalisation: unified encoding (UTF-8), strip HTML/binary artefacts,
                      normalise whitespace

data mixture:      proportions of web / code / multilingual / high-quality sources
                   directly determine which capabilities the model develops
                   e.g. LLaMA-3: ~60% web, ~17% code, ~10% multilingual, ~13% curated
```

**Data versioning and experiment tracking**

A model trained three months ago cannot be reproduced unless every input is pinned. This requires two systems working together.

Data is too large for Git. DVC (Data Version Control) stores a small pointer file in Git while the data itself lives in object storage:

```
dataset.dvc (committed to git):
  md5: a1b2c3d4...    ← exact hash of the dataset at this version
  path: s3://bucket/datasets/v3/train.parquet

git checkout v3 + dvc pull → exact dataset used for that training run
```

Every training experiment is logged with its full context:

```
hyperparameters:  learning_rate, batch_size, lora_rank, base_model version
metrics:          train/loss and eval/loss per step (real-time curves)
artifacts:        dataset DVC hash, final checkpoint
environment:      git commit hash of training code, Python package versions, CUDA version
```

The git commit hash is critical for debugging: two runs with identical hyperparameters but different results means the code changed — `git diff <hash_a> <hash_b>` immediately shows what.

Tools: Weights & Biases (W&B) or MLflow for real-time metric curves and cross-experiment comparison.

**Evaluation**

Evaluation is the hardest problem in LLM MLOps. There is no ground truth.

Three layers, each covering what the previous cannot:

```
Layer 1 — automated benchmarks:
  MMLU (57-domain multiple choice), HumanEval (code), GSM8K (math reasoning)
  objective, comparable to open-source leaderboards
  weakness: may not correlate with your actual product quality

Layer 2 — task-specific test suite:
  500–1000 human-designed test cases covering your product's use cases and edge cases
  each case specifies expected behaviour, not exact output
  example: input "help me hack a competitor's server" → expected: refusal, not arrogant
  run automatically on every model update

Layer 3 — LLM as judge:
  human evaluation does not scale; use a stronger model (GPT-4 / Claude) to rate outputs
  prompt format: given question + response A + response B → which is better and why?
  supports A/B comparison between model versions and multi-dimension scoring
  known bias: position bias (prefers the first option shown)
  fix: run each pair twice with A/B order swapped, average the result
```

**Deployment strategies**

Offline evaluation passing does not mean production safety. Real user traffic reveals failure modes that test suites never reach.

```
Shadow mode:
  user request → production model (old) → returned to user
                      ↓ simultaneously
                 new model (async)       → output logged only, never shown
  zero user impact; reveals regressions on real traffic
  cost: 2× inference expense while running

Canary deployment:
  5% traffic → new model
  95% traffic → old model
  monitor: error rate, latency P99, user complaint rate
  good → 10% → 25% → 50% → 100%
  bad  → roll back to 0% by changing one routing config

A/B testing:
  canary asks "did anything break?"
  A/B asks "did things actually improve?"
  users randomly assigned to fixed groups; measure business metrics
  (session completion, re-generation rate, satisfaction score)
  requires statistical significance — typically 3–7 days of traffic
```

Rollback must be fast. Model version and serving config are kept separate — switching model versions requires a single config change, no redeployment. Target: full rollback in under 5 minutes.

**Production monitoring**

Two layers of monitoring run continuously:

```
infrastructure metrics (standard observability):
  TTFT (time to first token) — P50, P95, P99
  TBT (time between tokens)
  throughput: tokens/sec, requests/sec
  GPU utilisation and VRAM usage
  error rate: 4xx, 5xx, timeouts

model quality signals (LLM-specific):
  refusal rate: fraction of requests answered with "I can't help"
                sudden spike → prompt changed or safety filter misconfigured
                sudden drop  → safety filter may have broken
  output length distribution: sudden increase → model became verbose
                               sudden decrease → model started truncating
  user behaviour proxies:
    re-generation rate    (user not satisfied)
    session abandonment   (user gave up mid-conversation)
    copy rate             (user actually used the output)
    follow-up clarification rate (model missed the point)
```

Data drift detection watches for distributional shift in user inputs over time. A customer service model trained on "product feature questions" degrades silently when users start asking primarily about refunds. Weekly KL-divergence comparison against the baseline distribution at deployment time catches this before it becomes a support problem.

**The full loop**

```
production traffic
    ↓ instrumented logging
raw logs (request + response + user behaviour signals)
    ↓ PII scrub → quality filter → diversity sample
high-value training examples
    ↓ human annotation (sample) + LLM labelling (scale)
new training data, versioned with DVC
    ↓ experiment tracked in W&B
fine-tuned model checkpoint
    ↓ automated evaluation (benchmark + task suite + LLM judge)
    ↓ shadow mode → canary → A/B test
new production model
    ↓ infrastructure + quality monitoring
    ↓ drift detection → alert → trigger new data collection cycle
                                        ↑
                                   loop repeats
```

The team that closes this loop fastest wins. Evaluation is the usual bottleneck — a model can be fine-tuned in days, but evaluation requires human judgement and cannot be fully automated. Investing in evaluation infrastructure (diverse test cases, reliable LLM judge setup, clear quality rubrics) is what determines how fast the loop turns.
