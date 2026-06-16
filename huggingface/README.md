# Hugging Face Hub & CLI

Personal notes on the Hugging Face ecosystem — how models actually get downloaded, cached, and managed. Starting from misconceptions and working toward clarity.

---

## Sections

---

### 1. `from_pretrained` doesn't just load into memory

Initial misconception: `AutoModel.from_pretrained("gpt2")` pulls the model into memory and that's it — gone when the process exits.

What actually happens — two steps:

1. **Download to disk** — first call fetches files from the Hub and persists them to local cache
2. **Load from disk into memory** — reads the files, builds the model object

Second call hits local cache directly, no network request.

---

### 2. Local cache structure

Default path: `~/.cache/huggingface/hub/`

Each repo gets a directory named `models--{owner}--{name}` (datasets use `datasets--`):

```
~/.cache/huggingface/hub/
  models--gpt2/
    refs/
      main              ← plain text file containing a commit hash
    blobs/              ← actual file contents, named by sha256
    snapshots/
      607a30d.../       ← directory named by commit hash
        config.json          → symlink -> ../../blobs/<sha256>
        model.safetensors    → symlink -> ../../blobs/<sha256>
        tokenizer.json       → symlink -> ../../blobs/<sha256>
        ...
```

**Why this design**: files that don't change across revisions (e.g. the tokenizer) are stored once as a blob. Multiple snapshots point to the same blob via symlinks — no wasted disk space.

**Windows caveat**: creating symlinks on Windows requires admin rights or Developer Mode. HF falls back to copying files directly into the snapshot directory, leaving `blobs/` empty. The deduplication benefit is lost, but everything still works.

---

### 3. What's inside the cached files

Using GPT-2 as a concrete example. Every downloaded model snapshot contains the same categories of files.

**`config.json`** — model architecture as JSON, no weights

```json
{
  "model_type": "gpt2",
  "n_layer": 12,
  "n_embd": 768,
  "n_head": 12,
  "n_positions": 1024,
  "vocab_size": 50257
}
```

`from_pretrained` reads this first to know what structure to build, then loads weights into it.

**`model.safetensors`** — the actual weights, 160 tensors for GPT-2

```
h.0.attn.c_attn.weight    [768, 2304]   float32   ← layer 0, Q/K/V projection (3×768 merged)
h.0.mlp.c_fc.weight       [768, 3072]   float32   ← layer 0, MLP first layer
...                                                 ← repeated for all 12 layers
```

Safetensors is a binary format — safe to load (no arbitrary code execution unlike pickle), supports memory-mapping so you can load only the tensors you need.

**`vocab.json`** — token string → token ID mapping, 50257 entries for GPT-2

Token 0 is `!`, token 50256 is `<|endoftext|>`. The `Ġ` prefix on tokens like `Ġthe` means "preceded by a space" in GPT-2's byte-level encoding.

**`merges.txt`** — BPE merge rules, ordered by frequency

GPT-2 uses Byte-Pair Encoding. Training starts with individual characters and repeatedly merges the most frequent adjacent pair. `merges.txt` records that order:

```
Ġ t       ← most common: space + t → merged first
Ġ a
h e
...
Ġt he     ← later: "Ġt" + "he" → "Ġthe"
```

Tokenizing a string means applying these rules in order. High-frequency words like `hello` end up as one token; rare words like `unhappiness` get split:

```python
tok.tokenize("hello")        # → ['hello']         one token
tok.tokenize("unhappiness")  # → ['un', 'h', 'appiness']
tok.tokenize("GPT2")         # → ['G', 'PT', '2']  very rare, split aggressively
```

**`tokenizer.json`** — the above two combined plus pre-tokenization rules, special tokens, etc.

This is the canonical format. `AutoTokenizer.from_pretrained` reads this. `vocab.json` and `merges.txt` are the legacy format kept for compatibility.

---

### 4. `hf_hub_download` vs `snapshot_download`

**`snapshot_download`** — downloads the entire repo (all files), returns the snapshot directory path. This is what `from_pretrained` uses internally.

```python
from huggingface_hub import snapshot_download

path = snapshot_download("gpt2")
# ~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d.../
```

**`hf_hub_download`** — downloads a single file, returns its local path. Everything else in the repo is untouched.

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("gpt2", filename="config.json")
# only config.json is downloaded, not the 548MB weights
```

**When single-file download matters:**

*Model selection before committing* — check architecture params before pulling a large model:

```python
import json
from huggingface_hub import hf_hub_download

config = json.load(open(hf_hub_download("meta-llama/Llama-3-70b", "config.json")))
print(config["num_hidden_layers"])  # verify it fits your use case before downloading 70B
```

*Training from scratch* — when you want the architecture definition but not the pretrained weights:

```python
from transformers import AutoConfig, GPT2LMHeadModel

config = AutoConfig.from_pretrained("gpt2")  # downloads only config.json
model = GPT2LMHeadModel(config)              # random init, no pretrained weights loaded
```

*Sampling a large dataset* — a dataset split across 100 shards; grab one to inspect the schema:

```python
hf_hub_download("some/dataset", "data/train-00000.parquet", repo_type="dataset")
```

---

### 5. Authentication

HF uses a bearer token, not SSH keys. The token is just a string sent directly in the HTTP header on every request:

```
Authorization: Bearer hf_xxx...
```

Contrast with SSH: SSH sends a *signature* (the private key never leaves your machine). With HF tokens, the token itself is the credential — whoever has it can use it.

**Token types**

- **read** — access your private repos and gated models (e.g. Llama, Gemma require applying on the Hub page first)
- **write** — read + upload files, create repos, push changes

**Where the token lives — priority order**

`get_token()` checks three places in order:

1. `HF_TOKEN` environment variable (recommended — won't accidentally get committed)
2. `~/.cache/huggingface/token` file (created by `huggingface-cli login`)
3. Google Colab secrets (if running in Colab)

**How to log in**

```bash
# interactive — prompts for token, saves to ~/.cache/huggingface/token
huggingface-cli login
```

```python
# programmatic — only valid for current process, not persisted
from huggingface_hub import login
login(token="hf_xxx")
```

```bash
# environment variable — best for CI / servers
export HF_TOKEN="hf_xxx"
# HF reads this automatically, no login() call needed
```

**Gated models** — some models (Llama, Gemma) require you to accept terms on the Hub website first. Your account gets flagged as approved, and then your read token can download them. Without approval, requests return 403.

**Token hygiene** — never commit the token to git, never print it to logs. If leaked, revoke it on the HF website and generate a new one.

---

### 6. Uploading to the Hub

Hub repos are git repos under the hood (large files via git-lfs). Every upload creates a commit.

```python
from huggingface_hub import HfApi
api = HfApi()

# create repo first
api.create_repo(
    repo_id="your-username/my-model",
    private=True,
    repo_type="model",   # or "dataset", "space"
)

# upload a single file
api.upload_file(
    path_or_fileobj="./my_model/config.json",
    path_in_repo="config.json",
    repo_id="your-username/my-model",
)

# upload an entire folder
api.upload_folder(
    folder_path="./my_model/",
    repo_id="your-username/my-model",
    num_workers=4,
)
```

**What `num_workers` actually does**

Upload is I/O-bound — CPU sits idle while waiting for the network. `num_workers=4` opens a `ThreadPoolExecutor` with 4 threads, each handling one file's HTTP PUT concurrently:

```
thread-1: PUT shard-1.safetensors  [====waiting for server====]
thread-2: PUT shard-2.safetensors  [====waiting for server====]
thread-3: PUT shard-3.safetensors  [====waiting for server====]
thread-4: PUT shard-4.safetensors  [====waiting for server====]
```

Threads work here because Python's GIL is *released* during network I/O — other threads run while one is waiting. More threads isn't always better: the real ceiling is your upload bandwidth or the Hub's rate limit.

**Contrast with PyTorch DataLoader's `num_workers`**

Same parameter name, opposite mechanism:

| | HF `upload_folder(num_workers=4)` | DataLoader `num_workers=4` |
|---|---|---|
| Underlying primitive | threads (`ThreadPoolExecutor`) | **processes** (`multiprocessing`) |
| Task type | I/O-bound (network) | CPU-bound (decode images, augment) |
| Why this choice | GIL released during I/O — threads are enough | Preprocessing is CPU-heavy; threads would be serialized by GIL; processes each get their own GIL |

DataLoader prefetches batches in parallel worker processes so the GPU never starves waiting for data. If it used threads, decoding images would serialize on the GIL and the GPU would stall.

The general rule: **I/O-bound → threads, CPU-bound → processes**.

**`num_workers` doesn't parallelize gradient updates — it pipelines data loading with GPU compute**

Gradient updates are inherently serial: batch N must finish before batch N+1 can start (weights change after every update). `num_workers` doesn't change this.

What it does: hides data loading latency behind GPU computation.

Without `num_workers` (serial):
```
CPU: [load B1]──────[load B2]──────[load B3]
GPU:             [train B1]  idle   [train B2]  idle
```
GPU finishes B1, CPU only then starts loading B2 — GPU idles.

With `num_workers=4` (pipelined):
```
workers: [load B1][load B2][load B3][load B4]...
GPU:              [train B1][train B2][train B3]
```
Workers prefetch ahead while GPU trains. By the time GPU finishes B1, B2 is already queued in memory.

This is a producer-consumer pipeline:
```
workers (producers) ──→ prefetch queue ──→ GPU (consumer)
```

`num_workers=0` collapses the pipeline entirely — data loading and training alternate serially, GPU utilization drops. The right value depends on how expensive your data preprocessing is relative to your training step time.

---

### 7. `datasets` library — Arrow columnar storage + mmap

`load_dataset` does more than download files. It converts raw data (JSON, CSV, Parquet) into Apache Arrow format, stored in a separate cache at `~/.cache/huggingface/datasets/`.

**Why columnar storage matters**

Row storage (JSON) interleaves columns:

```
[row 1]  question="How do I..." answer="To do this..."
[row 2]  question="What is..." answer="This is..."
```

To read all `question` values you must skip over all `answer` bytes to find each next question.

Arrow stores each column contiguously:

```
[question col]  q1  q2  q3  q4 ... q200
[answer col]    a1  a2  a3  a4 ... a200
```

Reading only `question` touches zero bytes of `answer`. On a 100M-row dataset where you only need one column, this is the difference between reading 1GB vs 50GB.

**How memory mapping works**

`load_dataset` opens the Arrow file via mmap rather than `file.read()`.

Regular `read()`:
```
disk → kernel buffer → copy → process RAM    (data in RAM = full file size)
```

mmap:
```
disk → virtual address space (no copy yet)
```

`mmap()` reserves a region of the process's *virtual* address space and records which file it maps to. Nothing is in RAM yet.

When you actually access data, the CPU looks up the page table:

- **Page table hit** → physical RAM address known → read directly, fast path
- **Page table miss (page fault)** → OS reads the 4KB page at the corresponding file offset from disk → loads into a physical RAM page → updates the page table → CPU retries → fast path from now on

The analogy to Redis cache is exact:

```
Redis:  cache hit → return  |  cache miss → query DB → populate cache → return
mmap:   page hit  → read RAM  |  page fault → read disk → load page → update table → read RAM
```

**Two practical consequences:**

*Dataset larger than RAM* — OS evicts cold pages (read-only pages are just discarded, no write-back needed) and loads new ones on demand. A 200GB dataset is usable on a 16GB machine.

*Multi-process zero-copy sharing* — DataLoader's 4 worker processes all mmap the same Arrow file. The OS maps them to the same physical pages. The data exists once in RAM regardless of worker count.

**Streaming mode** — skips the download entirely:

```python
ds = load_dataset("some/huge-dataset", streaming=True)
for row in ds["train"]:   # pulls rows from the Hub on the fly, nothing saved locally
    ...
```

---

### 8. Spaces

Spaces are web apps hosted by HF — you push code, they deploy it.

Minimal structure:
```
my-space/
  app.py            ← application code
  requirements.txt  ← dependencies
```

HF spins up a container from these two files and gives you a public URL.

Most common framework is **Gradio** — designed for ML demos, wraps a Python function into a web UI:

```python
import gradio as gr
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer(question, context):
    return qa(question=question, context=context)["answer"]

gr.Interface(fn=answer, inputs=["text", "text"], outputs="text").launch()
```

Key details:
- **Hardware**: free tier is CPU-only; GPU requires paid upgrade
- **Models**: `from_pretrained()` downloads weights at container startup — no need to bundle weights in the Space repo
- **Sleep**: free Spaces hibernate after inactivity, cold-start on next visit (first load is slow)

---

### 9. Hub REST API

`HfApi` is just a Python wrapper around HF's HTTP API. Anything it can do, a plain HTTP request can do too.

```bash
# model metadata
curl https://huggingface.co/api/models/gpt2

# list files in a repo
curl https://huggingface.co/api/models/gpt2/tree/main

# private repo — pass token in header
curl -H "Authorization: Bearer hf_xxx" \
  https://huggingface.co/api/models/your-username/private-model

# search models
curl "https://huggingface.co/api/models?search=bert&limit=5"
```

**Downloading a file without any HF library:**

```bash
curl -L https://huggingface.co/gpt2/resolve/main/config.json -o config.json
```

URL pattern: `huggingface.co/{repo}/resolve/{revision}/{filename}` — revision can be a branch name or commit hash.

**Uploading from CI:**

```bash
curl -X PUT \
  -H "Authorization: Bearer $HF_TOKEN" \
  -F "file=@checkpoint.pt" \
  "https://huggingface.co/api/models/your/model/upload/main/checkpoint.pt"
```

The key point: everything on the Hub has a URL, and a token is all you need to read or write it. The REST API matters when you're outside Python — shell scripts, CI pipelines, or any language with an HTTP client.

---

### 10. Inspecting the local cache

```python
from huggingface_hub import scan_cache_dir

info = scan_cache_dir()
print(f"total size: {info.size_on_disk_str}")

for repo in info.repos:
    print(f"{repo.repo_id:50s} {repo.size_on_disk_str:10s} revisions:{len(repo.revisions)}")
```

Or from the CLI:

```bash
huggingface-cli scan-cache
```

---

### 11. Offline mode

Set `TRANSFORMERS_OFFLINE=1` to force HF to use only local cache and make zero network requests:

```bash
export TRANSFORMERS_OFFLINE=1
```

Or in Python:

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

Useful in production deployments — models are pre-downloaded during a build/setup step, and the running service is airgapped. Without this flag, `from_pretrained` always pings the Hub to check for updates even when the model is already cached, which can cause unexpected failures or slowdowns if the server has no outbound network access.
