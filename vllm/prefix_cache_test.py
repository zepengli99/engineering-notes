"""
Prefix cache hit rate demo.

Sends N requests that all share the same long system prompt.
After each batch, polls vLLM's /metrics endpoint to read
  - gpu_kv_cache_usage_perc
  - prefix_cache_hit_rate

The hit rate should climb from 0% toward 100% as vLLM reuses
the cached KV blocks for the shared prefix instead of recomputing.
"""

import urllib.request
import urllib.error
import json
import time
import threading

BASE_URL   = "http://localhost:8000"
MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"

# Long system prompt — the more tokens, the more blocks get shared.
# ~120 tokens so it spans several KV cache blocks.
SYSTEM_PROMPT = (
    "You are a helpful assistant specialised in explaining computer science concepts. "
    "When answering, be concise and precise. Focus on the core idea first, then add "
    "detail only if needed. Avoid unnecessary filler phrases. Every answer should "
    "demonstrate deep understanding of the underlying mechanism, not just surface-level "
    "description. This system prompt is intentionally long to occupy multiple KV cache "
    "blocks so that prefix caching has a measurable effect on the hit rate."
)

QUESTIONS = [
    "What is a hash table?",
    "Explain binary search.",
    "What is a deadlock?",
    "What does O(n log n) mean?",
    "What is a stack overflow?",
    "Explain TCP vs UDP in one sentence each.",
    "What is memoisation?",
    "What is a race condition?",
]


def chat(question: str) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": question},
        ],
        "max_tokens": 80,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_metrics() -> dict:
    """Parse vLLM's Prometheus metrics text into a dict of name -> float."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=5) as resp:
            text = resp.read().decode()
    except Exception:
        return {}

    metrics = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return metrics


def print_metrics(label: str) -> None:
    m       = get_metrics()
    prefix  = '{engine="0",model_name="Qwen/Qwen2.5-1.5B-Instruct"}'
    queries = m.get(f"vllm:prefix_cache_queries_total{prefix}", 0)
    hits    = m.get(f"vllm:prefix_cache_hits_total{prefix}", 0)
    kv      = m.get(f"vllm:kv_cache_usage_perc{prefix}", None)
    hit_str = f"{hits/queries*100:.1f}%" if queries > 0 else "0.0%"
    kv_str  = f"{kv*100:.1f}%" if kv is not None else "n/a"
    print(f"  [{label}]  prefix hit rate: {hit_str} ({int(hits)}/{int(queries)} tokens)   KV cache: {kv_str}")


# ── main ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Prefix Cache Hit Rate Demo")
print(f"System prompt: ~{len(SYSTEM_PROMPT.split())} words")
print("=" * 60)

print_metrics("before")

for i, q in enumerate(QUESTIONS, 1):
    t0  = time.time()
    res = chat(q)
    ms  = (time.time() - t0) * 1000
    ans = res["choices"][0]["message"]["content"][:60].replace("\n", " ")
    print(f"\nQ{i}: {q}")
    print(f"  -> {ans}...")
    print(f"  latency: {ms:.0f}ms  |  tokens: {res['usage']['total_tokens']}")
    print_metrics(f"after Q{i}")
