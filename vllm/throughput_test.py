"""
Serial vs concurrent throughput comparison.

Serial:     send requests one at a time, wait for each to finish
Concurrent: send all requests simultaneously, vLLM batches them

Metric: total tokens / total wall-clock seconds
"""

import urllib.request
import json
import time
import threading

BASE_URL = "http://localhost:8000"
MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPTS = [
    "Explain what a binary tree is.",
    "What is the difference between a process and a thread?",
    "Explain how TCP handshake works.",
    "What is a database index and why is it useful?",
    "What is the difference between stack and heap memory?",
    "Explain what REST API means.",
    "What is a mutex?",
    "What is garbage collection in programming?",
]


def chat(prompt: str) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ── serial ────────────────────────────────────────────────────────────────────

def run_serial():
    print("SERIAL  (one request at a time)")
    print("-" * 40)
    results = []
    t_start = time.time()
    for i, p in enumerate(PROMPTS, 1):
        r = chat(p)
        results.append(r)
        toks = r["usage"]["total_tokens"]
        print(f"  req {i}: {toks} tokens")
    elapsed = time.time() - t_start
    total   = sum(r["usage"]["total_tokens"] for r in results)
    print(f"\n  total tokens : {total}")
    print(f"  wall time    : {elapsed:.1f}s")
    print(f"  throughput   : {total/elapsed:.1f} tokens/s")
    return total / elapsed


# ── concurrent ────────────────────────────────────────────────────────────────

def run_concurrent():
    print("\nCONCURRENT  (all requests at once)")
    print("-" * 40)
    results = [None] * len(PROMPTS)

    def worker(i, prompt):
        results[i] = chat(prompt)

    threads = [
        threading.Thread(target=worker, args=(i, p))
        for i, p in enumerate(PROMPTS)
    ]

    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t_start

    total = sum(r["usage"]["total_tokens"] for r in results)
    for i, r in enumerate(results, 1):
        print(f"  req {i}: {r['usage']['total_tokens']} tokens")
    print(f"\n  total tokens : {total}")
    print(f"  wall time    : {elapsed:.1f}s")
    print(f"  throughput   : {total/elapsed:.1f} tokens/s")
    return total / elapsed


# ── main ──────────────────────────────────────────────────────────────────────

print("=" * 40)
print(f"Throughput test  ({len(PROMPTS)} requests)")
print("=" * 40 + "\n")

serial_tps     = run_serial()
concurrent_tps = run_concurrent()

print("\n" + "=" * 40)
print(f"  serial     : {serial_tps:.1f} tok/s")
print(f"  concurrent : {concurrent_tps:.1f} tok/s")
print(f"  speedup    : {concurrent_tps/serial_tps:.1f}x")
print("=" * 40)
