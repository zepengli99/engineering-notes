"""
Streaming demo — shows TTFT vs TBT in real time.

vLLM streams tokens as Server-Sent Events (SSE).
We measure:
  TTFT: time from request sent to first token received
  TBT:  average time between subsequent tokens
"""

import urllib.request
import json
import time

BASE_URL = "http://localhost:8000"
MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT = (
    "Explain how the Linux kernel manages virtual memory, "
    "covering page tables, TLB, and page faults."
)


def stream_chat(prompt: str) -> None:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t_request   = time.perf_counter()
    t_first     = None
    token_times = []

    print(f"Prompt: {prompt}\n")
    print("-" * 60)

    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if not delta:
                continue

            now = time.perf_counter()
            if t_first is None:
                t_first = now
                ttft_ms = (t_first - t_request) * 1000
                print(f"\n[TTFT: {ttft_ms:.0f}ms] ", end="", flush=True)
            else:
                token_times.append(now)

            print(delta, end="", flush=True)

    print("\n" + "-" * 60)

    if t_first and token_times:
        ttft_ms = (t_first - t_request) * 1000
        intervals = [
            (token_times[i] - token_times[i-1]) * 1000
            for i in range(1, len(token_times))
        ]
        avg_tbt = sum(intervals) / len(intervals) if intervals else 0
        total_s = (token_times[-1] - t_request)

        print(f"\nTTFT (time to first token) : {ttft_ms:.0f} ms")
        print(f"TBT  (avg between tokens)  : {avg_tbt:.0f} ms")
        print(f"Total tokens generated     : {len(token_times) + 1}")
        print(f"Effective throughput       : {(len(token_times)+1)/total_s:.1f} tok/s")


if __name__ == "__main__":
    stream_chat(PROMPT)
