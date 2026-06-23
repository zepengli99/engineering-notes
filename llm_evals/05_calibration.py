"""
LLM-as-Judge: Calibration

When the judge says "8/10", does that mean anything consistent? Three problems:

1. Scale compression — naive 1-10 prompts cluster scores in 7-9. The model
   rarely gives below 6 ("too harsh") or exactly 10 ("nothing is perfect").
   The effective range is 2-3 points wide, not 9.

2. Self-inconsistency — the same answer rated twice (temperature > 0) can
   produce different scores. A difference of 7 vs 8 may be pure noise.

3. Cross-question incomparability — a 7/10 for Q1 and a 7/10 for Q2 are not
   the same standard. Each question anchors the scale differently.

Mitigation: reference-anchored scoring (used by G-Eval).
Show the judge concrete examples of 1, 5, and 10 before rating.
Scores spread out and are more consistent across runs.
"""

from langchain_groq import ChatGroq
from common import parse_json
from dotenv import load_dotenv

load_dotenv()

# ── question and answer ladder ────────────────────────────────────────────
# Five answers with clearly different quality — from wrong to perfect.

QUESTION = "How long does NovaTech store my API request logs?"

ANSWERS = [
    {
        "label": "Terrible (factually wrong)",
        "text": "NovaTech stores logs indefinitely and may share them with third-party partners for analytics.",
    },
    {
        "label": "Poor (too vague to be useful)",
        "text": "Not very long. It depends on your plan.",
    },
    {
        "label": "Mediocre (correct but incomplete)",
        "text": "API logs are stored for 30 days by default.",
    },
    {
        "label": "Good (correct and covers enterprise)",
        "text": "30 days by default. Enterprise customers can configure retention between 0 and 365 days.",
    },
    {
        "label": "Excellent (complete and precise)",
        "text": "API request logs are retained for 30 days by default. Enterprise customers can configure this between 0 and 365 days. Prompt content is not logged.",
    },
]

# ── prompts ───────────────────────────────────────────────────────────────

NAIVE_PROMPT = """\
Question: {question}
Answer: {answer}

Rate the quality of this answer from 1 to 10.
Output only JSON: {{"score": N, "reason": "one sentence"}}"""

ANCHORED_PROMPT = """\
You are evaluating answers to the question: "{question}"

Scoring guide:
  1  — Factually wrong or completely unhelpful
  5  — Partially correct but missing important details
  10 — Fully accurate, covers all relevant aspects, well-structured

Answer to rate: {answer}

Rate from 1 to 10 using the guide above.
Output only JSON: {{"score": N, "reason": "one sentence"}}"""

# ── functions ─────────────────────────────────────────────────────────────

def rate(question: str, answer: str, prompt_template: str, llm) -> dict:
    prompt = prompt_template.format(question=question, answer=answer)
    return parse_json(llm.invoke(prompt).content)

def rate_n_times(question: str, answer: str, prompt_template: str, llm, n: int) -> list[int]:
    scores = []
    for _ in range(n):
        result = rate(question, answer, prompt_template, llm)
        scores.append(result["score"])
    return scores

# ── run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm_cold = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm_warm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    # ── scale compression: naive vs anchored ─────────────────────────
    print("=" * 65)
    print("SCALE COMPRESSION: naive vs reference-anchored scoring")
    print("=" * 65)
    print(f"\nQuestion: {QUESTION}\n")
    print(f"{'Quality':<32} {'Naive':>6}  {'Anchored':>9}")
    print("-" * 52)

    for item in ANSWERS:
        naive = rate(QUESTION, item["text"], NAIVE_PROMPT, llm_cold)
        anchored = rate(QUESTION, item["text"], ANCHORED_PROMPT, llm_cold)
        naive_s = naive["score"]
        anchored_s = anchored["score"]
        flag = "  ← spread" if abs(naive_s - anchored_s) >= 2 else ""
        print(f"  {item['label']:<30} {naive_s:>6}  {anchored_s:>9}{flag}")

    # ── self-inconsistency: same answer, multiple runs ────────────────
    print("\n" + "=" * 65)
    print("SELF-INCONSISTENCY: same answer rated 5 times (temperature=0.7)")
    print("=" * 65)

    target = ANSWERS[3]["text"]  # the "Good" answer
    print(f"\nAnswer: {target}\n")

    naive_scores = rate_n_times(QUESTION, target, NAIVE_PROMPT, llm_warm, n=5)
    anchored_scores = rate_n_times(QUESTION, target, ANCHORED_PROMPT, llm_warm, n=5)

    naive_range = max(naive_scores) - min(naive_scores)
    anchored_range = max(anchored_scores) - min(anchored_scores)

    print(f"Naive    scores: {naive_scores}  (range: {naive_range})")
    print(f"Anchored scores: {anchored_scores}  (range: {anchored_range})")
    print()
    if naive_range > 0:
        print(f"  Naive range = {naive_range} — a difference of {naive_range} point(s) between runs may be noise, not signal")
    if anchored_range < naive_range:
        print(f"  Anchored range = {anchored_range} — more consistent with reference anchors")
