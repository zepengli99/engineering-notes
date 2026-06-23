"""
LLM-as-Judge — can you trust the judge's scores?

We've already used LLM-as-judge twice:
  02_faithfulness.py  — verify each claim against context
  03_answer_relevance.py — generate back-questions

This script examines the judge itself. Two well-documented failure modes:

1. Verbosity bias — the judge rates longer answers higher even when the
   shorter one is equally correct. Length signals effort; effort signals quality.
   The judge conflates the two.

2. Position bias — in pairwise comparison ("A or B?"), the judge favours
   whichever answer appears first, independent of content quality.

Mitigation: swap augmentation — run each pair in both orderings.
If the winner flips, position drove the result, not content.
"""

from common import make_llm, parse_json

# ── scenarios ────────────────────────────────────────────────────────────

QUESTION_LOG = "How long does NovaTech store my API request logs?"

# Same facts, very different length — used to surface verbosity bias
BRIEF = "30 days by default. Enterprise customers can configure 0 to 365 days."

VERBOSE = (
    "NovaTech's data retention policy specifies that API request logs are retained "
    "for a default period of 30 days. Enterprise customers have the flexibility to "
    "configure this retention period anywhere between 0 and 365 days based on their "
    "specific compliance and operational requirements. It is also worth noting that "
    "prompt content is not included in these logs by default, providing an additional "
    "layer of privacy for your data."
)

QUESTION_PRO = "What does the NovaTech Pro plan include?"

# Two correct answers phrased differently — used to surface position bias
ANSWER_A = (
    "The Pro plan costs $99/month and includes 10,000 API calls per day, "
    "priority support with a 4-hour response time, and access to Nova-1 and Nova-2."
)
ANSWER_B = (
    "For $99 a month, NovaTech Pro gives you 10,000 daily API calls, "
    "access to both Nova-1 and Nova-2 models, and priority support with a 4-hour SLA."
)

# ── prompts ──────────────────────────────────────────────────────────────

RATE_PROMPT = """\
Question: {question}
Answer: {answer}

Rate the quality of this answer from 1 to 10. Be objective.
Output only JSON: {{"score": N, "reason": "one sentence"}}"""

PAIRWISE_PROMPT = """\
Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Which answer better addresses the question? Output only JSON:
{{"winner": "A" or "B", "reason": "one sentence"}}"""

# ── judge functions ───────────────────────────────────────────────────────

def rate(question: str, answer: str, llm) -> dict:
    response = llm.invoke(RATE_PROMPT.format(question=question, answer=answer))
    return parse_json(response.content)

def pairwise(question: str, answer_a: str, answer_b: str, llm) -> dict:
    response = llm.invoke(PAIRWISE_PROMPT.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    ))
    return parse_json(response.content)

def swap_augment(question: str, answer_a: str, answer_b: str, llm) -> dict:
    """Run pairwise comparison in both orderings. Flag if winner changes."""
    result_ab = pairwise(question, answer_a, answer_b, llm)
    result_ba = pairwise(question, answer_b, answer_a, llm)

    # normalise: result_ba's winner is in terms of B/A labels, flip back to A/B
    winner_ab = result_ab["winner"]
    winner_ba = "A" if result_ba["winner"] == "B" else "B"  # flip perspective

    consistent = winner_ab == winner_ba
    return {
        "ab_winner": winner_ab, "ab_reason": result_ab["reason"],
        "ba_winner": winner_ba, "ba_reason": result_ba["reason"],
        "consistent": consistent,
        "verdict": winner_ab if consistent else "INCONSISTENT — likely position bias",
    }

# ── run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = make_llm()

    # ── verbosity bias ────────────────────────────────────────────────
    print("=" * 60)
    print("VERBOSITY BIAS: same facts, different length")
    print("=" * 60)
    print(f"\nQuestion: {QUESTION_LOG}\n")
    print(f"BRIEF:   {BRIEF}\n")
    print(f"VERBOSE: {VERBOSE}\n")

    score_brief = rate(QUESTION_LOG, BRIEF, llm)
    score_verbose = rate(QUESTION_LOG, VERBOSE, llm)

    print(f"Brief score:   {score_brief['score']}/10 — {score_brief['reason']}")
    print(f"Verbose score: {score_verbose['score']}/10 — {score_verbose['reason']}")
    if score_verbose["score"] > score_brief["score"]:
        print("\n  → Verbose scored higher despite identical information (verbosity bias)")
    elif score_brief["score"] > score_verbose["score"]:
        print("\n  → Brief scored higher (no verbosity bias this time)")
    else:
        print("\n  → Equal scores")

    # ── position bias + swap augmentation ────────────────────────────
    print("\n" + "=" * 60)
    print("POSITION BIAS: two equivalent answers, swap augmentation")
    print("=" * 60)
    print(f"\nQuestion: {QUESTION_PRO}")
    print(f"\nAnswer A: {ANSWER_A}")
    print(f"\nAnswer B: {ANSWER_B}\n")

    result = swap_augment(QUESTION_PRO, ANSWER_A, ANSWER_B, llm)

    print(f"A then B → winner: {result['ab_winner']}  ({result['ab_reason']})")
    print(f"B then A → winner: {result['ba_winner']}  ({result['ba_reason']})")
    print(f"\nConsistent: {result['consistent']}")
    print(f"Verdict:    {result['verdict']}")
