"""
Context Precision and Context Recall — measuring retrieval quality.

Both metrics require ground truth: which doc IDs *should* be retrieved
for each question. In practice this comes from human annotation or
a curated eval dataset.

Context Precision = relevant_retrieved / k
  "Of the k chunks we gave the generator, how many were actually useful?"
  High precision = low noise in context.

Context Recall    = relevant_retrieved / total_relevant
  "Of all the chunks that matter, how many did we find?"
  High recall = we didn't miss anything important.

Key tension: increasing k improves recall but hurts precision.
The right k depends on whether your generator handles noise well.
"""

from common import DOCS, load_embedder, build_index, retrieve

# ── ground truth ────────────────────────────────────────────────────────
# Each entry: which doc IDs are truly relevant for this question.
# Everything else retrieved is noise.

TEST_SET = [
    {
        "question": "What does the Pro plan cost and what does it include?",
        "relevant_ids": {"pricing_pro"},
        # 00_rag_pipeline showed: retriever also pulls pricing_basic, pricing_enterprise
    },
    {
        "question": "Which NovaTech models support image inputs?",
        "relevant_ids": {"nova2_specs"},
        # nova1 and nova3 were retrieved but neither mentions image support
    },
    {
        "question": "How long does NovaTech store my API request logs?",
        "relevant_ids": {"data_retention"},
        # clean retrieval — high similarity score means less noise
    },
    {
        "question": "How does NovaTech handle support across different plan tiers?",
        "relevant_ids": {"support_policy", "pricing_basic", "pricing_pro", "pricing_enterprise"},
        # support info is spread across multiple docs — high k needed for full recall
    },
]

# ── metrics ─────────────────────────────────────────────────────────────

def context_precision(retrieved: list[dict], relevant_ids: set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for c in retrieved if c["id"] in relevant_ids)
    return hits / len(retrieved)

def context_recall(retrieved: list[dict], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 1.0
    hits = sum(1 for c in retrieved if c["id"] in relevant_ids)
    return hits / len(relevant_ids)

# ── run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embedder = load_embedder()
    index = build_index(DOCS, embedder)

    for item in TEST_SET:
        q = item["question"]
        relevant = item["relevant_ids"]

        print("=" * 65)
        print(f"Q: {q}")
        print(f"   Ground truth relevant: {relevant}")
        print()
        print(f"  {'k':>2}  {'Precision':>10}  {'Recall':>8}  Retrieved IDs")
        print("  " + "-" * 62)

        for k in [1, 2, 3, 5]:
            chunks = retrieve(q, k=k, index=index, docs=DOCS, embedder=embedder)
            retrieved_ids = [c["id"] for c in chunks]
            prec = context_precision(chunks, relevant)
            rec = context_recall(chunks, relevant)
            print(f"  {k:>2}  {prec:>10.2f}  {rec:>8.2f}  {retrieved_ids}")
        print()
