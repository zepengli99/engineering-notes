"""
RAG pipeline walkthrough — the system we'll evaluate.

Runs 3 questions through the full pipeline and prints every step:
  question → retrieved chunks (with similarity scores) → generated answer

Run this first to build intuition for what the eval scripts are measuring.
"""

from common import DOCS, load_embedder, build_index, retrieve, generate, make_llm

QUESTIONS = [
    "What does the Pro plan cost and what does it include?",
    "Which NovaTech models support image inputs?",
    "How long does NovaTech store my API request logs?",
]

if __name__ == "__main__":
    print("Loading embedder and building index...")
    embedder = load_embedder()
    index = build_index(DOCS, embedder)
    llm = make_llm()
    print(f"Index built: {len(DOCS)} documents\n")

    for q in QUESTIONS:
        print("=" * 60)
        print(f"QUESTION: {q}")

        chunks = retrieve(q, k=3, index=index, docs=DOCS, embedder=embedder)

        print("\nRETRIEVED CHUNKS (top 3 by cosine similarity):")
        for c in chunks:
            print(f"  [{c['score']:.3f}] {c['id']}")
            print(f"           {c['text'][:90]}...")

        answer = generate(q, chunks, llm)
        print(f"\nANSWER:\n  {answer}")
        print()
