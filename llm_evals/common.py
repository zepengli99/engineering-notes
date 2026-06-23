"""
Shared RAG pipeline used by all eval scripts.

NovaTech knowledge base: 10 documents covering products, pricing, and policies.
Embeddings: sentence-transformers all-MiniLM-L6-v2 (local, no API key needed).
Vector store: FAISS with cosine similarity.
LLM: Groq (requires GROQ_API_KEY in .env).
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── corpus ──────────────────────────────────────────────────────────────

DOCS = [
    {"id": "pricing_basic",      "text": "NovaTech Basic plan costs $29/month. Includes 1,000 API calls per day and access to the Nova-1 model. Support responds within 48 hours via email."},
    {"id": "pricing_pro",        "text": "NovaTech Pro plan costs $99/month. Includes 10,000 API calls per day, priority support with 4-hour response time, and access to Nova-1 and Nova-2 models."},
    {"id": "pricing_enterprise", "text": "NovaTech Enterprise plan is custom-priced. Includes unlimited API calls, a dedicated Slack support channel with 1-hour SLA, and access to all models including Nova-3."},
    {"id": "nova1_specs",        "text": "Nova-1 is NovaTech's base model with 7B parameters. Supports an 8k context window and processes approximately 85 tokens per second."},
    {"id": "nova2_specs",        "text": "Nova-2 is NovaTech's mid-tier model with 35B parameters. Supports a 32k context window, accepts image inputs, and processes 40 tokens per second."},
    {"id": "nova3_specs",        "text": "Nova-3 is NovaTech's flagship model with 175B parameters. Supports a 128k context window with built-in reasoning and tool use. Available on Enterprise plan only."},
    {"id": "data_retention",     "text": "NovaTech retains API request logs for 30 days by default. Enterprise customers can configure retention between 0 and 365 days. Prompt content is not logged."},
    {"id": "rate_limits",        "text": "Rate limits are enforced per API key. Exceeding the daily quota returns HTTP 429. Unused quota does not roll over to the next day."},
    {"id": "support_policy",     "text": "Standard support (Basic plan) responds within 48 hours via email. Priority support (Pro plan) responds within 4 hours. Enterprise gets a dedicated Slack channel with 1-hour SLA."},
    {"id": "refund_policy",      "text": "NovaTech offers a 14-day money-back guarantee for Basic and Pro plans. Enterprise contracts are non-refundable but include a 30-day pilot period before billing starts."},
]

# ── embedder and index ──────────────────────────────────────────────────

def load_embedder() -> SentenceTransformer:
    # downloads ~80MB on first run, cached afterwards
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index(docs: list[dict], embedder: SentenceTransformer) -> faiss.IndexFlatIP:
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim = dot product on unit vectors
    index.add(embeddings)
    return index

# ── retrieve and generate ───────────────────────────────────────────────

def retrieve(query: str, k: int, index: faiss.IndexFlatIP, docs: list[dict], embedder: SentenceTransformer) -> list[dict]:
    query_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_emb, k)
    return [
        {**docs[i], "score": float(scores[0][j])}
        for j, i in enumerate(indices[0])
    ]

def make_llm() -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def generate(query: str, chunks: list[dict], llm: ChatGroq) -> str:
    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    prompt = f"""Answer the question using only the provided context. Be concise and factual.

Context:
{context}

Question: {query}"""
    return llm.invoke(prompt).content

# ── JSON parsing helper ─────────────────────────────────────────────────

def parse_json(text: str) -> dict | list:
    """Strip markdown code fences then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])  # drop first and last fence lines
    return json.loads(text)
