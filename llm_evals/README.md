# LLM Evals

How to measure whether an AI system is actually working. Starts with RAG evaluation, the most structured and common eval problem in production systems.

## Code examples

| File | What it demonstrates |
|---|---|
| [common.py](common.py) | Shared RAG pipeline: corpus, FAISS index, retriever, generator |
| [00_rag_pipeline.py](00_rag_pipeline.py) | The system we're evaluating — run this first |
| [01_context_metrics.py](01_context_metrics.py) | Context Precision and Context Recall |
| [02_faithfulness.py](02_faithfulness.py) | Faithfulness — detecting hallucination without ground truth |
| [03_answer_relevance.py](03_answer_relevance.py) | Answer Relevance — did the answer address the question? |
| [04_llm_judge.py](04_llm_judge.py) | Position bias, verbosity bias, swap augmentation |
| [05_calibration.py](05_calibration.py) | Scale compression, reference-anchored scoring, self-inconsistency |

---

## Why RAG evaluation is structured differently from normal software testing

A regular function either returns the right value or it doesn't. A RAG pipeline has two independent failure modes:

```
Question → [Retriever] → Chunks → [Generator] → Answer
               ↑                        ↑
     Did we fetch the right info?  Did we use it correctly?
```

These two steps can fail independently:
- Good retrieval + bad generation → hallucination despite having the right context
- Bad retrieval + good generation → confident, well-formed answer that's missing key info
- Bad retrieval + bad generation → both problems compound

Measuring the final answer alone doesn't tell you which component failed. You need separate metrics for each stage. The pipeline being evaluated here — retriever, embeddings, vector search, reranker, generator — is built in [LLM Architecture → RAG](../llm_architecture/README.md#rag-retrieval-augmented-generation).

---

## The four RAG metrics

| Metric | Measures | Needs ground truth? |
|---|---|---|
| Context Precision | Of retrieved chunks, what fraction are relevant? | Yes |
| Context Recall | Of all relevant chunks, what fraction were retrieved? | Yes |
| Faithfulness | Is every claim in the answer supported by the context? | No |
| Answer Relevance | Does the answer actually address the question? | No |

Precision and Recall measure the retriever. Faithfulness and Answer Relevance measure the generator.

The reference-free metrics (Faithfulness, Answer Relevance) are especially valuable in production because you don't need a labelled dataset — you can run them continuously on live traffic.

---

## Context Precision and Context Recall

### Definitions

```
Context Precision = relevant_retrieved / k
Context Recall    = relevant_retrieved / total_relevant
```

Both require knowing which documents *should* have been retrieved — ground truth that typically comes from human annotation.

### The k tradeoff

From running `01_context_metrics.py` on the NovaTech corpus:

```
Q: "What does the Pro plan cost and what does it include?"
Ground truth relevant: {pricing_pro}

  k   Precision    Recall  Retrieved
  1        1.00      1.00  [pricing_pro]
  2        0.50      1.00  [pricing_pro, pricing_enterprise]
  3        0.33      1.00  [pricing_pro, pricing_enterprise, pricing_basic]
  5        0.20      1.00  [pricing_pro, ..., refund_policy, support_policy]
```

As k increases: recall stays at 1.0 (correct doc was always ranked #1), precision drops linearly. Every additional chunk is noise.

### Recall has a ceiling set by retriever quality, not k

The support-across-tiers question has four relevant documents. Ground truth: `{pricing_basic, pricing_enterprise, pricing_pro, support_policy}`.

```
  k   Precision    Recall  Retrieved
  1        1.00      0.25  [pricing_enterprise]
  2        1.00      0.50  [pricing_enterprise, pricing_pro]
  3        1.00      0.75  [pricing_enterprise, pricing_pro, pricing_basic]
  5        0.60      0.75  [..., refund_policy, nova3_specs]   ← recall stuck at 0.75
```

At k=5, recall is still 0.75 — `support_policy` was never retrieved. The embedding model ranked `refund_policy` and `nova3_specs` higher because the query "support across plan tiers" was interpreted semantically as a plan comparison, not a support policy lookup. Increasing k further won't fix this; it would only add more noise.

**The ceiling on recall is a retriever quality problem, not a k problem.**

### Precision vs Recall: which to optimise first?

**Prioritise recall first, then fix precision with a reranker.**

```
Stage 1 — Retriever (high recall):   fetch top 50–100 candidates
Stage 2 — Reranker (high precision): cross-encoder scores each against the query, keep top k
```

Why recall first: a chunk that wasn't retrieved cannot be recovered downstream. A noisy chunk can still be ignored by a capable generator or filtered by a reranker. Missing context is an unrecoverable loss; extra noise is manageable.

The only exception: extremely small context windows or a generator known to be heavily distracted by irrelevant content.

---

## Faithfulness

**Business meaning: did the generator stay within what it was given?**

For every claim in the answer, ask: does this come from the retrieved chunks? That's it. Whether the claim is factually correct in the real world is irrelevant — Faithfulness only cares about whether each claim has a traceable source in the context. A claim with no source in the chunks is a hallucination, regardless of whether it happens to be true.

Reference-free — no ground-truth answer needed, so it can run continuously on live traffic.

### Algorithm

```
1. Extract every factual claim from the answer        (LLM call)
2. For each claim: can this be inferred from context? (LLM call per claim)
3. Faithfulness = supported_claims / total_claims
```

### What it detects

```
Retrieved: "Pro plan costs $99/month, includes 10,000 API calls"
Answer:    "Pro plan costs $99/month, includes 10,000 API calls, plus a free 7-day trial"
                                                                        ↑
                                                       not in any retrieved chunk → ✗
```

The "free 7-day trial" claim gets flagged even if it were actually true. The generator went beyond what it was given — that's the problem Faithfulness catches.

### Faithfulness ≠ Accuracy

A low faithfulness score means the generator went beyond its context. That's not the same as being wrong:

| | Faithful | Not faithful |
|---|---|---|
| **Correct** | answer came from context, context was right | LLM used world knowledge, happened to be right |
| **Incorrect** | answer came from context, context was wrong | LLM hallucinated something false |

Faithfulness catches hallucination relative to context. A separate accuracy metric (which needs ground truth) is required to catch cases where the context itself is wrong or outdated.

---

## Answer Relevance

**Business meaning: did the answer actually address the question that was asked?**

An answer can be factually correct and fully faithful to its context, yet still be useless — if it answers the wrong question. Answer Relevance catches this. Like Faithfulness, it is reference-free and can run on live traffic.

### Algorithm

```
1. Generate n questions from the answer (pretend the original question is unknown)
2. Embed the original question and all n generated questions
3. Answer Relevance = mean cosine similarity(original, generated_i)
```

The intuition: if the answer is on-topic, you can reverse-engineer the original question from it, and the generated questions will cluster near the original in embedding space. If the answer wanders off-topic, the generated questions diverge.

### Results on the NovaTech corpus

```
Original question: "How long does NovaTech store my API request logs?"

SCENARIO A — on-topic answer
  [0.463] What is the default data retention period?
  [0.406] How long are records kept by default?
  [0.246] What are the data retention options for enterprise customers?
  Answer Relevance: 0.372

SCENARIO B — off-topic answer (answers a pricing question instead)
  [0.328] What are the different pricing plans offered by NovaTech?
  [0.428] How much does NovaTech charge for its services?
  [0.324] What are the pricing tiers and their respective costs?
  Answer Relevance: 0.360
```

Direction is correct (A > B), but the gap is only 0.012 — much smaller than expected.

### Why the gap is small: embedding model quality matters

`all-MiniLM-L6-v2` captures surface-level similarity (both answers mention NovaTech, both are question-shaped) and misses deeper semantic differences. The scores converge toward a baseline similarity regardless of topic.

RAGAS uses `text-embedding-ada-002` or `text-embedding-3-small`, which produce much sharper semantic distinctions and wider score separation between on-topic and off-topic answers.

**The embedding model is not just an implementation detail — it directly determines how sensitive this metric is.** A weak embedder produces scores that are technically correct in direction but too compressed to be actionable.

---

## LLM-as-Judge

Faithfulness and Answer Relevance both rely on an LLM to make judgments (verify claims, generate back-questions). So do many other eval patterns. The question is: can you trust the judge?

### Two documented failure modes

**Verbosity bias** — the judge conflates length with quality. Longer answers score higher even when they contain identical information. The judge learned from training data where thoroughness correlated with quality; it transfers that signal to length as a proxy.

From `04_llm_judge.py`:

```
Q: "How long does NovaTech store my API request logs?"

Brief answer:   "30 days by default. Enterprise can configure 0 to 365 days."
Verbose answer: (same facts + padding about compliance and privacy)

Brief score:   8/10 — "lacks additional context"
Verbose score: 9/10
```

The verbose answer gained a point for "additional context" that added no new information.

**Position bias** — in pairwise comparison ("A or B?"), the judge favours whichever answer appears in a particular position, independent of content. The same pair in reversed order often produces a different winner.

From `04_llm_judge.py` (two equivalent answers about the Pro plan):

```
A then B → winner: B  "B is more concise, A unnecessarily mentions unit of time for monthly cost"
B then A → winner: A  "B unnecessarily includes the specific SLA timeframe"
```

The winner flipped — but more importantly, the **reasons contradict each other**. Both times the judge picked the answer presented **second** (recency bias). It didn't reason and then decide; it decided based on position and then constructed a reason that sounds logical. This is post-hoc rationalization: the explanation is decoration, not the actual basis for the verdict.

Position bias can manifest as primacy (prefer first) or recency (prefer last) depending on the model and prompt. What's consistent is that position — not content — drove the result.

**Why post-hoc rationalization matters more than bias itself**: if you use the judge's stated reasoning to decide which component of your system to fix, you may be acting on fabricated logic.

### Mitigation: swap augmentation

Run each pairwise comparison twice with the order reversed. If the winner changes, position drove the result.

```python
result_ab = judge(question, A, B)   # A first
result_ba = judge(question, B, A)   # B first, flip winner label back to A/B

if result_ab.winner != result_ba.winner:
    verdict = "INCONSISTENT — do not trust"
else:
    verdict = result_ab.winner  # consistent across orderings, more reliable
```

Swap augmentation detects bias but doesn't eliminate it. Consistent results across orderings still doesn't guarantee the judge is right — it might be consistently biased toward one answer for a non-content reason.

### Calibration: does a score actually mean something?

Three calibration problems, tested in `05_calibration.py`:

**1. Scale compression (most practically harmful)**

Naive 1-10 rating on five answers of clearly different quality:

```
Quality                        Naive   Anchored
Terrible (factually wrong)         2          5
Poor (too vague)                   2          5
Mediocre (correct, incomplete)     8          5  ← anchoring helped here
Good (correct + enterprise info)   9         10
Excellent (complete, precise)      9         10  ← indistinguishable from Good
```

Good and Excellent both score 9/10 — a system using scores to rank outputs cannot distinguish them. Anchoring moved Mediocre from 8→5 (more realistic), but pushed Good and Excellent both to 10/10 — still indistinguishable at the top. Extreme low scores are also resisted: a factually wrong answer still scored 5/10 with anchors, not 1/10.

**2. Self-inconsistency depends on task ambiguity, not temperature**

The same answer rated 5 times at temperature=0.7:
```
Naive:    [9, 9, 9, 9, 9]   range = 0
Anchored: [10, 10, 10, 10, 10]  range = 0
```

No variance despite high temperature. Self-inconsistency is a function of task ambiguity, not temperature. Evaluating a clear factual question produces confident, stable scores. Inconsistency surfaces on subjective tasks or genuine edge cases where even a human would hesitate.

**3. Cross-question incomparability**

A score is anchored to an implicit standard set by the question. A 7/10 on Q1 and a 7/10 on Q2 are not the same thing. Scores cannot be compared across questions without explicit shared anchors.

**Practical implication**: use judge scores for within-question ranking, not cross-question comparison. For tracking quality over time, fix the question set and compare scores on the same questions across model versions.

### When LLM-as-judge is and isn't reliable

| Use case | Reliability | Why |
|---|---|---|
| Detecting obvious failures (hallucination, off-topic) | High | Large signal, hard to miss |
| Pairwise ranking of clearly different outputs | Medium | Direction usually right, magnitude unreliable |
| Pairwise ranking of similar outputs | Low | Position bias dominates when content is close |
| Precise numeric scores (7.2 vs 7.8) | Low | Scores are not calibrated across runs |
| Explaining *why* one answer is better | Low | Post-hoc rationalization |

**Rule of thumb**: use LLM-as-judge to catch failures and rough ordering. Use human evaluation for precise comparisons and to validate that the judge's rankings match human preferences on your specific task.

---

## Online Evaluation

Everything above is **offline eval**: curate a fixed test set, run the system on it, collect scores. Useful for pre-launch regression testing, but has two fundamental limits:

1. **Test sets go stale.** User query patterns drift over time. A test set written at launch doesn't represent what users ask six months later.
2. **You don't know what you don't know.** A curated test set covers scenarios you thought of. Production traffic surfaces scenarios you didn't.

Online eval runs evaluation continuously on sampled live traffic, catching what offline eval misses. This is the eval half of the production loop in [LLM Architecture → MLOps](../llm_architecture/README.md#mlops-from-research-to-production) — versioning code, data, and weights, then monitoring live behaviour.

### Which metrics work online

Only **reference-free** metrics are viable — Faithfulness and Answer Relevance. Context Precision and Recall both require ground truth (which chunks are truly relevant), which doesn't exist in production without dedicated annotation. Faithfulness and Answer Relevance only need `(question, chunks, answer)`, which the system already produces as natural logs.

### Architecture

Eval must never block the user-facing request. The pattern is async consumption from a queue:

```
User request
    │
    ├──► RAG system handles normally, returns answer to user
    │
    └──► Write (question, chunks, answer) to queue — fire and forget

              │  (async, separate process)
              ▼
         Eval worker consumes queue
              │
              ├── Sample (e.g. 5% of requests)
              ├── Run Faithfulness
              ├── Run Answer Relevance
              └── Write to time-series metrics store → dashboard + alerts
```

### Sampling strategy

Evaluating every request is expensive — eval itself calls an LLM. 5% random sampling is enough to detect trends. But sampling rate should vary by context:

| Situation | Strategy |
|---|---|
| High-stakes query types (medical, legal, financial) | Full or near-full sampling — error cost is high |
| Large variety of query categories | Stratified sampling — random sampling over-represents high-frequency categories, leaving rare but important types uncovered |
| Requests with anomaly signals | Always 100% — user gave negative feedback, request timed out, retrieval similarity score was abnormally low. These are already flagged as likely problems; eval cost here has the highest return |

### The offline / online feedback loop

```
Offline eval ── deployment gate ──► blocks regressions from shipping

Online eval ── production monitor ──► discovers new failures offline didn't cover
                    │
                    ├── triggers alert → rollback / traffic cutover / graceful degradation
                    │
                    └── failing examples fed back into offline test set
                                   │
                                   ▼
                         next deployment: offline eval now covers this case
                         same failure mode cannot slip through again
```

**Offline eval is necessary but not sufficient.** Passing offline eval only proves the problems you knew about aren't present. Online eval is the only way to close the gap between known and unknown failures.

Concretely: set a threshold on a rolling-window metric (e.g. Faithfulness 30-minute average drops below 0.75) → alert fires → on-call investigates → rollback or hotfix.

### Drift vs Anomaly

These are two distinct problems:

- **Anomaly** — sudden spike. Faithfulness drops from 0.85 to 0.5 after a deploy. Easy to detect with threshold alerts.
- **Drift** — slow, gradual shift over weeks. No obvious breakpoint. Threshold alerts miss it because no single moment crosses the line.

Three types of drift, each with a different cause and detection strategy:

| | Cause | Do metrics catch it? | Detection |
|---|---|---|---|
| Query drift | User behaviour changed | Not necessarily | Cluster query embeddings, watch cluster distribution shift |
| Knowledge drift | Knowledge base went stale | No | Document timestamps + external signals (user complaints, manual review) |
| Model drift | LLM provider silently updated | Sometimes | Canary eval set — run fixed questions daily, compare scores |

**Query drift**: the types of questions users ask shift over time. Faithfulness and Answer Relevance may still look fine — the system is answering a new category of question, just badly, and the metrics haven't surfaced it yet. Detection: embed every incoming query, cluster periodically, watch for new clusters emerging. A growing new cluster means a query type the system wasn't built for is arriving.

**Knowledge drift**: the knowledge base is outdated — documents say one thing, reality has changed. The system faithfully retrieves stale information and generates an answer that scores high on Faithfulness (every claim came from context) but is factually wrong (the context itself is wrong). Metrics cannot detect this. Mitigations: timestamp every document and enforce review cycles; rely on external signals like user negative feedback; for high-traffic documents, set a forced re-verification interval.

**Model drift**: the LLM provider does a silent model update — same version string, different behaviour. Detection: maintain a canary eval set, a small fixed set of questions with known correct answers, and run it on a schedule (daily or per-deploy). Score changes signal that the model changed, not the queries. The canary set is not for measuring overall quality — it is specifically for detecting system-level behavioural change.

---

## Synthetic Eval Dataset Generation

Hand-writing a test set like `TEST_SET` in `01_context_metrics.py` doesn't scale. A knowledge base with thousands of documents needs ground truth generated automatically.

### The basic pipeline

Instead of: question → human labels → relevant doc IDs

Do: document → LLM → (question, answer, relevant doc IDs)

```
For each document:
  1. LLM reads the document
  2. Generates candidate questions whose answers are in that document
  3. Ground truth (relevant_doc_ids) is known by construction
  4. A second LLM pass validates the questions before they enter the dataset
```

### Generator/evaluator leakage

Do not use the same model to generate questions and to evaluate answers. The same model tends to generate answers that match its own style, then scores those answers more favourably — a self-serving closed loop that produces inflated scores with no real signal.

In practice: use different model families (e.g. llama generates, qwen validates), or use a stronger model to validate what a weaker model generated.

### What synthetic data misses

LLM-generated questions are clean, well-formed, and cluster around the most salient parts of documents. Real user queries are:

- Vague: "how does the Pro thing work?"
- Cross-document: "if I upgrade from Basic to Pro, what changes?"
- Out of scope: questions the knowledge base doesn't cover at all
- Typo-filled and informal

**Synthetic eval performance is not a reliable predictor of real-world performance.** The query distributions are different.

### How to use synthetic data in practice

| Phase | Source | Purpose |
|---|---|---|
| Bootstrap | Synthetic generation | Get an eval set fast, before any production traffic |
| Validate | Small human-annotated set | Confirm the LLM judge's scores correlate with human judgment |
| Scale | Production logs | Add real user queries that exposed failures |

Start with synthetic, validate with humans, supplement with production over time. Never rely on synthetic alone.

---

## Agent Evaluation

Agent evaluation is a different problem from RAG evaluation: tasks have many valid paths, execution is multi-step, and there is often no single ground-truth answer. The key metrics are trajectory-level (tool call count, redundant calls, correct tool selection rate) rather than output-level.

Covered in [agents/README.md → Agent evaluation](../agents/README.md#agent-evaluation).
