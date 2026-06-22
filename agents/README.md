# Agents

Modern agent architecture beyond the basic ReAct loop. Implementation patterns live in [LangGraph notes](../langgraph/README.md); this covers cross-framework concepts and 2024–2025 developments.

---

## A2A Protocol (Agent-to-Agent)

[MCP](../llm_architecture/README.md) connects an agent to tools and data sources — synchronous, fast, like calling an API. **A2A** (Google, April 2025) connects agents to other agents — asynchronous, long-running, like delegating work to a contractor.

| | MCP | A2A |
|---|---|---|
| Direction | agent → tool | agent → agent |
| Execution | synchronous | asynchronous |
| Duration | milliseconds | seconds to minutes |
| Analogy | calling an API | hiring a contractor |

### Core concepts

**Agent Card** — a self-describing file each agent publishes, declaring its capabilities, accepted inputs, and communication protocol. Other agents discover and understand it through the card, similar to an OpenAPI spec.

**Task** — the unit of delegated work. The initiating agent creates a Task; the receiving agent executes it asynchronously and streams progress back via SSE. Tasks have states: `pending → running → completed / failed`.

### Concrete example

A customer service orchestrator receives: "My order #12345 shipped last week but hasn't arrived."

```
orchestrator agent
    → reads logistics agent's Agent Card
    → creates A2A Task: "investigate order #12345, customer hasn't received it"

logistics agent (async, ~30 seconds):
    → uses MCP to query order database
    → uses MCP to query shipping API
    → discovers package stuck at warehouse

    → streams progress: "contacting carrier..."
    → Task complete: "package delayed, recommend reship"

orchestrator
    → receives result
    → replies to customer
```

The orchestrator doesn't block for 30 seconds — A2A is async. The logistics agent internally uses MCP tools, but its relationship to the orchestrator is A2A.

### MCP and A2A are complementary

Production systems use both:

```
orchestrator
    ├── MCP: search tool, database tool  (synchronous, fast)
    └── A2A: researcher agent            (asynchronous, long-running)
```

MCP = equip the agent with tools (hammer, drill)
A2A = delegate work to a specialist agent (hire a subcontractor)

### A2A vs framework-internal communication

Agent frameworks (AutoGen, LangGraph, CrewAI) each have their own internal agent coordination mechanisms. A2A is the cross-system bridge:

```
AutoGen agent  ←→  AutoGen agent      ← AutoGen's internal mechanism
LangGraph agent ←→ AutoGen agent      ← needs A2A as the bridge
LangGraph agent ←→ CrewAI agent       ← needs A2A as the bridge
```

A2A is framework-agnostic — it lets agents built on different stacks speak the same language.

### Transport comparison: MCP vs A2A

Both protocols use HTTP + SSE at the transport layer, but MCP also supports a local mode:

```
MCP (local):   client → subprocess → stdio → MCP server
MCP (remote):  client → HTTP + SSE → MCP server
A2A:           agent  → HTTP + SSE → agent   (always remote)
```

A2A is always HTTP because it's designed for cross-system communication from the start. MCP has a local stdio mode because many tools (filesystem, local DB) run on the same machine as the agent.

The server side is what differs:
- MCP server = a tool or data source
- A2A server = a full agent (with its own state, tools, and logic)

---

## Memory Systems

Agents need different kinds of memory for different purposes. Three layers, each solving a different problem.

| Layer | What it stores | Analogy | Implementation |
|---|---|---|---|
| **Semantic** | Facts about the world | Knowing what Paris is the capital of | RAG + vector DB |
| **Episodic** | Past experiences — what the agent tried and what happened | Remembering you got lost on that road last time | Experience records + retrieval |
| **Procedural** | How to do things — reusable instruction sets | Knowing how to ride a bike | Skills / SKILL.md files |

Semantic memory is already covered in [LLM Architecture](../llm_architecture/README.md) under RAG.

### Episodic Memory

Stores past agent runs so future runs can avoid repeated mistakes.

**What to store after each run:**

```python
episode = {
    "task": "query sales database",
    "failed_approaches": ["query_sql"],
    "lesson": "Direct SQL is blocked with PermissionError. Use /api/sales via query_api.",
}
```

**How to use at the start of the next run:**

Embed the task description, search for similar past episodes, inject relevant ones into the system prompt:

```
You are a data analyst agent.

Past relevant experiences:
- Task: query sales database
  Failed: query_sql
  Lesson: Direct SQL blocked — use /api/sales via query_api instead.

Now help with: get this month's sales figures.
```

The model skips the failed approach without needing to discover it again.

**Key insight:** Episodic memory is just RAG where the "documents" are past agent experiences instead of knowledge articles. Same storage and retrieval mechanics, different data.

**LangGraph's Memory Store** (`InMemoryStore`, `PostgresStore`) provides the backing storage for cross-thread persistence. Retrieval logic, what counts as an episode, and what lesson to extract are all your responsibility.

### Procedural Memory (Skills)

Stores *how to do things* — reusable instruction sets that are loaded on demand.

**The problem with naive prompting:** putting all skill instructions in the system prompt at startup means:
- Context window cost scales linearly with number of skills
- Lost-in-the-Middle effect — instructions buried in the middle are underutilised

**Progressive disclosure solves this:**

```
1. Discovery (startup)
   Agent sees only: name + one-line description per skill
   Cost: ~2 tokens per skill regardless of instruction length

2. Activation (task arrives)
   Agent selects the relevant skill
   Full instructions loaded into context — only for that skill

3. Execution
   Agent follows the skill's steps
```

**Context comparison:**

```
3 skills, naive:   ~217 instruction words always in context
3 skills, dynamic: ~33 words (discovery) + ~73 words (one activated skill)
20 skills, naive:  ~1400 words always
20 skills, dynamic: same 33 + 73 regardless
```

**Skill vs Tool:**

- **Tool** — executes an action (`get_weather("tokyo")` → result)
- **Skill** — knows how to approach a task ("when doing code review, check security first, then error handling…")

A skill is a procedure. A tool is a function call. Skills typically orchestrate multiple tools internally.

**In practice:** Skills are stored as markdown files (SKILL.md), loaded dynamically when matched. Claude Code's slash commands (`/code-review`, `/deep-research`) are Agent Skills — each loads a SKILL.md that tells the model how to perform that workflow.

---

## Reasoning Models in Agents

Reasoning models (o1, Claude extended thinking) do extended internal chain-of-thought before generating a response. The thinking is hidden — you see only the final output — but the model has already worked through the problem in depth.

### How this changes the ReAct loop

Standard model in ReAct — each step of "thinking" is shallow, so the loop needs many iterations to correct mistakes:

```
sees problem → guesses a tool call → sees result → guesses next step → ...
```

Reasoning model — internal reasoning replaces many explicit loop iterations:

```
sees problem → [internal: thinks through the full solution] → outputs correct tool sequence
```

Fewer loop iterations needed because the model front-loads the thinking into a single call.

### The emerging architecture

```
Reasoning model  (strong, slow, expensive) — planning and complex decisions
      ↓ produces execution plan
Standard model   (fast, cheap)             — executes each step
```

This is Plan-and-Execute where the planner is a reasoning model. Plan quality improves significantly; replanning is rarely needed because the reasoning model already considered edge cases during planning.

### When NOT to use reasoning models

| Scenario | Why standard model is better |
|---|---|
| Latency-sensitive (<2s) | Extended thinking takes 10–30s |
| Tool-heavy tasks (20 API calls) | Reasoning doesn't help mechanical execution |
| Simple lookups | Overkill — pays thinking cost for zero gain |
| High-volume, repetitive tasks | Cost per call is much higher |

Reasoning models earn their cost on tasks where getting the plan right the first time saves multiple failed attempts — complex research, multi-constraint optimisation, debugging subtle issues.

---

## Framework Comparison

| Framework | Core idea | Best for | Weakness |
|---|---|---|---|
| **LangGraph** | Explicit graph — nodes, edges, state | Complex agents, human-in-the-loop, production | More code, steeper learning curve |
| **OpenAI Agents SDK** | Minimal API, built-in agent handoffs | Fast product shipping, OpenAI ecosystem | Less control, tied to OpenAI |
| **CrewAI** | Role-based orchestration | Quick prototyping, role-driven multi-agent | Poor customisation for production |
| **PydanticAI** | Type-safe, FastAPI-style | Strict structured output, data extraction | Not suited for complex multi-step agents |
| **AutoGen** | Async message-passing between agents | Long-running tasks, waiting on external events | High learning curve, different mental model |

### One-line selector

```
Complex flow + production control   → LangGraph
Fast prototype + OpenAI models      → OpenAI Agents SDK
Role-based multi-agent + speed      → CrewAI
Strict typed output                 → PydanticAI
Long async tasks + external events  → AutoGen
```

### Why LangGraph for production

LangGraph's explicit graph makes behaviour auditable — you can read the code and know exactly what the agent can and cannot do. Every branching decision is a named conditional edge. Every state change is a named node. This matters for debugging, testing, and compliance.

Other frameworks hide control flow behind higher-level abstractions that are convenient until something goes wrong — then you're fighting the framework to understand what actually happened.

The trade-off: LangGraph requires more code upfront. The payoff is full control at every step, which is why teams typically prototype with CrewAI or OpenAI Agents SDK and migrate to LangGraph when they need production reliability.
