# LangGraph

LangGraph is the harness layer for LLM agents. The model handles reasoning; the harness handles everything around it — assembling context, executing tools, managing state, enforcing constraints. Most agent engineering work happens in the harness, not the model.

Where [LangChain](../langchain/README.md) supplies components (models, tools, prompts), LangGraph defines how they run — the control flow, state management, and execution loop.

The core insight: **an agent is a state machine**. The ReAct loop that AgentExecutor hid in a black box is just a graph with two nodes and a conditional edge. LangGraph makes that explicit.

## Code examples

| File | What it demonstrates |
|---|---|
| [01_manual_loop.py](01_manual_loop.py) | Hand-written ReAct loop — no LangGraph, shows the raw tool-calling cycle |
| [02_simple_agent.py](02_simple_agent.py) | Same loop rewritten as a LangGraph StateGraph |
| [03_checkpointing.py](03_checkpointing.py) | MemorySaver, thread_id, multi-turn conversation |
| [04_human_in_the_loop.py](04_human_in_the_loop.py) | `interrupt()` and `Command(resume=...)` |
| [05_multi_agent.py](05_multi_agent.py) | Supervisor pattern with two worker subgraphs, `with_structured_output` |
| [06_streaming.py](06_streaming.py) | Event stream vs token stream |
| [07_error_handling.py](07_error_handling.py) | `handle_tool_errors`, max iterations guard |
| [08_observability.py](08_observability.py) | LangSmith tracing, stream-based local logging |
| [09_send_api.py](09_send_api.py) | Parallel fan-out with `Send`, custom reducers |
| [10_react_vs_plan.py](10_react_vs_plan.py) | ReAct vs Plan-and-Execute, LLM call count comparison |

---

## Core concepts

### State

State is the shared memory of the graph. Every node reads the current state and returns an update; LangGraph merges the update and passes it to the next node.

For a basic agent, state is a list of messages — the full conversation history:

```python
from langgraph.graph import MessagesState

# MessagesState is LangGraph's built-in. Equivalent to:
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
```

The `add_messages` **reducer** defines how updates merge into state. Instead of replacing the message list, it appends. This means nodes only need to return what's new:

```python
def agent_node(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # just the new AIMessage — LangGraph appends it
```

### Nodes

A node is any Python function that takes state and returns a partial update:

```python
def agent_node(state: MessagesState) -> dict:
    ...
    return {"messages": [...]}
```

### Edges

Two kinds:

- `add_edge(a, b)` — always go from a to b
- `add_conditional_edges(a, routing_fn)` — call `routing_fn(state)`, use the returned string as the next node name

```python
def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END
```

`return END` means the model's turn ended with no pending tool calls — not that the user's goal is satisfied. The model may have returned a clarifying question or a partial result. If goal completion matters, the harness needs an explicit check beyond the absence of tool calls.

---

## A minimal tool-calling agent

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 25°C, sunny"

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
model_with_tools = model.bind_tools([get_weather])

def agent_node(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([get_weather]))  # LangGraph built-in

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()
```

Execution trace for "What's the weather in Tokyo?":

```
State: [HumanMessage("Tokyo weather?")]
  → agent_node  → [..., AIMessage(tool_calls=[get_weather("Tokyo")])]
  → tools node  → [..., ToolMessage("Tokyo: 25°C")]
  → agent_node  → [..., AIMessage("Tokyo is 25°C right now.")]
  → END
```

`ToolNode` handles the full tool-calling protocol automatically: extracts `args` from each `tool_call`, calls the matching tool, wraps results in `ToolMessage` with the correct `tool_call_id`, and handles parallel tool calls in one shot.

The graph structure:

```
           ┌─────────┐
 START ───▶│  agent  │
           └────┬────┘
                │
        ┌───────┴───────┐
    tool_calls?        no
        │               │
        ▼             END
   ┌─────────┐
   │  tools  │
   └────┬────┘
        └──────▶ agent
```

---

## Checkpointing

Without checkpointing, every `invoke()` starts fresh — state is gone when the call returns. Checkpointing saves State after each node execution so conversations can resume across multiple calls.

The key is `thread_id` — it's the identity of a conversation. Same `thread_id` = same conversation, different `thread_id` = new conversation.

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "user-alice"}}

app.invoke({"messages": [HumanMessage("My name is Alice.")]}, config)
app.invoke({"messages": [HumanMessage("What's my name?")]}, config)
# agent replies: "Your name is Alice."
```

On the second call, LangGraph loads the State saved under `"user-alice"`, appends the new `HumanMessage`, runs the graph, and saves the updated State back. The model receives the full message history and knows the name.

### How it works internally

Checkpointing stores a State snapshot after **each node**, not just at the end of a full run. This per-node granularity is what enables human-in-the-loop (pause mid-graph, wait for approval, resume) and time travel (rewind to a specific node and replay).

```
invoke()
  agent_node runs → snapshot saved
  tools_node runs → snapshot saved
  agent_node runs → snapshot saved
```

### Checkpointer options

| Checkpointer | Use case |
|---|---|
| `MemorySaver` | In-memory, lost on restart — development only |
| `SqliteSaver` | Local file, simple persistence — testing |
| `PostgresSaver` | Production |

### Custom DB vs LangGraph Checkpointer

You can build the same thing yourself — store per-node State snapshots in your own tables, keyed by `thread_id`. The only friction is **serialization**: LangChain message objects (`AIMessage`, `ToolMessage`) aren't plain dicts and need proper handling when writing to and reading from a DB. LangGraph's checkpointers handle this for you.

The choice comes down to coupling. If agent state needs to join with business data (query all conversations per user, analytics on tool usage), owning the schema is worth it. If you just need persistence and human-in-the-loop, `PostgresSaver` is good enough.

### Context window growth

Every `invoke()` appends messages to State. Long conversations eventually exceed the model's context window. Two standard strategies:

**Sliding window** — keep only the last N messages. Simple but loses early context entirely.

**Summarization** — periodically ask the model to compress old messages into a summary string. State holds both `summary` and recent messages. The model receives "summary of earlier conversation + recent messages", so context is preserved without growing unboundedly.

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
```

---

## Human-in-the-loop

Some actions are irreversible — writing files, sending emails, calling external APIs. Rather than letting the agent execute them autonomously, you pause the graph, show the human what's about to happen, and only proceed on approval.

### The two primitives

`interrupt(value)` — called inside a node. Pauses graph execution immediately, saves a checkpoint, and returns control to the caller. `value` is anything you want the human to see (a string, a dict).

`Command(resume=value)` — passed to the next `invoke()` instead of new messages. Resumes execution from the interrupt point; `value` becomes the return value of `interrupt()`.

```python
def approve_node(state: State):
    draft = state["messages"][-1].content
    decision = interrupt(f"Draft:\n{draft}\n\nProceed? (yes/no)")  # pauses here
    if decision.strip().lower() == "yes":
        return {"messages": [AIMessage("Approved.")]}
    return {"messages": [AIMessage("Cancelled.")]}

# first invoke — pauses at interrupt()
result = app.invoke({"messages": [HumanMessage("...")]}, config)
print(result["__interrupt__"][0].value)   # show the human

# second invoke — resumes with human's answer
final = app.invoke(Command(resume="yes"), config)
```

Both calls use the same `config` (same `thread_id`). The second `invoke()` doesn't pass new messages — only `Command(resume=...)`.

### What's in the checkpoint at interrupt time

LangGraph saves a checkpoint after each completed node. When `interrupt()` fires inside `approve_node`:

- The checkpoint contains State after the last **completed** node (e.g. `draft_node`)
- A pending interrupt record is stored alongside it
- `approve_node` has not completed yet

On resume, `approve_node` re-executes from the top — but `interrupt()` returns the resume value immediately instead of pausing again.

### Keep interrupt nodes atomic

Because the interrupt node re-runs on resume, any code before `interrupt()` executes twice. Put expensive work (LLM calls, API calls) in a preceding node so results land in State before the interrupt node runs.

```python
# correct — draft_node does the heavy work, approve_node just reads State
def draft_node(state):
    response = model.invoke(state["messages"])   # runs once
    return {"messages": [response]}

def approve_node(state):
    draft = state["messages"][-1].content        # read from State, no recomputation
    decision = interrupt(f"Draft: {draft}\nProceed?")
    ...

# wrong — model called twice (once before pause, once on resume)
def approve_node(state):
    draft = model.invoke(state["messages"])      # runs twice
    decision = interrupt(f"Draft: {draft}\nProceed?")
    ...
```

---

## Multi-agent: Supervisor pattern

A single agent hits two limits: one context window and serial execution. Multi-agent systems split work across specialised agents that can run independently.

**Supervisor pattern** — one supervisor decides who acts next; workers return results to the supervisor, which decides the next step or ends.

```
START → supervisor → researcher → supervisor → writer → supervisor → FINISH → END
```

The supervisor is just a node that calls an LLM to pick the next worker:

```python
class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]
    reason: str

supervisor_model = model.with_structured_output(Route)

def supervisor_node(state: ParentState):
    messages = [SystemMessage(SUPERVISOR_PROMPT)] + state["messages"]
    route = supervisor_model.invoke(messages)
    return {"next": route.next}
```

Each worker is a **compiled subgraph** — a complete agent loop with its own tools and internal state. The parent graph calls it as a black box and only receives the final message:

```python
def researcher_node(state: ParentState):
    result = researcher_app.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}   # only the final answer, not internal tool calls
```

Routing wires everything together:

```python
parent_graph.add_conditional_edges("supervisor", route_next, {
    "researcher": "researcher",
    "writer": "writer",
    "FINISH": END,
})
parent_graph.add_edge("researcher", "supervisor")
parent_graph.add_edge("writer", "supervisor")
```

Workers always return to supervisor. Supervisor decides if more work is needed or if the task is done.

---

## Structured output and the Instructor pattern

### `with_structured_output`

Forces the model to return a Pydantic object via function calling. Three methods, in decreasing order of reliability:

| method | mechanism | guarantee |
|---|---|---|
| `"function_calling"` (default) | schema treated as a forced tool call | probabilistic — ValidationError on failure |
| `"json_schema"` + `strict=True` | API-level constrained decoding | near 100%, few models support it |
| `"json_mode"` | just asks for JSON, no schema | weakest — prompt must specify format |

No retry logic. Validation failure raises immediately. The `json_schema` + `strict` row is constrained decoding at the sampling layer — how it works (logit masking, FSM/grammar) is in [LLM Architecture → Structured output and constrained decoding](../llm_architecture/README.md#structured-output-and-constrained-decoding).

### Instructor: smart retry

`instructor` wraps the LLM client and retries on validation failure — critically, it feeds the error message back to the model so it can fix its output:

```python
import instructor
from groq import Groq

client = instructor.from_groq(Groq())

route = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[...],
    response_model=Route,
    max_retries=3,
)
```

Retry loop:

```
attempt 1: {"next": "research"}
  → ValidationError: 'research' is not a valid enum value

retry — error sent back to model:
  "next: Input should be 'researcher', 'writer' or 'FINISH'"

attempt 2: {"next": "researcher"}  → success
```

This is smarter than `.with_retry()` — blind retries hit the same failure. Instructor retries with context, so the model can correct itself.

### Pydantic validators as business logic

Validation rules in the schema act as self-correcting constraints:

```python
from pydantic import field_validator

class ExtractedData(BaseModel):
    revenue: float
    year: int

    @field_validator('year')
    def must_be_recent(cls, v):
        if v < 2000 or v > 2030:
            raise ValueError("year must be between 2000 and 2030")
        return v
```

Model outputs `year: 1985` → validator raises → error fed back → model corrects. Business rules live in the schema, not scattered across prompt engineering.

For production supervisor nodes, Instructor is more robust than `with_structured_output` alone.

---

## Streaming

Two modes, solving different problems.

### Mode 1: stream graph events

`app.stream()` yields one chunk per completed node. Use for progress updates — knowing which step just finished.

```python
for chunk in app.stream({"messages": [HumanMessage("...")]}):
    node_name = list(chunk.keys())[0]
    messages = chunk[node_name]["messages"]
    print(f"[{node_name}] {messages[-1].content[:80]}")
```

Output structure — one dict per node completion:

```python
{"agent": {"messages": [AIMessage(tool_calls=[...])]}}
{"tools": {"messages": [ToolMessage("18C")]}}
{"agent": {"messages": [AIMessage("Tokyo is 18C")]}}
```

### Mode 2: stream tokens

`stream_mode="messages"` yields one chunk per token as the model generates. Use for typewriter-style UI output.

```python
for chunk, metadata in app.stream(
    {"messages": [HumanMessage("...")]},
    stream_mode="messages",
):
    if (
        metadata.get("langgraph_node") == "agent"
        and hasattr(chunk, "content")
        and chunk.content
    ):
        print(chunk.content, end="", flush=True)
```

Filter by `langgraph_node` to skip `ToolMessage` chunks — those contain tool results, not the model's text response.

---

## Custom State

`MessagesState` is a shortcut. For anything beyond a message list, define your own `TypedDict`:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # append reducer
    steps:    int                            # replace reducer (default)
    next:     str                            # replace reducer (default)
    user_id:  str                            # set once, passed through
    summary:  str                            # updated by summarise node
```

### Reducers

Every State field has a reducer that defines how a node's returned value merges with the existing value. Default is **replace** (last write wins). `add_messages` is a built-in reducer that appends instead.

You can write your own:

```python
def add_int(existing: int, new: int) -> int:
    return existing + new

class State(TypedDict):
    steps: Annotated[int, add_int]  # returns delta, reducer accumulates
```

With `add_int`, a node returns `{"steps": 1}` instead of `{"steps": state["steps"] + 1}`. The reducer applies the increment — the node doesn't need to read the old value.

### Why custom reducers matter in parallel execution

In a sequential graph the two styles are equivalent. In a parallel fan-out (multiple nodes running simultaneously), manual increment causes a race condition:

```
node_A reads steps=3, returns {"steps": 4}
node_B reads steps=3, returns {"steps": 4}
→ one increment is lost, result is 4 instead of 5
```

With a reducer each node returns its delta (`{"steps": 1}`) and LangGraph applies them both:

```
add_int(add_int(3, 1), 1) → 5  ← correct
```

### messages vs State fields

**messages** are for the LLM's context window — conversation history, tool calls, tool results.

**State fields** are for application logic — routing decisions, counters, metadata, intermediate results. The model never sees them directly.

Don't put everything in messages. `user_id`, `next`, `steps` don't belong in the conversation history — they belong in State fields where only your code reads them.

---

## Error handling

### Tool failures

By default `ToolNode` lets exceptions propagate and crash the graph. `handle_tool_errors=True` catches them and wraps the error as a `ToolMessage` so the model can recover:

```python
tool_node = ToolNode(tools, handle_tool_errors=True)
```

The model receives:

```
ToolMessage(content="Error: ZeroDivisionError('Cannot divide by zero.') Please fix your mistakes.")
```

It can then retry with corrected arguments, explain the failure to the user, or take an alternative path — all without the graph crashing.

Pass a function for custom error formatting:

```python
def format_error(error: Exception) -> str:
    return f"Tool failed ({type(error).__name__}): {error}"

ToolNode(tools, handle_tool_errors=format_error)
```

### Max iterations guard

An agent calling a tool that always returns "not done yet" will loop forever. Add a `steps` counter to State and force `END` when it hits the limit:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int

def agent_node(state: State):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response], "steps": state["steps"] + 1}

def should_continue(state: State):
    if state["steps"] >= 10:
        return END          # hard stop — no more tool calls
    if state["messages"][-1].tool_calls:
        return "tools"
    return END
```

The guard fires when `steps` reaches the limit regardless of what the model wants to do next. Without it, a tool that always returns "still running" will run the agent until the context window fills up or the API bill explodes.

### Stuck loop detection

Iteration cap is a quantity guard — it fires after N steps regardless of what happened. A complementary check catches a different failure: the agent repeating the same action without progress.

The signal is tool call repetition: the agent invokes the same tool with identical arguments on consecutive iterations. This is not work — it is a stall. A well-instrumented harness tracks recent calls, detects the pattern, and exits early rather than burning the remaining iterations on a stuck run. Oscillation between two states belongs to the same family.

Two stop conditions, different concerns:

- **Iteration cap** — fires after N steps regardless of content
- **Repetition detection** — fires when the agent stops making progress

---

## Observability

### LangSmith

Set three env vars — every `invoke()` automatically uploads a trace. No code changes needed.

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here   # free tier at smith.langchain.com
LANGCHAIN_PROJECT=my-agent            # groups runs in the UI
```

To trace only specific runs, use the context manager:

```python
from langchain_core.tracers.context import tracing_v2_enabled

with tracing_v2_enabled(project_name="my-project"):
    result = app.invoke(...)
```

Add metadata and tags via `config` for filtering in the UI:

```python
app.invoke(
    {"messages": [HumanMessage("...")]},
    config={
        "run_name": "weather-query",
        "tags": ["prod", "weather"],
        "metadata": {"user_id": "u-123"},
    },
)
```

LangSmith shows: each node's input/output, full LLM prompts and responses, token usage, latency, and execution path.

### Stream-based local logging

No external service. `stream_mode="updates"` yields one dict per node completion:

```python
for chunk in app.stream(input, stream_mode="updates"):
    for node_name, update in chunk.items():
        for m in update.get("messages", []):
            print(f"[{node_name}] {type(m).__name__}: {m.content[:80]}")
```

Lightweight alternative when LangSmith is unavailable or data cannot leave the system.

---

## Send API — parallel fan-out

`Send` dispatches multiple tasks simultaneously instead of routing to one node at a time. All branches run in parallel; an aggregate node collects results after all complete.

```python
from langgraph.types import Send

def dispatch(state: OverallState):
    return [Send("research", {"city": city}) for city in state["cities"]]
```

`dispatch` returns a list of `Send(node_name, state)` objects. Each runs its own isolated copy of `research_node` in parallel.

### State design for fan-out

Parallel nodes write to the same parent State simultaneously. A custom reducer prevents overwrites:

```python
def append_result(existing: list, new) -> list:
    if isinstance(new, list):
        return existing + new   # handles initial state setup
    return existing + [new]     # appends a single result

class OverallState(TypedDict):
    cities:  list[str]
    results: Annotated[list, append_result]   # parallel nodes append safely
    summary: str
```

Each `research_node` returns `{"results": "Tokyo: 18C"}`. The reducer appends it to the shared list — no overwrite, no race condition.

### Graph wiring

```python
graph.add_conditional_edges(START, dispatch)   # fan-out
graph.add_edge("research", "aggregate")        # all branches converge here
graph.add_edge("aggregate", END)
```

`aggregate` only runs after **all** parallel `research` branches complete.

### Common pitfalls

**Reducer called on initialisation.** LangGraph applies the reducer when the initial State is set up. A reducer that only handles strings will crash on the initial `[]` value. Always handle both `list` and single-item inputs.

**Result order is not guaranteed.** Parallel branches finish in arbitrary order. If order matters, sort in the aggregate node.

### Sequential vs parallel

```
Supervisor (sequential):  dispatcher → A → dispatcher → B → dispatcher → END
Send API (parallel):      dispatcher → A ──┐
                                      → B ──┼── aggregate → END
                                      → C ──┘

wall-clock time:
  sequential: time(A) + time(B) + time(C)
  parallel:   max(time(A), time(B), time(C))
```

---

## ReAct vs Plan-and-Execute

### ReAct (LangGraph's default agent)

Think → Act → Observe → repeat. The model decides the next action after every tool result.

**When model parallelises tool calls, ReAct is already efficient** — 3 independent lookups collapse into 1 LLM call + 1 tool execution + 1 final LLM call = 2 total.

**Expensive when:** steps are sequential and interdependent — every step needs a full LLM call.

### Plan-and-Execute

Strong model plans all steps upfront. Executor runs tool steps mechanically — no LLM per step. Final LLM call synthesises results.

Cost: always 2 LLM calls regardless of number of steps.

### State structure

```python
# ReAct — plan lives in the model's head
class ReActState(TypedDict):
    messages: Annotated[list, add_messages]

# Plan-and-Execute — plan explicit in State
class PlanState(TypedDict):
    messages: Annotated[list, add_messages]
    plan:     list        # remaining steps
    results:  list        # collected tool outputs
```

### When the cost gap matters

| Scenario | ReAct calls | Plan-and-Execute calls |
|---|---|---|
| 3 independent tool calls (parallel) | 2 | 2 |
| 5 sequential dependent steps | 6 | 2 |
| Unexpected mid-task event | handles naturally | replan = +1 call |

Parallel tool calling is the key variable. If the model batches independent calls, ReAct is already near-optimal. Plan-and-Execute earns its keep only when steps must be serial.
```
