"""
Observability — two approaches:

1. LangSmith (recommended for production)
   Set env vars → every invoke() auto-uploads a trace.
   No code changes needed beyond the env vars.

2. Stream-based local logging (no external service)
   Use stream_mode="updates" to print each node's input/output.

To use LangSmith:
  1. Sign up at https://smith.langchain.com
  2. Create an API key
  3. Add to .env:
       LANGCHAIN_TRACING_V2=true
       LANGCHAIN_API_KEY=ls__your_key_here
       LANGCHAIN_PROJECT=langgraph-notes   (optional, groups runs)
  4. Run this file — traces appear in the LangSmith UI automatically.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tracers.context import tracing_v2_enabled
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

# --- tools ---

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {
        "tokyo": "18C, cloudy",
        "london": "12C, rainy",
        "sydney": "26C, sunny",
    }
    return data.get(city.lower(), "unknown city")

@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"

tools = [get_weather, calculate]
model_with_tools = model.bind_tools(tools)

# --- graph ---

def agent_node(state: MessagesState):
    return {"messages": [model_with_tools.invoke(state["messages"])]}

def should_continue(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()

# ── approach 1: LangSmith ──────────────────────────────────────────────
#
# If LANGCHAIN_TRACING_V2=true is set in .env, every invoke() below
# automatically uploads a trace. Nothing else to change.
#
# To trace only specific runs (not everything), use the context manager:
#
#   with tracing_v2_enabled(project_name="my-project"):
#       result = app.invoke(...)
#
# Each run also accepts metadata and tags for filtering in the UI:

def run_with_langsmith():
    print("=== approach 1: LangSmith tracing ===\n")
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2") == "true"
    if tracing_on:
        print("LangSmith tracing is ON — check https://smith.langchain.com\n")
    else:
        print("LangSmith tracing is OFF (LANGCHAIN_TRACING_V2 not set)\n")
        print("Add to .env:")
        print("  LANGCHAIN_TRACING_V2=true")
        print("  LANGCHAIN_API_KEY=ls__your_key_here\n")

    result = app.invoke(
        {"messages": [HumanMessage("What's the weather in Tokyo and what is 15 * 7?")]},
        # optional: tag and name this run for easier filtering in LangSmith UI
        config={
            "run_name": "weather-and-math",
            "tags": ["demo", "multi-tool"],
            "metadata": {"user_id": "user-123", "env": "development"},
        },
    )
    print("answer:", result["messages"][-1].content[:120])

# ── approach 2: stream-based local logging ─────────────────────────────
#
# No external service. stream_mode="updates" yields one dict per node:
# {node_name: {field: new_value}} — shows exactly what each node produced.

def run_with_local_logging():
    print("\n=== approach 2: stream-based local logging ===\n")

    for chunk in app.stream(
        {"messages": [HumanMessage("What's the weather in London?")]},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            msgs = update.get("messages", [])
            for m in msgs:
                label = type(m).__name__
                content = (m.content or str(getattr(m, "tool_calls", "")))[:100]
                print(f"  [{node_name}] {label}: {content}")

# ── run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_with_langsmith()
    run_with_local_logging()
