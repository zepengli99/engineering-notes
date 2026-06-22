"""
Error handling — two problems:

1. Tool failures: ToolNode catches exceptions and feeds them back to the model
   as ToolMessage so the agent can recover instead of crashing.

2. Infinite loops: a step counter in State enforces a hard iteration limit.

Run this and observe:
- The agent receives the error message and corrects its tool call
- After hitting max steps the graph terminates cleanly
"""

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

# --- tools ---

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Supported: tokyo, london, sydney."""
    data = {
        "tokyo": "18C, cloudy",
        "london": "12C, rainy",
        "sydney": "26C, sunny",
    }
    city_lower = city.lower()
    if city_lower not in data:
        raise ValueError(
            f"City '{city}' not supported. Available cities: {', '.join(data.keys())}"
        )
    return data[city_lower]

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return a / b

@tool
def check_status() -> str:
    """Check if the background report generation task is complete."""
    return "Task still running. Please check again in a moment."

tools = [get_weather, divide, check_status]

# --- state with step counter ---

class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int

# --- nodes ---

model_with_tools = model.bind_tools(tools)

def agent_node(state: State):
    response = model_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "steps": state["steps"] + 1,
    }

def should_continue(state: State):
    MAX_STEPS = 6
    if state["steps"] >= MAX_STEPS:
        print(f"  [guard] max steps ({MAX_STEPS}) reached — forcing END")
        return END
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# handle_tool_errors=True: catches exceptions inside tools and wraps them
# as ToolMessage content instead of crashing the graph.
tool_node = ToolNode(tools, handle_tool_errors=True)

# --- graph ---

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()

# --- helpers ---

def run(label: str, question: str):
    print(f"\n=== {label} ===\n")
    result = app.invoke({"messages": [HumanMessage(question)], "steps": 0})
    print("\n--- message trace ---")
    for m in result["messages"]:
        label_m = type(m).__name__
        content = (m.content or str(getattr(m, "tool_calls", "")))[:120]
        print(f"  [{label_m}] {content}")
    print(f"\nsteps used: {result['steps']}")

# --- demos ---

if __name__ == "__main__":
    # Demo 1: tool raises ValueError — agent receives error and corrects itself
    run(
        "tool error recovery",
        "What's the weather in Tokio?",   # intentional typo: Tokio vs tokyo
    )

    # Demo 2: division by zero — agent receives error and explains to user
    run(
        "division by zero",
        "What is 42 divided by 0?",
    )

    # Demo 3: max iterations guard
    # check_status always returns "still running", forcing the agent to keep
    # polling until the step counter hits MAX_STEPS and END is forced.
    run(
        "max steps guard",
        "Check the task status and keep checking until it reports as done.",
    )
