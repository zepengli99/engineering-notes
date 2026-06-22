"""
Same agent as 01_manual_loop.py, rewritten with LangGraph.

The loop is now an explicit graph: two nodes, one conditional edge.
State (the message list) is managed by LangGraph instead of by hand.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- tools (identical to 01_manual_loop.py) ---

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {
        "tokyo": "18°C, cloudy",
        "london": "12°C, rainy",
        "sydney": "26°C, sunny",
    }
    return data.get(city.lower(), "unknown city")

@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '3 * 7 + 2'."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"

tools = [get_weather, calculate]

model = ChatGroq(model="qwen/qwen3-32b")
model_with_tools = model.bind_tools(tools)

# --- graph ---

def agent_node(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

# --- run ---

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    result = app.invoke({"messages": [HumanMessage("What's the weather in Tokyo? Also, what is 123 * 456?")]})

    print("\n--- full message history ---")
    for msg in result["messages"]:
        label = type(msg).__name__
        content = msg.content or str(getattr(msg, "tool_calls", ""))
        print(f"[{label}] {content}")
