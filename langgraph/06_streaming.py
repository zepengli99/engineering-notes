"""
Streaming — two modes compared.

Mode 1: stream graph events
  app.stream(...) — yields one chunk per completed node
  use when: you want to know which step just finished (progress updates)

Mode 2: stream tokens
  app.stream(..., stream_mode="messages") — yields one chunk per token
  use when: you want typewriter-style output in a UI
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
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

tools = [get_weather]
model_with_tools = model.bind_tools(tools)

# --- graph (same as 02_simple_agent.py) ---

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

# --- mode 1: stream graph events ---

def demo_event_stream():
    print("=== mode 1: stream graph events ===\n")
    print("each line = one node completing\n")

    for chunk in app.stream(
        {"messages": [HumanMessage("What is the weather in Tokyo and London?")]},
    ):
        node_name = list(chunk.keys())[0]
        messages = chunk[node_name]["messages"]
        last = messages[-1]
        content = last.content or str(getattr(last, "tool_calls", ""))
        print(f"[{node_name}] {content[:100]}")

# --- mode 2: stream tokens ---

def demo_token_stream():
    print("\n=== mode 2: stream tokens (typewriter effect) ===\n")

    for chunk, metadata in app.stream(
        {"messages": [HumanMessage("Tell me about the weather in Tokyo in two sentences.")]},
        stream_mode="messages",
    ):
        # only print tokens coming from the agent node, skip tool messages
        if (
            metadata.get("langgraph_node") == "agent"
            and hasattr(chunk, "content")
            and chunk.content
        ):
            print(chunk.content, end="", flush=True)

    print()  # newline after stream ends

# --- run ---

if __name__ == "__main__":
    demo_event_stream()
    demo_token_stream()
