"""
Checkpointing — persistent state across multiple turns.

Run this and observe:
- Same thread_id: agent remembers previous messages
- Different thread_id: fresh conversation, no memory
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

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
model = ChatGroq(model="qwen/qwen3-32b")
model_with_tools = model.bind_tools(tools)

# --- graph (identical to 02_simple_agent.py) ---

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

# MemorySaver: in-memory checkpointer, state survives across invoke() calls
app = graph.compile(checkpointer=MemorySaver())

# --- helpers ---

def chat(thread_id: str, user_input: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(user_input)]}, config)
    reply = result["messages"][-1].content
    print(f"[{thread_id}] user : {user_input}")
    print(f"[{thread_id}] agent: {reply}")
    print()

def show_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    print(f"--- history for thread '{thread_id}' ({len(state.values['messages'])} messages) ---")
    for m in state.values["messages"]:
        label = type(m).__name__
        content = m.content or str(m.tool_calls)
        print(f"  {label}: {content[:80]}")
    print()

# --- demo ---

if __name__ == "__main__":
    print("=== thread: alice (multi-turn) ===\n")
    chat("alice", "Hi, my name is Alice.")
    chat("alice", "What's the weather in Tokyo?")
    chat("alice", "What's my name?")   # agent should remember

    print("=== thread: bob (separate conversation) ===\n")
    chat("bob", "What's my name?")     # no memory of Alice

    print("=== full state of alice's thread ===\n")
    show_history("alice")
