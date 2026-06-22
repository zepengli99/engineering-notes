"""
Human-in-the-loop — approve before writing a file.

Flow:
  draft_node  : LLM generates file content based on user request
  approve_node: interrupt — human reviews the draft and decides yes or no
  write_node  : write to disk only if approved

Writing a file is irreversible (or at least side-effectful), so it sits
behind a human gate. The safe action (drafting) runs first without any risk.
"""

from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

model = ChatGroq(model="qwen/qwen3-32b")

# --- state ---

class State(TypedDict):
    messages: Annotated[list, add_messages]
    filename: str

# --- nodes ---

def draft_node(state: State):
    """Ask the LLM to write the file content."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def approve_node(state: State):
    """Show the draft to the human and pause for approval."""
    draft = state["messages"][-1].content
    decision = interrupt(
        f"--- Draft for '{state['filename']}' ---\n{draft}\n\nWrite this file? (yes/no)"
    )
    if decision.strip().lower() == "yes":
        return {"messages": [AIMessage("Approved.")]}
    else:
        return {"messages": [AIMessage(f"Cancelled. Reason: {decision}")]}


def should_write(state: State):
    if state["messages"][-1].content == "Approved.":
        return "write"
    return END


def write_node(state: State):
    """Write the approved draft to disk."""
    # draft is the message just before the "Approved." message
    draft = state["messages"][-2].content
    path = OUTPUT_DIR / state["filename"]
    path.write_text(draft, encoding="utf-8")
    return {"messages": [AIMessage(f"Done. Written to: {path}")]}


# --- graph ---

graph = StateGraph(State)
graph.add_node("draft", draft_node)
graph.add_node("approve", approve_node)
graph.add_node("write", write_node)

graph.add_edge(START, "draft")
graph.add_edge("draft", "approve")
graph.add_conditional_edges("approve", should_write)
graph.add_edge("write", END)

app = graph.compile(checkpointer=MemorySaver())

# --- run ---

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "write-1"}}
    filename = "note.txt"

    print("=== step 1: drafting content ===\n")
    result = app.invoke(
        {
            "messages": [HumanMessage("Write a short motivational note about learning new things.")],
            "filename": filename,
        },
        config,
    )

    interrupts = result.get("__interrupt__", [])
    if interrupts:
        print(interrupts[0].value)
        print()

    answer = input("Your decision (yes/no): ").strip()

    print(f"\n=== step 2: resuming with '{answer}' ===\n")
    final = app.invoke(Command(resume=answer), config)
    print(final["messages"][-1].content)
