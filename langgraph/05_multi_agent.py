"""
Multi-agent: Supervisor + two worker subgraphs.

Supervisor decides which worker to invoke next.
Each worker is a compiled subgraph with its own internal tool-calling loop.

  researcher : gathers information using a search tool
  writer     : writes and saves a report based on the research
  supervisor : routes between workers until the task is done

Flow:
  START -> supervisor -> researcher -> supervisor -> writer -> supervisor -> FINISH -> END
"""

from pathlib import Path
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

model = ChatGroq(model="qwen/qwen3-32b")

# ── tools ──────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    db = {
        "exercise benefits": "Regular exercise improves cardiovascular health, boosts mood via endorphins, strengthens muscles and bones, and reduces risk of chronic diseases.",
        "exercise mental health": "Exercise reduces anxiety and depression, improves cognitive function, and increases self-esteem.",
        "exercise recommendations": "WHO recommends 150 minutes of moderate aerobic activity per week for adults.",
    }
    for key, val in db.items():
        if key in query.lower():
            return val
    return f"No specific results for '{query}'."

@tool
def save_report(content: str, filename: str = "report.txt") -> str:
    """Save a written report to a file."""
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    return f"Report saved to {path}"

# ── researcher subgraph ────────────────────────────────────────────────
# A complete agent loop: model -> tools -> model, until no more tool calls.

researcher_model = model.bind_tools([search_web])

def _researcher_agent(state: MessagesState):
    return {"messages": [researcher_model.invoke(state["messages"])]}

def _researcher_route(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END

researcher_graph = StateGraph(MessagesState)
researcher_graph.add_node("agent", _researcher_agent)
researcher_graph.add_node("tools", ToolNode([search_web]))
researcher_graph.add_edge(START, "agent")
researcher_graph.add_conditional_edges("agent", _researcher_route)
researcher_graph.add_edge("tools", "agent")
researcher_app = researcher_graph.compile()

# ── writer subgraph ────────────────────────────────────────────────────

writer_model = model.bind_tools([save_report])

def _writer_agent(state: MessagesState):
    return {"messages": [writer_model.invoke(state["messages"])]}

def _writer_route(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END

writer_graph = StateGraph(MessagesState)
writer_graph.add_node("agent", _writer_agent)
writer_graph.add_node("tools", ToolNode([save_report]))
writer_graph.add_edge(START, "agent")
writer_graph.add_conditional_edges("agent", _writer_route)
writer_graph.add_edge("tools", "agent")
writer_app = writer_graph.compile()

# ── parent graph ───────────────────────────────────────────────────────

class ParentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

# with_structured_output forces the model to return a validated Pydantic object.
# No parsing needed — if the model returns invalid JSON, it retries automatically.
class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]
    reason: str

supervisor_model = model.with_structured_output(Route)

SUPERVISOR_PROMPT = """You are a supervisor managing two workers:
- researcher: searches for information on topics
- writer: writes and saves reports based on gathered information

Decide who should act next:
- 'researcher' if more information needs to be gathered
- 'writer' if enough research is done and a report should be written
- 'FINISH' if the report has been saved and the task is complete
"""

def supervisor_node(state: ParentState):
    messages = [SystemMessage(SUPERVISOR_PROMPT)] + state["messages"]
    route = supervisor_model.invoke(messages)
    print(f"  [supervisor] -> {route.next}  ({route.reason})")
    return {"next": route.next}

def researcher_node(state: ParentState):
    """Run the researcher subgraph; append only its final message to parent state."""
    print("  [researcher] running...")
    result = researcher_app.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}

def writer_node(state: ParentState):
    """Run the writer subgraph; append only its final message to parent state."""
    print("  [writer] running...")
    result = writer_app.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}

def route_next(state: ParentState):
    return state["next"]

parent_graph = StateGraph(ParentState)
parent_graph.add_node("supervisor", supervisor_node)
parent_graph.add_node("researcher", researcher_node)
parent_graph.add_node("writer", writer_node)

parent_graph.add_edge(START, "supervisor")
parent_graph.add_conditional_edges("supervisor", route_next, {
    "researcher": "researcher",
    "writer": "writer",
    "FINISH": END,
})
parent_graph.add_edge("researcher", "supervisor")
parent_graph.add_edge("writer", "supervisor")

app = parent_graph.compile()

# ── run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== starting multi-agent run ===\n")
    result = app.invoke({
        "messages": [HumanMessage("Research the benefits of exercise and write a short report.")],
        "next": "",
    })

    print("\n=== final answer ===")
    print(result["messages"][-1].content)

    print("\n=== message trace ===")
    for m in result["messages"]:
        label = type(m).__name__
        content = (m.content or str(getattr(m, "tool_calls", "")))[:100]
        print(f"  [{label}] {content}")
