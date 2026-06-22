"""
Send API — parallel fan-out.

Instead of one worker at a time (sequential supervisor), Send dispatches
multiple tasks simultaneously. All run in parallel; results are collected
by an aggregate node after all complete.

Flow:
  START → dispatch → research(tokyo)  ──┐
                   → research(london) ──┼── aggregate → END
                   → research(sydney) ──┘

Key difference from 05_multi_agent.py:
  - Supervisor routes one worker at a time (sequential)
  - Send dispatches all workers at once (parallel)
"""

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

# --- tools ---

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {
        "tokyo":  "18C, cloudy",
        "london": "12C, rainy",
        "sydney": "26C, sunny",
    }
    return data.get(city.lower(), "unknown city")

# --- state ---

# Custom reducer: each parallel research node appends one string to the list.
# Without this reducer, parallel writes would overwrite each other.
def append_result(existing: list, new) -> list:
    if isinstance(new, list):
        return existing + new   # merge lists (initial state setup)
    return existing + [new]     # append single string (from research nodes)

class OverallState(TypedDict):
    cities: list[str]                           # input: cities to research
    results: Annotated[list, append_result]     # collected from parallel nodes
    summary: str                                # final aggregate output

class ResearchState(TypedDict):
    city: str   # each parallel branch gets its own isolated state

# --- nodes ---

def dispatch(state: OverallState):
    """Fan out: send one Research task per city, all in parallel."""
    return [Send("research", {"city": city}) for city in state["cities"]]

def research_node(state: ResearchState):
    """Runs in parallel for each city. Writes one result back to OverallState."""
    city = state["city"]
    weather = get_weather.invoke({"city": city})
    result = f"{city.capitalize()}: {weather}"
    print(f"  [research] {result}")
    return {"results": result}   # reducer appends this to OverallState.results

def aggregate_node(state: OverallState):
    """Runs once after ALL parallel research nodes complete."""
    print(f"\n  [aggregate] collected {len(state['results'])} results")
    prompt = (
        "Summarise the following weather reports in one short paragraph:\n"
        + "\n".join(state["results"])
    )
    response = model.invoke([HumanMessage(prompt)])
    return {"summary": response.content}

# --- graph ---

graph = StateGraph(OverallState)
graph.add_node("research", research_node)
graph.add_node("aggregate", aggregate_node)

# dispatch returns a list of Send objects → fan-out
graph.add_conditional_edges(START, dispatch)

# all research branches converge on aggregate
graph.add_edge("research", "aggregate")
graph.add_edge("aggregate", END)

app = graph.compile()

# --- run ---

if __name__ == "__main__":
    print("=== parallel fan-out with Send API ===\n")

    result = app.invoke({
        "cities": ["tokyo", "london", "sydney"],
        "results": [],
        "summary": "",
    })

    print("\n=== summary ===")
    print(result["summary"])

    print("\n=== raw results (order may vary — they ran in parallel) ===")
    for r in result["results"]:
        print(f"  {r}")
