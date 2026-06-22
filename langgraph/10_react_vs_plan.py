"""
ReAct vs Plan-and-Execute — side by side comparison.

Same task, two approaches. Watch the LLM call counter to see the cost difference.

Task: check weather in 3 cities and recommend the best for a weekend trip.

ReAct:
  - LLM decides next action after every tool result
  - N tool calls = N+1 LLM calls (one per step + final answer)

Plan-and-Execute:
  - LLM called ONCE upfront to produce the full plan
  - Executor runs tool steps mechanically (no LLM needed)
  - LLM called ONCE at the end to synthesize results
  - Total: 2 LLM calls regardless of how many tools
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

# --- tools ---

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {
        "beijing": "28C, sunny",
        "shanghai": "22C, typhoon warning — outdoor activities cancelled",
        "guangzhou": "25C, partly cloudy",
    }
    return data.get(city.lower(), f"no data for {city}")

tools = [get_weather]
tool_map = {"get_weather": get_weather}

# ── APPROACH 1: ReAct ──────────────────────────────────────────────────

react_llm_calls = 0

class ReActState(TypedDict):
    messages: Annotated[list, add_messages]

model_with_tools = model.bind_tools(tools)

def react_agent_node(state: ReActState):
    global react_llm_calls
    react_llm_calls += 1
    print(f"  [ReAct] LLM call #{react_llm_calls}")
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def react_should_continue(state: ReActState):
    return "tools" if state["messages"][-1].tool_calls else END

react_graph = StateGraph(ReActState)
react_graph.add_node("agent", react_agent_node)
react_graph.add_node("tools", ToolNode(tools))
react_graph.add_edge(START, "agent")
react_graph.add_conditional_edges("agent", react_should_continue)
react_graph.add_edge("tools", "agent")
react_app = react_graph.compile()

# ── APPROACH 2: Plan-and-Execute ───────────────────────────────────────

plan_llm_calls = 0

# --- plan schema ---

class ToolStep(BaseModel):
    action: str      # name of the function to invoke (e.g. "get_weather")
    params: dict     # parameters to pass (e.g. {"city": "beijing"})
    reason: str      # why this step is needed

class Plan(BaseModel):
    steps: list[ToolStep]
    final_prompt: str   # instruction for the summarise step

# --- state ---

def append_str(existing: list, new) -> list:
    if isinstance(new, list):
        return existing + new
    return existing + [new]

class PlanState(TypedDict):
    messages:  Annotated[list, add_messages]
    plan:      list          # remaining steps
    results:   Annotated[list, append_str]  # collected tool results

# --- nodes ---

PLANNER_PROMPT = """You are a planner. Output a structured plan — do NOT execute anything.
For each city mentioned, create a step with action='get_weather' and params={'city': '<name>'}.
Add a final_prompt describing how to summarise all weather results into a recommendation."""

planner_model = model.with_structured_output(Plan)

def planner_node(state: PlanState):
    global plan_llm_calls
    plan_llm_calls += 1
    print(f"  [Plan] LLM call #{plan_llm_calls} — planning")
    plan = planner_model.invoke(
        [SystemMessage(PLANNER_PROMPT)] + state["messages"]
    )
    print(f"  [Plan] plan has {len(plan.steps)} steps + 1 summarise")
    for i, step in enumerate(plan.steps):
        print(f"    step {i+1}: {step.action}({step.params}) — {step.reason}")
    return {
        "plan": [{"type": "tool", "action": s.action, "params": s.params, "reason": s.reason} for s in plan.steps]
              + [{"type": "summarise", "final_prompt": plan.final_prompt}]
    }

def executor_node(state: PlanState):
    global plan_llm_calls
    step = state["plan"][0]
    remaining = state["plan"][1:]

    if step["type"] == "tool":
        # No LLM needed — just execute the tool directly
        tool_fn = tool_map.get(step["action"])
        result = tool_fn.invoke(step["params"]) if tool_fn else f"unknown action: {step['action']}"
        print(f"  [Exec] {step['action']}({step['params']}) → {result}")
        return {
            "plan": remaining,
            "results": f"{step['params']}: {result}",
        }

    elif step["type"] == "summarise":
        # Final step — one LLM call to synthesise all results
        plan_llm_calls += 1
        print(f"  [Plan] LLM call #{plan_llm_calls} — summarising")
        context = "\n".join(state["results"])
        response = model.invoke([
            HumanMessage(f"{step['final_prompt']}\n\nResults:\n{context}")
        ])
        return {
            "plan": remaining,
            "messages": [response],
        }

def plan_should_continue(state: PlanState):
    return "executor" if state["plan"] else END

plan_graph = StateGraph(PlanState)
plan_graph.add_node("planner", planner_node)
plan_graph.add_node("executor", executor_node)
plan_graph.add_edge(START, "planner")
plan_graph.add_edge("planner", "executor")
plan_graph.add_conditional_edges("executor", plan_should_continue)
plan_app = plan_graph.compile()

# ── run both and compare ───────────────────────────────────────────────

QUESTION = "Check the weather in Beijing, Shanghai, and Guangzhou. Which is best for a weekend outdoor trip?"

if __name__ == "__main__":
    print("=" * 55)
    print("APPROACH 1: ReAct")
    print("=" * 55)
    react_result = react_app.invoke({"messages": [HumanMessage(QUESTION)]})
    print(f"\nAnswer: {react_result['messages'][-1].content[:200]}")
    print(f"Total LLM calls: {react_llm_calls}")

    print()
    print("=" * 55)
    print("APPROACH 2: Plan-and-Execute")
    print("=" * 55)
    plan_result = plan_app.invoke({
        "messages": [HumanMessage(QUESTION)],
        "plan": [],
        "results": [],
    })
    print(f"\nAnswer: {plan_result['messages'][-1].content[:200]}")
    print(f"Total LLM calls: {plan_llm_calls}")

    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  ReAct             : {react_llm_calls} LLM calls")
    print(f"  Plan-and-Execute  : {plan_llm_calls} LLM calls")
    print(f"  Saved             : {react_llm_calls - plan_llm_calls} calls")
