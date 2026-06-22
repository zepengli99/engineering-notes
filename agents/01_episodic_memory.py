"""
Episodic Memory — agent learns from past failures.

Without memory: agent tries SQL → gets permission error → figures out API → succeeds (3 LLM calls)
With memory:    agent reads past episode → skips SQL → goes straight to API (2 LLM calls)

Episodic memory = RAG, but the "documents" are past agent experiences,
not knowledge articles. Same storage and retrieval mechanism, different data.
"""

import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

# ── fake tools ─────────────────────────────────────────────────────────

@tool
def query_sql(query: str) -> str:
    """Query the sales database directly with SQL."""
    # Always fails — direct SQL is not allowed
    raise PermissionError(f"Permission denied: direct SQL queries are blocked. Use the API instead.")

@tool
def query_api(endpoint: str) -> str:
    """Query sales data through the internal API."""
    data = {
        "/api/sales/monthly": "Jan: $120k, Feb: $135k, Mar: $142k",
        "/api/sales/summary": "Q1 total: $397k, up 12% YoY",
    }
    return data.get(endpoint, f"No data at {endpoint}")

tools = [query_sql, query_api]
tool_map = {t.name: t for t in tools}

# ── simple in-memory episode store ─────────────────────────────────────
# In production: embed the task description and store in a vector DB.
# Here: keyword matching is enough to demonstrate the concept.

episode_store: list[dict] = []

def save_episode(task: str, failed_approaches: list[str], lesson: str):
    episode_store.append({
        "task": task,
        "failed_approaches": failed_approaches,
        "lesson": lesson,
    })
    print(f"\n  [memory] episode saved: {lesson}")

def retrieve_episodes(task: str, top_k: int = 2) -> list[dict]:
    """Find relevant past episodes by keyword overlap (simplified)."""
    task_words = set(task.lower().split())
    scored = []
    for ep in episode_store:
        ep_words = set(ep["task"].lower().split())
        overlap = len(task_words & ep_words)
        if overlap > 0:
            scored.append((overlap, ep))
    scored.sort(reverse=True)
    return [ep for _, ep in scored[:top_k]]

# ── agent ──────────────────────────────────────────────────────────────

llm_calls = 0
model_with_tools = model.bind_tools(tools)

def build_system_prompt(episodes: list[dict]) -> str:
    if not episodes:
        return "You are a helpful data analyst assistant."
    ep_text = "\n".join(
        f"- Task: {e['task']}\n  Failed: {', '.join(e['failed_approaches'])}\n  Lesson: {e['lesson']}"
        for e in episodes
    )
    return f"""You are a helpful data analyst assistant.

Past relevant experiences (learn from these):
{ep_text}

Apply these lessons to avoid repeating past mistakes."""

def run_agent(task: str, use_memory: bool) -> list:
    global llm_calls
    episodes = retrieve_episodes(task) if use_memory else []
    system_prompt = build_system_prompt(episodes)

    messages = [SystemMessage(system_prompt), HumanMessage(task)]
    failed_tools = []

    while True:
        llm_calls += 1
        print(f"  LLM call #{llm_calls}")
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            print(f"    → {tc['name']}({tc['args']})")
            try:
                result = tool_map[tc["name"]].invoke(tc["args"])
                print(f"    ← {result[:60]}")
            except Exception as e:
                result = f"Error: {e}"
                failed_tools.append(tc["name"])
                print(f"    ← {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages, failed_tools

# ── run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TASK = "Get me the monthly sales figures for this quarter."

    # ── run 1: no memory, agent figures it out the hard way ───────────
    print("=" * 55)
    print("RUN 1: no episodic memory")
    print("=" * 55)

    calls_before = llm_calls
    messages, failed = run_agent(TASK, use_memory=False)
    calls_run1 = llm_calls - calls_before

    print(f"\nAnswer: {messages[-1].content[:150]}")
    print(f"LLM calls: {calls_run1}")

    # save what we learned from this run
    if failed:
        save_episode(
            task=TASK,
            failed_approaches=failed,
            lesson="Direct SQL queries are blocked with PermissionError. Use /api/sales/monthly via query_api instead.",
        )

    # ── run 2: with memory, agent skips the mistake ───────────────────
    print()
    print("=" * 55)
    print("RUN 2: with episodic memory")
    print("=" * 55)

    calls_before = llm_calls
    messages, _ = run_agent(TASK, use_memory=True)
    calls_run2 = llm_calls - calls_before

    print(f"\nAnswer: {messages[-1].content[:150]}")
    print(f"LLM calls: {calls_run2}")

    # ── summary ───────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Without memory: {calls_run1} LLM calls (tried SQL, got error, then API)")
    print(f"  With memory:    {calls_run2} LLM calls (skipped SQL, went straight to API)")
    print(f"  Saved:          {calls_run1 - calls_run2} LLM call(s)")
