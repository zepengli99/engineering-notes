"""
Manual agent loop — no LangGraph.

Same behaviour as 02_simple_agent.py but written by hand.
Run this first to understand what LangGraph is actually doing.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# --- tools ---

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
tool_map = {t.name: t for t in tools}

model = ChatGroq(model="qwen/qwen3-32b")
model_with_tools = model.bind_tools(tools)

# --- manual loop ---

def run(user_input: str):
    messages = [HumanMessage(user_input)]
    step = 0

    while True:
        step += 1
        print(f"\n[step {step}] messages sent to model:")
        for m in messages:
            print(f"  {type(m).__name__}: {m.content or m.tool_calls}")
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"[done] {response.content}")
            break

        for tc in response.tool_calls:
            print(f"  > tool call: {tc['name']}({tc['args']})")
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"  < result: {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


if __name__ == "__main__":
    run("What's the weather in Tokyo? Also, what is 123 * 456?")
