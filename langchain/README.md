# LangChain

LangChain is the component layer for LLM applications — models, prompts, parsers, tools, retrievers. It doesn't orchestrate agents; that's [LangGraph](../langgraph/README.md)'s job. LangChain supplies the building blocks that LangGraph nodes use to actually do work.

In practice, most of LangChain's heavier abstractions (complex chains, legacy agents, community integrations) go unused in production. What actually pulls its weight:

- **Provider packages** (`langchain-anthropic`, `langchain-openai`) — maintained wrappers that handle streaming, retries, and tool-calling protocol details correctly.
- **`@tool` and message types** (`HumanMessage`, `AIMessage`, `ToolMessage`) — shared vocabulary that LangGraph uses directly.

Everything else is optional. Many teams use `langchain-anthropic` + LangGraph and skip the `langchain` main package almost entirely. Some go further and use the Anthropic SDK directly with LangGraph, cutting LangChain out altogether.

---

## LCEL and the Runnable interface

Everything in modern LangChain implements the `Runnable` interface:

```python
runnable.invoke(input)      # single call
runnable.stream(input)      # token-by-token streaming
runnable.batch([...])       # parallel batch
```

This uniformity is the point. Because prompt templates, models, parsers, and custom functions all share the same interface, they can be composed freely.

**LCEL** (LangChain Expression Language) uses `|` to chain Runnables into a pipeline:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is LCEL?"})
```

Data flows left to right. Each step's output becomes the next step's input — types must be compatible.

### Type flow

```
dict  →  ChatPromptValue  →  AIMessage  →  str
      prompt            model          parser
```

- `ChatPromptTemplate.invoke()` returns a `ChatPromptValue`, which wraps a list of typed messages.
- `ChatModel` receives the `ChatPromptValue`, unpacks it, calls the API, and returns an `AIMessage`.
- `StrOutputParser` extracts `AIMessage.content` as a plain string.

### Message types

| Type | Role |
|---|---|
| `SystemMessage` | system |
| `HumanMessage` | user |
| `AIMessage` | assistant (model response) |
| `ToolMessage` | result of a tool call |

These types show up throughout LangGraph as well — agent state usually holds a list of them representing conversation history.

### Why LCEL over the old API

The old `LLMChain(llm=model, prompt=prompt)` was opaque — nested objects, no visibility into data flow. LCEL makes the pipeline explicit:

- The `|` operator reads left to right, matching how data actually moves.
- A chain is itself a `Runnable`, so it can be composed into a larger chain.
- `.stream()` works end-to-end automatically as long as each component supports it.

LCEL is good for linear, stateless pipelines. For anything with branching, loops, or shared state — use LangGraph.

---

## Tools and tool calling

### Defining a tool

`@tool` turns a Python function into a LangChain tool. The function name becomes the tool name; the docstring becomes the description sent to the model; the type annotations become the args schema.

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 25°C, sunny"

get_weather.name         # "get_weather"
get_weather.description  # "Get current weather for a city."
get_weather.args_schema  # {"city": {"type": "string"}}
```

### Binding tools to a model

`model.bind_tools()` attaches a list of tools to a model and returns a new Runnable. Every call to this model will include the tool schemas in the API request.

```python
model_with_tools = model.bind_tools([get_weather])
```

### The tool-calling protocol

The model does not execute tools — it only signals intent. When it decides to call a tool, it returns an `AIMessage` with a `tool_calls` field instead of a text response:

```python
response = model_with_tools.invoke("What's the weather in Tokyo?")
response.content      # "" (empty when calling a tool)
response.tool_calls   # [{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "abc123"}]
```

Your code executes the tool, wraps the result in a `ToolMessage`, and feeds it back. The `tool_call_id` links the result to the original call — required when multiple tools run in parallel.

```python
messages = [HumanMessage("What's the weather in Tokyo?")]

response = model_with_tools.invoke(messages)
messages.append(response)

for tc in response.tool_calls:
    result = get_weather.invoke(tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

final = model_with_tools.invoke(messages)
```

Skipping the `ToolMessage` is a protocol violation — the Anthropic API returns a 400 error if `tool_calls` in the history have no corresponding results.

### Why this loop is tedious to manage manually

The model may call tools across multiple turns. After receiving tool results, it might call another tool before giving a final answer. Maintaining the message list, matching `tool_call_id`s, and deciding when to stop all adds up fast.

This is exactly what LangGraph handles — it models the loop as an explicit graph so you define the logic once rather than writing it by hand each time.

### Tool result caching

Tool calls can be expensive — hitting external APIs, querying databases. If many users ask about the same city's weather, there's no reason to call the API every time.

Cache at the implementation level, not the LangChain level. `@tool` can't use `@lru_cache` directly (LangChain wraps the function), so extract the actual work into a separate cached function:

```python
import time
from functools import lru_cache
from langchain_core.tools import tool

# Simple in-memory cache with TTL
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes

def _fetch_weather(city: str) -> str:
    key = city.lower()
    if key in _cache:
        result, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return result
    result = call_weather_api(city)   # the real API call
    _cache[key] = (result, time.time())
    return result

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return _fetch_weather(city)
```

For multi-process deployments (multiple workers), use Redis instead of an in-memory dict so all workers share the same cache. The pattern is identical to any other API caching — tool calls are just function calls.
