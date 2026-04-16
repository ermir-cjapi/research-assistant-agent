# LangChain vs LangGraph: Understanding the Concepts

This document explains the relationship between LangChain and LangGraph, and when to use each.

## The Key Insight

**LangChain** = Building blocks + simple chains  
**LangGraph** = Stateful workflows with loops and branching (built ON TOP of LangChain)  
**Prebuilt Agents** = LangGraph features with simple LangChain-like API

You can use tools with LangChain alone! But LangGraph (or prebuilt agents) give you more power.

---

## The Easiest Path: `create_agent` from LangChain

Before diving into the details, know that **you don't have to choose**. LangChain now provides `create_agent` which is built on LangGraph under the hood:

```python
from langchain.agents import create_agent

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    import wikipedia
    return wikipedia.summary(query, sentences=3)

# This is ALL you need for most cases!
agent = create_agent(
    model="openai:gpt-4o-mini",  # or "anthropic:claude-sonnet-4-6"
    tools=[search_wikipedia],
    system_prompt="You are a helpful research assistant",
)

# Use it
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Who created Python?"}]}
)
```

This is the **recommended starting point** - you get LangGraph features (memory, tool execution, human-in-the-loop) with a simple API.

### Alternative: `create_react_agent` from LangGraph

For more control, you can use the LangGraph prebuilt directly:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_react_agent(
    model=llm,
    tools=[search_wikipedia, calculate],
    checkpointer=MemorySaver(),
)

result = agent.invoke(
    {"messages": [("user", "Who created Python?")]},
    {"configurable": {"thread_id": "user-123"}}
)
```

---

## Part 1: Tools with Pure LangChain (No LangGraph)

### Simple Tool Calling

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# Define a tool
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    import wikipedia
    return wikipedia.summary(query, sentences=3)

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Create LLM with tools bound
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([search_wikipedia, calculate])

# The LLM will decide if it needs to use a tool
response = llm_with_tools.invoke([
    HumanMessage(content="What is 25 * 4?")
])

print(response.tool_calls)
# [{'name': 'calculate', 'args': {'expression': '25 * 4'}, 'id': '...'}]
```

### The Problem: LangChain Alone Doesn't Execute Tools

When you call `llm_with_tools.invoke()`, the LLM returns a **request** to call a tool, but it doesn't actually execute it. You need to:

1. Check if the response has `tool_calls`
2. Execute each tool manually
3. Send the results back to the LLM
4. Repeat until the LLM gives a final answer

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def run_agent_manually(question: str):
    """Manual agent loop - this is what LangGraph automates!"""
    messages = [HumanMessage(content=question)]
    
    while True:
        # Step 1: Ask the LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Step 2: Check if it wants to use tools
        if not response.tool_calls:
            # No tools needed - we have the final answer
            return response.content
        
        # Step 3: Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Find and execute the tool
            if tool_name == "search_wikipedia":
                result = search_wikipedia.invoke(tool_args)
            elif tool_name == "calculate":
                result = calculate.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Step 4: Add tool result to messages
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))
        
        # Loop back to Step 1 - LLM will see the tool results

# Usage
answer = run_agent_manually("Search Wikipedia for Python language and count the words")
print(answer)
```

### This Works But Has Limitations

1. **No built-in memory** - each call is independent
2. **Manual loop management** - you write the while loop
3. **No branching logic** - hard to add conditional paths
4. **No checkpointing** - can't pause/resume

---

## Part 2: Why LangGraph Exists

LangGraph solves these problems by modeling the agent as a **graph**:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    ┌───────────┐         ┌───────────┐                 │
│    │   Agent   │────────▶│  Should   │                 │
│    │   Node    │         │  Use      │                 │
│    │  (LLM)    │◀────────│  Tool?    │                 │
│    └───────────┘         └───────────┘                 │
│         │                      │                        │
│         │                      │ Yes                    │
│         │                      ▼                        │
│         │               ┌───────────┐                  │
│         │               │   Tool    │                  │
│         │               │   Node    │                  │
│         │               └───────────┘                  │
│         │ No                   │                        │
│         ▼                      │                        │
│    ┌───────────┐               │                        │
│    │    END    │◀──────────────┘                        │
│    └───────────┘                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### LangGraph Equivalent of Manual Loop

```python
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# State flows through the graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

tools = [search_wikipedia, calculate]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# Node 1: Call the LLM
def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Routing function: tools or end?
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))  # LangGraph handles tool execution!

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
graph.add_edge("tools", "agent")  # After tools, go back to agent

agent = graph.compile()

# Usage - same result, but with graph benefits
result = agent.invoke({
    "messages": [HumanMessage(content="Search Wikipedia for Python and count words")]
})
```

---

## Part 3: What LangGraph Adds

### 1. Automatic Tool Execution
```python
# LangChain: You manually execute tools
result = my_tool.invoke(args)

# LangGraph: ToolNode does it for you
graph.add_node("tools", ToolNode(tools))
```

### 2. Built-in Memory with Checkpointers
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

# Each thread_id maintains separate conversation history
config = {"configurable": {"thread_id": "user-123"}}

# First message
agent.invoke({"messages": [HumanMessage("Who is Guido?")]}, config)

# Follow-up - agent remembers context!
agent.invoke({"messages": [HumanMessage("When was he born?")]}, config)
```

### 3. Complex Branching
```python
def route_by_intent(state):
    # Analyze the message and route to different nodes
    if "weather" in state["messages"][-1].content.lower():
        return "weather_agent"
    elif "search" in state["messages"][-1].content.lower():
        return "search_agent"
    else:
        return "general_agent"

graph.add_conditional_edges("classifier", route_by_intent, {
    "weather_agent": "weather_node",
    "search_agent": "search_node",
    "general_agent": "general_node"
})
```

### 4. Human-in-the-Loop
```python
from langgraph.checkpoint.memory import MemorySaver

# Interrupt before executing dangerous operations
agent = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["dangerous_tool_node"]
)

# Run until interrupt point
result = agent.invoke(input, config)
# ... user reviews and approves ...
# Resume from checkpoint
result = agent.invoke(None, config)  # Continues from where it stopped
```

---

## Part 4: Decision Guide

### Use Pure LangChain When:

| Scenario | Example |
|----------|---------|
| Simple Q&A | Chatbot answering questions |
| Single tool call | "Translate this text" |
| Linear pipeline | prompt → LLM → parser |
| RAG | Retrieve docs → generate answer |
| No memory needed | Stateless API calls |

```python
# Simple chain - no graph needed
chain = prompt | llm | parser
result = chain.invoke({"question": "What is AI?"})
```

### Use LangGraph When:

| Scenario | Example |
|----------|---------|
| Multi-step reasoning | Research that requires multiple searches |
| Tool loops | Agent keeps calling tools until satisfied |
| Conversation memory | Remember context across messages |
| Conditional flows | Different paths based on user intent |
| Human approval | Pause for user to approve actions |
| Multi-agent | Multiple specialized agents working together |

```python
# Complex agent - graph makes it manageable
graph = StateGraph(AgentState)
graph.add_node("researcher", research_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)
# ... complex routing logic
agent = graph.compile(checkpointer=memory)
```

---

## Part 5: Code Comparison

### Task: "Search Wikipedia for Python and tell me when it was created"

#### Pure LangChain (Manual Loop)
```python
def langchain_agent(question):
    messages = [HumanMessage(content=question)]
    
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
            
        for tc in response.tool_calls:
            if tc["name"] == "search_wikipedia":
                result = search_wikipedia.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
```
- ✅ Works fine for simple cases
- ❌ No memory between calls
- ❌ Manual tool dispatch
- ❌ Hard to extend

#### LangGraph
```python
def langgraph_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())

agent = langgraph_agent()
agent.invoke({"messages": [HumanMessage(question)]}, {"configurable": {"thread_id": "1"}})
```
- ✅ Automatic tool execution
- ✅ Built-in memory
- ✅ Easy to extend with more nodes
- ✅ Visual graph representation

---

## Part 6: Prebuilt Agents - Best of Both Worlds

### Option A: `langchain.agents.create_agent` (Simplest)

The modern LangChain way - built on LangGraph under the hood:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",           # Model string format
    tools=[search_wikipedia, calculate],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Who invented Python?"}]}
)
```

### Option B: `langgraph.prebuilt.create_react_agent` (More Control)

When you need explicit checkpointer/memory control:

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_react_agent(
    model=llm,
    tools=[search_wikipedia, calculate],
    checkpointer=MemorySaver(),
)

# Invoke with thread_id for memory
result = agent.invoke(
    {"messages": [("user", "Who invented Python?")]},
    {"configurable": {"thread_id": "session-123"}}
)
```

### Advanced: Human-in-the-Loop

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    # Interrupt before dangerous operations
    interrupt_before=["send_email_tool"],
)

# Run until interrupt
config = {"configurable": {"thread_id": "email-session"}}
result = agent.invoke({"messages": [{"role": "user", "content": "Send email to boss"}]}, config)

# Agent pauses before send_email_tool
# ... user reviews and approves ...

# Resume execution
result = agent.invoke(None, config)  # Continues from checkpoint
```

### When to Use Each Approach

| Approach | Code Complexity | Flexibility | Use Case |
|----------|-----------------|-------------|----------|
| Pure LangChain (manual loop) | Low | Low | Learning, simple scripts |
| `create_react_agent` | Low | Medium | Most production agents |
| Custom StateGraph | High | High | Complex multi-agent systems |

### Progression Path

```
1. Start here for most cases:
   create_react_agent(model, tools, checkpointer)

2. Need more control? Add interrupt points:
   create_react_agent(..., interrupt_before=["dangerous_tool"])

3. Need custom logic? Build your own graph:
   StateGraph → add_node → add_edge → compile
```

---

## Summary

| Feature | Manual Loop | `create_agent` | `create_react_agent` | Custom StateGraph |
|---------|-------------|----------------|---------------------|-------------------|
| Import | `langchain_openai` | `langchain.agents` | `langgraph.prebuilt` | `langgraph.graph` |
| Tool execution | ❌ Manual | ✅ Automatic | ✅ Automatic | ✅ Automatic |
| Agent loop | ❌ Write yourself | ✅ Built-in | ✅ Built-in | ✅ You design it |
| Memory | ❌ Manual | ✅ Built-in | ✅ checkpointer | ✅ checkpointer |
| Human-in-the-loop | ❌ Manual | ✅ interrupt_before | ✅ interrupt_before | ✅ Full control |
| Custom branching | ❌ If/else | ❌ Limited | ❌ Limited | ✅ Conditional edges |
| Code complexity | Medium | **Very Low** | Low | High |

### The Recommendation

```
┌─────────────────────────────────────────────────────────────┐
│  RECOMMENDED: langchain.agents.create_agent()              │
│                                                             │
│   from langchain.agents import create_agent                │
│   agent = create_agent(                                    │
│       model="openai:gpt-4o-mini",                          │
│       tools=[tool1, tool2],                                │
│   )                                                         │
│                                                             │
│   ✅ Simplest API                                          │
│   ✅ LangGraph power under the hood                        │
│   ✅ Memory, interrupts, tool execution                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  FOR MORE CONTROL: langgraph.prebuilt.create_react_agent   │
│  - Explicit checkpointer configuration                     │
│  - Thread-based memory management                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  FOR COMPLEX CASES: Custom StateGraph                       │
│  - Multi-agent orchestration                                │
│  - Complex routing logic                                    │
│  - Custom state management                                  │
│  - Parallel node execution                                  │
└─────────────────────────────────────────────────────────────┘
```

**Bottom line**: 
- Start with `from langchain.agents import create_agent` - simplest API, built on LangGraph
- Use `create_react_agent` when you need explicit memory/checkpointer control
- Build custom StateGraph only for complex multi-agent workflows
- Manual loops are mainly for learning how it works under the hood
