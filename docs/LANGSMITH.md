# LangSmith: Observability & Debugging for LLM Applications

## What is LangSmith?

**LangSmith** is a platform for **debugging, testing, and monitoring** LLM applications. Think of it as "DevTools for AI" - it lets you see exactly what's happening inside your agent.

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR APPLICATION                         │
│                                                                 │
│   User → Agent → LLM → Tools → LLM → Response                  │
│            │       │      │      │                              │
│            ▼       ▼      ▼      ▼                              │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              LANGSMITH (observability)                   │  │
│   │  • Traces every step                                     │  │
│   │  • Records inputs/outputs                                │  │
│   │  • Measures latency & tokens                             │  │
│   │  • Captures errors                                       │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Use LangSmith?

### The Problem Without Observability

When your agent doesn't work correctly, you ask:
- What did the LLM actually see?
- Which tools were called and with what arguments?
- Why did it make that decision?
- Where did the error occur?
- How long did each step take?

Without observability, you're debugging blind.

### What LangSmith Shows You

```
Trace: "Who invented Python and when?"
├── Agent Node (245ms)
│   ├── Input: [HumanMessage: "Who invented Python..."]
│   ├── LLM Call: gpt-4o-mini (189ms, 47 tokens)
│   └── Output: [AIMessage with tool_calls: search_wikipedia]
│
├── Tool Node (1,203ms)
│   ├── Tool: search_wikipedia
│   ├── Input: {"query": "Python programming language"}
│   └── Output: "Python is a programming language created by..."
│
├── Agent Node (312ms)
│   ├── Input: [Previous messages + ToolMessage]
│   ├── LLM Call: gpt-4o-mini (267ms, 156 tokens)
│   └── Output: [AIMessage: "Python was created by Guido..."]
│
└── Total: 1,760ms | 203 tokens | $0.0004
```

## LangSmith vs LangSmith Studio

| Feature | LangSmith | LangSmith Studio |
|---------|-----------|------------------|
| **Purpose** | Observability & debugging | Building & testing agents visually |
| **Interface** | Web dashboard for traces | Visual graph editor |
| **Use Case** | Monitor production, debug issues | Design agent workflows |
| **When** | During/after development | During development |
| **Cost** | Free tier available | Part of LangSmith |

### LangSmith (Tracing & Monitoring)
- **What it does**: Records every LLM call, tool invocation, and state change
- **When to use**: Development debugging, production monitoring
- **How it works**: Set environment variables, traces appear in dashboard

### LangSmith Studio (Visual Builder)
- **What it does**: Visual interface to build and test LangGraph agents
- **When to use**: Designing complex agent workflows
- **How it works**: Drag-and-drop graph builder, instant testing

## Quick Setup

### 1. Get API Key

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Create an account (free tier available)
3. Go to Settings → API Keys → Create API Key

### 2. Set Environment Variables

```bash
# Add to your .env file
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxx
LANGSMITH_PROJECT=research-assistant  # Optional: organize by project
```

### 3. That's It!

LangChain and LangGraph automatically detect these variables and start sending traces.

```python
# No code changes needed!
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_wikipedia],
)

# This call is automatically traced
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

## What You'll See in LangSmith Dashboard

### 1. Trace View
Every agent invocation creates a trace showing:
- Full conversation flow
- Each LLM call with prompts and completions
- Tool invocations with inputs/outputs
- Timing for each step
- Token counts and estimated costs
- Errors with stack traces

### 2. Runs Table
List of all executions with:
- Status (success/error)
- Duration
- Token usage
- Feedback scores

### 3. Debugging Features
- **Playground**: Re-run any trace with modified inputs
- **Compare**: Side-by-side comparison of runs
- **Filter**: Find traces by status, latency, model, etc.

## Integration Levels

### Level 1: Basic Tracing (Zero Code)
Just set environment variables:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxx
```

All LangChain/LangGraph calls are automatically traced.

### Level 2: Custom Metadata
Add context to your traces:

```python
from langsmith import traceable

@traceable(
    name="research_query",
    metadata={"user_id": "123", "feature": "wikipedia_search"}
)
def handle_research_query(question: str):
    return agent.invoke({"messages": [{"role": "user", "content": question}]})
```

### Level 3: Manual Spans
For non-LangChain code:

```python
from langsmith import trace

with trace("my_custom_operation") as span:
    # Your code here
    result = do_something()
    span.metadata["result_count"] = len(result)
```

### Level 4: Evaluation & Testing
Create datasets and run evaluations:

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create a dataset
dataset = client.create_dataset("research_questions")
client.create_examples(
    inputs=[{"question": "Who invented Python?"}],
    outputs=[{"answer": "Guido van Rossum"}],
    dataset_id=dataset.id
)

# Run evaluation
results = evaluate(
    lambda x: agent.invoke({"messages": [{"role": "user", "content": x["question"]}]}),
    data=dataset,
    evaluators=[correctness_evaluator]
)
```

## Common Use Cases

### 1. Debugging Why Agent Failed
```
Trace shows: Tool returned error → Agent didn't handle it → Wrong response
Fix: Add error handling in tool
```

### 2. Finding Slow Steps
```
Trace shows: Wikipedia search taking 3s (90% of total time)
Fix: Add caching or use faster API
```

### 3. Understanding Token Usage
```
Trace shows: System prompt = 500 tokens on every call
Fix: Shorten system prompt or use caching
```

### 4. Monitoring Production
```
Dashboard shows: Error rate spiked at 3pm
Investigation: External API was down
```

## Best Practices

### 1. Use Projects to Organize
```bash
LANGSMITH_PROJECT=research-assistant-prod  # Production
LANGSMITH_PROJECT=research-assistant-dev   # Development
```

### 2. Add Metadata for Filtering
```python
@traceable(metadata={"user_id": user_id, "session_id": session_id})
def chat(message: str):
    ...
```

### 3. Don't Trace in Tests (Optional)
```bash
# In test environment
LANGSMITH_TRACING=false
```

### 4. Use Feedback for Quality
```python
from langsmith import Client

client = Client()
client.create_feedback(
    run_id=run_id,
    key="user_rating",
    score=1.0,  # thumbs up
)
```

## Pricing

| Tier | Traces/Month | Features |
|------|-------------|----------|
| Free | 5,000 | Basic tracing, 14-day retention |
| Plus | 50,000 | Longer retention, more users |
| Enterprise | Unlimited | SSO, dedicated support |

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│  LangSmith = See inside your AI application                │
│                                                             │
│  • Set 2 env vars → automatic tracing                      │
│  • See every LLM call, tool use, decision                  │
│  • Debug issues, optimize performance                       │
│  • Monitor production                                       │
│                                                             │
│  LangSmith Studio = Visual builder for agents              │
│  (A feature within LangSmith for designing workflows)       │
└─────────────────────────────────────────────────────────────┘
```

The key insight: **LangSmith is observability** (like DataDog for AI), while **LangSmith Studio is a visual builder** (like a no-code tool for creating agents).
