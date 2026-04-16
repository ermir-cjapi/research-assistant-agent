# Research Assistant Agent

A complete LangChain + LangGraph project demonstrating how to build a stateful AI agent with:

- **Wikipedia Search**: Search and retrieve information from Wikipedia
- **Conversation Memory**: Remember context across multiple turns
- **Multi-step Reasoning**: Answer complex questions by chaining tool calls
- **Modern UI**: Beautiful Next.js React frontend

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│                    http://localhost:3000                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + Python)                    │
│                    http://localhost:8000                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    LangGraph Agent                          ││
│  │  ┌──────────┐    ┌──────────────┐    ┌──────────┐          ││
│  │  │  Agent   │───▶│  Should Use  │───▶│  Tools   │          ││
│  │  │  Node    │◀───│    Tool?     │◀───│  Node    │          ││
│  │  └──────────┘    └──────────────┘    └──────────┘          ││
│  │       │                                    │                 ││
│  │       └────────────────────────────────────┘                 ││
│  │                   (loop until done)                          ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Tools: search_wikipedia, get_wikipedia_page, calculate     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Key Learning Concepts

### LangChain Fundamentals
- **ChatOpenAI**: Wrapper for OpenAI chat models
- **Messages**: HumanMessage, AIMessage, SystemMessage
- **Tools**: @tool decorator to create custom tools
- **Tool Binding**: `llm.bind_tools()` to give LLM access to functions

### LangGraph Concepts
- **StateGraph**: Define workflow as a graph of nodes
- **State**: TypedDict that flows through the graph
- **Nodes**: Functions that process state
- **Edges**: Connections between nodes (fixed or conditional)
- **ToolNode**: Prebuilt node for executing tool calls
- **MemorySaver**: Checkpointer for conversation persistence

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- OpenAI API key

### 1. Set up the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Start the server
python server.py
```

The API will be available at http://localhost:8000

### 2. Set up the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The UI will be available at http://localhost:3000

## Project Structure

```
research-assistant-agent/
├── backend/
│   ├── .env.example      # Template for environment variables
│   ├── requirements.txt  # Python dependencies
│   ├── tools.py          # Custom LangChain tools
│   ├── agent.py          # LangGraph agent definition
│   └── server.py         # FastAPI server
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx      # Main page
│   │   │   ├── layout.tsx    # Root layout
│   │   │   └── globals.css   # Global styles
│   │   ├── components/
│   │   │   ├── Chat.tsx      # Main chat component
│   │   │   ├── ChatInput.tsx # Message input
│   │   │   └── ChatMessage.tsx # Message display
│   │   └── lib/
│   │       └── api.ts        # API client
│   └── package.json
│
├── langgraph_tutorial.html   # Reference tutorial
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/sessions` | Create new chat session |
| POST | `/sessions/{id}/chat` | Send message |
| GET | `/sessions/{id}/history` | Get conversation history |
| DELETE | `/sessions/{id}` | Delete session |

## Example Conversations

### Multi-step Research
```
User: Who invented Python and when was he born?
Agent: [searches Wikipedia for "Python programming language"]
       [searches Wikipedia for "Guido van Rossum"]
       Python was created by Guido van Rossum. He was born on 
       January 31, 1956, in Haarlem, Netherlands...

User: What's 2024 minus his birth year?
Agent: [calculates 2024 - 1956]
       2024 - 1956 = 68. Guido van Rossum is 68 years old in 2024.
```

### Context Memory
```
User: Tell me about the Transformer neural network
Agent: [searches Wikipedia]
       The Transformer is a deep learning architecture...
       introduced in 2017 by researchers at Google...

User: Who were the main authors of that paper?
Agent: The main authors of "Attention Is All You Need" were
       Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
       Llion Jones, Aidan Gomez, Łukasz Kaiser, and Illia Polosukhin.
```

## Tools Available

| Tool | Description |
|------|-------------|
| `search_wikipedia` | Search Wikipedia and get a summary |
| `get_wikipedia_page` | Get detailed content from a specific page |
| `calculate` | Evaluate mathematical expressions |
| `word_count` | Count words in a text |

## Extending the Agent

### Adding New Tools

1. Create a new tool in `backend/tools.py`:

```python
from langchain_core.tools import tool

@tool
def my_new_tool(param: str) -> str:
    """Description of what the tool does."""
    # Your implementation
    return result
```

2. Add it to `ALL_TOOLS` list in `tools.py`

### Modifying Agent Behavior

Edit the `SYSTEM_PROMPT` in `backend/agent.py` to change how the agent behaves.

### Using Different Models

Change the model in `create_agent()`:
```python
agent = create_agent(model_name="gpt-4o", temperature=0.5)
```

## Understanding LangChain vs LangGraph

**Important**: Tools can be used with LangChain alone! LangGraph is NOT required for tool usage.

### Quick Comparison

| What You Need | Use |
|---------------|-----|
| Simple Q&A, RAG, linear pipelines | **LangChain** (chains) |
| Tool calling (basic) | **LangChain** (manual loop) |
| Agent with memory + tools | **`create_react_agent`** ← Start here! |
| Human-in-the-loop approval | **`create_react_agent`** with `interrupt_before` |
| Complex multi-agent systems | **Custom StateGraph** |

### The Recommended Path

```python
# For 90% of use cases, this is all you need:
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",  # or "anthropic:claude-sonnet-4-6"
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant",
)

# That's it! LangGraph power with minimal code.
result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

### Learning Resources (in this repo)

1. **[docs/CONCEPTS.md](docs/CONCEPTS.md)** - LangChain vs LangGraph explained
2. **[docs/LANGSMITH.md](docs/LANGSMITH.md)** - Observability & debugging with LangSmith
3. **[backend/examples/langchain_only.py](backend/examples/langchain_only.py)** - Tools with pure LangChain (manual loop)
4. **[backend/examples/prebuilt_agent.py](backend/examples/prebuilt_agent.py)** - **RECOMMENDED** using `create_agent`
5. **[backend/examples/langgraph_version.py](backend/examples/langgraph_version.py)** - Custom StateGraph (full control)
6. **[backend/examples/langsmith_tracing.py](backend/examples/langsmith_tracing.py)** - LangSmith integration examples

Run the examples:
```bash
cd backend
python examples/langchain_only.py    # Manual loop - understand the basics
python examples/prebuilt_agent.py    # RECOMMENDED - simple API, full power
python examples/langgraph_version.py # Custom graph - when you need control
python examples/langsmith_tracing.py # See LangSmith integration
```

### The Three Levels

```python
# Level 1: Manual loop - for learning how it works
while True:
    response = llm_with_tools.invoke(messages)
    if not response.tool_calls: break
    for tc in response.tool_calls:
        result = execute_tool(tc)  # You handle this
        messages.append(ToolMessage(result))

# Level 2: create_agent (RECOMMENDED) - simple + powerful
from langchain.agents import create_agent
agent = create_agent(model="openai:gpt-4o-mini", tools=tools)

# Level 3: Custom StateGraph - full control for complex cases
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("agent", router, {...})
agent = graph.compile(checkpointer=memory)
```

## LangSmith Integration (Observability)

LangSmith provides debugging, tracing, and monitoring for your agent. See [docs/LANGSMITH.md](docs/LANGSMITH.md) for details.

### Quick Setup

```bash
# Add to your .env file
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your-key-here
LANGSMITH_PROJECT=research-assistant
```

### What You Get

- **Traces**: See every LLM call, tool invocation, and decision
- **Timing**: Know which steps are slow
- **Debugging**: Full input/output for each step
- **Monitoring**: Track errors and usage in production

### LangSmith vs LangSmith Studio

| | LangSmith | LangSmith Studio |
|--|-----------|------------------|
| **Purpose** | Observability & debugging | Visual agent builder |
| **When** | During/after development | During design |
| **How** | Set env vars, view dashboard | Drag-and-drop editor |

## Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## Troubleshooting

### Backend won't start
- Ensure you have Python 3.11+
- Check that `.env` file exists with valid API key
- Make sure all dependencies are installed

### Frontend can't connect
- Verify backend is running on port 8000
- Check for CORS issues in browser console
- Ensure no firewall is blocking connections

### Agent errors
- Check OpenAI API key is valid
- Verify you have API credits available
- Look at server logs for detailed error messages
