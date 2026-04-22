# Learning Guide: LangChain + LangGraph + RAG

This simplified implementation demonstrates all core concepts in one clean flow.

## 🎯 What You'll Learn

### 1. **LangChain Fundamentals**
- **Tools**: Functions your AI can call (`@tool` decorator)
- **Prompts**: Structured instructions to guide the AI
- **Chains**: Connecting LLM with other components

### 2. **LangGraph Concepts** 
- **State Management**: How data flows through the agent
- **Node-based Workflow**: Agent → Tools → Agent (loop until done)
- **Memory**: Conversation persistence across turns

### 3. **LangSmith Integration**
- **Automatic Tracing**: Every interaction is logged
- **Observability**: See exactly what your agent is thinking
- **Debugging**: Understand tool calls and reasoning

### 4. **RAG (Retrieval-Augmented Generation)**
- **Document Processing**: Split text → Create embeddings → Store in vector DB
- **Semantic Search**: Find relevant chunks based on meaning
- **Context Injection**: Provide retrieved info to the LLM

## 🚀 Quick Start

```bash
# 1. Start the server
cd backend
python start_server.py

# 2. Start the frontend (new terminal)
cd frontend  
npm run dev

# 3. Open http://localhost:3000
```

## 📁 File Structure (Simplified)

```
research-assistant-agent/
├── backend/
│   ├── server.py              # ⭐ FastAPI server
│   ├── agent.py               # LangGraph agent implementation  
│   ├── rag_manager.py         # RAG document management
│   ├── tools.py               # LangChain tools
│   ├── start_server.py        # Simple startup script
│   ├── requirements.txt       # Dependencies
│   └── .env                   # Your OPENAI_API_KEY
├── frontend/                  # React UI (unchanged)
└── LEARNING_GUIDE.md         # This guide
```

## 🔍 Core Concepts Explained

### LangChain Tools

Tools are functions your AI can call. Look at `tools.py`:

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search uploaded documents for relevant information."""
    # RAG search logic here
    return relevant_context

@tool  
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    return wikipedia.summary(query, sentences=3)
```

**Key Learning**: Tools extend your AI's capabilities beyond just text generation.

### LangGraph Workflow

The agent follows a simple loop:

```python
# 1. Agent decides what to do
def agent_node(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 2. Check if tools are needed  
def should_continue(state):
    if last_message.tool_calls:
        return "tools"  # Execute tools
    return "end"       # Done!

# 3. Execute tools and loop back to agent
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",  # → Execute tools → Back to agent
    "end": END         # → Finish
})
```

**Key Learning**: LangGraph manages the "thinking → acting → thinking" loop automatically.

### RAG Implementation

RAG adds your private knowledge:

```python
# 1. Process documents
docs = loader.load()                    # Load file
chunks = text_splitter.split(docs)      # Split into pieces  
vectorstore = FAISS.from_documents(chunks, embeddings)  # Create searchable index

# 2. Search when needed
def search_knowledge_base(query):
    results = vectorstore.similarity_search(query)  # Find relevant chunks
    return format_context(results)                  # Return as text
```

**Key Learning**: RAG = Retrieval (find relevant docs) + Augmented Generation (include in prompt).

### LangSmith Tracing

Just set environment variables:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "your-project-name"
```

**Key Learning**: LangSmith automatically captures every LLM call, tool use, and reasoning step.

## 🎓 Learning Exercises

### Exercise 1: Add a New Tool
Try adding a weather tool to `tools.py`:

```python
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    # You could integrate with a real weather API
    return f"The weather in {city} is sunny and 72°F"

# Don't forget to add to tools list!
tools = [..., get_weather]
```

### Exercise 2: Modify the System Prompt
Change the `system_prompt` in `agent.py` to make the agent more specialized (e.g., focus on a specific domain).

### Exercise 3: Upload Different Documents
Try uploading:
- Research papers (PDF)
- Technical documentation (TXT)
- Your own notes

Ask questions that require combining information from multiple sources.

### Exercise 4: Explore LangSmith
1. Go to [LangSmith](https://smith.langchain.com/)
2. Find your project traces
3. Click on a conversation to see the full workflow
4. Notice how tool calls and reasoning steps are captured

## 🔧 How It All Works Together

Here's what happens when you ask a question:

```
You: "What does my document say about machine learning?"

1. LangGraph receives your message
2. Agent node calls LLM with system prompt + your question
3. LLM decides to use search_knowledge_base tool
4. Tool searches your uploaded documents using RAG
5. Tool returns relevant text chunks
6. Agent gets tool result, generates final answer
7. You receive answer with source attribution
8. LangSmith logs the entire conversation flow
```

## 🚀 Next Steps

1. **Start Simple**: Run the unified server, upload a document, ask questions
2. **Explore Traces**: Look at LangSmith to see how decisions are made
3. **Modify & Experiment**: Add tools, change prompts, try different documents
4. **Understand the Flow**: Follow the code from API endpoint → LangGraph → Tools → Response

## 💡 Key Insights

- **LangChain**: Provides the building blocks (tools, prompts, vector stores)
- **LangGraph**: Orchestrates the workflow (agent loop, state management, memory)
- **RAG**: Gives the AI access to your specific knowledge
- **LangSmith**: Shows you exactly what's happening under the hood

The magic is in how these components work together to create an intelligent, observable, and extensible AI assistant!

## 📚 Further Reading

- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/)
- [LangSmith Tracing](https://docs.smith.langchain.com/)
- [RAG Concepts](https://python.langchain.com/docs/use_cases/question_answering)

Remember: The best way to learn is to experiment! 🧪