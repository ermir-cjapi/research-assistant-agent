"""
Example: Using Prebuilt Agents

This is the RECOMMENDED approach for most use cases.
You get LangGraph features (memory, tool execution, interrupts) 
with a simple API.

Two options shown:
1. langchain.agents.create_agent - SIMPLEST (recommended)
2. langgraph.prebuilt.create_react_agent - More control

Run: python examples/prebuilt_agent.py
"""
from dotenv import load_dotenv
load_dotenv()

import wikipedia


# ============================================================
# Define Tools (plain functions work with create_agent!)
# ============================================================

def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error: {e}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (simulated)."""
    print(f"  [SENDING EMAIL] To: {to}, Subject: {subject}")
    return f"Email sent to {to} with subject '{subject}'"


# ============================================================
# OPTION A: langchain.agents.create_agent (SIMPLEST)
# ============================================================

from langchain.agents import create_agent

# This is the RECOMMENDED way - minimal code!
# Plain functions work directly - no @tool decorator needed!
simple_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_wikipedia, calculate],
    system_prompt="You are a helpful research assistant.",
)


# ============================================================
# OPTION B: langgraph.prebuilt.create_react_agent (More Control)
# ============================================================

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# For create_react_agent, wrap functions with @tool decorator
@tool
def search_wikipedia_tool(query: str) -> str:
    """Search Wikipedia and return a summary."""
    return search_wikipedia(query)

@tool
def calculate_tool(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return calculate(expression)

@tool 
def send_email_tool(to: str, subject: str, body: str) -> str:
    """Send an email (simulated)."""
    return send_email(to, subject, body)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# More explicit control over memory with checkpointer
react_agent = create_react_agent(
    model=llm,
    tools=[search_wikipedia_tool, calculate_tool],
    checkpointer=MemorySaver(),
)

# Agent with interrupt before dangerous operation
agent_with_approval = create_react_agent(
    model=llm,
    tools=[search_wikipedia_tool, calculate_tool, send_email_tool],
    checkpointer=MemorySaver(),
    interrupt_before=["send_email_tool"],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def chat_simple(agent, message: str):
    """Chat with simple_agent (langchain.agents.create_agent)."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


def chat_react(agent, message: str, thread_id: str = "default"):
    """Chat with react_agent (langgraph.prebuilt.create_react_agent)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [("user", message)]},
        config
    )
    return result["messages"][-1].content


def chat_with_state(agent, message: str, thread_id: str):
    """Send message and return full state for inspection."""
    config = {"configurable": {"thread_id": thread_id}}
    if message:
        result = agent.invoke(
            {"messages": [("user", message)]},
            config
        )
    else:
        result = agent.invoke(None, config)
    return result


# ============================================================
# EXAMPLE 1: Simple Agent (langchain.agents.create_agent)
# ============================================================

def example_simple_agent():
    print("\n" + "=" * 60)
    print("EXAMPLE 1: langchain.agents.create_agent (SIMPLEST)")
    print("=" * 60)
    
    print("\nCode:")
    print("""
    from langchain.agents import create_agent
    
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_wikipedia, calculate],  # Plain functions!
        system_prompt="You are a helpful assistant",
    )
    """)
    
    answer = chat_simple(simple_agent, "What is 42 * 17?")
    print(f"\nQ: What is 42 * 17?")
    print(f"A: {answer}")


# ============================================================
# EXAMPLE 2: Memory with create_react_agent
# ============================================================

def example_memory():
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Memory with create_react_agent")
    print("=" * 60)
    
    print("\nCode:")
    print("""
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    
    agent = create_react_agent(
        model=llm,
        tools=[...],
        checkpointer=MemorySaver(),  # Enables memory!
    )
    
    # Use thread_id to maintain conversation context
    agent.invoke(..., {"configurable": {"thread_id": "session-1"}})
    """)
    
    thread = "memory-example"
    
    # First question
    a1 = chat_react(react_agent, "Who created Python programming language?", thread)
    print(f"\nQ1: Who created Python?")
    print(f"A1: {a1}")
    
    # Follow-up - agent remembers context!
    a2 = chat_react(react_agent, "When was he born?", thread)
    print(f"\nQ2: When was he born?")
    print(f"A2: {a2}")
    
    # Another follow-up
    a3 = chat_react(react_agent, "Calculate 2024 minus that year", thread)
    print(f"\nQ3: Calculate 2024 minus that year")
    print(f"A3: {a3}")


# ============================================================
# EXAMPLE 3: Human-in-the-Loop (Interrupt Before)
# ============================================================

def example_human_in_loop():
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Human-in-the-Loop (Interrupt)")
    print("=" * 60)
    
    print("\nCode:")
    print("""
    agent = create_react_agent(
        model=llm,
        tools=[send_email_tool],
        checkpointer=MemorySaver(),
        interrupt_before=["send_email_tool"],  # Pause here!
    )
    """)
    
    thread = "email-approval"
    
    print("\nUser: Send an email to boss@company.com about the project update")
    print("\n[Agent processing...]")
    
    # This will pause BEFORE send_email is executed
    result = chat_with_state(
        agent_with_approval, 
        "Send an email to boss@company.com with subject 'Project Update' and body 'The project is on track.'",
        thread
    )
    
    # Check if we're at an interrupt point
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print("\n⚠️  INTERRUPT: Agent wants to send an email!")
        print(f"   Tool: {last_msg.tool_calls[0]['name']}")
        print(f"   Args: {last_msg.tool_calls[0]['args']}")
        
        # Simulate user approval
        print("\n[User reviews and approves...]")
        user_approval = input("Approve? (y/n): ").strip().lower()
        
        if user_approval == 'y':
            print("\n[Resuming agent execution...]")
            # Resume - pass None to continue from checkpoint
            config = {"configurable": {"thread_id": thread}}
            final_result = agent_with_approval.invoke(None, config)
            print(f"\nFinal: {final_result['messages'][-1].content}")
        else:
            print("\n❌ Email sending cancelled by user")
    else:
        print(f"Response: {last_msg.content}")


# ============================================================
# EXAMPLE 4: Comparing Complexity
# ============================================================

def example_comparison():
    print("\n" + "=" * 60)
    print("COMPLEXITY COMPARISON")
    print("=" * 60)
    
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │ PURE LANGCHAIN (manual loop) - ~40 lines                │
    │                                                         │
    │   while True:                                           │
    │       response = llm.invoke(messages)                   │
    │       if not response.tool_calls:                       │
    │           break                                         │
    │       for tc in response.tool_calls:                    │
    │           result = tool_map[tc.name].invoke(tc.args)    │
    │           messages.append(ToolMessage(...))             │
    │                                                         │
    │   ❌ No memory                                          │
    │   ❌ Manual tool dispatch                               │
    │   ❌ No interrupts                                      │
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │ langchain.agents.create_agent - ~5 lines (SIMPLEST!)    │
    │                                                         │
    │   from langchain.agents import create_agent             │
    │   agent = create_agent(                                 │
    │       model="openai:gpt-4o-mini",                       │
    │       tools=[func1, func2],  # Plain functions!         │
    │   )                                                     │
    │                                                         │
    │   ✅ LangGraph under the hood                           │
    │   ✅ Automatic tool execution                           │
    │   ✅ Memory built-in                                    │
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │ langgraph.prebuilt.create_react_agent - ~8 lines        │
    │                                                         │
    │   agent = create_react_agent(                           │
    │       model=llm,                                        │
    │       tools=[tool1, tool2],  # @tool decorated          │
    │       checkpointer=MemorySaver(),                       │
    │       interrupt_before=["dangerous_tool"],              │
    │   )                                                     │
    │                                                         │
    │   ✅ Explicit memory control                            │
    │   ✅ Human-in-the-loop                                  │
    │   ✅ Thread-based sessions                              │
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │ CUSTOM STATEGRAPH - ~30 lines                           │
    │                                                         │
    │   graph = StateGraph(AgentState)                        │
    │   graph.add_node("agent", agent_fn)                     │
    │   graph.add_node("tools", ToolNode(tools))              │
    │   graph.add_conditional_edges(...)                      │
    │   graph.add_edge("tools", "agent")                      │
    │   agent = graph.compile(checkpointer=...)               │
    │                                                         │
    │   ✅ Full control over flow                             │
    │   ✅ Custom branching logic                             │
    │   ✅ Multi-agent possible                               │
    │   ⚠️  More code to maintain                             │
    └─────────────────────────────────────────────────────────┘
    """)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    example_simple_agent()
    example_memory()
    example_comparison()
    
    # Uncomment to try interactive approval:
    # example_human_in_loop()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
    Start with: from langchain.agents import create_agent
    
    This gives you:
    - Simple API (plain functions as tools!)
    - LangGraph power under the hood
    - Memory, tool execution, all automatic
    
    Use create_react_agent when you need:
    - Explicit checkpointer/memory control
    - interrupt_before for human-in-the-loop
    - Thread-based session management
    
    Use custom StateGraph only for:
    - Multi-agent systems
    - Complex routing logic
    - Parallel node execution
    """)
