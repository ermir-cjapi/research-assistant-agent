"""
Example: Using Tools with LangGraph

This is the SAME functionality as langchain_only.py, but using LangGraph.
Compare the two to see what LangGraph adds.

Run: python examples/langgraph_version.py
"""
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Literal
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import wikipedia


# ============================================================
# STEP 1: Define Tools (SAME as LangChain version)
# ============================================================

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error: {e}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


tools = [search_wikipedia, calculate]


# ============================================================
# STEP 2: Define State (LangGraph-specific)
# ============================================================

class AgentState(TypedDict):
    """
    State that flows through the graph.
    The Annotated[list, operator.add] means messages accumulate.
    """
    messages: Annotated[list[BaseMessage], operator.add]


# ============================================================
# STEP 3: Create LLM with Tools (SAME as LangChain)
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# STEP 4: Define Nodes (LangGraph-specific)
# ============================================================

def agent_node(state: AgentState) -> dict:
    """
    The agent node - calls the LLM.
    This is similar to one iteration of our manual loop.
    """
    print("  [Agent Node] Calling LLM...")
    response = llm_with_tools.invoke(state["messages"])
    
    if response.tool_calls:
        print(f"  [Agent Node] LLM wants to call: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print("  [Agent Node] LLM finished")
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Routing function - decides next step.
    This replaces our 'if not response.tool_calls' check.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ============================================================
# STEP 5: Build the Graph (LangGraph-specific)
# ============================================================

def create_agent():
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))  # Automatic tool execution!
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")  # Loop back after tools
    
    # Compile with memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# STEP 6: Use the Agent
# ============================================================

agent = create_agent()


def run_agent(question: str, thread_id: str = "default") -> str:
    """Run the agent with a question."""
    print(f"\n--- Running Agent (thread: {thread_id}) ---")
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}}
    )
    
    return result["messages"][-1].content


# ============================================================
# MAIN: Run Examples
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE 1: Single Tool Call")
    print("=" * 60)
    answer = run_agent("What is 42 * 17?")
    print(f"\nFinal Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Wikipedia Search")
    print("=" * 60)
    answer = run_agent("Search Wikipedia for Python programming language")
    print(f"\nFinal Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Memory Demo (Same Thread)")
    print("=" * 60)
    # First question
    answer1 = run_agent("Who created Python?", thread_id="memory-demo")
    print(f"\nAnswer 1: {answer1}")
    
    # Follow-up - agent remembers!
    answer2 = run_agent("When was he born?", thread_id="memory-demo")
    print(f"\nAnswer 2: {answer2}")
    
    # Another follow-up
    answer3 = run_agent("What's 2024 minus that year?", thread_id="memory-demo")
    print(f"\nAnswer 3: {answer3}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Different Thread (No Memory)")
    print("=" * 60)
    # This won't know about previous questions
    answer = run_agent("When was he born?", thread_id="new-thread")
    print(f"\nAnswer (confused - no context): {answer}")
    
    print("\n" + "=" * 60)
    print("KEY ADVANTAGES OF LANGGRAPH")
    print("=" * 60)
    print("""
    Compare to langchain_only.py:
    
    1. NO manual while loop - graph handles iteration
    2. NO manual tool dispatch - ToolNode does it
    3. MEMORY works across calls (same thread_id)
    4. EASY to add branches with conditional_edges
    5. VISUAL representation possible (graph.get_graph())
    
    The trade-off: More setup code, but much more maintainable
    for complex agents.
    """)
