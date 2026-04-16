"""
Example: Using Tools with Pure LangChain (No LangGraph)

This demonstrates that you CAN use tools without LangGraph.
LangGraph adds stateful workflows, but for simple cases, LangChain works fine.

Run: python examples/langchain_only.py
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import wikipedia


# ============================================================
# STEP 1: Define Tools (same as LangGraph version)
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
tool_map = {t.name: t for t in tools}


# ============================================================
# STEP 2: Create LLM with Tools
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# STEP 3: Manual Agent Loop (what LangGraph automates)
# ============================================================

def run_agent(question: str, verbose: bool = True) -> str:
    """
    Run a simple agent loop with LangChain only.
    
    This is the MANUAL version of what LangGraph does automatically.
    """
    messages = [HumanMessage(content=question)]
    iteration = 0
    max_iterations = 10  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
        
        # Ask the LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if LLM wants to use tools
        if not response.tool_calls:
            if verbose:
                print("LLM finished (no tool calls)")
            return response.content
        
        # Execute each tool call
        if verbose:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if verbose:
                print(f"  Calling: {tool_name}({tool_args})")
            
            # Execute the tool
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            if verbose:
                print(f"  Result: {result[:100]}...")
            
            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            ))
    
    return "Max iterations reached"


# ============================================================
# STEP 4: Simple Chain (No Tools, Just Prompt → LLM)
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def simple_chain_example():
    """
    Simplest LangChain usage - no tools at all.
    Just prompt → LLM → output
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"question": "What is 2 + 2?"})
    print(f"Simple chain result: {result}")


# ============================================================
# MAIN: Run Examples
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE 1: Simple Chain (No Tools)")
    print("=" * 60)
    simple_chain_example()
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Single Tool Call")
    print("=" * 60)
    answer = run_agent("What is 42 * 17?")
    print(f"\nFinal Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Wikipedia Search")
    print("=" * 60)
    answer = run_agent("Search Wikipedia for 'Python programming language' and tell me who created it")
    print(f"\nFinal Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multi-Step (Multiple Tool Calls)")
    print("=" * 60)
    answer = run_agent("Search Wikipedia for Guido van Rossum, then calculate 2024 minus his birth year")
    print(f"\nFinal Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
    This works! But notice:
    
    1. We had to write the while loop ourselves
    2. We had to manually dispatch tool calls
    3. There's no memory between run_agent() calls
    4. Adding branching logic would be messy
    
    LangGraph solves these problems with:
    - StateGraph for the loop
    - ToolNode for automatic execution
    - Checkpointers for memory
    - Conditional edges for branching
    """)
