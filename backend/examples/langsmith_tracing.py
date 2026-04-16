"""
Example: LangSmith Integration for Observability

LangSmith provides tracing, debugging, and monitoring for LLM applications.
This example shows different levels of integration.

Setup:
1. Create account at https://smith.langchain.com
2. Get API key from Settings → API Keys
3. Set environment variables (see below)

Run: python examples/langsmith_tracing.py
"""
import os
from dotenv import load_dotenv

# Load environment variables BEFORE importing langchain
load_dotenv()

# ============================================================
# SETUP: Environment Variables for LangSmith
# ============================================================

# These can be set in .env file or here for demo:
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_xxx"
# os.environ["LANGSMITH_PROJECT"] = "research-assistant"

# Check if LangSmith is configured
langsmith_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
print(f"LangSmith Tracing: {'ENABLED' if langsmith_enabled else 'DISABLED'}")
if langsmith_enabled:
    print(f"Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
print()


# ============================================================
# LEVEL 1: Automatic Tracing (Zero Code)
# ============================================================

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def level1_automatic_tracing():
    """
    Just by setting LANGSMITH_TRACING=true, all LangChain calls are traced.
    No code changes needed!
    """
    print("=" * 60)
    print("LEVEL 1: Automatic Tracing")
    print("=" * 60)
    print("""
    With LANGSMITH_TRACING=true, this call is automatically traced:
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm.invoke([HumanMessage(content="Hello")])
    
    Go to smith.langchain.com to see the trace!
    """)
    
    if not langsmith_enabled:
        print("⚠️  Set LANGSMITH_TRACING=true to enable tracing")
        return
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke([HumanMessage(content="Say 'Hello from LangSmith!' in one line")])
    print(f"Response: {response.content}")
    print("\n✅ Check smith.langchain.com for the trace!")


# ============================================================
# LEVEL 2: Custom Trace Names with @traceable
# ============================================================

from langsmith import traceable

@traceable(name="wikipedia_research")
def level2_custom_trace_name(question: str):
    """
    Use @traceable to give your functions meaningful names in traces.
    This helps identify specific operations in the dashboard.
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke([HumanMessage(content=question)])
    return response.content


def demo_level2():
    print("\n" + "=" * 60)
    print("LEVEL 2: Custom Trace Names with @traceable")
    print("=" * 60)
    print("""
    @traceable(name="wikipedia_research")
    def my_function(question: str):
        ...
    
    Now the trace shows "wikipedia_research" instead of "my_function"
    """)
    
    if not langsmith_enabled:
        print("⚠️  Set LANGSMITH_TRACING=true to enable tracing")
        return
    
    result = level2_custom_trace_name("What is Python?")
    print(f"Response: {result[:100]}...")
    print("\n✅ Trace named 'wikipedia_research' visible in dashboard!")


# ============================================================
# LEVEL 3: Adding Metadata for Filtering
# ============================================================

@traceable(
    name="user_query",
    metadata={"feature": "research", "version": "1.0"}
)
def level3_with_metadata(user_id: str, question: str):
    """
    Add metadata to traces for filtering and analysis.
    Useful for: user tracking, A/B tests, feature flags, etc.
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Metadata is attached to this trace
    response = llm.invoke([HumanMessage(content=question)])
    return response.content


def demo_level3():
    print("\n" + "=" * 60)
    print("LEVEL 3: Adding Metadata")
    print("=" * 60)
    print("""
    @traceable(
        name="user_query",
        metadata={"user_id": "123", "feature": "research"}
    )
    def my_function(...):
        ...
    
    Filter traces in dashboard by metadata!
    """)
    
    if not langsmith_enabled:
        print("⚠️  Set LANGSMITH_TRACING=true to enable tracing")
        return
    
    result = level3_with_metadata("user_123", "Explain AI briefly")
    print(f"Response: {result[:100]}...")
    print("\n✅ Trace has metadata for filtering!")


# ============================================================
# LEVEL 4: Manual Spans for Custom Operations
# ============================================================

from langsmith import trace

def level4_manual_spans():
    """
    Use trace() context manager for non-LangChain code.
    Useful for: database calls, API requests, custom logic.
    """
    print("\n" + "=" * 60)
    print("LEVEL 4: Manual Spans")
    print("=" * 60)
    print("""
    from langsmith import trace
    
    with trace("custom_operation") as span:
        result = do_something()
        span.metadata["custom_key"] = "value"
    """)
    
    if not langsmith_enabled:
        print("⚠️  Set LANGSMITH_TRACING=true to enable tracing")
        return
    
    with trace("data_processing") as span:
        # Simulate some processing
        data = ["item1", "item2", "item3"]
        processed = [item.upper() for item in data]
        
        # Add metadata to span
        span.metadata["input_count"] = len(data)
        span.metadata["output_count"] = len(processed)
    
    print(f"Processed: {processed}")
    print("\n✅ Custom span 'data_processing' visible in trace!")


# ============================================================
# LEVEL 5: Tracing Agent Runs with Run ID
# ============================================================

from langsmith import Client

def level5_feedback_and_evaluation():
    """
    Capture run IDs for feedback and evaluation.
    This enables: user ratings, quality scoring, A/B testing.
    """
    print("\n" + "=" * 60)
    print("LEVEL 5: Feedback & Evaluation")
    print("=" * 60)
    print("""
    from langsmith import Client
    
    # After a run, you can add feedback
    client = Client()
    client.create_feedback(
        run_id=run_id,
        key="user_rating",
        score=1.0,  # thumbs up
    )
    """)
    
    if not langsmith_enabled:
        print("⚠️  Set LANGSMITH_TRACING=true to enable tracing")
        return
    
    print("Feedback allows you to:")
    print("  • Track user satisfaction (thumbs up/down)")
    print("  • Score response quality")
    print("  • Build datasets for fine-tuning")
    print("  • Run automated evaluations")


# ============================================================
# FULL EXAMPLE: Agent with Tracing
# ============================================================

from langchain.agents import create_agent

def full_agent_example():
    """
    Complete example: Agent with full LangSmith tracing.
    """
    print("\n" + "=" * 60)
    print("FULL EXAMPLE: Traced Agent")
    print("=" * 60)
    
    if not langsmith_enabled:
        print("⚠️  Set LANGSMITH_TRACING=true to enable tracing")
        print("\nTo enable:")
        print("  1. Get API key from smith.langchain.com")
        print("  2. Add to .env file:")
        print("     LANGSMITH_TRACING=true")
        print("     LANGSMITH_API_KEY=lsv2_pt_xxx")
        print("     LANGSMITH_PROJECT=research-assistant")
        return
    
    import wikipedia
    
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for information."""
        try:
            return wikipedia.summary(query, sentences=2)
        except Exception as e:
            return f"Error: {e}"
    
    # Create agent - automatically traced!
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_wikipedia],
        system_prompt="You are a helpful research assistant.",
    )
    
    print("\nRunning agent (check LangSmith for trace)...")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Who invented Python?"}]
    })
    
    print(f"\nResponse: {result['messages'][-1].content}")
    print("\n" + "=" * 60)
    print("CHECK YOUR LANGSMITH DASHBOARD!")
    print("=" * 60)
    print("""
    Go to: https://smith.langchain.com
    
    You'll see a trace showing:
    ├── Agent invocation
    │   ├── LLM call (deciding to use tool)
    │   ├── Tool: search_wikipedia
    │   ├── LLM call (generating response)
    │   └── Final response
    │
    With timing, token counts, and full I/O for each step!
    """)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              LANGSMITH INTEGRATION EXAMPLES                   ║
║                                                               ║
║  LangSmith = Observability for LLM applications               ║
║  See every LLM call, tool use, and decision in your agent     ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    level1_automatic_tracing()
    demo_level2()
    demo_level3()
    level4_manual_spans()
    level5_feedback_and_evaluation()
    full_agent_example()
    
    print("\n" + "=" * 60)
    print("SUMMARY: LangSmith Integration Levels")
    print("=" * 60)
    print("""
    Level 1: Set LANGSMITH_TRACING=true (automatic, zero code)
    Level 2: @traceable decorator (custom names)
    Level 3: Metadata (filtering, user tracking)
    Level 4: Manual spans (non-LangChain code)
    Level 5: Feedback & evaluation (quality tracking)
    
    Start with Level 1 - it's free and requires no code changes!
    """)
