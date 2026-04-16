"""
LangGraph Research Assistant Agent.

This module implements a stateful agent using LangGraph that can:
- Search Wikipedia for information
- Remember conversation context across multiple turns
- Answer multi-step questions by chaining tool calls

LangSmith Integration:
- Set LANGSMITH_TRACING=true in .env for automatic tracing
- All LLM calls and tool invocations are traced
- View traces at https://smith.langchain.com
"""
from typing import TypedDict, Annotated, Literal
import operator
import uuid
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Optional: LangSmith traceable decorator for custom trace names
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from tools import ALL_TOOLS

# Log LangSmith status on import
_langsmith_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
if _langsmith_enabled:
    print(f"[LangSmith] Tracing enabled (project: {os.getenv('LANGSMITH_PROJECT', 'default')})")
else:
    print("[LangSmith] Tracing disabled (set LANGSMITH_TRACING=true to enable)")


SYSTEM_PROMPT = """You are a helpful research assistant with access to Wikipedia and other tools.

Your capabilities:
1. Search Wikipedia for factual information about any topic
2. Get detailed Wikipedia page content when needed
3. Perform mathematical calculations
4. Get current date and time
5. Count words in text

Guidelines:
- Use the search_wikipedia tool when asked about facts, people, events, or concepts
- For multi-step questions, break them down and use tools as needed
- Cite Wikipedia as your source when providing facts
- Be conversational and helpful
- Remember context from the conversation to provide coherent responses
"""


class AgentState(TypedDict):
    """
    The state that flows through the graph.
    
    messages: List of conversation messages (accumulates over time)
    """
    messages: Annotated[list[BaseMessage], operator.add]


def create_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.7):
    """
    Create and return a compiled LangGraph agent.
    
    Args:
        model_name: The OpenAI model to use
        temperature: Sampling temperature for the LLM
        
    Returns:
        A compiled LangGraph agent with memory checkpointing
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    def call_model(state: AgentState) -> dict:
        """The agent node - calls the LLM to decide next action."""
        messages = state["messages"]
        
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """
        Routing function to decide if we should call tools or end.
        
        Returns:
            "tools" if the LLM wants to use a tool
            "end" if the LLM is done and has a final response
        """
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(AgentState)
    
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    
    graph.set_entry_point("agent")
    
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    graph.add_edge("tools", "agent")
    
    memory = MemorySaver()
    agent = graph.compile(checkpointer=memory)
    
    return agent


research_agent = create_agent()


class ConversationManager:
    """
    Manages multiple conversation sessions with the research agent.
    Each session maintains its own thread_id for memory persistence.
    """
    
    def __init__(self):
        self.agent = research_agent
        self.sessions: dict[str, list[dict]] = {}
    
    def create_session(self) -> str:
        """Create a new conversation session and return its ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id
    
    @traceable(name="research_assistant_chat", metadata={"component": "conversation_manager"})
    def chat(self, session_id: str, message: str) -> dict:
        """
        Send a message in a conversation session.
        
        Args:
            session_id: The session identifier
            message: The user's message
            
        Returns:
            Dict with 'response' and 'tool_calls' used
            
        Note: This method is traced by LangSmith when LANGSMITH_TRACING=true
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        config = {
            "configurable": {"thread_id": session_id},
            # Add metadata for LangSmith filtering
            "metadata": {"session_id": session_id}
        }
        
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        tool_calls_made = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_made.append({
                        "name": tc["name"],
                        "args": tc["args"]
                    })
        
        final_response = result["messages"][-1].content
        
        self.sessions[session_id].append({
            "role": "user",
            "content": message
        })
        self.sessions[session_id].append({
            "role": "assistant",
            "content": final_response,
            "tool_calls": tool_calls_made
        })
        
        return {
            "response": final_response,
            "tool_calls": tool_calls_made
        }
    
    def get_history(self, session_id: str) -> list[dict]:
        """Get the conversation history for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


conversation_manager = ConversationManager()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    session_id = conversation_manager.create_session()
    print(f"Created session: {session_id}\n")
    
    questions = [
        "Who invented Python programming language?",
        "When was he born and where?",
        "What's 2024 minus his birth year?"
    ]
    
    for q in questions:
        print(f"User: {q}")
        result = conversation_manager.chat(session_id, q)
        print(f"Assistant: {result['response']}")
        if result['tool_calls']:
            print(f"  [Tools used: {[tc['name'] for tc in result['tool_calls']]}]")
        print()
