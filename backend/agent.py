"""
Research Assistant Agent

This module contains the LangGraph agent implementation.
It orchestrates the conversation flow using tools and maintains state.
"""
import os
from typing import List, Dict, Any, TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tools import create_tools


class AgentState(TypedDict):
    """State that flows through the LangGraph agent."""
    messages: Annotated[List, operator.add]


class ResearchAgent:
    """
    LangGraph-based research assistant agent.
    
    Features:
    - Stateful conversation management
    - Tool integration (RAG, Wikipedia, Calculator)
    - Memory persistence across sessions
    - LangSmith tracing integration
    """
    
    def __init__(self, rag_manager):
        """
        Initialize the research agent.
        
        Args:
            rag_manager: RAGManager instance for document operations
        """
        self.rag_manager = rag_manager
        
        # LangSmith tracing configuration
        self._setup_tracing()
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create tools
        self.tools = create_tools(rag_manager)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
        
        # Build and compile the agent
        self.agent_executor = self._build_agent()
        
        print(f"[AGENT] LangGraph agent initialized with {len(self.tools)} tools and memory")
    
    def _setup_tracing(self):
        """Configure LangSmith tracing."""
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "research-assistant")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return """You are an intelligent research assistant with access to multiple information sources.

Available tools:
1. search_knowledge_base: Search uploaded documents for specific information
2. search_wikipedia: Search Wikipedia for general knowledge
3. calculate: Perform mathematical calculations  
4. get_knowledge_base_info: See what documents are available

Guidelines:
- Always search the knowledge base FIRST for domain-specific questions
- Use Wikipedia for general knowledge not in your documents
- Cite your sources clearly
- If asked about concepts, provide comprehensive explanations with examples
- For complex questions, break them down and use multiple tools if needed

Be helpful, accurate, and educational in your responses."""
    
    def _agent_node(self, state: AgentState):
        """
        Main agent node - calls the LLM with tools.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with LLM response
        """
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentState):
        """
        Decide whether to continue with tool calls or end.
        
        Args:
            state: Current agent state
            
        Returns:
            "tools" to execute tools, "end" to finish
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"
    
    def _build_agent(self):
        """
        Build and compile the LangGraph agent.
        
        Returns:
            Compiled agent executor
        """
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        
        # Set entry point
        graph.set_entry_point("agent")
        
        # Add edges
        graph.add_conditional_edges(
            "agent", 
            self._should_continue, 
            {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")
        
        # Compile with memory
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    
    def chat(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a chat message through the agent.
        
        Args:
            message: User message
            session_id: Session identifier for memory persistence
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Invoke the agent
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=message)]},
                {"configurable": {"thread_id": session_id}}
            )
            
            # Extract the final response
            final_message = result["messages"][-1]
            response_text = final_message.content
            
            # Extract sources used from tool calls
            sources_used = self._extract_sources(result["messages"])
            
            return {
                "response": response_text,
                "session_id": session_id,
                "sources_used": sources_used
            }
            
        except Exception as e:
            raise Exception(f"Agent error: {str(e)}")
    
    def _extract_sources(self, messages: List) -> List[str]:
        """
        Extract sources used from tool calls in the conversation.
        
        Args:
            messages: List of messages from the conversation
            
        Returns:
            List of source names
        """
        sources_used = []
        
        for message in messages:
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call["name"]
                    if tool_name == "search_knowledge_base":
                        sources_used.append("Knowledge Base")
                    elif tool_name == "search_wikipedia":
                        sources_used.append("Wikipedia")
                    elif tool_name == "calculate":
                        sources_used.append("Calculator")
                    elif tool_name == "get_knowledge_base_info":
                        sources_used.append("Knowledge Base Info")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(sources_used))