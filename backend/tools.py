"""
Research Assistant Tools

This module contains all the tools available to the LangGraph agent.
Each tool is a function the AI can call to perform specific tasks.
"""
from langchain_core.tools import tool
import wikipedia


def create_tools(rag_manager):
    """
    Create and return all tools for the research assistant.
    
    Args:
        rag_manager: Instance of RAGManager for document operations
        
    Returns:
        List of LangChain tools
    """
    
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the uploaded documents for relevant information."""
        results = rag_manager.search(query)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            filename = doc.metadata.get('original_filename', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[Document {i}: {filename}]\n{content}")
        
        return "\n\n".join(context_parts)

    @tool  
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for general information."""
        try:
            return wikipedia.summary(query, sentences=3)
        except Exception as e:
            return f"Wikipedia search failed: {str(e)}"

    @tool
    def calculate(expression: str) -> str:
        """Perform mathematical calculations."""
        try:
            # Simple eval for demo - in production use safer math parser
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"

    @tool
    def get_knowledge_base_info() -> str:
        """Get information about the current knowledge base."""
        info = rag_manager.get_info()
        if info["total_documents"] == 0:
            return "Knowledge base is empty. Upload documents to get started."
        
        summary = f"Knowledge Base:\n"
        summary += f"- {info['total_documents']} documents\n"
        summary += f"- {info['total_chunks']} searchable chunks\n\n"
        
        for doc_id, doc_info in info["documents"].items():
            summary += f"- {doc_info['filename']}: {doc_info['chunks']} chunks\n"
        
        return summary
    
    return [search_knowledge_base, search_wikipedia, calculate, get_knowledge_base_info]