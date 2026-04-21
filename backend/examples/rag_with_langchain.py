"""
Example: RAG (Retrieval-Augmented Generation) with LangChain and LangGraph

This demonstrates how to enhance your research assistant with RAG capabilities:
1. Document loading and chunking
2. Vector embeddings and storage
3. Retrieval-based context injection
4. Integration with existing agent workflow

Run: python examples/rag_with_langchain.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import TypedDict, Annotated, Literal
import operator

# LangChain RAG imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Other tools from previous examples
import wikipedia


# ============================================================
# STEP 1: RAG Components Setup
# ============================================================

class RAGManager:
    """Manages document loading, vectorization, and retrieval."""
    
    def __init__(self, persist_directory: str = "rag_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        self.retriever = None
        
    def load_documents_from_texts(self, texts: list[str], metadatas: list[dict] = None) -> list[Document]:
        """Load documents from raw text strings."""
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {"source": f"text_{i}"}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
    
    def load_documents_from_files(self, file_paths: list[str]) -> list[Document]:
        """Load documents from various file types."""
        docs = []
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: File {path} not found, skipping...")
                continue
                
            try:
                if path.endswith('.txt'):
                    loader = TextLoader(path, encoding='utf-8')
                elif path.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                else:
                    print(f"Unsupported file type: {path}")
                    continue
                    
                file_docs = loader.load()
                docs.extend(file_docs)
                print(f"Loaded {len(file_docs)} documents from {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        return docs
    
    def load_documents_from_urls(self, urls: list[str]) -> list[Document]:
        """Load documents from web URLs."""
        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                url_docs = loader.load()
                docs.extend(url_docs)
                print(f"Loaded {len(url_docs)} documents from {url}")
            except Exception as e:
                print(f"Error loading {url}: {e}")
        
        return docs
    
    def setup_vectorstore(self, documents: list[Document]):
        """Create vector store from documents."""
        if not documents:
            print("No documents to process!")
            return
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")
        
        # Create or load vector store
        if os.path.exists(self.persist_directory):
            print("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                self.persist_directory, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new vector store...")
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            self.vectorstore.save_local(self.persist_directory)
        
        # Setup retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Return top 4 relevant chunks
        )
        print("Vector store setup complete!")
    
    def search_documents(self, query: str, k: int = 4) -> list[Document]:
        """Search for relevant documents."""
        if not self.retriever:
            return []
        
        results = self.retriever.invoke(query)
        return results


# ============================================================
# STEP 2: RAG-Enhanced Tools
# ============================================================

# Global RAG manager (in practice, you'd inject this properly)
rag_manager = RAGManager()

@tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base for relevant information."""
    try:
        if not rag_manager.retriever:
            return "Knowledge base not initialized. Please load documents first."
        
        results = rag_manager.search_documents(query)
        if not results:
            return "No relevant information found in knowledge base."
        
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in results
        ])
        
        return f"Found {len(results)} relevant documents:\n\n{context}"
    
    except Exception as e:
        return f"Error searching knowledge base: {e}"

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


# All available tools
tools = [search_knowledge_base, search_wikipedia, calculate]


# ============================================================
# STEP 3: RAG-Enhanced Agent State
# ============================================================

class RAGAgentState(TypedDict):
    """
    Enhanced state that includes RAG context.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    context: str  # Retrieved context from knowledge base


# ============================================================
# STEP 4: RAG-Enhanced LLM Setup
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Enhanced system prompt for RAG
SYSTEM_PROMPT = """You are a research assistant with access to multiple information sources:

1. **Local Knowledge Base**: Use search_knowledge_base() to find information from uploaded documents
2. **Wikipedia**: Use search_wikipedia() for general knowledge and facts  
3. **Calculator**: Use calculate() for mathematical operations

When answering questions:
- ALWAYS try the local knowledge base first for domain-specific information
- Use Wikipedia for general facts not in your knowledge base
- Cite your sources when possible
- If information conflicts between sources, mention both and explain the discrepancy
- Be honest when information is not available in any source

Prioritize accuracy and cite your sources clearly."""


# ============================================================
# STEP 5: RAG-Enhanced Agent Nodes
# ============================================================

def rag_agent_node(state: RAGAgentState) -> dict:
    """
    Enhanced agent node that includes system context about RAG capabilities.
    """
    print("  [RAG Agent Node] Processing with RAG context...")
    
    # Prepare messages with system context
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        print(f"  [RAG Agent Node] LLM wants to call: {tool_names}")
    else:
        print("  [RAG Agent Node] LLM finished")
    
    return {"messages": [response]}


def should_continue(state: RAGAgentState) -> Literal["tools", "end"]:
    """Routing function - decides next step."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ============================================================
# STEP 6: Build RAG Agent Graph
# ============================================================

def create_rag_agent():
    """Build and compile the RAG-enhanced agent graph."""
    graph = StateGraph(RAGAgentState)
    
    # Add nodes
    graph.add_node("agent", rag_agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# STEP 7: Simple RAG Chain (Alternative Approach)
# ============================================================

def create_simple_rag_chain():
    """
    Creates a simple RAG chain without the full agent loop.
    Useful for direct question-answering scenarios.
    """
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use the following context to answer the user's question. 
        If the context doesn't contain relevant information, say so clearly.
        
        Context: {context}"""),
        ("user", "{question}")
    ])
    
    if not rag_manager.retriever:
        print("Warning: RAG retriever not initialized")
        return None
    
    rag_chain = (
        {
            "context": rag_manager.retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# ============================================================
# STEP 8: Demo Setup and Usage Functions
# ============================================================

def setup_demo_knowledge_base():
    """Setup a demo knowledge base with sample documents."""
    
    # Sample documents about AI and Machine Learning
    sample_docs = [
        """
        LangChain is a framework for developing applications powered by language models. 
        It provides tools for document loading, text splitting, embeddings, vector stores, 
        and retrieval. LangChain enables developers to create complex applications that 
        can reason over private data and take actions. The framework supports various 
        language models including OpenAI's GPT models, Anthropic's Claude, and open-source 
        models through Hugging Face.
        """,
        """
        Retrieval-Augmented Generation (RAG) is a technique that combines the power of 
        large language models with external knowledge retrieval. In RAG, relevant documents 
        are first retrieved from a knowledge base using similarity search, then this context 
        is provided to the language model to generate more accurate and informative responses. 
        This approach helps reduce hallucinations and allows models to access up-to-date information.
        """,
        """
        Vector embeddings are numerical representations of text that capture semantic meaning. 
        These high-dimensional vectors allow for similarity search and retrieval of relevant 
        documents. Popular embedding models include OpenAI's text-embedding-ada-002, 
        sentence-transformers, and various open-source alternatives. Vector databases like 
        FAISS, Chroma, and Pinecone are used to store and efficiently search these embeddings.
        """,
        """
        LangGraph is a library for building stateful, multi-actor applications with language models. 
        It extends LangChain's functionality by providing a graph-based approach to orchestrating 
        complex workflows. LangGraph allows developers to create agents that can maintain state, 
        use tools, and follow complex branching logic. It's particularly useful for building 
        research assistants, customer service bots, and other applications requiring multi-step reasoning.
        """
    ]
    
    metadatas = [
        {"source": "langchain_docs", "topic": "framework"},
        {"source": "rag_tutorial", "topic": "technique"},
        {"source": "embedding_guide", "topic": "vectors"},
        {"source": "langgraph_docs", "topic": "workflow"}
    ]
    
    # Load documents
    docs = rag_manager.load_documents_from_texts(sample_docs, metadatas)
    
    # You could also load from files:
    # docs = rag_manager.load_documents_from_files(['path/to/doc1.txt', 'path/to/doc2.pdf'])
    
    # Or from URLs:
    # docs = rag_manager.load_documents_from_urls(['https://example.com/article'])
    
    # Setup vector store
    rag_manager.setup_vectorstore(docs)
    print("Demo knowledge base setup complete!")


def run_rag_agent(question: str, thread_id: str = "rag-demo") -> str:
    """Run the RAG-enhanced agent with a question."""
    print(f"\n--- Running RAG Agent (thread: {thread_id}) ---")
    
    agent = create_rag_agent()
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "context": ""
        },
        {"configurable": {"thread_id": thread_id}}
    )
    
    return result["messages"][-1].content


def run_simple_rag_chain(question: str) -> str:
    """Run the simple RAG chain (non-agent approach)."""
    print(f"\n--- Running Simple RAG Chain ---")
    
    chain = create_simple_rag_chain()
    if not chain:
        return "RAG chain not available - knowledge base not initialized"
    
    return chain.invoke(question)


# ============================================================
# MAIN: Comprehensive RAG Demo
# ============================================================

def main():
    print("=" * 80)
    print("RAG WITH LANGCHAIN AND LANGGRAPH DEMO")
    print("=" * 80)
    
    # Setup demo knowledge base
    print("\n1. Setting up demo knowledge base...")
    setup_demo_knowledge_base()
    
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Simple RAG Chain (Direct Q&A)")
    print("=" * 60)
    answer = run_simple_rag_chain("What is LangChain and what does it provide?")
    print(f"\nAnswer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: RAG Agent (Can use multiple tools)")
    print("=" * 60)
    answer = run_rag_agent("What is RAG and how does it help with language models?")
    print(f"\nAnswer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Knowledge Base + External Search")
    print("=" * 60)
    answer = run_rag_agent(
        "Compare what you know about LangChain from your knowledge base with "
        "what Wikipedia says about it"
    )
    print(f"\nAnswer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Mixed Query (RAG + Calculation)")
    print("=" * 60)
    answer = run_rag_agent(
        "Based on your knowledge base, how many main components does LangChain have? "
        "Then calculate 2024 + that number."
    )
    print(f"\nAnswer: {answer}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Follow-up with Memory")
    print("=" * 60)
    # First question
    answer1 = run_rag_agent("What are vector embeddings?", "memory-demo")
    print(f"\nAnswer 1: {answer1}")
    
    # Follow-up question
    answer2 = run_rag_agent("What databases are mentioned for storing them?", "memory-demo")
    print(f"\nAnswer 2: {answer2}")
    
    print("\n" + "=" * 60)
    print("KEY RAG CONCEPTS DEMONSTRATED")
    print("=" * 60)
    print("""
    This example shows:
    
    1. 📄 DOCUMENT LOADING: From texts, files, URLs
    2. ✂️  TEXT SPLITTING: Chunking for better retrieval
    3. 🧮 EMBEDDINGS: Converting text to vectors  
    4. 🗄️  VECTOR STORAGE: FAISS for similarity search
    5. 🔍 RETRIEVAL: Finding relevant context
    6. 🤖 RAG INTEGRATION: Context + LLM generation
    7. 🔧 TOOL COMBINATION: RAG + Wikipedia + Calculator
    8. 🧠 MEMORY: Conversation history across turns
    9. 📊 SIMPLE vs AGENT: Two approaches to RAG
    
    RAG Benefits:
    ✅ Access to private/domain-specific knowledge
    ✅ Reduced hallucinations  
    ✅ Up-to-date information
    ✅ Source attribution
    ✅ Scalable knowledge management
    """)


if __name__ == "__main__":
    main()