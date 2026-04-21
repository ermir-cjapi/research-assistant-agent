"""
Example: RAG-Enhanced Research Assistant Server

This shows how to integrate RAG capabilities into your existing FastAPI server.
It extends the basic agent with document management and retrieval capabilities.

Features:
- Document upload and management endpoints
- RAG-enhanced question answering
- Persistent vector storage
- Integration with existing agent workflow

Run: python examples/rag_enhanced_server.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# RAG imports (from previous example)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# Agent imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
import operator

import wikipedia
import uuid
import json


# ============================================================
# RAG MANAGER (Enhanced version)
# ============================================================

class EnhancedRAGManager:
    """Enhanced RAG manager with document management capabilities."""
    
    def __init__(self, base_dir: str = "rag_storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.vector_dir = self.base_dir / "vectors"
        self.docs_dir = self.base_dir / "documents"
        self.metadata_file = self.base_dir / "metadata.json"
        
        self.vector_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.vectorstore = None
        self.retriever = None
        self.document_metadata = self._load_metadata()
        
        # Initialize vector store if it exists
        self._initialize_vectorstore()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load document metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"documents": {}, "total_chunks": 0}
    
    def _save_metadata(self):
        """Save document metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.document_metadata, f, indent=2)
    
    def _initialize_vectorstore(self):
        """Initialize vector store if it exists."""
        if (self.vector_dir / "index.faiss").exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.vector_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                print(f"Loaded existing vector store with {self.document_metadata['total_chunks']} chunks")
            except Exception as e:
                print(f"Error loading vector store: {e}")
    
    async def add_document_from_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Add a document from uploaded file."""
        # Generate unique filename
        doc_id = str(uuid.uuid4())
        file_extension = Path(file.filename or "unknown").suffix
        stored_filename = f"{doc_id}{file_extension}"
        file_path = self.docs_dir / stored_filename
        
        # Save uploaded file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Process document
        try:
            docs = self._load_document(str(file_path), file.filename or "unknown")
            chunks = self.text_splitter.split_documents(docs)
            
            # Add to vector store
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            # Save vector store
            self.vectorstore.save_local(str(self.vector_dir))
            
            # Update metadata
            self.document_metadata["documents"][doc_id] = {
                "filename": file.filename,
                "stored_as": stored_filename,
                "chunks": len(chunks),
                "content_type": file.content_type,
                "size": len(content)
            }
            self.document_metadata["total_chunks"] += len(chunks)
            self._save_metadata()
            
            return {
                "doc_id": doc_id,
                "filename": file.filename,
                "chunks_created": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, f"Error processing document: {e}")
    
    def _load_document(self, file_path: str, original_filename: str) -> List[Document]:
        """Load document from file path."""
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        docs = loader.load()
        # Add original filename to metadata
        for doc in docs:
            doc.metadata["original_filename"] = original_filename
        
        return docs
    
    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """Search for relevant documents."""
        if not self.retriever:
            return []
        
        try:
            results = self.retriever.invoke(query)
            return results[:k]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def get_documents_info(self) -> Dict[str, Any]:
        """Get information about stored documents."""
        return {
            "total_documents": len(self.document_metadata["documents"]),
            "total_chunks": self.document_metadata["total_chunks"],
            "documents": {
                doc_id: {
                    "filename": info["filename"],
                    "chunks": info["chunks"],
                    "content_type": info.get("content_type", "unknown"),
                    "size": info["size"]
                }
                for doc_id, info in self.document_metadata["documents"].items()
            }
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document (note: doesn't remove from vector store)."""
        if doc_id not in self.document_metadata["documents"]:
            return False
        
        doc_info = self.document_metadata["documents"][doc_id]
        file_path = self.docs_dir / doc_info["stored_as"]
        
        if file_path.exists():
            file_path.unlink()
        
        self.document_metadata["total_chunks"] -= doc_info["chunks"]
        del self.document_metadata["documents"][doc_id]
        self._save_metadata()
        
        return True


# Global RAG manager
rag_manager = EnhancedRAGManager()


# ============================================================
# ENHANCED TOOLS
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information from uploaded documents."""
    try:
        results = rag_manager.search_documents(query)
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('original_filename', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    except Exception as e:
        return f"Error searching knowledge base: {e}"

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for general information."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia search error: {e}"

@tool  
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Simple eval for demo - in production, use a safer math parser
        allowed_chars = set('0123456789+-*/()%. ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def get_knowledge_base_info() -> str:
    """Get information about the current knowledge base contents."""
    try:
        info = rag_manager.get_documents_info()
        if info["total_documents"] == 0:
            return "The knowledge base is empty. No documents have been uploaded."
        
        summary = f"Knowledge Base Summary:\n"
        summary += f"- Total documents: {info['total_documents']}\n"
        summary += f"- Total chunks: {info['total_chunks']}\n\n"
        summary += "Documents:\n"
        
        for doc_id, doc_info in info["documents"].items():
            summary += f"- {doc_info['filename']}: {doc_info['chunks']} chunks\n"
        
        return summary
    except Exception as e:
        return f"Error getting knowledge base info: {e}"


tools = [search_knowledge_base, search_wikipedia, calculate, get_knowledge_base_info]


# ============================================================
# AGENT SETUP
# ============================================================

class RAGAgentState(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an advanced research assistant with access to multiple information sources:

🔍 **Available Tools:**
1. **search_knowledge_base**: Search uploaded documents for specific information
2. **search_wikipedia**: Search Wikipedia for general knowledge  
3. **calculate**: Perform mathematical calculations
4. **get_knowledge_base_info**: See what documents are available

🎯 **Best Practices:**
- Always check the knowledge base FIRST for domain-specific questions
- Use Wikipedia for general facts not in your knowledge base
- Cite sources clearly (mention document names when using knowledge base)
- If information conflicts between sources, explain the discrepancy
- Use get_knowledge_base_info to understand available documents

📚 **Response Quality:**
- Provide comprehensive, well-structured answers
- Include relevant context from multiple sources when beneficial
- Be honest about limitations and missing information
- Always cite your sources"""

def agent_node(state: RAGAgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: RAGAgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
graph = StateGraph(RAGAgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)


# ============================================================
# API MODELS
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    thread_id: Optional[str] = Field(default="default", description="Thread ID for conversation memory")

class QuestionResponse(BaseModel):
    answer: str
    thread_id: str
    sources_used: List[str] = []

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int
    status: str

class KnowledgeBaseInfo(BaseModel):
    total_documents: int
    total_chunks: int
    documents: Dict[str, Dict[str, Any]]


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RAG-Enhanced Research Assistant",
    description="A research assistant with RAG capabilities for document-based question answering",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "RAG-Enhanced Research Assistant API",
        "version": "1.0.0",
        "features": [
            "Document upload and processing",
            "Vector search and retrieval",
            "Multi-source question answering",
            "Conversation memory",
            "Tool integration"
        ]
    }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a document for the knowledge base."""
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    allowed_extensions = ['.txt', '.pdf']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        result = await rag_manager.add_document_from_upload(file)
        return DocumentUploadResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base", response_model=KnowledgeBaseInfo)
async def get_knowledge_base():
    """Get information about the current knowledge base."""
    info = rag_manager.get_documents_info()
    return KnowledgeBaseInfo(**info)

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base."""
    success = rag_manager.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully", "doc_id": doc_id}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG-enhanced research assistant."""
    
    try:
        # Run the agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            {"configurable": {"thread_id": request.thread_id}}
        )
        
        answer = result["messages"][-1].content
        
        # Extract sources used (simple heuristic based on tool calls)
        sources_used = []
        for message in result["messages"]:
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call["name"] == "search_knowledge_base":
                        sources_used.append("Knowledge Base")
                    elif tool_call["name"] == "search_wikipedia":
                        sources_used.append("Wikipedia")
                    elif tool_call["name"] == "calculate":
                        sources_used.append("Calculator")
        
        sources_used = list(set(sources_used))  # Remove duplicates
        
        return QuestionResponse(
            answer=answer,
            thread_id=request.thread_id,
            sources_used=sources_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/search")
async def search_knowledge_base_endpoint(query: str, limit: int = 4):
    """Direct search of the knowledge base (for debugging/testing)."""
    results = rag_manager.search_documents(query, k=limit)
    
    return {
        "query": query,
        "results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    }


# ============================================================
# STARTUP/SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    print("🚀 RAG-Enhanced Research Assistant API starting up...")
    
    # Setup demo knowledge base if empty
    info = rag_manager.get_documents_info()
    if info["total_documents"] == 0:
        print("📚 Setting up demo knowledge base...")
        await setup_demo_documents()
    
    print(f"📊 Knowledge Base: {info['total_documents']} documents, {info['total_chunks']} chunks")
    print("✅ API ready to serve requests!")

async def setup_demo_documents():
    """Setup demo documents if knowledge base is empty."""
    
    demo_content = """
    # LangChain Framework Overview
    
    LangChain is a comprehensive framework designed for developing applications powered by large language models (LLMs). 
    It provides a modular approach to building complex AI applications through several key components:
    
    ## Core Components
    1. **Document Loaders**: Support for various file formats (PDF, HTML, JSON, etc.)
    2. **Text Splitters**: Intelligent chunking of documents for optimal retrieval
    3. **Embeddings**: Converting text to numerical vectors for similarity search
    4. **Vector Stores**: Databases for storing and querying embeddings (FAISS, Chroma, Pinecone)
    5. **Retrievers**: Components for finding relevant documents
    6. **Chains**: Connecting LLMs with other components in workflows
    7. **Agents**: AI systems that can use tools and make decisions
    
    ## RAG Implementation
    Retrieval-Augmented Generation (RAG) in LangChain follows this pattern:
    1. Document ingestion and preprocessing
    2. Vector embedding creation
    3. Storage in vector database
    4. Query embedding and similarity search
    5. Context injection into LLM prompts
    6. Response generation with citations
    
    This approach significantly reduces hallucinations and enables LLMs to work with private, domain-specific knowledge.
    """
    
    # Create a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(demo_content)
        temp_path = f.name
    
    try:
        # Simulate file upload
        class MockUploadFile:
            def __init__(self, content: str, filename: str):
                self.content = content.encode()
                self.filename = filename
                self.content_type = "text/plain"
            
            async def read(self):
                return self.content
        
        mock_file = MockUploadFile(demo_content, "langchain_overview.txt")
        result = await rag_manager.add_document_from_upload(mock_file)
        print(f"✅ Demo document added: {result}")
        
    finally:
        # Clean up temp file
        os.unlink(temp_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting RAG-Enhanced Research Assistant Server...")
    print("Features:")
    print("  📤 Document Upload: POST /upload")
    print("  ❓ Ask Questions: POST /ask") 
    print("  📋 Knowledge Base Info: GET /knowledge-base")
    print("  🔍 Direct Search: GET /search?query=...")
    print("  🗑️  Delete Document: DELETE /documents/{doc_id}")
    print()
    
    uvicorn.run(
        "rag_enhanced_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )