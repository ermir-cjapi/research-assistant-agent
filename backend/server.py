"""
Research Assistant Server

FastAPI server demonstrating LangChain, LangGraph, LangSmith, and RAG integration.

This server provides a clean API for:
- Chat interactions with the research agent
- Document upload and management
- Knowledge base operations
- Health monitoring
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_manager import RAGManager
from agent import ResearchAgent

# Load environment variables
load_dotenv()

# ============================================================================
# API MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    session_id: str
    sources_used: List[str] = []

class DocumentUploadResponse(BaseModel):
    """Response model for document uploads."""
    doc_id: str
    filename: str
    chunks_created: int
    status: str

class KnowledgeBaseInfo(BaseModel):
    """Response model for knowledge base information."""
    total_documents: int
    total_chunks: int
    documents: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    rag_available: bool
    knowledge_base_status: Dict[str, Any]

# ============================================================================
# APPLICATION SETUP
# ============================================================================

# Initialize RAG manager and agent
rag_manager = RAGManager()
agent = ResearchAgent(rag_manager)

# Create FastAPI application
app = FastAPI(
    title="Research Assistant",
    description="LangChain + LangGraph + LangSmith + RAG demonstration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Get API information and current status."""
    kb_info = rag_manager.get_info()
    return {
        "name": "Research Assistant",
        "description": "LangChain + LangGraph + LangSmith + RAG",
        "version": "2.0.0",
        "features": [
            "LangChain tools and chains",
            "LangGraph stateful workflows", 
            "LangSmith tracing",
            "RAG document upload and search",
            "Conversation memory",
            "Multi-source information synthesis"
        ],
        "knowledge_base": {
            "documents": kb_info["total_documents"],
            "chunks": kb_info["total_chunks"]
        },
        "endpoints": {
            "POST /chat": "Send a message to the agent",
            "POST /upload": "Upload a document to knowledge base",
            "GET /knowledge-base": "Get knowledge base information",
            "GET /health": "Health check"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and knowledge base status."""
    kb_info = rag_manager.get_info()
    return HealthResponse(
        status="healthy",
        rag_available=True,
        knowledge_base_status=kb_info
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the research assistant.
    
    The agent will:
    1. Process your message using LangGraph workflow
    2. Search knowledge base and/or Wikipedia as needed  
    3. Perform calculations if required
    4. Return a comprehensive response with source attribution
    
    All interactions are traced with LangSmith for observability.
    """
    try:
        result = agent.chat(request.message, request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the knowledge base.
    
    Supported formats: .txt, .pdf
    The document will be:
    1. Split into chunks for better retrieval
    2. Converted to vector embeddings  
    3. Stored in FAISS vector database
    4. Made available for search via the knowledge base tool
    """
    # Validate file
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
        result = await rag_manager.add_document(file)
        return DocumentUploadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base", response_model=KnowledgeBaseInfo)
async def get_knowledge_base():
    """Get information about the current knowledge base."""
    info = rag_manager.get_info()
    return KnowledgeBaseInfo(**info)

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base."""
    success = rag_manager.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "message": "Document deleted successfully",
        "doc_id": doc_id
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Display startup information."""
    print("\n" + "="*60)
    print("[STARTUP] RESEARCH ASSISTANT SERVER")
    print("="*60)
    print("Architecture:")
    print("  - LangChain: Tools, prompts, chains")
    print("  - LangGraph: Stateful agent workflow")  
    print("  - LangSmith: Automatic tracing & observability")
    print("  - RAG: Document upload, vector search, retrieval")
    print()
    
    kb_info = rag_manager.get_info()
    if kb_info["total_documents"] > 0:
        print(f"[KB] Knowledge Base: {kb_info['total_documents']} documents, {kb_info['total_chunks']} chunks")
    else:
        print("[KB] Knowledge Base: Empty (upload documents via /upload)")
    
    print()
    print("[API] Endpoints:")
    print("  - Chat: POST /chat")
    print("  - Upload: POST /upload") 
    print("  - Knowledge Base: GET /knowledge-base")
    print("  - Health: GET /health")
    print()
    print("[EXAMPLES] Try asking:")
    print('  - "What documents do I have?"')
    print('  - "Search my documents for [topic]"')
    print('  - "What is machine learning?" (Wikipedia)')
    print('  - "Calculate 42 * 1337"')
    print()
    print("[READY] Server ready!")
    print("="*60)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)