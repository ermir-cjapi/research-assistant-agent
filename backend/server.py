"""
FastAPI server for the Research Assistant Agent.

Provides REST API endpoints for:
- Creating chat sessions
- Sending messages and receiving responses
- Managing conversation history
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent import conversation_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Research Assistant Agent starting up...")
    yield
    print("Research Assistant Agent shutting down...")


app = FastAPI(
    title="Research Assistant Agent API",
    description="A LangGraph-powered research assistant with Wikipedia search capabilities",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict]


class SessionResponse(BaseModel):
    session_id: str


class HistoryMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[dict] | None = None


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server is running."""
    return HealthResponse(status="ok", message="Research Assistant Agent is running")


@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new conversation session."""
    session_id = conversation_manager.create_session()
    return SessionResponse(session_id=session_id)


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest):
    """
    Send a message in a conversation session.
    
    The agent will:
    1. Process your message
    2. Search Wikipedia if needed
    3. Perform calculations if needed
    4. Return a response based on context
    """
    try:
        result = conversation_manager.chat(session_id, request.message)
        return ChatResponse(
            response=result["response"],
            tool_calls=result["tool_calls"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get the conversation history for a session."""
    try:
        history = conversation_manager.get_history(session_id)
        messages = [
            HistoryMessage(
                role=msg["role"],
                content=msg["content"],
                tool_calls=msg.get("tool_calls")
            )
            for msg in history
        ]
        return HistoryResponse(messages=messages)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session."""
    if conversation_manager.delete_session(session_id):
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Research Assistant Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /sessions": "Create a new chat session",
            "POST /sessions/{id}/chat": "Send a message",
            "GET /sessions/{id}/history": "Get conversation history",
            "DELETE /sessions/{id}": "Delete a session",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
