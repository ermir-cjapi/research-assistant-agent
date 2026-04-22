# 🚀 Quick Start Guide

## One-Command Full Stack Startup

### Option 1: Both Servers (Recommended)
```bash
python start_server.py
```
Starts both backend and frontend automatically.

### Option 2: Backend Only
```bash
python start_backend.py
```
Just starts the Python API server.

### Option 3: Windows Users
Double-click `start.bat` file.

## What Gets Started

### 🔧 Backend (Python)
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs  
- **Features**: LangChain + LangGraph + RAG + LangSmith
- **Location**: Uses `backend/venv` automatically

### 🌐 Frontend (React)
- **URL**: http://localhost:3000
- **Features**: Document upload, chat interface, source attribution
- **Requirements**: Node.js and npm (auto-installs dependencies)

## Manual Startup (If Needed)

### Backend
```bash
cd backend
venv\Scripts\python.exe server.py
```

### Frontend
```bash
cd frontend
npm run dev
```

## Project Structure
```
research-assistant-agent/
├── start_server.py        # 🚀 Full stack startup
├── start_backend.py       # 🔧 Backend only
├── start.bat             # 🪟 Windows quick start
├── backend/
│   ├── server.py         # FastAPI server
│   ├── agent.py          # LangGraph agent
│   ├── rag_manager.py    # RAG functionality
│   ├── tools.py          # LangChain tools
│   └── venv/             # Virtual environment
└── frontend/             # React application
```

## Troubleshooting

**Port 8000 already in use**: Stop other servers or change port in server.py
**npm not found**: Install Node.js from https://nodejs.org
**Python not found**: Ensure Python 3.8+ is installed
**Missing dependencies**: Run `pip install -r backend/requirements.txt`

## Features Available

- 📤 **Document Upload**: Upload PDFs and text files
- 💬 **Intelligent Chat**: Ask questions about your documents
- 🔍 **Multi-source Search**: Knowledge base + Wikipedia + Calculator
- 📚 **Source Attribution**: See which sources were used
- 🧠 **Memory**: Conversation history across sessions
- 📊 **Tracing**: LangSmith observability (automatic)

Ready to explore LangChain, LangGraph, and RAG concepts! 🎉