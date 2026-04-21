# RAG Integration with LangChain

This document explains how to add Retrieval-Augmented Generation (RAG) capabilities to your research assistant using LangChain and LangGraph.

## Overview

RAG enhances your research assistant with the ability to:

- 📄 **Ingest Documents**: Load and process various document types (PDF, TXT, web pages)
- 🧮 **Create Embeddings**: Convert documents to vector representations for semantic search
- 🔍 **Semantic Search**: Find relevant information based on meaning, not just keywords
- 🤖 **Augmented Generation**: Provide context-aware answers using retrieved information
- 📚 **Source Attribution**: Track and cite where information comes from
- 💾 **Persistent Storage**: Maintain knowledge base across sessions

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   RAG Manager    │───▶│  Vector Store   │
│ (PDF, TXT, Web) │    │ - Loading        │    │ (FAISS/Chroma)  │
└─────────────────┘    │ - Chunking       │    └─────────────────┘
                       │ - Embedding      │
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     User        │───▶│   Agent with     │───▶│   LLM with      │
│   Question      │    │   RAG Tools      │    │   Context       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Examples Provided

### 1. Basic RAG with LangChain (`rag_with_langchain.py`)

**Features:**
- Document loading from multiple sources
- Vector store creation and management
- RAG-enhanced agent with multiple tools
- Simple RAG chain for direct Q&A
- Demo knowledge base setup

**Key Components:**
```python
# RAG Manager for document handling
rag_manager = RAGManager()

# Enhanced tools
@tool
def search_knowledge_base(query: str) -> str:
    """Search uploaded documents for relevant information"""

# RAG-enhanced agent with system prompt
SYSTEM_PROMPT = """You are a research assistant with access to:
1. Local Knowledge Base (search_knowledge_base)
2. Wikipedia (search_wikipedia) 
3. Calculator (calculate)"""
```

**Run:** `python examples/rag_with_langchain.py`

### 2. RAG-Enhanced Server (`rag_enhanced_server.py`)

**Features:**
- Full FastAPI server with RAG capabilities
- Document upload API endpoints
- Knowledge base management
- Conversation memory
- Production-ready structure

**API Endpoints:**
```
POST /upload          - Upload documents
GET  /knowledge-base   - Get KB information
POST /ask             - Ask questions
GET  /search          - Direct search
DELETE /documents/{id} - Delete documents
```

**Run:** `python examples/rag_enhanced_server.py`

### 3. Client Demo (`rag_client_demo.py`)

**Features:**
- Complete client interaction example
- Document upload demonstration
- Multi-turn conversation examples
- Various query types (single-source, multi-source, calculation)

**Run:** `python examples/rag_client_demo.py`

## Installation

Add RAG dependencies to your requirements.txt:

```txt
# RAG-specific dependencies
faiss-cpu>=1.7.4          # Vector database
chromadb>=0.4.0           # Alternative vector database
pypdf>=3.17.0             # PDF processing
beautifulsoup4>=4.12.0    # Web scraping
lxml>=4.9.0               # XML/HTML parsing
```

Install:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic RAG Setup

```python
from examples.rag_with_langchain import RAGManager, setup_demo_knowledge_base

# Initialize RAG
rag_manager = RAGManager()
setup_demo_knowledge_base()

# Ask questions
question = "What is LangChain?"
answer = run_rag_agent(question)
```

### 2. Server Setup

```bash
# Start server
python examples/rag_enhanced_server.py

# Upload document
curl -X POST "http://localhost:8000/upload" -F "file=@document.pdf"

# Ask question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about...?"}'
```

### 3. Custom Integration

```python
# Add to your existing agent
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Add RAG tool
@tool
def search_docs(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# Update your agent's tools
tools.append(search_docs)
```

## Integration Patterns

### Pattern 1: RAG as Additional Tool

Add RAG as another tool alongside existing capabilities:

```python
tools = [
    search_knowledge_base,  # RAG tool
    search_wikipedia,       # External search
    calculate,             # Computation
    get_weather,           # API calls
    # ... other tools
]
```

### Pattern 2: RAG-First Agent

Prioritize knowledge base search before external sources:

```python
SYSTEM_PROMPT = """
Always search the knowledge base FIRST for domain-specific information.
Only use external sources if information is not in the knowledge base.
"""
```

### Pattern 3: Hybrid Retrieval

Combine multiple retrieval strategies:

```python
def hybrid_search(query: str):
    # Semantic search
    semantic_results = vector_retriever.invoke(query)
    
    # Keyword search  
    keyword_results = keyword_retriever.invoke(query)
    
    # Combine and rank
    return merge_and_rank(semantic_results, keyword_results)
```

## Configuration Options

### Vector Store Options

```python
# FAISS (Local, fast)
vectorstore = FAISS.from_documents(documents, embeddings)

# Chroma (Local, persistent)
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./db")

# Pinecone (Cloud, scalable)
vectorstore = Pinecone.from_documents(documents, embeddings, index_name="my-index")
```

### Text Splitting Strategies

```python
# Default: Recursive splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Semantic splitting
splitter = SemanticChunker(embeddings)

# Custom splitting by headers
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1")])
```

### Retrieval Parameters

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",           # similarity, mmr, similarity_score_threshold
    search_kwargs={
        "k": 4,                        # Number of documents
        "score_threshold": 0.5,        # Minimum similarity score
        "fetch_k": 20,                 # Candidates before filtering
        "lambda_mult": 0.5             # Diversity vs relevance (MMR)
    }
)
```

## Best Practices

### 1. Document Preprocessing

```python
# Clean and normalize documents
def preprocess_document(doc):
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', doc.page_content)
    
    # Add metadata
    doc.metadata.update({
        'processed_date': datetime.now().isoformat(),
        'word_count': len(content.split())
    })
    
    return doc
```

### 2. Chunking Strategy

```python
# Optimal chunk size depends on domain
CHUNK_SIZES = {
    'technical_docs': 1500,    # Larger for detailed explanations
    'news_articles': 800,      # Medium for news content
    'chat_logs': 300,         # Smaller for conversational data
}

chunk_size = CHUNK_SIZES.get(document_type, 1000)
```

### 3. Query Enhancement

```python
def enhance_query(query: str, conversation_history: list) -> str:
    """Add context from conversation history"""
    
    # Extract entities and topics from history
    context = extract_key_topics(conversation_history)
    
    # Expand query with context
    enhanced_query = f"{query}\nContext: {context}"
    
    return enhanced_query
```

### 4. Response Post-processing

```python
def format_rag_response(answer: str, sources: list) -> str:
    """Format response with proper citations"""
    
    # Add source citations
    citations = []
    for i, source in enumerate(sources, 1):
        filename = source.metadata.get('source', 'Unknown')
        citations.append(f"[{i}] {filename}")
    
    if citations:
        answer += f"\n\nSources:\n" + "\n".join(citations)
    
    return answer
```

## Performance Optimization

### 1. Batch Processing

```python
# Process documents in batches
def add_documents_batch(documents: list, batch_size: int = 100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)
```

### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, k: int = 4):
    return vectorstore.similarity_search(query, k=k)
```

### 3. Async Processing

```python
import asyncio
from langchain_community.vectorstores import AsyncChroma

async def async_rag_search(queries: list):
    vectorstore = AsyncChroma(...)
    tasks = [vectorstore.asimilarity_search(query) for query in queries]
    return await asyncio.gather(*tasks)
```

## Troubleshooting

### Common Issues

1. **"No module named 'faiss'"**
   ```bash
   pip install faiss-cpu  # or faiss-gpu for GPU support
   ```

2. **"Vector store not initialized"**
   ```python
   # Ensure documents are loaded first
   if not rag_manager.retriever:
       rag_manager.setup_vectorstore(documents)
   ```

3. **Poor retrieval quality**
   - Adjust chunk size and overlap
   - Try different embedding models
   - Implement hybrid search
   - Add metadata filtering

4. **Memory issues with large documents**
   - Use batch processing
   - Implement document streaming
   - Consider cloud vector databases

### Debugging

```python
# Check vector store contents
print(f"Vector store has {vectorstore.index.ntotal} vectors")

# Examine retrieved documents
results = retriever.invoke("test query")
for doc in results:
    print(f"Source: {doc.metadata}")
    print(f"Content: {doc.page_content[:200]}...")
```

## Next Steps

1. **Experiment** with the provided examples
2. **Customize** the RAG tools for your domain
3. **Integrate** with your existing agent workflow
4. **Optimize** performance for your use case
5. **Scale** with cloud vector databases when needed

For advanced features like multi-modal RAG, conversational retrieval, or agent memory integration, see the LangChain documentation and community examples.