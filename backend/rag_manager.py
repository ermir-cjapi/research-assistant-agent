"""
RAG Manager

This module handles all RAG (Retrieval-Augmented Generation) functionality:
- Document loading and processing
- Vector storage and retrieval
- Knowledge base management
"""
import os
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import UploadFile, HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGManager:
    """
    Manages document storage, processing, and retrieval for RAG functionality.
    
    Features:
    - Document upload and processing (PDF, TXT)
    - Text chunking and embedding
    - Vector storage with FAISS
    - Semantic search and retrieval
    """
    
    def __init__(self, storage_dir: str = "rag_storage"):
        """
        Initialize the RAG Manager.
        
        Args:
            storage_dir: Directory to store documents and vector database
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.vector_dir = self.storage_dir / "vectors"
        self.docs_dir = self.storage_dir / "documents" 
        self.metadata_file = self.storage_dir / "metadata.json"
        
        self.vector_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        
        # Components
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Vector store and retriever
        self.vectorstore = None
        self.retriever = None
        
        # Document metadata
        self.metadata = self._load_metadata()
        
        # Initialize existing vector store
        self._initialize_vectorstore()
    
    def _load_metadata(self) -> Dict:
        """Load document metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"documents": {}, "total_chunks": 0}
    
    def _save_metadata(self):
        """Save document metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _initialize_vectorstore(self):
        """Initialize vector store if it exists."""
        vector_file = self.vector_dir / "index.faiss"
        if vector_file.exists():
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
                print(f"[RAG] Loaded existing knowledge base: {self.metadata['total_chunks']} chunks")
            except Exception as e:
                print(f"[RAG] Error loading vector store: {e}")
    
    async def add_document(self, file: UploadFile) -> Dict[str, Any]:
        """
        Add a document to the knowledge base.
        
        Args:
            file: Uploaded file (PDF or TXT)
            
        Returns:
            Dictionary with upload results
            
        Raises:
            HTTPException: If document processing fails
        """
        # Generate unique ID and filename
        doc_id = str(uuid.uuid4())
        file_extension = Path(file.filename or "unknown").suffix
        stored_filename = f"{doc_id}{file_extension}"
        file_path = self.docs_dir / stored_filename
        
        # Save uploaded file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        try:
            # Load and process document
            docs = self._load_document(str(file_path), file.filename or "unknown")
            chunks = self.text_splitter.split_documents(docs)
            
            # Create or update vector store
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
            self.metadata["documents"][doc_id] = {
                "filename": file.filename,
                "stored_as": stored_filename,
                "chunks": len(chunks),
                "size": len(content)
            }
            self.metadata["total_chunks"] += len(chunks)
            self._save_metadata()
            
            return {
                "doc_id": doc_id,
                "filename": file.filename,
                "chunks_created": len(chunks),
                "status": "success"
            }
        
        except Exception as e:
            # Cleanup on error
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing document: {str(e)}"
            )
    
    def _load_document(self, file_path: str, original_filename: str) -> List[Document]:
        """
        Load document from file path.
        
        Args:
            file_path: Path to the stored file
            original_filename: Original filename for metadata
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type not supported
        """
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        docs = loader.load()
        for doc in docs:
            doc.metadata["original_filename"] = original_filename
        
        return docs
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            return []
        
        try:
            return self.retriever.invoke(query)[:k]
        except Exception as e:
            print(f"[RAG] Search error: {e}")
            return []
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get knowledge base information.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        return {
            "total_documents": len(self.metadata["documents"]),
            "total_chunks": self.metadata["total_chunks"],
            "documents": self.metadata["documents"]
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        
        Note: This removes metadata and files but doesn't rebuild vector store.
        For production use, consider implementing vector store rebuilding.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if doc_id not in self.metadata["documents"]:
            return False
        
        doc_info = self.metadata["documents"][doc_id]
        file_path = self.docs_dir / doc_info["stored_as"]
        
        # Remove file
        if file_path.exists():
            file_path.unlink()
        
        # Update metadata
        self.metadata["total_chunks"] -= doc_info["chunks"]
        del self.metadata["documents"][doc_id]
        self._save_metadata()
        
        return True