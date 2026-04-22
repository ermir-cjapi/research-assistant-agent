const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ToolCall {
  name: string;
  args: Record<string, string>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  tool_calls?: ToolCall[];
  sources_used?: string[];
}

export interface ChatResponse {
  response: string;
  tool_calls: ToolCall[];
  sources_used?: string[];
}

export interface SessionResponse {
  session_id: string;
}

export interface HistoryResponse {
  messages: ChatMessage[];
}

// RAG-specific interfaces
export interface DocumentInfo {
  doc_id: string;
  filename: string;
  chunks: number;
  content_type: string;
  size: number;
}

export interface KnowledgeBaseInfo {
  total_documents: number;
  total_chunks: number;
  documents: Record<string, DocumentInfo>;
}

export interface DocumentUploadResponse {
  doc_id: string;
  filename: string;
  chunks_created: number;
  status: string;
}

// Removed RAGChatResponse - now ChatResponse includes sources_used

// RAG API functions
export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to upload document');
  }

  return response.json();
}

export async function getKnowledgeBase(): Promise<KnowledgeBaseInfo> {
  const response = await fetch(`${API_BASE}/knowledge-base`);
  
  if (!response.ok) {
    throw new Error('Failed to get knowledge base info');
  }
  
  return response.json();
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${docId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error('Failed to delete document');
  }
}

export async function searchKnowledgeBase(query: string, limit: number = 4) {
  const response = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}&limit=${limit}`);
  
  if (!response.ok) {
    throw new Error('Failed to search knowledge base');
  }
  
  return response.json();
}

// Note: sendRAGMessage is now unified with sendMessage since the server 
// automatically handles RAG when documents are available

export async function createSession(): Promise<string> {
  // With the new unified server, we don't need to create sessions
  // We just generate a client-side session ID
  const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  return sessionId;
}

export async function sendMessage(sessionId: string, message: string): Promise<ChatResponse> {
  // Use the new unified chat endpoint
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      message: message,
      session_id: sessionId 
    }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to send message');
  }
  
  const chatResponse = await response.json();
  
  // Convert to expected format
  return {
    response: chatResponse.response,
    tool_calls: [], // New server doesn't expose tool calls in this format
    sources_used: chatResponse.sources_used || []
  };
}

// Note: The new unified server handles session management automatically
// History and session deletion are managed internally by LangGraph

export interface HealthStatus {
  status: string;
  message: string;
  rag_available?: boolean;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function getHealthStatus(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/health`);
  
  if (!response.ok) {
    throw new Error('Failed to get health status');
  }
  
  const healthData = await response.json();
  
  // Convert unified server response to expected format
  return {
    status: healthData.status,
    message: `Unified server: ${healthData.knowledge_base_status.total_documents} docs`,
    rag_available: healthData.rag_available
  };
}
