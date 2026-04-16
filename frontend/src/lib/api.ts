const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ToolCall {
  name: string;
  args: Record<string, string>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  tool_calls?: ToolCall[];
}

export interface ChatResponse {
  response: string;
  tool_calls: ToolCall[];
}

export interface SessionResponse {
  session_id: string;
}

export interface HistoryResponse {
  messages: ChatMessage[];
}

export async function createSession(): Promise<string> {
  const response = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    throw new Error('Failed to create session');
  }
  
  const data: SessionResponse = await response.json();
  return data.session_id;
}

export async function sendMessage(sessionId: string, message: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to send message');
  }
  
  return response.json();
}

export async function getHistory(sessionId: string): Promise<ChatMessage[]> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/history`);
  
  if (!response.ok) {
    throw new Error('Failed to get history');
  }
  
  const data: HistoryResponse = await response.json();
  return data.messages;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error('Failed to delete session');
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
