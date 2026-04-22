'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  ChatMessage as ChatMessageType, 
  createSession, 
  sendMessage, 
  sendRAGMessage,
  checkHealth,
  getHealthStatus,
  getKnowledgeBase,
  KnowledgeBaseInfo,
  HealthStatus 
} from '@/lib/api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import DocumentManager from './DocumentManager';

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isDocManagerOpen, setIsDocManagerOpen] = useState(false);
  const [knowledgeBase, setKnowledgeBase] = useState<KnowledgeBaseInfo | null>(null);
  const [useRAG, setUseRAG] = useState(true); // Toggle between RAG and basic mode
  const [ragAvailable, setRagAvailable] = useState(false); // Server RAG capability
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const loadKnowledgeBase = async () => {
    try {
      const kb = await getKnowledgeBase();
      setKnowledgeBase(kb);
    } catch (error) {
      console.log('Knowledge base not available, using basic mode');
      setUseRAG(false);
    }
  };

  useEffect(() => {
    const init = async () => {
      const healthy = await checkHealth();
      setIsConnected(healthy);
      
      if (healthy) {
        try {
          // Check RAG availability
          const healthStatus = await getHealthStatus();
          setRagAvailable(healthStatus.rag_available || false);
          
          const id = await createSession();
          setSessionId(id);
          
          if (healthStatus.rag_available) {
            await loadKnowledgeBase();
          }
        } catch {
          setError('Failed to create session. Please refresh the page.');
        }
      }
    };
    
    init();
  }, []);
  
  const handleSend = async (message: string) => {
    if (!sessionId) return;
    
    setError(null);
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    setIsLoading(true);
    
    try {
      let response;
      
      if (useRAG && ragAvailable) {
        // Use RAG-enhanced messaging (with automatic fallback)
        response = await sendRAGMessage(sessionId, message);
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: response.response,
            tool_calls: response.tool_calls,
            sources_used: response.sources_used,
          },
        ]);
      } else {
        // Use basic messaging
        response = await sendMessage(sessionId, message);
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: response.response,
            tool_calls: response.tool_calls,
          },
        ]);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleNewChat = async () => {
    setMessages([]);
    setError(null);
    try {
      const id = await createSession();
      setSessionId(id);
    } catch {
      setError('Failed to create new session.');
    }
  };

  const handleDocumentsUpdate = () => {
    loadKnowledgeBase();
  };
  
  if (!isConnected) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <div className="w-16 h-16 mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
          <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-white mb-2">Backend Not Connected</h2>
        <p className="text-gray-400 max-w-md">
          The Python backend server is not running. Please start it with:
        </p>
        <code className="mt-4 px-4 py-2 bg-gray-800 rounded-lg text-green-400 text-sm">
          cd backend && python server.py
        </code>
        <button
          onClick={() => window.location.reload()}
          className="mt-6 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          Retry Connection
        </button>
      </div>
    );
  }
  
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">Research Assistant</h1>
            <p className="text-xs text-gray-400">
              {ragAvailable ? (
                useRAG && knowledgeBase && knowledgeBase.total_documents > 0 
                  ? `RAG Mode • ${knowledgeBase.total_documents} docs • ${knowledgeBase.total_chunks} chunks`
                  : `RAG Available • ${knowledgeBase?.total_documents || 0} docs`
              ) : (
                'Basic Mode • Wikipedia + Calculator'
              )}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Document Manager Button */}
          <button
            onClick={() => setIsDocManagerOpen(true)}
            className={`px-3 py-2 text-sm rounded-lg transition-colors flex items-center gap-2 ${
              ragAvailable 
                ? 'bg-purple-600 hover:bg-purple-700 text-white'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            }`}
            title={ragAvailable ? "Manage Documents" : "RAG not available on server"}
            disabled={!ragAvailable}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            {knowledgeBase && knowledgeBase.total_documents > 0 && (
              <span className="bg-purple-800 text-xs px-2 py-0.5 rounded-full">
                {knowledgeBase.total_documents}
              </span>
            )}
          </button>
          
          {/* Mode Toggle */}
          <button
            onClick={() => setUseRAG(!useRAG)}
            className={`px-3 py-2 text-sm rounded-lg transition-colors flex items-center gap-2 ${
              useRAG 
                ? 'bg-green-600 hover:bg-green-700 text-white' 
                : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
            }`}
            title={ragAvailable ? (useRAG ? 'RAG Mode Active' : 'Basic Mode Active') : 'RAG not available on server'}
            disabled={!ragAvailable}
          >
            {useRAG ? (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                RAG
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Basic
              </>
            )}
          </button>
          
          {/* New Chat Button */}
          <button
            onClick={handleNewChat}
            className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-20 h-20 mb-6 rounded-full bg-gradient-to-br from-blue-500/20 to-purple-600/20 flex items-center justify-center">
              <svg className="w-10 h-10 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Welcome to Research Assistant</h2>
            <p className="text-gray-400 max-w-md mb-4">
              {useRAG && knowledgeBase && knowledgeBase.total_documents > 0 ? (
                <>
                  I can search your uploaded documents, Wikipedia, and perform calculations. 
                  I have access to <strong className="text-white">{knowledgeBase.total_documents} documents</strong> with <strong className="text-white">{knowledgeBase.total_chunks} searchable chunks</strong>.
                </>
              ) : (
                <>
                  I can search Wikipedia for information, remember our conversation context, and help you with multi-step research questions.
                  <strong className="text-yellow-400"> Upload documents to enable RAG mode!</strong>
                </>
              )}
            </p>
            
            {/* Quick upload hint */}
            {ragAvailable && (!knowledgeBase || knowledgeBase.total_documents === 0) && (
              <div className="mb-6 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                <div className="flex items-center gap-2 text-sm text-purple-300">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Click "Manage Documents" to upload PDFs or text files for personalized answers
                </div>
              </div>
            )}
            
            {!ragAvailable && (
              <div className="mb-6 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <div className="flex items-center gap-2 text-sm text-yellow-300">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  RAG features not available. Start server with: python backend/server_with_rag.py
                </div>
              </div>
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-lg">
              {(ragAvailable && useRAG && knowledgeBase && knowledgeBase.total_documents > 0 ? [
                "What information do you have in your knowledge base?",
                "Summarize the key concepts from my documents",
                "Search my documents for specific information",
                "Compare information from my docs with Wikipedia",
              ] : [
                "Who invented Python and when was he born?",
                "Tell me about the Transformer neural network", 
                "What is LangChain used for?",
                "Search for information about Alan Turing",
              ]).map((suggestion, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(suggestion)}
                  className="px-4 py-3 text-sm text-left bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700 rounded-xl text-gray-300 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg} />
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-md px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-sm text-gray-400">Researching...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {error && (
        <div className="mx-6 mb-2 px-4 py-2 bg-red-500/20 border border-red-500/30 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}
      
      <ChatInput
        onSend={handleSend}
        disabled={isLoading || !sessionId}
        placeholder={isLoading ? "Thinking..." : "Ask me anything..."}
      />
      
      {/* Document Manager Modal */}
      <DocumentManager
        isOpen={isDocManagerOpen}
        onClose={() => setIsDocManagerOpen(false)}
        onDocumentsUpdate={handleDocumentsUpdate}
      />
    </div>
  );
}
