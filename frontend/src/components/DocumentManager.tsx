'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  uploadDocument, 
  getKnowledgeBase, 
  deleteDocument, 
  DocumentInfo, 
  KnowledgeBaseInfo 
} from '@/lib/api';

interface DocumentManagerProps {
  isOpen: boolean;
  onClose: () => void;
  onDocumentsUpdate: () => void;
}

export default function DocumentManager({ isOpen, onClose, onDocumentsUpdate }: DocumentManagerProps) {
  const [knowledgeBase, setKnowledgeBase] = useState<KnowledgeBaseInfo | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadKnowledgeBase = async () => {
    try {
      const kb = await getKnowledgeBase();
      setKnowledgeBase(kb);
    } catch (error) {
      console.error('Failed to load knowledge base:', error);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadKnowledgeBase();
    }
  }, [isOpen]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['text/plain', 'application/pdf'];
    if (!allowedTypes.includes(file.type)) {
      setUploadError('Only .txt and .pdf files are supported');
      return;
    }

    // Validate file size (10MB limit)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      setUploadError('File size must be less than 10MB');
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const result = await uploadDocument(file);
      setUploadSuccess(`Successfully uploaded "${result.filename}" with ${result.chunks_created} chunks`);
      await loadKnowledgeBase();
      onDocumentsUpdate();
      
      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Failed to upload document');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    try {
      await deleteDocument(docId);
      await loadKnowledgeBase();
      onDocumentsUpdate();
      setUploadSuccess(`Deleted "${filename}"`);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Failed to delete document');
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">Document Manager</h2>
              <p className="text-sm text-gray-400">Upload and manage your knowledge base</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Knowledge Base Stats */}
        {knowledgeBase && (
          <div className="p-6 border-b border-gray-700">
            <h3 className="text-lg font-medium text-white mb-3">Knowledge Base Status</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-2xl font-bold text-blue-400">{knowledgeBase.total_documents}</div>
                <div className="text-sm text-gray-400">Documents</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-2xl font-bold text-green-400">{knowledgeBase.total_chunks}</div>
                <div className="text-sm text-gray-400">Searchable Chunks</div>
              </div>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="p-6 border-b border-gray-700">
          <h3 className="text-lg font-medium text-white mb-3">Upload New Document</h3>
          
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-gray-500 transition-colors">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf"
                onChange={handleFileUpload}
                disabled={isUploading}
                className="hidden"
                id="document-upload"
              />
              <label
                htmlFor="document-upload"
                className={`cursor-pointer ${isUploading ? 'cursor-not-allowed' : ''}`}
              >
                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gray-700 flex items-center justify-center">
                  {isUploading ? (
                    <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  )}
                </div>
                <div className="text-white font-medium mb-1">
                  {isUploading ? 'Uploading...' : 'Choose file to upload'}
                </div>
                <div className="text-sm text-gray-400">
                  Supports .txt and .pdf files (max 10MB)
                </div>
              </label>
            </div>

            {uploadError && (
              <div className="p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-300 text-sm">
                {uploadError}
              </div>
            )}

            {uploadSuccess && (
              <div className="p-3 bg-green-500/20 border border-green-500/30 rounded-lg text-green-300 text-sm">
                {uploadSuccess}
              </div>
            )}
          </div>
        </div>

        {/* Document List */}
        <div className="p-6">
          <h3 className="text-lg font-medium text-white mb-3">Uploaded Documents</h3>
          
          {knowledgeBase && Object.keys(knowledgeBase.documents).length > 0 ? (
            <div className="space-y-2">
              {Object.entries(knowledgeBase.documents).map(([docId, doc]) => (
                <div
                  key={docId}
                  className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-gray-700 flex items-center justify-center">
                      <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div>
                      <div className="text-white font-medium">{doc.filename}</div>
                      <div className="text-sm text-gray-400">
                        {doc.chunks} chunks • {formatFileSize(doc.size)}
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => handleDeleteDocument(docId, doc.filename)}
                    className="p-2 hover:bg-red-500/20 rounded-lg text-gray-400 hover:text-red-400 transition-colors"
                    title="Delete document"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <svg className="w-12 h-12 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
              <p>No documents uploaded yet</p>
              <p className="text-sm">Upload your first document to get started</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}