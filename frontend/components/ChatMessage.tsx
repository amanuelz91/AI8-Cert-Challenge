'use client'

import { useState } from 'react'
import { DocumentChunk } from '@/lib/streaming'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  confidence?: number | { confidence_score?: number; reasoning?: string }
  sources?: number
  chunks?: DocumentChunk[]
  timestamp: string
}

interface ChatMessageProps {
  message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const [showSources, setShowSources] = useState(false)

  const getConfidenceScore = (): number | undefined => {
    if (typeof message.confidence === 'number') {
      return message.confidence
    }
    if (message.confidence && typeof message.confidence === 'object') {
      return message.confidence.confidence_score
    }
    return undefined
  }

  const confidenceScore = getConfidenceScore()

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-primary-500 text-white'
            : 'bg-white text-gray-900 shadow-md border border-gray-200'
        }`}
      >
        <div className="whitespace-pre-wrap break-words">{message.content}</div>
        
        {!isUser && (message.sources !== undefined || message.chunks) && (
          <div className="mt-2 pt-2 border-t border-gray-200">
            <div className="text-xs text-gray-500 flex items-center gap-3 flex-wrap">
              {message.sources !== undefined && message.sources > 0 && (
                <span className="flex items-center gap-1">
                  <svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  {message.sources} source{message.sources !== 1 ? 's' : ''}
                </span>
              )}
              {confidenceScore !== undefined && (
                <span className="flex items-center gap-1">
                  <svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  Confidence: {Math.round(confidenceScore * 100)}%
                </span>
              )}
              {message.chunks && message.chunks.length > 0 && (
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="flex items-center gap-1 text-primary-600 hover:text-primary-700 underline"
                >
                  <svg
                    className={`w-3 h-3 transition-transform ${showSources ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                  {showSources ? 'Hide' : 'Show'} sources ({message.chunks.length})
                </button>
              )}
            </div>

            {/* Document chunks display */}
            {showSources && message.chunks && message.chunks.length > 0 && (
              <div className="mt-3 space-y-2">
                {message.chunks.map((chunk, index) => (
                  <div
                    key={chunk.metadata.id || index}
                    className="bg-gray-50 rounded-lg p-3 border border-gray-200 text-xs"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <div className="font-semibold text-gray-700 mb-1">
                          {chunk.metadata.title || 'Document Chunk'}
                        </div>
                        {chunk.metadata.source && (
                          <div className="text-gray-600 truncate">
                            {chunk.metadata.source.split('/').pop() || chunk.metadata.source}
                            {chunk.metadata.page && ` (Page ${chunk.metadata.page})`}
                          </div>
                        )}
                        {chunk.metadata.retrieval_method && (
                          <span className="inline-block mt-1 px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                            {chunk.metadata.retrieval_method}
                          </span>
                        )}
                      </div>
                      {chunk.metadata.url && (
                        <a
                          href={chunk.metadata.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-2 text-primary-600 hover:text-primary-700"
                        >
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                            />
                          </svg>
                        </a>
                      )}
                    </div>
                    <div className="text-gray-700 line-clamp-3">
                      {chunk.content}
                      {chunk.content.length > 500 && '...'}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

