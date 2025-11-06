'use client'

import { useState, useRef, useEffect } from 'react'
import ChatMessage from '@/components/ChatMessage'
import ChatInput from '@/components/ChatInput'
import { streamQuery, DocumentChunk } from '@/lib/streaming'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  confidence?: number | { confidence_score?: number; reasoning?: string }
  sources?: number
  chunks?: DocumentChunk[]
  timestamp: string
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedMethods, setSelectedMethods] = useState<string[]>(['production'])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Available methods
  const availableMethods = [
    { id: 'production', name: 'Production', description: 'Multi-retrieval workflow (naive, semantic, tool, parent document)' },
    { id: 'naive', name: 'Naive', description: 'Simple vector similarity search' },
    { id: 'semantic', name: 'Semantic', description: 'Semantic chunking with similarity search' },
    { id: 'tool', name: 'Tool', description: 'Web search based retrieval' },
    { id: 'hybrid', name: 'Hybrid', description: 'Combination of knowledge base and web search' },
    { id: 'bm25', name: 'BM25', description: 'Keyword-based retrieval using BM25 algorithm' },
  ]

  const toggleMethod = (methodId: string) => {
    setSelectedMethods((prev) => {
      if (prev.includes(methodId)) {
        // Don't allow deselecting all methods
        if (prev.length === 1) return prev
        return prev.filter((m) => m !== methodId)
      } else {
        return [...prev, methodId]
      }
    })
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (question: string) => {
    if (!question.trim() || isLoading) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)
    setError(null)

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString()
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      let fullContent = ''
      let documentChunks: DocumentChunk[] = []
      
      await streamQuery(
        question,
        selectedMethods,
        (chunk) => {
          if (chunk.chunk_type === 'content') {
            fullContent += chunk.content || ''
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: fullContent }
                  : msg
              )
            )
          } else if (chunk.chunk_type === 'sources') {
            // Store document chunks
            if (chunk.chunks) {
              documentChunks = chunk.chunks
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, chunks: documentChunks }
                    : msg
                )
              )
            }
          } else if (chunk.chunk_type === 'end') {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? {
                      ...msg,
                      content: fullContent,
                      confidence: chunk.confidence,
                      sources: chunk.sources,
                      chunks: documentChunks.length > 0 ? documentChunks : msg.chunks,
                    }
                  : msg
              )
            )
            setIsLoading(false)
          } else if (chunk.chunk_type === 'error') {
            setError(chunk.error || 'An error occurred')
            setIsLoading(false)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: 'Sorry, an error occurred while processing your question.' }
                  : msg
              )
            )
          }
        },
        () => {
          setIsLoading(false)
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      setIsLoading(false)
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, content: 'Sorry, an error occurred while processing your question.' }
            : msg
        )
      )
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Student Loan Q&A Chatbot
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Ask questions about student loans and financial aid
              </p>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm font-medium text-gray-700">Methods:</label>
              <div className="flex gap-2 flex-wrap">
                {availableMethods.map((method) => (
                  <label
                    key={method.id}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-sm border rounded-lg cursor-pointer transition-colors ${
                      selectedMethods.includes(method.id)
                        ? 'bg-primary-500 text-white border-primary-500'
                        : 'bg-white text-gray-700 border-gray-300 hover:border-primary-300'
                    } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    title={method.description}
                  >
                    <input
                      type="checkbox"
                      checked={selectedMethods.includes(method.id)}
                      onChange={() => toggleMethod(method.id)}
                      disabled={isLoading || (selectedMethods.length === 1 && selectedMethods.includes(method.id))}
                      className="sr-only"
                    />
                    <span>{method.name}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="inline-block p-4 bg-white rounded-full shadow-lg mb-4">
                <svg
                  className="w-12 h-12 text-primary-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                  />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-gray-700 mb-2">
                Welcome to the Student Loan Q&A Chatbot
              </h2>
              <p className="text-gray-600">
                Ask me anything about student loans, financial aid, or FSA programs
              </p>
            </div>
          )}

          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex items-center gap-2 text-gray-500">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary-500 border-t-transparent"></div>
                <span className="text-sm">Thinking...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mx-4 mb-2">
          <div className="max-w-4xl mx-auto bg-red-50 border border-red-200 text-red-700 px-4 py-2 rounded-lg text-sm">
            {error}
          </div>
        </div>
      )}

      {/* Chat Input */}
      <div className="bg-white border-t border-gray-200 px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <ChatInput
            onSendMessage={handleSendMessage}
            disabled={isLoading}
          />
        </div>
      </div>
    </div>
  )
}

