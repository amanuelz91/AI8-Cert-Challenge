export interface DocumentChunk {
  content: string
  metadata: {
    id: string | number
    page?: number
    source?: string
    document_id?: string
    retrieval_method?: string
    title?: string
    url?: string
  }
}

export interface StreamChunk {
  chunk_type: 'start' | 'content' | 'end' | 'error' | 'sources'
  content?: string
  question?: string
  method?: string
  confidence?: number | { confidence_score?: number; reasoning?: string; context_utilization?: number; answer_completeness?: number }
  sources?: number
  chunks?: DocumentChunk[]
  error?: string
  metadata?: any
  timestamp?: string
}

export async function streamQuery(
  question: string,
  methods: string[] = ['production'],
  onChunk: (chunk: StreamChunk) => void,
  onError?: () => void
): Promise<void> {
  try {
    const response = await fetch('/api/stream/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        methods,
        include_confidence: true,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No reader available')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6) // Remove 'data: ' prefix
            const chunk: StreamChunk = JSON.parse(jsonStr)
            onChunk(chunk)

            if (chunk.chunk_type === 'end' || chunk.chunk_type === 'error') {
              return
            }
          } catch (e) {
            console.error('Error parsing chunk:', e)
          }
        }
      }
    }
  } catch (error) {
    console.error('Streaming error:', error)
    onChunk({
      chunk_type: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
    })
    onError?.()
    throw error
  }
}

