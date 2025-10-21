# Production-Grade RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and Qdrant, featuring multiple retrieval strategies and production-ready architecture.

## Features

### 🚀 Production Retrieval Methods

- **Naive Retriever**: Fast cosine similarity with Qdrant vector store
- **Semantic Retriever**: Semantic chunking for better document boundaries
- **Tool-Based Retriever**: Real-time web search integration (Tavily)

### 🏗️ Architecture

- **SOLID Principles**: Clean separation of concerns and modular design
- **DRY Implementation**: Reusable components and shared utilities
- **LCEL Chains**: LangChain Expression Language for chain composition
- **LangGraph Workflows**: Complex multi-step RAG operations
- **Cloud Qdrant**: Production-ready vector database

### 📊 Evaluation Framework

- **Multiple Retrievers**: BM25, Contextual Compression, Multi-Query, Parent Document, Ensemble
- **RAGAS Integration**: Comprehensive evaluation metrics
- **LangSmith Monitoring**: Observability and performance tracking

## Quick Start

### 1. Installation

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies
uv sync
```

### 2. Quick API Start

```bash
# Check environment
python start.py check

# Start API server
python start.py start

# Test the API
python start.py test
```

**That's it!** Your RAG API is now running at http://localhost:8000

### 3. Environment Setup

Create a `.env` file:

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional (for tool-based search)
TAVILY_API_KEY=your_tavily_api_key

# Optional (for evaluation retrievers)
COHERE_API_KEY=your_cohere_api_key

# Optional (for cloud Qdrant)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
```

**Note**: The system will automatically detect and use Tavily search if `TAVILY_API_KEY` is provided. Tool-based retrieval will be disabled if no Tavily API key is available.

### 4. Data Preparation

Place your documents in the `data/` folder:

- PDF files (`.pdf`)
- CSV files (`complaints.csv`)

### 5. Basic Usage

#### **Direct Python Usage**

```python
from src.core.system import create_production_rag_system

# Initialize RAG system
rag_system = create_production_rag_system()

# Query the system
result = rag_system.query("What is FAFSA?", method="production")
print(result["response"])
```

#### **API Usage**

Start the API server:

```bash
# Using the startup script
python start.py start

# Or directly with uvicorn
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test the API:

```bash
# Run automated tests
python start.py test

# Interactive client
python start.py interactive

# Or use the client script directly
python client.py
```

#### **API Endpoints**

- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/system/stats` - System statistics
- **GET** `/system/methods` - Available retrieval methods
- **POST** `/query` - Single query
- **POST** `/query/batch` - Batch queries
- **POST** `/system/reload` - Reload system
- **POST** `/stream/query` - Streaming query
- **POST** `/stream/query/llm` - LLM streaming query
- **GET** `/stream/query/simple` - Simple streaming query

#### **Example API Calls**

```bash
# Health check
curl http://localhost:8000/health

# Single query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FAFSA?", "method": "production"}'

# System stats
curl http://localhost:8000/system/stats

# Batch queries
curl -X POST http://localhost:8000/query/batch \
  -H "Content-Type: application/json" \
  -d '{"questions": ["What is FAFSA?", "How do I apply for loans?"]}'

# Streaming query
curl -X POST http://localhost:8000/stream/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FAFSA?", "method": "production"}' \
  --no-buffer

# Simple streaming query
curl http://localhost:8000/stream/query/simple?question="What is FAFSA?"
```

#### **API Documentation**

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### **Streaming Support**

The API supports real-time streaming responses:

```python
# Python streaming client
from streaming_client import StreamingRAGClient

client = StreamingRAGClient()

# Stream a query
for chunk in client.stream_query("What is FAFSA?"):
    if chunk["chunk_type"] == "content":
        print(chunk["content"], end="", flush=True)
    elif chunk["chunk_type"] == "end":
        print(f"\nSources: {chunk['sources']}")
```

```javascript
// JavaScript streaming example
const eventSource = new EventSource(
  "/stream/query/simple?question=What is FAFSA?"
);

eventSource.onmessage = function (event) {
  const chunk = JSON.parse(event.data);
  if (chunk.chunk_type === "content") {
    document.getElementById("response").innerHTML += chunk.content;
  }
};
```

**Streaming Features:**

- **Real-time responses** as they're generated
- **Server-Sent Events (SSE)** format
- **LLM streaming** for true real-time generation
- **Chunk-based delivery** for smooth user experience
- **Error handling** with graceful degradation

## Architecture Overview

```
src/
├── api/                           # FastAPI application
│   ├── endpoints/                 # API route handlers
│   │   ├── health.py             # Health check endpoints
│   │   ├── query.py              # Query endpoints
│   │   └── system.py             # System management endpoints
│   ├── models/                    # Pydantic schemas
│   │   └── schemas.py            # Request/response models
│   ├── services/                  # Business logic services
│   │   └── rag_service.py        # RAG service layer
│   └── main.py                   # FastAPI app initialization
├── core/                          # Core business logic
│   ├── data/                      # Document processing
│   ├── embeddings/                # Embedding management
│   ├── vectorstore/               # Vector database operations
│   ├── retrieval/                 # Retrieval strategies
│   └── system/                    # Main RAG system
├── chains/                        # LCEL chain compositions
├── workflows/                     # LangGraph workflows
├── config/                        # Configuration management
└── utils/                         # Utility functions
```

## Retrieval Methods

### Production Methods

1. **Naive Retriever**

   - Fast cosine similarity search
   - Qdrant vector store integration
   - Configurable similarity thresholds

2. **Semantic Retriever**

   - Semantic chunking for better boundaries
   - Preserves semantic meaning across chunks
   - Improved retrieval quality

3. **Tool-Based Retriever**
   - Real-time web search (Tavily)
   - Handles questions not in knowledge base
   - Hybrid approach (KB + web search)

### Evaluation Methods

4. **BM25 Retriever**

   - Keyword-based retrieval
   - Traditional information retrieval
   - Baseline comparison

5. **Contextual Compression**

   - AI-powered reranking (Cohere)
   - Improved semantic relevance
   - Premium quality results

6. **Multi-Query Retriever**

   - LLM query expansion
   - Multiple query perspectives
   - Comprehensive coverage

7. **Parent Document Retriever**

   - Hierarchical retrieval
   - Maximum context preservation
   - Full document context

8. **Ensemble Retriever**
   - Weighted combination of methods
   - Best of all approaches
   - Comprehensive coverage

## Configuration

The system uses Pydantic for configuration management:

```python
from src.config.settings import get_config

config = get_config()
print(config.llm.model_name)  # gpt-4o-mini
print(config.retrieval.default_k)  # 5
```

## Advanced Usage

### Custom Retrievers

```python
from src.core.retrieval import create_naive_retriever

# Create custom retriever
retriever = create_naive_retriever(
    vector_store=vector_store,
    k=10,
    similarity_threshold=0.8
)
```

### Custom Workflows

```python
from src.workflows.rag_workflow import create_rag_workflow_builder

# Create custom workflow
builder = create_rag_workflow_builder()
workflow = builder.create_production_rag_workflow(
    naive_retriever, semantic_retriever, tool_retriever
)
```

### Evaluation

```python
# Run evaluation (separate module)
from src.evaluation import run_ragas_evaluation

results = run_ragas_evaluation(
    dataset=test_dataset,
    retrievers=[naive_retriever, semantic_retriever],
    llm_model="gpt-4o-mini"
)
```

## Performance Characteristics

| Method          | Speed      | Quality    | Cost       | Use Case          |
| --------------- | ---------- | ---------- | ---------- | ----------------- |
| Naive           | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | General purpose   |
| Semantic        | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | Complex queries   |
| Tool-Based      | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐     | Real-time info    |
| Compression     | ⭐⭐       | ⭐⭐⭐⭐⭐ | ⭐⭐       | Premium quality   |
| Multi-Query     | ⭐⭐       | ⭐⭐⭐⭐   | ⭐⭐       | Ambiguous queries |
| Parent Document | ⭐⭐       | ⭐⭐⭐     | ⭐⭐⭐     | Full context      |
| Ensemble        | ⭐         | ⭐⭐⭐⭐⭐ | ⭐         | Best results      |

## Monitoring and Observability

- **Structured Logging**: Comprehensive logging with emojis and context
- **Health Checks**: System component monitoring
- **Performance Metrics**: Response times and token usage
- **Error Handling**: Graceful failure handling
- **LangSmith Integration**: Production monitoring

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black src/
flake8 src/
mypy src/
```

### Adding New Retrievers

1. Create retriever class inheriting from `BaseRAGRetriever`
2. Implement `retrieve_documents()` and `retrieve_with_scores()`
3. Add to retrieval module exports
4. Update production chains if needed

## License

This project is part of the AI Makerspace curriculum and follows academic integrity guidelines.

## Contributing

This is an educational project. Please ensure all code is original and follows the established patterns.
