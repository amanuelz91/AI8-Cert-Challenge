"""
Retrieval module initialization.

Exports all retrieval classes and functions for the RAG system.
"""

# Base classes
from .base import (
    BaseRAGRetriever,
    LangChainRetrieverWrapper,
    RetrievalResult,
    RetrievalPipeline
)

# Production retrievers
from .naive import (
    NaiveRetriever,
    create_naive_retriever
)

from .semantic import (
    SemanticRetriever,
    create_semantic_retriever
)

from .tools import (
    ToolBasedRetriever,
    create_tool_based_retriever
)

# Evaluation retrievers
from .bm25 import (
    BM25RetrieverWrapper,
    create_bm25_retriever
)

from .compression import (
    CompressionRetriever,
    create_compression_retriever
)

from .multi_query import (
    MultiQueryRetrieverWrapper,
    create_multi_query_retriever
)

from .parent_document import (
    ParentDocumentRetrieverWrapper,
    create_parent_document_retriever
)

from .ensemble import (
    EnsembleRetrieverWrapper,
    create_ensemble_retriever
)

__all__ = [
    # Base classes
    "BaseRAGRetriever",
    "LangChainRetrieverWrapper",
    "RetrievalResult",
    "RetrievalPipeline",
    
    # Production retrievers
    "NaiveRetriever",
    "create_naive_retriever",
    "SemanticRetriever", 
    "create_semantic_retriever",
    "ToolBasedRetriever",
    "create_tool_based_retriever",
    
    # Evaluation retrievers
    "BM25RetrieverWrapper",
    "create_bm25_retriever",
    "CompressionRetriever",
    "create_compression_retriever",
    "MultiQueryRetrieverWrapper",
    "create_multi_query_retriever",
    "ParentDocumentRetrieverWrapper",
    "create_parent_document_retriever",
    "EnsembleRetrieverWrapper",
    "create_ensemble_retriever"
]
