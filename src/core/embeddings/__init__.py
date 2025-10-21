"""
Embedding module initialization.

Exports main classes and functions for embedding management.
"""

from .providers import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    EmbeddingManager,
    create_embedding_provider,
    get_default_embedding_provider
)

from .cache import (
    EmbeddingCache,
    BatchEmbeddingCache
)

__all__ = [
    # Providers
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider", 
    "EmbeddingManager",
    "create_embedding_provider",
    "get_default_embedding_provider",
    
    # Cache
    "EmbeddingCache",
    "BatchEmbeddingCache"
]
