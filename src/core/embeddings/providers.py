"""
Embedding providers and management for the RAG system.

Handles embedding model initialization, caching, and batch processing.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.utils.logging import get_logger
from src.utils.exceptions import EmbeddingError, APIKeyError
from src.utils.decorators import timing_decorator, retry_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with caching and error handling."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model_name: OpenAI embedding model name
            batch_size: Batch size for embedding requests
            cache_enabled: Whether to enable caching
            
        Raises:
            APIKeyError: If OpenAI API key is missing
        """
        self.config = get_config()
        
        # Validate API key
        if not self.config.openai_api_key:
            raise APIKeyError("OpenAI API key is required")
        
        self.model_name = model_name or self.config.embedding.model_name
        self.batch_size = batch_size or self.config.embedding.batch_size
        self.cache_enabled = cache_enabled
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.config.openai_api_key
        )
        
        # Simple in-memory cache
        self._cache: Dict[str, List[float]] = {}
        
        logger.info(f"🤖 Initialized OpenAI embedding provider: {self.model_name}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"{self.model_name}:{hash(text)}"
    
    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches."""
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts with caching and batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            logger.info(f"🔤 Embedding {len(texts)} documents")
            
            # Check cache first
            cached_results = []
            texts_to_embed = []
            cache_indices = []
            
            for i, text in enumerate(texts):
                if self.cache_enabled:
                    cache_key = self._get_cache_key(text)
                    if cache_key in self._cache:
                        cached_results.append((i, self._cache[cache_key]))
                        continue
                
                texts_to_embed.append(text)
                cache_indices.append(i)
            
            # Embed uncached texts
            if texts_to_embed:
                logger.info(f"📡 Embedding {len(texts_to_embed)} uncached texts")
                
                # Process in batches
                all_embeddings = []
                batches = self._batch_texts(texts_to_embed)
                
                for batch in batches:
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Cache results
                    if self.cache_enabled:
                        for text, embedding in zip(batch, batch_embeddings):
                            cache_key = self._get_cache_key(text)
                            self._cache[cache_key] = embedding
                
                logger.info(f"✅ Generated {len(all_embeddings)} new embeddings")
            else:
                logger.info("🎯 All texts found in cache")
                all_embeddings = []
            
            # Combine cached and new results
            result_embeddings = [None] * len(texts)
            
            # Add cached results
            for i, embedding in cached_results:
                result_embeddings[i] = embedding
            
            # Add new results
            for i, embedding in zip(cache_indices, all_embeddings):
                result_embeddings[i] = embedding
            
            logger.info(f"📊 Embedding complete: {len(cached_results)} cached, {len(all_embeddings)} new")
            return result_embeddings
            
        except Exception as e:
            error_msg = f"Failed to embed documents: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            logger.info(f"🔍 Embedding query: {text[:50]}...")
            
            # Check cache first
            if self.cache_enabled:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    logger.info("🎯 Query embedding found in cache")
                    return self._cache[cache_key]
            
            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            
            # Cache result
            if self.cache_enabled:
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = embedding
            
            logger.info("✅ Query embedding generated")
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to embed query: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_enabled": self.cache_enabled,
            "cached_embeddings": len(self._cache),
            "model_name": self.model_name,
            "batch_size": self.batch_size
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("🗑️ Embedding cache cleared")


class EmbeddingManager:
    """Manager for embedding operations and document processing."""
    
    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        """
        Initialize embedding manager.
        
        Args:
            provider: Embedding provider to use
        """
        self.provider = provider or OpenAIEmbeddingProvider()
        logger.info("📚 Initialized embedding manager")
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of documents with embeddings
            
        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            logger.info(f"📄 Embedding {len(documents)} documents")
            
            # Extract texts
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.provider.embed_documents(texts)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['embedding'] = embedding
            
            logger.info(f"✅ Successfully embedded {len(documents)} documents")
            return documents
            
        except Exception as e:
            error_msg = f"Failed to embed documents: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding vector
            
        Raises:
            EmbeddingError: If embedding fails
        """
        return self.provider.embed_query(query)
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get embedding provider statistics."""
        return self.provider.get_cache_stats()


def create_embedding_provider(
    provider_type: str = "openai",
    **kwargs
) -> EmbeddingProvider:
    """
    Create an embedding provider instance.
    
    Args:
        provider_type: Type of provider to create
        **kwargs: Provider-specific arguments
        
    Returns:
        Embedding provider instance
        
    Raises:
        ValueError: If provider type is not supported
    """
    if provider_type.lower() == "openai":
        return OpenAIEmbeddingProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider type: {provider_type}")


def get_default_embedding_provider() -> EmbeddingProvider:
    """Get the default embedding provider."""
    return OpenAIEmbeddingProvider()
