"""
Embedding caching utilities for performance optimization.

Provides persistent caching for embeddings to avoid redundant API calls.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.utils.logging import get_logger
from src.utils.exceptions import EmbeddingError
from src.config.settings import get_config

logger = get_logger(__name__)


class EmbeddingCache:
    """Persistent cache for embeddings."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.config = get_config()
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache", "embeddings")
        self._ensure_cache_dir()
        
        logger.info(f"💾 Initialized embedding cache: {self.cache_dir}")
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path for a key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def _generate_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        return f"{model_name}:{hashlib.sha256(text.encode()).hexdigest()}"
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Text that was embedded
            model_name: Model used for embedding
            
        Returns:
            Cached embedding or None if not found
        """
        try:
            cache_key = self._generate_cache_key(text, model_name)
            cache_path = self._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"🎯 Cache hit for text: {text[:50]}...")
                    return data['embedding']
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Cache read error: {str(e)}")
            return None
    
    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Text that was embedded
            model_name: Model used for embedding
            embedding: Embedding vector to cache
        """
        try:
            cache_key = self._generate_cache_key(text, model_name)
            cache_path = self._get_cache_path(cache_key)
            
            data = {
                'text': text,
                'model_name': model_name,
                'embedding': embedding,
                'cached_at': str(os.path.getctime(cache_path)) if os.path.exists(cache_path) else None
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"💾 Cached embedding for text: {text[:50]}...")
            
        except Exception as e:
            logger.warning(f"⚠️ Cache write error: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, file))
                logger.info("🗑️ Embedding cache cleared")
        except Exception as e:
            logger.warning(f"⚠️ Cache clear error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if not os.path.exists(self.cache_dir):
                return {
                    "cache_dir": self.cache_dir,
                    "cached_embeddings": 0,
                    "cache_size_mb": 0
                }
            
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in files
            )
            
            return {
                "cache_dir": self.cache_dir,
                "cached_embeddings": len(files),
                "cache_size_mb": total_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Cache stats error: {str(e)}")
            return {
                "cache_dir": self.cache_dir,
                "cached_embeddings": 0,
                "cache_size_mb": 0,
                "error": str(e)
            }


class BatchEmbeddingCache:
    """Cache for batch embedding operations."""
    
    def __init__(self, cache: Optional[EmbeddingCache] = None):
        """
        Initialize batch embedding cache.
        
        Args:
            cache: Embedding cache instance
        """
        self.cache = cache or EmbeddingCache()
        self._batch_cache: Dict[str, List[List[float]]] = {}
    
    def get_batch(self, texts: List[str], model_name: str) -> List[Optional[List[float]]]:
        """
        Get batch of embeddings from cache.
        
        Args:
            texts: List of texts to get embeddings for
            model_name: Model name
            
        Returns:
            List of cached embeddings (None if not cached)
        """
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text, model_name)
            results.append(cached_embedding)
            
            if cached_embedding is None:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        logger.info(f"📊 Batch cache: {len(texts) - len(uncached_texts)} cached, {len(uncached_texts)} uncached")
        return results, uncached_texts, uncached_indices
    
    def set_batch(self, texts: List[str], model_name: str, embeddings: List[List[float]]) -> None:
        """
        Cache batch of embeddings.
        
        Args:
            texts: List of texts
            model_name: Model name
            embeddings: List of embeddings
        """
        for text, embedding in zip(texts, embeddings):
            self.cache.set(text, model_name, embedding)
        
        logger.info(f"💾 Cached {len(embeddings)} embeddings in batch")
    
    def clear(self) -> None:
        """Clear batch cache."""
        self.cache.clear()
        self._batch_cache.clear()
        logger.info("🗑️ Batch embedding cache cleared")
