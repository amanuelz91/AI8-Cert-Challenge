"""
Naive retriever implementation using cosine similarity.

Production-grade naive retriever with Qdrant vector store.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.utils.decorators import timing_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class NaiveRetriever(BaseRAGRetriever):
    """Naive retriever using cosine similarity with Qdrant."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        k: int = 5,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialize naive retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
        """
        super().__init__("naive", k)
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold or get_config().retrieval.similarity_threshold
        
        logger.info(f"🔍 Initialized naive retriever (k={k}, threshold={self.similarity_threshold})")
    
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using cosine similarity.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔍 [Naive] Retrieving documents for: {query[:50]}...")
            
            # Use similarity search with score
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=self.k
            )
            
            # Filter by similarity threshold
            filtered_docs = []
            for doc, score in docs_with_scores:
                # Convert distance to similarity (Qdrant returns distance)
                similarity = 1 - score
                if similarity >= self.similarity_threshold:
                    # Add similarity score to metadata
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['similarity_score'] = similarity
                    doc.metadata['relevance_score'] = similarity
                    filtered_docs.append(doc)
            
            logger.info(f"📚 [Naive] Retrieved {len(filtered_docs)} documents "
                       f"(threshold={self.similarity_threshold})")
            return filtered_docs
            
        except Exception as e:
            error_msg = f"Naive retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔍 [Naive] Retrieving documents with scores for: {query[:50]}...")
            
            # Use similarity search with score
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=self.k
            )
            
            # Convert distance to similarity and filter
            results = []
            for doc, distance in docs_with_scores:
                similarity = 1 - distance
                if similarity >= self.similarity_threshold:
                    # Add similarity score to metadata
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['similarity_score'] = similarity
                    doc.metadata['relevance_score'] = similarity
                    results.append((doc, similarity))
            
            logger.info(f"📚 [Naive] Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            error_msg = f"Naive retrieval with scores failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def retrieve_with_result(self, query: str) -> RetrievalResult:
        """
        Retrieve documents and return as RetrievalResult.
        
        Args:
            query: Query string
            
        Returns:
            RetrievalResult with documents and metadata
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            docs_with_scores = self.retrieve_with_scores(query)
            
            if not docs_with_scores:
                return RetrievalResult(
                    documents=[],
                    scores=[],
                    retriever_name=self.name,
                    query=query,
                    metadata={"warning": "No documents found above similarity threshold"}
                )
            
            documents, scores = zip(*docs_with_scores)
            
            return RetrievalResult(
                documents=list(documents),
                scores=list(scores),
                retriever_name=self.name,
                query=query,
                metadata={
                    "similarity_threshold": self.similarity_threshold,
                    "avg_similarity": sum(scores) / len(scores),
                    "max_similarity": max(scores),
                    "min_similarity": min(scores)
                }
            )
            
        except Exception as e:
            error_msg = f"Naive retrieval with result failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def get_retriever_stats(self) -> dict:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with retriever statistics
        """
        return {
            "name": self.name,
            "k": self.k,
            "similarity_threshold": self.similarity_threshold,
            "type": "cosine_similarity",
            "vector_store_type": "QdrantVectorStore"
        }


def create_naive_retriever(
    vector_store: QdrantVectorStore,
    k: int = 5,
    similarity_threshold: Optional[float] = None
) -> NaiveRetriever:
    """
    Create a naive retriever instance.
    
    Args:
        vector_store: Qdrant vector store instance
        k: Number of documents to retrieve
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Naive retriever instance
    """
    return NaiveRetriever(vector_store, k, similarity_threshold)
