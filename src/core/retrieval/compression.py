"""
Contextual compression retriever implementation for evaluation.

AI-powered reranking using Cohere for improved semantic relevance.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError, APIKeyError
from src.utils.decorators import timing_decorator, retry_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class CompressionRetriever(BaseRAGRetriever):
    """Contextual compression retriever with AI reranking."""
    
    def __init__(
        self,
        base_retriever,
        k: int = 5,
        base_k: int = 20,
        model: str = "rerank-v3.5"
    ):
        """
        Initialize compression retriever.
        
        Args:
            base_retriever: Base retriever to use
            k: Number of final documents to retrieve
            base_k: Number of candidates to retrieve before reranking
            model: Cohere reranking model
            
        Raises:
            APIKeyError: If Cohere API key is missing
        """
        super().__init__("compression", k)
        self.base_retriever = base_retriever
        self.base_k = base_k
        self.model = model
        
        # Validate API key
        config = get_config()
        if not config.cohere_api_key:
            raise APIKeyError("Cohere API key is required for compression retriever")
        
        # Initialize compressor
        self.compressor = CohereRerank(model=model)
        
        # Initialize compression retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever,
            search_kwargs={"k": k}
        )
        
        logger.info(f"🎯 Initialized compression retriever (k={k}, base_k={base_k}, model={model})")
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using contextual compression.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🎯 [Compression] Retrieving documents for: {query[:50]}...")
            
            # Use compression retriever
            documents = self.compression_retriever.invoke(query)
            
            # Add compression metadata
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'contextual_compression'
                doc.metadata['rerank_model'] = self.model
                doc.metadata['base_k'] = self.base_k
            
            logger.info(f"📚 [Compression] Retrieved {len(documents)} documents after reranking")
            return documents
            
        except Exception as e:
            error_msg = f"Compression retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with reranking scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🎯 [Compression] Retrieving documents with scores for: {query[:50]}...")
            
            # Get base documents first
            base_documents = self.base_retriever.invoke(query)
            
            if not base_documents:
                return []
            
            # Limit base documents
            if len(base_documents) > self.base_k:
                base_documents = base_documents[:self.base_k]
            
            # Use compressor to rerank
            reranked_docs = self.compressor.compress_documents(base_documents, query)
            
            # Limit to k documents
            if len(reranked_docs) > self.k:
                reranked_docs = reranked_docs[:self.k]
            
            # Add compression metadata and scores
            results = []
            for i, doc in enumerate(reranked_docs):
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'contextual_compression'
                doc.metadata['rerank_model'] = self.model
                doc.metadata['base_k'] = self.base_k
                doc.metadata['rerank_position'] = i + 1
                
                # Use position-based score (higher position = higher score)
                score = 1.0 - (i / len(reranked_docs))
                results.append((doc, score))
            
            logger.info(f"📚 [Compression] Retrieved {len(results)} documents with reranking scores")
            return results
            
        except Exception as e:
            error_msg = f"Compression retrieval with scores failed: {str(e)}"
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
                    metadata={"warning": "No documents found"}
                )
            
            documents, scores = zip(*docs_with_scores)
            
            return RetrievalResult(
                documents=list(documents),
                scores=list(scores),
                retriever_name=self.name,
                query=query,
                metadata={
                    "avg_rerank_score": sum(scores) / len(scores),
                    "max_rerank_score": max(scores),
                    "min_rerank_score": min(scores),
                    "rerank_model": self.model,
                    "base_k": self.base_k,
                    "retrieval_method": "contextual_compression"
                }
            )
            
        except Exception as e:
            error_msg = f"Compression retrieval with result failed: {str(e)}"
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
            "base_k": self.base_k,
            "type": "contextual_compression",
            "rerank_model": self.model,
            "base_retriever": self.base_retriever.__class__.__name__
        }


def create_compression_retriever(
    base_retriever,
    k: int = 5,
    base_k: int = 20,
    model: str = "rerank-v3.5"
) -> CompressionRetriever:
    """
    Create a compression retriever instance.
    
    Args:
        base_retriever: Base retriever to use
        k: Number of final documents to retrieve
        base_k: Number of candidates to retrieve before reranking
        model: Cohere reranking model
        
    Returns:
        Compression retriever instance
    """
    return CompressionRetriever(base_retriever, k, base_k, model)
