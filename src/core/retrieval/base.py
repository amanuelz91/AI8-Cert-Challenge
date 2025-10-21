"""
Base retriever classes and interfaces for the RAG system.

Provides abstract base classes and common functionality for all retrieval methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.utils.decorators import timing_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class BaseRAGRetriever(ABC):
    """Abstract base class for RAG retrievers."""
    
    def __init__(self, name: str, k: int = 5):
        """
        Initialize base retriever.
        
        Args:
            name: Name of the retriever
            k: Number of documents to retrieve
        """
        self.name = name
        self.k = k
        self.config = get_config()
        logger.info(f"🔍 Initialized {self.name} retriever (k={k})")
    
    @abstractmethod
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        pass
    
    @abstractmethod
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
        pass
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get retriever information.
        
        Returns:
            Dictionary with retriever metadata
        """
        return {
            "name": self.name,
            "k": self.k,
            "type": self.__class__.__name__
        }


class LangChainRetrieverWrapper(BaseRAGRetriever):
    """Wrapper for LangChain retrievers."""
    
    def __init__(self, retriever: BaseRetriever, name: str, k: int = 5):
        """
        Initialize LangChain retriever wrapper.
        
        Args:
            retriever: LangChain retriever instance
            name: Name of the retriever
            k: Number of documents to retrieve
        """
        super().__init__(name, k)
        self.retriever = retriever
    
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using LangChain retriever.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔍 [{self.name}] Retrieving documents for: {query[:50]}...")
            
            documents = self.retriever.invoke(query)
            
            # Limit to k documents
            if len(documents) > self.k:
                documents = documents[:self.k]
            
            logger.info(f"📚 [{self.name}] Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            error_msg = f"Failed to retrieve documents with {self.name}: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with scores using LangChain retriever.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔍 [{self.name}] Retrieving documents with scores for: {query[:50]}...")
            
            # Try to get documents with scores
            if hasattr(self.retriever, 'similarity_search_with_score'):
                results = self.retriever.similarity_search_with_score(query, k=self.k)
                documents_with_scores = [(doc, float(score)) for doc, score in results]
            else:
                # Fallback to regular retrieval
                documents = self.retrieve_documents(query)
                documents_with_scores = [(doc, 1.0) for doc in documents]
            
            logger.info(f"📚 [{self.name}] Retrieved {len(documents_with_scores)} documents with scores")
            return documents_with_scores
            
        except Exception as e:
            error_msg = f"Failed to retrieve documents with scores using {self.name}: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e


class RetrievalResult:
    """Container for retrieval results with metadata."""
    
    def __init__(
        self,
        documents: List[Document],
        scores: Optional[List[float]] = None,
        retriever_name: str = "unknown",
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize retrieval result.
        
        Args:
            documents: List of retrieved documents
            scores: Optional similarity scores
            retriever_name: Name of the retriever used
            query: Original query
            metadata: Additional metadata
        """
        self.documents = documents
        self.scores = scores or [1.0] * len(documents)
        self.retriever_name = retriever_name
        self.query = query
        self.metadata = metadata or {}
        
        # Ensure scores and documents have same length
        if len(self.scores) != len(self.documents):
            self.scores = [1.0] * len(self.documents)
    
    def get_documents_with_scores(self) -> List[Tuple[Document, float]]:
        """
        Get documents with their scores.
        
        Returns:
            List of (document, score) tuples
        """
        return list(zip(self.documents, self.scores))
    
    def get_top_k(self, k: int) -> 'RetrievalResult':
        """
        Get top k results.
        
        Args:
            k: Number of top results to return
            
        Returns:
            New RetrievalResult with top k documents
        """
        if k >= len(self.documents):
            return self
        
        # Sort by scores (descending)
        sorted_results = sorted(
            zip(self.documents, self.scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_documents, top_scores = zip(*sorted_results[:k])
        
        return RetrievalResult(
            documents=list(top_documents),
            scores=list(top_scores),
            retriever_name=self.retriever_name,
            query=self.query,
            metadata=self.metadata
        )
    
    def get_context_text(self, separator: str = "\n\n") -> str:
        """
        Get concatenated context text from documents.
        
        Args:
            separator: Separator between documents
            
        Returns:
            Concatenated context text
        """
        return separator.join(doc.page_content for doc in self.documents)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of retrieval results.
        
        Returns:
            Dictionary with result summary
        """
        return {
            "retriever_name": self.retriever_name,
            "query": self.query,
            "num_documents": len(self.documents),
            "avg_score": sum(self.scores) / len(self.scores) if self.scores else 0,
            "max_score": max(self.scores) if self.scores else 0,
            "min_score": min(self.scores) if self.scores else 0,
            "metadata": self.metadata
        }


class RetrievalPipeline:
    """Pipeline for orchestrating multiple retrievers."""
    
    def __init__(self, retrievers: List[BaseRAGRetriever]):
        """
        Initialize retrieval pipeline.
        
        Args:
            retrievers: List of retrievers to use
        """
        self.retrievers = retrievers
        logger.info(f"🔗 Initialized retrieval pipeline with {len(retrievers)} retrievers")
    
    def retrieve_with_all(self, query: str) -> Dict[str, RetrievalResult]:
        """
        Retrieve documents using all retrievers.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary mapping retriever names to results
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔍 Running retrieval pipeline for: {query[:50]}...")
            
            results = {}
            for retriever in self.retrievers:
                try:
                    documents_with_scores = retriever.retrieve_with_scores(query)
                    documents, scores = zip(*documents_with_scores) if documents_with_scores else ([], [])
                    
                    result = RetrievalResult(
                        documents=list(documents),
                        scores=list(scores),
                        retriever_name=retriever.name,
                        query=query
                    )
                    
                    results[retriever.name] = result
                    
                except Exception as e:
                    logger.warning(f"⚠️ Retriever {retriever.name} failed: {str(e)}")
                    results[retriever.name] = RetrievalResult(
                        documents=[],
                        scores=[],
                        retriever_name=retriever.name,
                        query=query,
                        metadata={"error": str(e)}
                    )
            
            logger.info(f"✅ Pipeline complete: {len(results)} retrievers processed")
            return results
            
        except Exception as e:
            error_msg = f"Retrieval pipeline failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get pipeline information.
        
        Returns:
            Dictionary with pipeline metadata
        """
        return {
            "num_retrievers": len(self.retrievers),
            "retriever_names": [r.name for r in self.retrievers],
            "retriever_types": [r.__class__.__name__ for r in self.retrievers]
        }
