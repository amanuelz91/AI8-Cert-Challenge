"""
BM25 retriever implementation for evaluation.

Keyword-based retrieval using BM25 algorithm for comparison purposes.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.utils.decorators import timing_decorator

logger = get_logger(__name__)


class BM25RetrieverWrapper(BaseRAGRetriever):
    """BM25 retriever wrapper for keyword-based search."""
    
    def __init__(
        self,
        documents: List[Document],
        k: int = 5,
        k1: float = 1.2,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of documents to index
            k: Number of documents to retrieve
            k1: BM25 parameter k1
            b: BM25 parameter b
        """
        super().__init__("bm25", k)
        self.k1 = k1
        self.b = b
        
        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k1 = k1
        self.bm25_retriever.b = b
        
        logger.info(f"🔤 Initialized BM25 retriever (k={k}, k1={k1}, b={b})")
    
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using BM25 keyword matching.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔤 [BM25] Retrieving documents for: {query[:50]}...")
            
            # Use BM25 retriever
            documents = self.bm25_retriever.invoke(query)
            
            # Limit to k documents
            if len(documents) > self.k:
                documents = documents[:self.k]
            
            # Add BM25 metadata
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'bm25'
                doc.metadata['bm25_k1'] = self.k1
                doc.metadata['bm25_b'] = self.b
            
            logger.info(f"📚 [BM25] Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            error_msg = f"BM25 retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with BM25 scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔤 [BM25] Retrieving documents with scores for: {query[:50]}...")
            
            # Use BM25 retriever with scores
            docs_with_scores = self.bm25_retriever.similarity_search_with_score(query, k=self.k)
            
            # Add BM25 metadata
            results = []
            for doc, score in docs_with_scores:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'bm25'
                doc.metadata['bm25_k1'] = self.k1
                doc.metadata['bm25_b'] = self.b
                doc.metadata['bm25_score'] = float(score)
                results.append((doc, float(score)))
            
            logger.info(f"📚 [BM25] Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            error_msg = f"BM25 retrieval with scores failed: {str(e)}"
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
                    "avg_bm25_score": sum(scores) / len(scores),
                    "max_bm25_score": max(scores),
                    "min_bm25_score": min(scores),
                    "bm25_k1": self.k1,
                    "bm25_b": self.b,
                    "retrieval_method": "bm25"
                }
            )
            
        except Exception as e:
            error_msg = f"BM25 retrieval with result failed: {str(e)}"
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
            "type": "bm25_keyword_search",
            "bm25_k1": self.k1,
            "bm25_b": self.b,
            "indexed_documents": len(self.bm25_retriever.docs)
        }


def create_bm25_retriever(
    documents: List[Document],
    k: int = 5,
    k1: float = 1.2,
    b: float = 0.75
) -> BM25RetrieverWrapper:
    """
    Create a BM25 retriever instance.
    
    Args:
        documents: List of documents to index
        k: Number of documents to retrieve
        k1: BM25 parameter k1
        b: BM25 parameter b
        
    Returns:
        BM25 retriever instance
    """
    return BM25RetrieverWrapper(documents, k, k1, b)
