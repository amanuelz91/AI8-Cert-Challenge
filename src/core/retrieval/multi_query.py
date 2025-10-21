"""
Multi-query retriever implementation for evaluation.

LLM-powered query expansion for comprehensive document coverage.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError, APIKeyError
from src.utils.decorators import timing_decorator, retry_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class MultiQueryRetrieverWrapper(BaseRAGRetriever):
    """Multi-query retriever wrapper for query expansion."""
    
    def __init__(
        self,
        base_retriever,
        llm: Optional[ChatOpenAI] = None,
        k: int = 5
    ):
        """
        Initialize multi-query retriever.
        
        Args:
            base_retriever: Base retriever to use
            llm: Language model for query generation
            k: Number of documents to retrieve
            
        Raises:
            APIKeyError: If OpenAI API key is missing
        """
        super().__init__("multi_query", k)
        self.base_retriever = base_retriever
        
        # Validate API key
        config = get_config()
        if not config.openai_api_key:
            raise APIKeyError("OpenAI API key is required for multi-query retriever")
        
        # Initialize LLM if not provided
        if llm is None:
            self.llm = ChatOpenAI(
                model=config.llm.model_name,
                temperature=0.0,
                openai_api_key=config.openai_api_key
            )
        else:
            self.llm = llm
        
        # Initialize multi-query retriever
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm
        )
        
        logger.info(f"🔄 Initialized multi-query retriever (k={k})")
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using multi-query expansion.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔄 [Multi-Query] Retrieving documents for: {query[:50]}...")
            
            # Use multi-query retriever
            documents = self.multi_query_retriever.invoke(query)
            
            # Limit to k documents
            if len(documents) > self.k:
                documents = documents[:self.k]
            
            # Add multi-query metadata
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'multi_query'
                doc.metadata['llm_model'] = self.llm.model_name
            
            logger.info(f"📚 [Multi-Query] Retrieved {len(documents)} documents from expanded queries")
            return documents
            
        except Exception as e:
            error_msg = f"Multi-query retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with expansion scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔄 [Multi-Query] Retrieving documents with scores for: {query[:50]}...")
            
            # Get documents from multi-query retriever
            documents = self.multi_query_retriever.invoke(query)
            
            # Limit to k documents
            if len(documents) > self.k:
                documents = documents[:self.k]
            
            # Add multi-query metadata and scores
            results = []
            for i, doc in enumerate(documents):
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'multi_query'
                doc.metadata['llm_model'] = self.llm.model_name
                doc.metadata['query_expansion_position'] = i + 1
                
                # Use position-based score (earlier documents = higher score)
                score = 1.0 - (i / len(documents)) if documents else 1.0
                results.append((doc, score))
            
            logger.info(f"📚 [Multi-Query] Retrieved {len(results)} documents with expansion scores")
            return results
            
        except Exception as e:
            error_msg = f"Multi-query retrieval with scores failed: {str(e)}"
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
                    "avg_expansion_score": sum(scores) / len(scores),
                    "max_expansion_score": max(scores),
                    "min_expansion_score": min(scores),
                    "llm_model": self.llm.model_name,
                    "retrieval_method": "multi_query",
                    "query_expansion": True
                }
            )
            
        except Exception as e:
            error_msg = f"Multi-query retrieval with result failed: {str(e)}"
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
            "type": "multi_query_expansion",
            "llm_model": self.llm.model_name,
            "base_retriever": self.base_retriever.__class__.__name__
        }


def create_multi_query_retriever(
    base_retriever,
    llm: Optional[ChatOpenAI] = None,
    k: int = 5
) -> MultiQueryRetrieverWrapper:
    """
    Create a multi-query retriever instance.
    
    Args:
        base_retriever: Base retriever to use
        llm: Language model for query generation
        k: Number of documents to retrieve
        
    Returns:
        Multi-query retriever instance
    """
    return MultiQueryRetrieverWrapper(base_retriever, llm, k)
