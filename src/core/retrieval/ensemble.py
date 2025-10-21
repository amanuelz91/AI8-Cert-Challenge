"""
Ensemble retriever implementation for evaluation.

Weighted combination of multiple retrieval methods for comprehensive coverage.
"""

from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.utils.decorators import timing_decorator

logger = get_logger(__name__)


class EnsembleRetrieverWrapper(BaseRAGRetriever):
    """Ensemble retriever wrapper for combining multiple retrieval methods."""
    
    def __init__(
        self,
        retrievers: List[BaseRAGRetriever],
        weights: Optional[List[float]] = None,
        k: int = 5
    ):
        """
        Initialize ensemble retriever.
        
        Args:
            retrievers: List of retrievers to combine
            weights: Optional weights for each retriever
            k: Number of documents to retrieve
        """
        super().__init__("ensemble", k)
        self.retrievers = retrievers
        
        # Set equal weights if not provided
        if weights is None:
            self.weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        # Create LangChain ensemble retriever
        self._create_ensemble_retriever()
        
        logger.info(f"🎭 Initialized ensemble retriever with {len(retrievers)} retrievers (k={k})")
    
    def _create_ensemble_retriever(self) -> None:
        """Create the LangChain ensemble retriever."""
        try:
            # Extract LangChain retrievers
            langchain_retrievers = []
            for retriever in self.retrievers:
                if hasattr(retriever, 'retriever'):
                    # Wrapper retrievers
                    langchain_retrievers.append(retriever.retriever)
                elif hasattr(retriever, 'base_retriever'):
                    # Compression retrievers
                    langchain_retrievers.append(retriever.base_retriever)
                else:
                    # Direct retrievers
                    langchain_retrievers.append(retriever)
            
            # Create ensemble retriever
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=langchain_retrievers,
                weights=self.weights
            )
            
            logger.info(f"✅ Created ensemble retriever with {len(langchain_retrievers)} retrievers")
            
        except Exception as e:
            error_msg = f"Failed to create ensemble retriever: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using ensemble of retrievers.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🎭 [Ensemble] Retrieving documents for: {query[:50]}...")
            
            # Use ensemble retriever
            documents = self.ensemble_retriever.invoke(query)
            
            # Limit to k documents
            if len(documents) > self.k:
                documents = documents[:self.k]
            
            # Add ensemble metadata
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'ensemble'
                doc.metadata['ensemble_size'] = len(self.retrievers)
                doc.metadata['ensemble_weights'] = self.weights
            
            logger.info(f"📚 [Ensemble] Retrieved {len(documents)} documents from ensemble")
            return documents
            
        except Exception as e:
            error_msg = f"Ensemble retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with ensemble scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🎭 [Ensemble] Retrieving documents with scores for: {query[:50]}...")
            
            # Get results from individual retrievers
            all_results = {}
            for i, retriever in enumerate(self.retrievers):
                try:
                    docs_with_scores = retriever.retrieve_with_scores(query)
                    all_results[retriever.name] = docs_with_scores
                except Exception as e:
                    logger.warning(f"⚠️ Retriever {retriever.name} failed: {str(e)}")
                    all_results[retriever.name] = []
            
            # Combine results with weighted scores
            combined_results = self._combine_results(all_results)
            
            # Limit to k documents
            if len(combined_results) > self.k:
                combined_results = combined_results[:self.k]
            
            logger.info(f"📚 [Ensemble] Retrieved {len(combined_results)} documents with ensemble scores")
            return combined_results
            
        except Exception as e:
            error_msg = f"Ensemble retrieval with scores failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def _combine_results(self, all_results: Dict[str, List[Tuple[Document, float]]]) -> List[Tuple[Document, float]]:
        """
        Combine results from multiple retrievers with weighted scores.
        
        Args:
            all_results: Dictionary mapping retriever names to results
            
        Returns:
            Combined list of (document, score) tuples
        """
        # Document to weighted score mapping
        doc_scores = {}
        doc_metadata = {}
        
        for i, (retriever_name, results) in enumerate(all_results.items()):
            weight = self.weights[i]
            
            for doc, score in results:
                # Use document content as key for deduplication
                doc_key = doc.page_content[:200]  # First 200 chars as key
                
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = 0.0
                    doc_metadata[doc_key] = doc
                
                # Add weighted score
                doc_scores[doc_key] += score * weight
        
        # Convert back to list and sort by score
        combined_results = []
        for doc_key, weighted_score in doc_scores.items():
            doc = doc_metadata[doc_key]
            
            # Add ensemble metadata
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['retrieval_method'] = 'ensemble'
            doc.metadata['ensemble_size'] = len(self.retrievers)
            doc.metadata['ensemble_weights'] = self.weights
            doc.metadata['weighted_score'] = weighted_score
            
            combined_results.append((doc, weighted_score))
        
        # Sort by weighted score (descending)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
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
                    "avg_ensemble_score": sum(scores) / len(scores),
                    "max_ensemble_score": max(scores),
                    "min_ensemble_score": min(scores),
                    "ensemble_size": len(self.retrievers),
                    "ensemble_weights": self.weights,
                    "retriever_names": [r.name for r in self.retrievers],
                    "retrieval_method": "ensemble"
                }
            )
            
        except Exception as e:
            error_msg = f"Ensemble retrieval with result failed: {str(e)}"
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
            "type": "ensemble_weighted",
            "ensemble_size": len(self.retrievers),
            "retriever_names": [r.name for r in self.retrievers],
            "retriever_types": [r.__class__.__name__ for r in self.retrievers],
            "weights": self.weights
        }


def create_ensemble_retriever(
    retrievers: List[BaseRAGRetriever],
    weights: Optional[List[float]] = None,
    k: int = 5
) -> EnsembleRetrieverWrapper:
    """
    Create an ensemble retriever instance.
    
    Args:
        retrievers: List of retrievers to combine
        weights: Optional weights for each retriever
        k: Number of documents to retrieve
        
    Returns:
        Ensemble retriever instance
    """
    return EnsembleRetrieverWrapper(retrievers, weights, k)
