"""
Semantic chunking retriever implementation.

Production-grade retriever using semantic chunking for better document boundaries.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.utils.decorators import timing_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class SemanticRetriever(BaseRAGRetriever):
    """Semantic retriever using semantic chunking for better boundaries."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        breakpoint_threshold_type: str = "percentile"
    ):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            embeddings: Embedding model for semantic analysis
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
            breakpoint_threshold_type: Type of threshold for semantic breakpoints
        """
        super().__init__("semantic", k)
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold or get_config().retrieval.similarity_threshold
        self.breakpoint_threshold_type = breakpoint_threshold_type
        
        # Initialize semantic chunker
        self.semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type
        )
        
        logger.info(f"🧠 Initialized semantic retriever (k={k}, threshold={self.similarity_threshold})")
    
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using semantic similarity.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🧠 [Semantic] Retrieving documents for: {query[:50]}...")
            
            # Use similarity search with score
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=self.k
            )
            
            # Filter by similarity threshold and enhance with semantic info
            filtered_docs = []
            for doc, score in docs_with_scores:
                # Convert distance to similarity (Qdrant returns distance)
                similarity = 1 - score
                if similarity >= self.similarity_threshold:
                    # Add semantic metadata
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['similarity_score'] = similarity
                    doc.metadata['relevance_score'] = similarity
                    doc.metadata['chunking_method'] = 'semantic'
                    doc.metadata['breakpoint_threshold'] = self.breakpoint_threshold_type
                    filtered_docs.append(doc)
            
            logger.info(f"📚 [Semantic] Retrieved {len(filtered_docs)} documents "
                       f"(threshold={self.similarity_threshold})")
            return filtered_docs
            
        except Exception as e:
            error_msg = f"Semantic retrieval failed: {str(e)}"
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
            logger.info(f"🧠 [Semantic] Retrieving documents with scores for: {query[:50]}...")
            
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
                    # Add semantic metadata
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['similarity_score'] = similarity
                    doc.metadata['relevance_score'] = similarity
                    doc.metadata['chunking_method'] = 'semantic'
                    doc.metadata['breakpoint_threshold'] = self.breakpoint_threshold_type
                    results.append((doc, similarity))
            
            logger.info(f"📚 [Semantic] Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            error_msg = f"Semantic retrieval with scores failed: {str(e)}"
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
                    "min_similarity": min(scores),
                    "chunking_method": "semantic",
                    "breakpoint_threshold_type": self.breakpoint_threshold_type
                }
            )
            
        except Exception as e:
            error_msg = f"Semantic retrieval with result failed: {str(e)}"
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
            "type": "semantic_similarity",
            "vector_store_type": "QdrantVectorStore",
            "chunking_method": "semantic",
            "breakpoint_threshold_type": self.breakpoint_threshold_type
        }


def create_semantic_retriever(
    vector_store: QdrantVectorStore,
    embeddings,
    k: int = 5,
    similarity_threshold: Optional[float] = None,
    breakpoint_threshold_type: str = "percentile"
) -> SemanticRetriever:
    """
    Create a semantic retriever instance.
    
    Args:
        vector_store: Qdrant vector store instance
        embeddings: Embedding model for semantic analysis
        k: Number of documents to retrieve
        similarity_threshold: Minimum similarity threshold
        breakpoint_threshold_type: Type of threshold for semantic breakpoints
        
    Returns:
        Semantic retriever instance
    """
    return SemanticRetriever(
        vector_store, 
        embeddings, 
        k, 
        similarity_threshold, 
        breakpoint_threshold_type
    )
