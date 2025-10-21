"""
Parent document retriever implementation for evaluation.

Hierarchical retrieval for maximum context preservation.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.utils.decorators import timing_decorator

logger = get_logger(__name__)


class ParentDocumentRetrieverWrapper(BaseRAGRetriever):
    """Parent document retriever wrapper for hierarchical retrieval."""
    
    def __init__(
        self,
        parent_documents: List[Document],
        embeddings,
        k: int = 5,
        child_chunk_size: int = 750,
        child_chunk_overlap: int = 100,
        collection_name: str = "parent_documents"
    ):
        """
        Initialize parent document retriever.
        
        Args:
            parent_documents: List of parent documents
            embeddings: Embedding model
            k: Number of documents to retrieve
            child_chunk_size: Size of child chunks
            child_chunk_overlap: Overlap between child chunks
            collection_name: Qdrant collection name
        """
        super().__init__("parent_document", k)
        self.parent_documents = parent_documents
        self.embeddings = embeddings
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.collection_name = collection_name
        
        # Initialize components
        self._setup_parent_document_retriever()
        
        logger.info(f"📄 Initialized parent document retriever (k={k}, chunk_size={child_chunk_size})")
    
    def _setup_parent_document_retriever(self) -> None:
        """Set up the parent document retriever components."""
        try:
            # Create child splitter
            self.child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.child_chunk_size,
                chunk_overlap=self.child_chunk_overlap
            )
            
            # Create vector store for child chunks
            self.child_vectorstore = QdrantVectorStore(
                collection_name=self.collection_name,
                embedding=self.embeddings,
                location=":memory:"  # Using in-memory for simplicity
            )
            
            # Create document store for parent documents
            self.docstore = InMemoryStore()
            
            # Create parent document retriever
            self.parent_document_retriever = ParentDocumentRetriever(
                vectorstore=self.child_vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter
            )
            
            # Add parent documents
            self.parent_document_retriever.add_documents(self.parent_documents)
            
            logger.info(f"✅ Set up parent document retriever with {len(self.parent_documents)} parent documents")
            
        except Exception as e:
            error_msg = f"Failed to setup parent document retriever: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve parent documents using hierarchical retrieval.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved parent documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"📄 [Parent Document] Retrieving documents for: {query[:50]}...")
            
            # Use parent document retriever
            documents = self.parent_document_retriever.invoke(query)
            
            # Limit to k documents
            if len(documents) > self.k:
                documents = documents[:self.k]
            
            # Add parent document metadata
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'parent_document'
                doc.metadata['child_chunk_size'] = self.child_chunk_size
                doc.metadata['child_chunk_overlap'] = self.child_chunk_overlap
            
            logger.info(f"📚 [Parent Document] Retrieved {len(documents)} parent documents")
            return documents
            
        except Exception as e:
            error_msg = f"Parent document retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve parent documents with hierarchical scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"📄 [Parent Document] Retrieving documents with scores for: {query[:50]}...")
            
            # Get child chunks with scores first
            child_docs_with_scores = self.child_vectorstore.similarity_search_with_score(
                query, k=self.k
            )
            
            # Get parent documents
            parent_documents = self.parent_document_retriever.invoke(query)
            
            # Limit to k documents
            if len(parent_documents) > self.k:
                parent_documents = parent_documents[:self.k]
            
            # Map child chunk scores to parent documents
            child_score_map = {}
            for child_doc, score in child_docs_with_scores:
                # Use content snippet as key to match with parent docs
                content_key = child_doc.page_content[:100]
                if content_key not in child_score_map:
                    child_score_map[content_key] = float(score)
            
            # Add parent document metadata and scores
            results = []
            for i, doc in enumerate(parent_documents):
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'parent_document'
                doc.metadata['child_chunk_size'] = self.child_chunk_size
                doc.metadata['child_chunk_overlap'] = self.child_chunk_overlap
                doc.metadata['parent_document_position'] = i + 1
                
                # Find best matching child score
                best_score = 0.0
                for content_key, score in child_score_map.items():
                    if content_key in doc.page_content:
                        best_score = max(best_score, score)
                
                # Convert distance to similarity
                similarity_score = 1 - best_score if best_score > 0 else 0.5
                results.append((doc, similarity_score))
            
            logger.info(f"📚 [Parent Document] Retrieved {len(results)} parent documents with scores")
            return results
            
        except Exception as e:
            error_msg = f"Parent document retrieval with scores failed: {str(e)}"
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
                    "avg_parent_score": sum(scores) / len(scores),
                    "max_parent_score": max(scores),
                    "min_parent_score": min(scores),
                    "child_chunk_size": self.child_chunk_size,
                    "child_chunk_overlap": self.child_chunk_overlap,
                    "retrieval_method": "parent_document",
                    "hierarchical_retrieval": True
                }
            )
            
        except Exception as e:
            error_msg = f"Parent document retrieval with result failed: {str(e)}"
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
            "type": "parent_document_hierarchical",
            "child_chunk_size": self.child_chunk_size,
            "child_chunk_overlap": self.child_chunk_overlap,
            "parent_documents": len(self.parent_documents),
            "collection_name": self.collection_name
        }


def create_parent_document_retriever(
    parent_documents: List[Document],
    embeddings,
    k: int = 5,
    child_chunk_size: int = 750,
    child_chunk_overlap: int = 100,
    collection_name: str = "parent_documents"
) -> ParentDocumentRetrieverWrapper:
    """
    Create a parent document retriever instance.
    
    Args:
        parent_documents: List of parent documents
        embeddings: Embedding model
        k: Number of documents to retrieve
        child_chunk_size: Size of child chunks
        child_chunk_overlap: Overlap between child chunks
        collection_name: Qdrant collection name
        
    Returns:
        Parent document retriever instance
    """
    return ParentDocumentRetrieverWrapper(
        parent_documents, 
        embeddings, 
        k, 
        child_chunk_size, 
        child_chunk_overlap, 
        collection_name
    )
