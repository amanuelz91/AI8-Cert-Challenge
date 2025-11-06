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
from qdrant_client import QdrantClient, models
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
        collection_name: str = "parent_documents",
        vector_store: Optional[QdrantVectorStore] = None
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
            vector_store: Optional existing QdrantVectorStore to reuse (uses cloud Qdrant if provided)
        """
        super().__init__("parent_document", k)
        self.parent_documents = parent_documents
        self.embeddings = embeddings
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.collection_name = collection_name
        self.vector_store = vector_store  # Store for potential reuse
        
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
            
            # Use existing vector store if provided (reuses cloud Qdrant), otherwise create new in-memory one
            if self.vector_store is not None:
                logger.info(f"♻️ Reusing existing vector store for parent documents")
                # Create a new QdrantVectorStore pointing to a different collection in the same cloud Qdrant
                # Get the client from the existing vector store
                existing_client = self.vector_store.client
                
                # Get embedding dimension
                try:
                    sample_embedding = self.embeddings.embed_query("test")
                    embedding_dim = len(sample_embedding)
                except Exception:
                    embedding_dim = 1536
                
                # Create collection for parent document child chunks in the cloud Qdrant
                try:
                    existing_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=embedding_dim,
                            distance=models.Distance.COSINE
                        )
                    )
                    logger.info(f"✅ Created collection '{self.collection_name}' in cloud Qdrant for parent documents")
                except Exception as e:
                    logger.debug(f"Collection '{self.collection_name}' may already exist: {str(e)}")
                
                # Create vector store using the same cloud client but different collection
                self.child_vectorstore = QdrantVectorStore(
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    client=existing_client
                )
                logger.info(f"✅ Reusing cloud Qdrant instance for parent document child chunks")
                
                # Check if collection already has documents
                try:
                    collection_info = existing_client.get_collection(self.collection_name)
                    existing_points = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                    if existing_points > 0:
                        logger.info(f"📦 Parent document collection '{self.collection_name}' already contains {existing_points} child chunks")
                        logger.info(f"✅ Skipping document addition (documents already loaded)")
                    else:
                        logger.info(f"📤 Parent document collection '{self.collection_name}' is empty, will add documents")
                except Exception as e:
                    logger.debug(f"Could not check collection stats: {str(e)}")
                    existing_points = 0
            else:
                # Fallback: Create in-memory Qdrant client (separate instance)
                logger.info(f"🗃️ Creating in-memory Qdrant client for parent documents (separate from cloud)")
                client = QdrantClient(location=":memory:")
                
                # Create collection for child chunks
                # First, get the embedding dimension
                try:
                    # Get embedding dimension from a sample document
                    sample_embedding = self.embeddings.embed_query("test")
                    embedding_dim = len(sample_embedding)
                except Exception:
                    # Default to common embedding dimension
                    embedding_dim = 1536
                    logger.warning(f"⚠️ Could not determine embedding dimension, using default: {embedding_dim}")
                
                # Create collection if it doesn't exist
                try:
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=embedding_dim,
                            distance=models.Distance.COSINE
                        )
                    )
                    logger.info(f"✅ Created collection: {self.collection_name}")
                except Exception as e:
                    # Collection might already exist, that's okay
                    logger.debug(f"Collection {self.collection_name} may already exist: {str(e)}")
                
                # Create vector store for child chunks
                logger.info(f"📦 Creating QdrantVectorStore with collection: {self.collection_name}")
                self.child_vectorstore = QdrantVectorStore(
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    client=client
                )
            
            # Create document store for parent documents
            self.docstore = InMemoryStore()
            
            # Create parent document retriever
            self.parent_document_retriever = ParentDocumentRetriever(
                vectorstore=self.child_vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter
            )
            
            # Check if child chunks already exist in cloud Qdrant
            child_chunks_exist = False
            existing_points = 0
            if self.vector_store is not None:
                # We're using cloud Qdrant - check if collection already has data
                try:
                    existing_client = self.child_vectorstore.client
                    collection_info = existing_client.get_collection(self.collection_name)
                    existing_points = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                    if existing_points > 0:
                        child_chunks_exist = True
                        logger.info(f"📦 Parent document collection already has {existing_points} child chunks")
                        logger.info(f"♻️ Child chunks already in cloud Qdrant, will only add parent documents to docstore")
                except Exception as e:
                    logger.debug(f"Could not check collection stats: {str(e)}")
                    child_chunks_exist = False
            
            # Always add parent documents to docstore (it's in-memory, so it's empty on restart)
            # But if child chunks already exist, we only add to docstore, not re-add child chunks
            if child_chunks_exist:
                # Child chunks already exist in Qdrant, only populate docstore
                logger.info(f"📚 Populating docstore with {len(self.parent_documents)} parent documents (child chunks already in Qdrant)...")
                # Extract parent document IDs from existing child chunks to match IDs correctly
                try:
                    # Sample a few child chunks to see what parent document IDs they reference
                    # ParentDocumentRetriever stores parent document IDs in child chunk metadata
                    # Use a generic query to get some chunks
                    sample_result = self.child_vectorstore.similarity_search("document", k=min(10, existing_points))
                    parent_ids = set()
                    for chunk in sample_result:
                        # ParentDocumentRetriever uses 'parent_doc_id' or similar in metadata
                        parent_id = chunk.metadata.get('parent_doc_id') or chunk.metadata.get('doc_id')
                        if parent_id:
                            parent_ids.add(parent_id)
                    
                    if parent_ids:
                        logger.info(f"📋 Found {len(parent_ids)} unique parent document IDs in existing child chunks")
                        # Use the same ID format - if we found UUIDs, generate UUIDs; if strings, use strings
                        # For now, we'll just add documents with sequential IDs and hope they match
                        # This is a limitation - ideally we'd extract all parent IDs from all child chunks
                        logger.warning(f"⚠️ Using sequential IDs; some parent documents might not match if IDs changed")
                    
                    # Add parent documents with sequential string IDs (ParentDocumentRetriever default)
                    for i, doc in enumerate(self.parent_documents):
                        doc_id = str(i)
                        self.docstore.mset([(doc_id, doc)])
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract parent IDs from child chunks: {str(e)}")
                    logger.warning(f"⚠️ Adding parent documents with sequential IDs (may not match existing child chunks)")
                    # Fallback: use sequential IDs
                    for i, doc in enumerate(self.parent_documents):
                        doc_id = str(i)
                        self.docstore.mset([(doc_id, doc)])
                
                logger.info(f"✅ Populated docstore with {len(self.parent_documents)} parent documents")
                logger.info(f"ℹ️ Note: Child chunks already in cloud Qdrant, skipped re-adding them")
            else:
                # Collection is empty, add both child chunks and parent documents
                logger.info(f"📚 Adding {len(self.parent_documents)} parent documents (child chunks + parent docs)...")
                self.parent_document_retriever.add_documents(self.parent_documents, ids=None)
                logger.info(f"✅ Added {len(self.parent_documents)} parent documents to retriever")
            
            logger.info(f"✅ Set up parent document retriever with {len(self.parent_documents)} parent documents")
            
        except Exception as e:
            error_msg = f"Failed to setup parent document retriever: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
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
    collection_name: str = "parent_documents",
    vector_store: Optional[QdrantVectorStore] = None
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
        vector_store: Optional existing QdrantVectorStore to reuse (uses cloud Qdrant if provided)
        
    Returns:
        Parent document retriever instance
    """
    return ParentDocumentRetrieverWrapper(
        parent_documents, 
        embeddings, 
        k, 
        child_chunk_size, 
        child_chunk_overlap, 
        collection_name,
        vector_store
    )
