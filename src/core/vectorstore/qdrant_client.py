"""
Qdrant vector store client and operations.

Handles cloud Qdrant vector database operations with proper error handling.
"""

from typing import List, Optional, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from src.utils.logging import get_logger
from src.utils.exceptions import VectorStoreError, APIKeyError
from src.utils.decorators import timing_decorator, retry_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class QdrantManager:
    """Manager for Qdrant vector store operations."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize Qdrant manager.
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            collection_name: Collection name
            
        Raises:
            APIKeyError: If API key is missing for cloud deployment
        """
        self.config = get_config()
        
        self.url = url or self.config.database.qdrant_url
        self.api_key = api_key or self.config.database.qdrant_api_key
        self.collection_name = collection_name or self.config.database.collection_name
        self.vector_size = self.config.database.vector_size
        
        # Validate cloud deployment requirements
        if self.url != "http://localhost:6333" and not self.api_key:
            raise APIKeyError("Qdrant API key required for cloud deployment")
        
        # Initialize client
        self.client = self._create_client()
        
        logger.info(f"🗃️ Initialized Qdrant manager: {self.url}")
        logger.info(f"📦 Collection: {self.collection_name}")
    
    def _create_client(self) -> QdrantClient:
        """Create Qdrant client with proper configuration."""
        try:
            if self.api_key:
                client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key
                )
            else:
                client = QdrantClient(url=self.url)
            
            # Test connection
            client.get_collections()
            logger.info("✅ Qdrant connection successful")
            return client
            
        except Exception as e:
            error_msg = f"Failed to connect to Qdrant: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def create_collection(self, vector_size: Optional[int] = None) -> None:
        """
        Create a new collection.
        
        Args:
            vector_size: Vector dimension size
            
        Raises:
            VectorStoreError: If collection creation fails
        """
        try:
            vector_size = vector_size or self.vector_size
            
            # Check if collection exists
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name in existing_collections:
                logger.info(f"📦 Collection '{self.collection_name}' already exists")
                return
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"✅ Created collection '{self.collection_name}' with vector size {vector_size}")
            
        except Exception as e:
            error_msg = f"Failed to create collection: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def delete_collection(self) -> None:
        """
        Delete the collection.
        
        Raises:
            VectorStoreError: If collection deletion fails
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️ Deleted collection '{self.collection_name}'")
            
        except Exception as e:
            error_msg = f"Failed to delete collection: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the collection in batches.
        
        Args:
            documents: List of documents to add
            embeddings: List of embeddings for documents
            ids: Optional list of document IDs
            
        Raises:
            VectorStoreError: If document addition fails
        """
        try:
            logger.info(f"⬆️ Adding {len(documents)} documents to collection in batches")
            
            # Generate IDs if not provided (use integers for Qdrant compatibility)
            if ids is None:
                ids = [i for i in range(len(documents))]
            
            # Get batch size from config
            batch_size = self.config.database.batch_size
            total_documents = len(documents)
            
            # Process documents in batches
            for batch_start in range(0, total_documents, batch_size):
                batch_end = min(batch_start + batch_size, total_documents)
                batch_documents = documents[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end]
                batch_ids = ids[batch_start:batch_end]
                
                logger.info(f"📦 Processing batch {batch_start//batch_size + 1}: documents {batch_start+1}-{batch_end}")
                
                # Prepare points for this batch
                points = []
                for i, (doc, embedding, doc_id) in enumerate(zip(batch_documents, batch_embeddings, batch_ids)):
                    # Prepare metadata
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    metadata.update({
                        "content": doc.page_content,
                        "document_id": f"doc_{doc_id}",  # String ID for reference
                        "point_id": doc_id  # Integer ID used by Qdrant
                    })
                    
                    point = PointStruct(
                        id=doc_id,  # Use integer ID for Qdrant
                        vector=embedding,
                        payload=metadata
                    )
                    points.append(point)
                
                # Upload batch points
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                logger.info(f"✅ Successfully added batch {batch_start//batch_size + 1}: {len(batch_documents)} documents")
            
            logger.info(f"✅ Successfully added all {len(documents)} documents in batches")
            
        except Exception as e:
            error_msg = f"Failed to add documents: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def search_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.info(f"🔍 Searching for {k} similar documents")
            
            # Prepare search parameters
            search_params = {
                "vector": query_embedding,
                "limit": k,
                "with_payload": True,
                "with_vectors": False
            }
            
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                **search_params
            )
            
            # Convert results to documents
            documents_with_scores = []
            for result in results:
                # Create document
                doc = Document(
                    page_content=result.payload.get("content", ""),
                    metadata=result.payload
                )
                
                # Calculate similarity score (Qdrant returns distance, convert to similarity)
                similarity_score = 1 - result.score
                
                documents_with_scores.append((doc, similarity_score))
            
            logger.info(f"✅ Found {len(documents_with_scores)} similar documents")
            return documents_with_scores
            
        except Exception as e:
            error_msg = f"Failed to search documents: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Handle different response structures
            points_count = getattr(collection_info, 'points_count', 0)
            
            # Try to get vector size safely
            vector_size = None
            distance_metric = None
            status = getattr(collection_info, 'status', 'unknown')
            
            try:
                if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                    if hasattr(collection_info.config.params, 'vectors'):
                        vector_size = getattr(collection_info.config.params.vectors, 'size', None)
                        distance_metric = getattr(collection_info.config.params.vectors, 'distance', None)
            except AttributeError:
                # If the structure is different, just use defaults
                pass
            
            return {
                "collection_name": self.collection_name,
                "vector_size": vector_size,
                "distance_metric": distance_metric,
                "points_count": points_count,
                "status": status
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get collection info: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "points_count": 0,  # Default to 0 if we can't get info
                "error": str(e)
            }
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"❌ Qdrant health check failed: {str(e)}")
            return False


class VectorStoreManager:
    """High-level manager for vector store operations."""
    
    def __init__(self, qdrant_manager: Optional[QdrantManager] = None):
        """
        Initialize vector store manager.
        
        Args:
            qdrant_manager: Qdrant manager instance
        """
        self.qdrant_manager = qdrant_manager or QdrantManager()
        self.vector_store: Optional[QdrantVectorStore] = None
        
        logger.info("📚 Initialized vector store manager")
    
    def initialize_vector_store(self, embeddings) -> QdrantVectorStore:
        """
        Initialize LangChain QdrantVectorStore.
        
        Args:
            embeddings: Embedding model instance
            
        Returns:
            Initialized QdrantVectorStore
        """
        try:
            # Ensure collection exists
            self.qdrant_manager.create_collection()
            
            # Create LangChain vector store
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_manager.client,
                collection_name=self.qdrant_manager.collection_name,
                embedding=embeddings
            )
            
            logger.info("✅ Initialized LangChain QdrantVectorStore")
            return self.vector_store
            
        except Exception as e:
            error_msg = f"Failed to initialize vector store: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        Add documents to vector store.
        
        Args:
            documents: List of documents
            embeddings: List of embeddings
        """
        self.qdrant_manager.add_documents(documents, embeddings)
    
    def search_documents(
        self,
        query: str,
        embeddings,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents using query text.
        
        Args:
            query: Query text
            embeddings: Embedding model
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search similar documents
        return self.qdrant_manager.search_similar(query_embedding, k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.qdrant_manager.get_collection_info()


def create_qdrant_manager(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: Optional[str] = None
) -> QdrantManager:
    """
    Create a Qdrant manager instance.
    
    Args:
        url: Qdrant server URL
        api_key: Qdrant API key
        collection_name: Collection name
        
    Returns:
        Qdrant manager instance
    """
    return QdrantManager(url, api_key, collection_name)


def get_default_vector_store_manager() -> VectorStoreManager:
    """Get the default vector store manager."""
    return VectorStoreManager()
