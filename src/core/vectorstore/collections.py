"""
Collection management utilities for Qdrant vector store.

Handles collection creation, management, and operations.
"""

from typing import List, Optional, Dict, Any
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
from src.utils.logging import get_logger
from src.utils.exceptions import VectorStoreError
from src.utils.decorators import timing_decorator
from .qdrant_client import QdrantManager

logger = get_logger(__name__)


class CollectionManager:
    """Manager for Qdrant collection operations."""
    
    def __init__(self, qdrant_manager: QdrantManager):
        """
        Initialize collection manager.
        
        Args:
            qdrant_manager: Qdrant manager instance
        """
        self.qdrant_manager = qdrant_manager
        logger.info("📦 Initialized collection manager")
    
    @timing_decorator
    def create_collection_with_config(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance_metric: Distance = Distance.COSINE,
        on_disk_payload: bool = False
    ) -> None:
        """
        Create collection with specific configuration.
        
        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension size
            distance_metric: Distance metric for similarity
            on_disk_payload: Whether to store payload on disk
            
        Raises:
            VectorStoreError: If collection creation fails
        """
        try:
            logger.info(f"📦 Creating collection '{collection_name}' with config")
            
            # Check if collection exists
            collections = self.qdrant_manager.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                logger.info(f"📦 Collection '{collection_name}' already exists")
                return
            
            # Create collection with configuration
            self.qdrant_manager.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric,
                    on_disk=on_disk_payload
                )
            )
            
            logger.info(f"✅ Created collection '{collection_name}' "
                       f"(size={vector_size}, distance={distance_metric})")
            
        except Exception as e:
            error_msg = f"Failed to create collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections.
        
        Returns:
            List of collection information dictionaries
        """
        try:
            collections = self.qdrant_manager.client.get_collections()
            
            collection_info = []
            for collection in collections.collections:
                info = {
                    "name": collection.name,
                    "status": collection.status,
                    "points_count": getattr(collection, 'points_count', 0)
                }
                collection_info.append(info)
            
            logger.info(f"📋 Found {len(collection_info)} collections")
            return collection_info
            
        except Exception as e:
            logger.error(f"❌ Failed to list collections: {str(e)}")
            return []
    
    def get_collection_status(self, collection_name: str) -> Optional[CollectionStatus]:
        """
        Get collection status.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection status or None if not found
        """
        try:
            collection_info = self.qdrant_manager.client.get_collection(collection_name)
            return collection_info.status
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get status for collection '{collection_name}': {str(e)}")
            return None
    
    def wait_for_collection_ready(self, collection_name: str, timeout: int = 30) -> bool:
        """
        Wait for collection to be ready.
        
        Args:
            collection_name: Name of the collection
            timeout: Timeout in seconds
            
        Returns:
            True if ready, False if timeout
        """
        import time
        
        logger.info(f"⏳ Waiting for collection '{collection_name}' to be ready")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_collection_status(collection_name)
            if status == CollectionStatus.GREEN:
                logger.info(f"✅ Collection '{collection_name}' is ready")
                return True
            
            time.sleep(1)
        
        logger.warning(f"⚠️ Collection '{collection_name}' not ready after {timeout}s")
        return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed collection statistics.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.qdrant_manager.client.get_collection(collection_name)
            
            stats = {
                "name": collection_name,
                "status": collection_info.status,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "on_disk_payload": getattr(collection_info.config.params.vectors, 'on_disk', False)
            }
            
            return stats
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get stats for collection '{collection_name}': {str(e)}")
            return {
                "name": collection_name,
                "error": str(e)
            }
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            self.qdrant_manager.client.delete_collection(collection_name)
            logger.info(f"🗑️ Deleted collection '{collection_name}'")
            
        except Exception as e:
            error_msg = f"Failed to delete collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def recreate_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance_metric: Distance = Distance.COSINE
    ) -> None:
        """
        Recreate a collection (delete and create).
        
        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension size
            distance_metric: Distance metric
            
        Raises:
            VectorStoreError: If recreation fails
        """
        try:
            logger.info(f"🔄 Recreating collection '{collection_name}'")
            
            # Delete if exists
            try:
                self.delete_collection(collection_name)
            except VectorStoreError:
                pass  # Collection might not exist
            
            # Create new collection
            self.create_collection_with_config(
                collection_name=collection_name,
                vector_size=vector_size,
                distance_metric=distance_metric
            )
            
            logger.info(f"✅ Successfully recreated collection '{collection_name}'")
            
        except Exception as e:
            error_msg = f"Failed to recreate collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e


def create_collection_manager(qdrant_manager: QdrantManager) -> CollectionManager:
    """
    Create a collection manager instance.
    
    Args:
        qdrant_manager: Qdrant manager instance
        
    Returns:
        Collection manager instance
    """
    return CollectionManager(qdrant_manager)
