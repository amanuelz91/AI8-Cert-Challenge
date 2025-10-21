"""
System management endpoints.

Handles stats, methods, and system operations.
"""

from fastapi import APIRouter, HTTPException
from src.api.models import (
    StatsResponse, 
    MethodsResponse, 
    ReloadResponse
)
from src.api.services import rag_service
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        if not rag_service.is_ready():
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        stats = rag_service.get_stats()
        
        return StatsResponse(
            documents=stats["documents"],
            retrievers=stats["retrievers"],
            vector_store=stats["vector_store"],
            configuration=stats["configuration"]
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/methods", response_model=MethodsResponse)
async def get_available_methods():
    """Get available retrieval methods."""
    try:
        methods = rag_service.get_available_methods()
        return MethodsResponse(**methods)
        
    except Exception as e:
        logger.error(f"❌ Failed to get methods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get methods: {str(e)}")


@router.post("/reload", response_model=ReloadResponse)
async def reload_system():
    """Reload the RAG system."""
    try:
        result = rag_service.reload_system()
        return ReloadResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Failed to reload system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload system: {str(e)}")
