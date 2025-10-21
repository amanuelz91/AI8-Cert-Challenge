"""
Health check endpoint.

Provides system health monitoring.
"""

from fastapi import APIRouter, HTTPException
from src.api.models import HealthResponse
from src.api.services import rag_service
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_status = rag_service.health_check()
        
        return HealthResponse(
            status=health_status["overall"],
            components=health_status["components"],
            timestamp=health_status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
