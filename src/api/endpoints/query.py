"""
Query endpoints for RAG operations.

Handles single and batch query processing.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.models import (
    QueryRequest, 
    QueryResponse, 
    BatchQueryRequest, 
    BatchQueryResponse
)
from src.api.services import rag_service
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    try:
        if not rag_service.is_ready():
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        logger.info(f"❓ Processing query: {request.question[:50]}...")
        
        result = rag_service.query(
            question=request.question,
            method=request.method,
            include_confidence=request.include_confidence
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/batch", response_model=BatchQueryResponse)
async def batch_query_rag(request: BatchQueryRequest, background_tasks: BackgroundTasks):
    """Process multiple queries in batch."""
    try:
        if not rag_service.is_ready():
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        logger.info(f"📦 Processing batch of {len(request.questions)} queries")
        
        result = rag_service.batch_query(
            questions=request.questions,
            method=request.method,
            include_confidence=request.include_confidence
        )
        
        return BatchQueryResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Batch query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch query failed: {str(e)}")
