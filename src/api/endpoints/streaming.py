"""
Streaming query endpoints for RAG operations.

Handles streaming query processing with real-time responses.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.models import QueryRequest, StreamChunk
from src.api.services import rag_service
from src.utils.logging import get_logger
from datetime import datetime
import json

logger = get_logger(__name__)

router = APIRouter(prefix="/stream", tags=["streaming"])


@router.post("/query")
async def stream_query(request: QueryRequest):
    """Stream a RAG query response."""
    try:
        if not rag_service.is_ready():
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        logger.info(f"🌊 Starting stream for query: {request.question[:50]}...")
        
        async def generate_stream():
            try:
                async for chunk in rag_service.stream_query(
                    question=request.question,
                    method=request.method,
                    include_confidence=request.include_confidence
                ):
                    # Format as Server-Sent Events
                    chunk_json = json.dumps(chunk)
                    yield f"data: {chunk_json}\n\n"
                    
                    # Break on error or end
                    if chunk.get("chunk_type") in ["error", "end"]:
                        break
                        
            except Exception as e:
                logger.error(f"❌ Stream generation failed: {str(e)}")
                error_chunk = {
                    "chunk_type": "error",
                    "error": str(e),
                    "timestamp": str(datetime.now())
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Stream endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stream failed: {str(e)}")


@router.post("/query/llm")
async def stream_query_with_llm(request: QueryRequest):
    """Stream a RAG query with real LLM streaming."""
    try:
        if not rag_service.is_ready():
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        logger.info(f"🌊 Starting LLM stream for query: {request.question[:50]}...")
        
        async def generate_llm_stream():
            try:
                async for chunk in rag_service.stream_query_with_llm(
                    question=request.question,
                    method=request.method,
                    include_confidence=request.include_confidence
                ):
                    # Format as Server-Sent Events
                    chunk_json = json.dumps(chunk)
                    yield f"data: {chunk_json}\n\n"
                    
                    # Break on error or end
                    if chunk.get("chunk_type") in ["error", "end"]:
                        break
                        
            except Exception as e:
                logger.error(f"❌ LLM stream generation failed: {str(e)}")
                error_chunk = {
                    "chunk_type": "error",
                    "error": str(e),
                    "timestamp": str(datetime.now())
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_llm_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ LLM stream endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM stream failed: {str(e)}")


@router.get("/query/simple")
async def stream_query_simple(question: str, method: str = "production"):
    """Simple streaming endpoint for testing."""
    try:
        if not rag_service.is_ready():
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        logger.info(f"🌊 Starting simple stream for query: {question[:50]}...")
        
        async def generate_simple_stream():
            try:
                async for chunk in rag_service.stream_query(
                    question=question,
                    method=method,
                    include_confidence=True
                ):
                    # Format as Server-Sent Events
                    chunk_json = json.dumps(chunk)
                    yield f"data: {chunk_json}\n\n"
                    
                    # Break on error or end
                    if chunk.get("chunk_type") in ["error", "end"]:
                        break
                        
            except Exception as e:
                logger.error(f"❌ Simple stream generation failed: {str(e)}")
                error_chunk = {
                    "chunk_type": "error",
                    "error": str(e),
                    "timestamp": str(datetime.now())
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_simple_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Simple stream endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple stream failed: {str(e)}")
