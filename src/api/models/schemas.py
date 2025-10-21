"""
API models for request/response schemas.

Pydantic models for type-safe API communication.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="The question to ask the RAG system")
    method: str = Field(default="production", description="Retrieval method to use")
    include_confidence: bool = Field(default=True, description="Whether to include confidence scoring")
    max_results: Optional[int] = Field(default=None, description="Maximum number of results")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(..., description="The generated answer")
    method: str = Field(..., description="The retrieval method used")
    confidence: Optional[Dict[str, Any]] = Field(default=None, description="Confidence scores")
    sources: int = Field(..., description="Number of sources used")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Overall system status")
    components: Dict[str, Any] = Field(..., description="Component health status")
    timestamp: str = Field(..., description="Health check timestamp")


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    documents: Dict[str, int] = Field(..., description="Document statistics")
    retrievers: Dict[str, Any] = Field(..., description="Retriever statistics")
    vector_store: Dict[str, Any] = Field(..., description="Vector store statistics")
    configuration: Dict[str, Any] = Field(..., description="System configuration")


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    questions: List[str] = Field(..., description="List of questions to process")
    method: str = Field(default="production", description="Retrieval method to use")
    include_confidence: bool = Field(default=True, description="Whether to include confidence scoring")


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""
    results: List[QueryResponse] = Field(..., description="Results for each question")
    total_questions: int = Field(..., description="Total number of questions processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    processing_time: float = Field(..., description="Total processing time in seconds")


class MethodInfo(BaseModel):
    """Model for retrieval method information."""
    name: str = Field(..., description="Method name")
    description: str = Field(..., description="Method description")
    use_case: str = Field(..., description="Recommended use case")


class MethodsResponse(BaseModel):
    """Response model for available methods."""
    available_methods: List[MethodInfo] = Field(..., description="List of available methods")


class ReloadResponse(BaseModel):
    """Response model for system reload."""
    message: str = Field(..., description="Reload status message")
    timestamp: str = Field(..., description="Reload timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
    timestamp: str = Field(default_factory=lambda: str(datetime.now()), description="Error timestamp")
