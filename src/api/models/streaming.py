"""
Streaming response models for RAG operations.

Pydantic models for streaming API responses.
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field


class StreamChunk(BaseModel):
    """Model for streaming response chunks."""
    chunk_type: str = Field(..., description="Type of chunk (start, content, end)")
    content: Optional[str] = Field(default=None, description="Chunk content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Chunk metadata")
    timestamp: str = Field(default_factory=lambda: str(datetime.now()), description="Chunk timestamp")


class StreamStart(BaseModel):
    """Model for stream start event."""
    chunk_type: str = Field(default="start", description="Chunk type")
    question: str = Field(..., description="Original question")
    method: str = Field(..., description="Retrieval method")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Stream metadata")
    timestamp: str = Field(default_factory=lambda: str(datetime.now()), description="Start timestamp")


class StreamContent(BaseModel):
    """Model for stream content chunks."""
    chunk_type: str = Field(default="content", description="Chunk type")
    content: str = Field(..., description="Content chunk")
    chunk_index: int = Field(..., description="Chunk sequence number")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Chunk metadata")
    timestamp: str = Field(default_factory=lambda: str(datetime.now()), description="Chunk timestamp")


class StreamEnd(BaseModel):
    """Model for stream end event."""
    chunk_type: str = Field(default="end", description="Chunk type")
    total_chunks: int = Field(..., description="Total number of content chunks")
    sources: int = Field(..., description="Number of sources used")
    confidence: Optional[Dict[str, Any]] = Field(default=None, description="Confidence scores")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Final metadata")
    timestamp: str = Field(default_factory=lambda: str(datetime.now()), description="End timestamp")


class StreamError(BaseModel):
    """Model for stream error events."""
    chunk_type: str = Field(default="error", description="Chunk type")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Error metadata")
    timestamp: str = Field(default_factory=lambda: str(datetime.now()), description="Error timestamp")
