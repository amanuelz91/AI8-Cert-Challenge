"""
Pydantic models for configuration validation and type safety.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class DocumentMetadata(BaseModel):
    """Metadata model for documents."""
    
    source: str
    page_number: Optional[int] = None
    file_path: Optional[str] = None
    document_type: str = "unknown"
    created_at: Optional[str] = None
    additional_fields: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Model for retrieval results."""
    
    documents: List[str]
    scores: List[float]
    metadata: List[DocumentMetadata]
    retrieval_method: str
    query: str


class RAGResponse(BaseModel):
    """Model for RAG system responses."""
    
    answer: str
    sources: List[str]
    confidence_score: float
    retrieval_method: str
    context_used: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    """Model for evaluation metrics."""
    
    context_recall: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    context_utilization: float = Field(ge=0, le=1)
    answer_correctness: float = Field(ge=0, le=1)
    answer_similarity: float = Field(ge=0, le=1)
    answer_consistency: float = Field(ge=0, le=1)
    
    @validator('*', pre=True)
    def validate_scores(cls, v):
        """Validate that all scores are between 0 and 1."""
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return v


class PerformanceMetrics(BaseModel):
    """Model for performance metrics."""
    
    total_runs: int
    total_cost: float
    avg_cost_per_run: float
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_sec: float
    success_rate: float = Field(ge=0, le=1)


class RetrieverConfig(BaseModel):
    """Configuration for individual retrievers."""
    
    name: str
    enabled: bool = True
    k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    max_context_length: int = Field(default=4000, ge=100, le=8000)
    additional_params: Dict[str, Any] = Field(default_factory=dict)
