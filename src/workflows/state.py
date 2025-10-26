"""
State definitions for LangGraph workflows.

Contains TypedDict classes for state management in RAG workflows.
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from operator import add


class BaseRAGState(TypedDict):
    """Base state for RAG workflows."""
    question: str
    response: str
    metadata: Dict[str, Any]


class RetrievalState(BaseRAGState):
    """State for retrieval operations."""
    context: List[Document]
    retrieval_method: str
    retrieval_scores: List[float]


class GenerationState(RetrievalState):
    """State for text generation."""
    prompt: str
    generation_params: Dict[str, Any]
    confidence_score: Optional[float]


class EvaluationState(GenerationState):
    """State for evaluation operations."""
    evaluation_metrics: Dict[str, float]
    evaluation_scores: List[float]
    evaluation_method: str


class MultiRetrievalState(BaseRAGState):
    """State for multi-retrieval workflows."""
    naive_context: List[Document]
    semantic_context: List[Document]
    tool_context: List[Document]
    combined_context: List[Document]
    retrieval_results: Dict[str, Any]


class HybridRAGState(MultiRetrievalState):
    """State for hybrid RAG workflows."""
    knowledge_response: str
    search_response: str
    final_response: str
    source_attribution: Dict[str, List[str]]


class ProductionRAGState(HybridRAGState):
    """State for production RAG workflows."""
    confidence_scores: Dict[str, float]
    quality_metrics: Dict[str, float]
    performance_metrics: Dict[str, Any]
    error_handling: Optional[str]
    retrieval_method: Annotated[List[str], add]


class EvaluationWorkflowState(BaseRAGState):
    """State for evaluation workflows."""
    ground_truth: Optional[str]
    evaluation_dataset: List[Dict[str, Any]]
    evaluation_results: Dict[str, Any]
    comparison_metrics: Dict[str, float]
    retriever_performance: Dict[str, Dict[str, float]]


# Export all state classes
__all__ = [
    "BaseRAGState",
    "RetrievalState", 
    "GenerationState",
    "EvaluationState",
    "MultiRetrievalState",
    "HybridRAGState",
    "ProductionRAGState",
    "EvaluationWorkflowState"
]
