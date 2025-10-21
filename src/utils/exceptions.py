"""
Custom exceptions for the RAG system.

Provides specific exception types for different error scenarios.
"""


class RAGException(Exception):
    """Base exception for RAG system errors."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid or missing."""
    pass


class DataProcessingError(RAGException):
    """Raised when data processing fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding operations fail."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class RetrievalError(RAGException):
    """Raised when retrieval operations fail."""
    pass


class GenerationError(RAGException):
    """Raised when text generation fails."""
    pass


class EvaluationError(RAGException):
    """Raised when evaluation operations fail."""
    pass


class APIKeyError(RAGException):
    """Raised when API keys are missing or invalid."""
    pass


class ValidationError(RAGException):
    """Raised when data validation fails."""
    pass
