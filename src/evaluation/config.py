"""
Configuration for RAG evaluation using RAGAS and LangSmith.
"""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class EvaluationConfig(BaseSettings):
    """Configuration for RAG evaluation."""
    
    # Dataset generation
    testset_size: int = Field(default=10, env="EVAL_TESTSET_SIZE")
    generator_model: str = Field(default="gpt-4o", env="EVAL_GENERATOR_MODEL")
    generator_temperature: float = Field(default=0.7, env="EVAL_GENERATOR_TEMP")
    
    # Evaluation models
    evaluator_model: str = Field(default="gpt-4o-mini", env="EVAL_EVALUATOR_MODEL")
    evaluator_temperature: float = Field(default=0.0, env="EVAL_EVALUATOR_TEMP")
    
    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small", env="EVAL_EMBEDDING_MODEL")
    
    # Evaluation settings
    timeout: int = Field(default=360, env="EVAL_TIMEOUT")
    batch_size: int = Field(default=5, env="EVAL_BATCH_SIZE")
    
    # LangSmith settings
    langsmith_project: str = Field(default="rag-evaluation", env="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, env="LANGSMITH_TRACING")
    
    # Output settings
    results_dir: str = Field(default="./evaluation_results", env="EVAL_RESULTS_DIR")
    save_dataset: bool = Field(default=True, env="EVAL_SAVE_DATASET")
    save_results: bool = Field(default=True, env="EVAL_SAVE_RESULTS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration."""
    return EvaluationConfig()
