"""
Environment settings and configuration management.

Provides centralized configuration using Pydantic models
for type safety and validation.
"""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    collection_name: str = Field(default="rag_documents", env="QDRANT_COLLECTION")
    vector_size: int = Field(default=1536, env="VECTOR_SIZE")
    batch_size: int = Field(default=100, env="QDRANT_BATCH_SIZE")
    
    class Config:
        extra = "ignore"


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    cache_enabled: bool = Field(default=True, env="EMBEDDING_CACHE")
    
    class Config:
        env_prefix = "EMBEDDING_"
        extra = "ignore"


class LLMConfig(BaseSettings):
    """Large Language Model configuration."""
    
    model_name: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=1000, env="LLM_MAX_TOKENS")
    timeout: int = Field(default=120, env="LLM_TIMEOUT")
    
    class Config:
        env_prefix = "LLM_"
        extra = "ignore"


class RetrievalConfig(BaseSettings):
    """Retrieval strategy configuration."""
    
    default_k: int = Field(default=5, env="RETRIEVAL_K")
    similarity_threshold: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")
    max_context_length: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    
    class Config:
        env_prefix = "RETRIEVAL_"
        extra = "ignore"


class DataConfig(BaseSettings):
    """Data processing configuration."""
    
    chunk_size: int = Field(default=750, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    data_folder: str = Field(default="./src/data", env="DATA_FOLDER")
    
    class Config:
        env_prefix = "DATA_"
        extra = "ignore"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE")
    
    class Config:
        env_prefix = "LOG_"
        extra = "ignore"


class EvaluationConfig(BaseSettings):
    """RAG evaluation configuration."""
    
    testset_size: int = Field(default=10, env="EVAL_TESTSET_SIZE")
    generator_model: str = Field(default="gpt-4o", env="EVAL_GENERATOR_MODEL")
    generator_temperature: float = Field(default=0.7, env="EVAL_GENERATOR_TEMP")
    evaluator_model: str = Field(default="gpt-4o-mini", env="EVAL_EVALUATOR_MODEL")
    evaluator_temperature: float = Field(default=0.0, env="EVAL_EVALUATOR_TEMP")
    embedding_model: str = Field(default="text-embedding-3-small", env="EVAL_EMBEDDING_MODEL")
    timeout: int = Field(default=360, env="EVAL_TIMEOUT")
    batch_size: int = Field(default=5, env="EVAL_BATCH_SIZE")
    langsmith_project: str = Field(default="rag-evaluation", env="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, env="LANGSMITH_TRACING")
    results_dir: str = Field(default="./evaluation_results", env="EVAL_RESULTS_DIR")
    save_dataset: bool = Field(default=True, env="EVAL_SAVE_DATASET")
    save_results: bool = Field(default=True, env="EVAL_SAVE_RESULTS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


class RAGConfig(BaseSettings):
    """Main RAG system configuration."""
    
    database: DatabaseConfig = DatabaseConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


# Global configuration instance
config = RAGConfig()


def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    return config


def validate_api_keys() -> None:
    """Validate that required API keys are present."""
    required_keys = ["openai_api_key"]
    missing_keys = []
    
    for key in required_keys:
        if not getattr(config, key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")


def get_data_path() -> str:
    """Get the configured data folder path."""
    data_path = config.data.data_folder
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return data_path
