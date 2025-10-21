"""
Logging configuration for the RAG system.

Provides structured logging with proper formatting and levels.
"""

import logging
import sys
from typing import Optional
from src.config.settings import get_config


def setup_logging(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for a module.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level override
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    config = get_config()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or config.logging.level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=config.logging.format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file or config.logging.file_path:
        file_path = log_file or config.logging.file_path
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)
