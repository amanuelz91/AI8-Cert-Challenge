"""
Utility decorators for the RAG system.

Provides common decorators for caching, timing, and error handling.
"""

import time
import functools
from typing import Any, Callable, Dict, Optional
from src.utils.logging import get_logger
from src.utils.exceptions import RAGException


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"⏱️ {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


def retry_decorator(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"❌ {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator


def cache_decorator(cache: Optional[Dict] = None):
    """
    Simple in-memory cache decorator.
    
    Args:
        cache: Optional cache dictionary to use
        
    Returns:
        Decorator function
    """
    if cache is None:
        cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            if cache_key in cache:
                logger = get_logger(func.__module__)
                logger.debug(f"🎯 Cache hit for {func.__name__}")
                return cache[cache_key]
            
            result = func(*args, **kwargs)
            cache[cache_key] = result
            
            logger = get_logger(func.__module__)
            logger.debug(f"💾 Cached result for {func.__name__}")
            return result
        
        return wrapper
    return decorator


def error_handler_decorator(exception_type: type = RAGException):
    """
    Decorator to handle exceptions gracefully.
    
    Args:
        exception_type: Type of exception to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                logger = get_logger(func.__module__)
                logger.error(f"❌ {func.__name__} failed: {str(e)}")
                raise
            except Exception as e:
                logger = get_logger(func.__module__)
                logger.error(f"❌ Unexpected error in {func.__name__}: {str(e)}")
                raise exception_type(f"Unexpected error in {func.__name__}: {str(e)}")
        
        return wrapper
    return decorator


def validate_inputs_decorator(validator_func: Callable):
    """
    Decorator to validate function inputs.
    
    Args:
        validator_func: Function to validate inputs
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validator_func(*args, **kwargs):
                raise ValueError(f"Invalid inputs for {func.__name__}")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
