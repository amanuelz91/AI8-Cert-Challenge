"""
High-level tool cache manager for Tavily search tools.
This module provides a hierarchical caching system for different tool categories.
"""

from typing import Dict, Any, Optional, Callable
from functools import wraps
import threading


class ToolCacheManager:
    """High-level cache manager for different tool categories."""
    
    def __init__(self):
        self._caches: Dict[str, Dict[str, Any]] = {}
        self._cache_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
    
    def _get_cache_lock(self, cache_name: str) -> threading.Lock:
        """Get or create a lock for a specific cache."""
        with self._lock:
            if cache_name not in self._cache_locks:
                self._cache_locks[cache_name] = threading.Lock()
            return self._cache_locks[cache_name]
    
    def get_or_create_tool(
        self, 
        cache_name: str, 
        tool_name: str, 
        tool_factory: Callable[[], Any],
        api_key: Optional[str] = None
    ) -> Any:
        """
        Get tool from cache or create it using the factory function.
        
        Args:
            cache_name (str): Name of the cache category (e.g., 'student_loan')
            tool_name (str): Name of the specific tool (e.g., 'studentaid')
            tool_factory (Callable): Function that creates the tool
            api_key (Optional[str]): API key for cache invalidation
            
        Returns:
            Any: The cached or newly created tool
        """
        cache_key = f"{cache_name}_{api_key or 'default'}"
        
        with self._get_cache_lock(cache_name):
            if cache_name not in self._caches:
                self._caches[cache_name] = {}
            
            if cache_key not in self._caches[cache_name]:
                self._caches[cache_name][cache_key] = {}
            
            if tool_name not in self._caches[cache_name][cache_key]:
                self._caches[cache_name][cache_key][tool_name] = tool_factory()
            
            return self._caches[cache_name][cache_key][tool_name]
    
    def get_all_tools_in_category(
        self, 
        cache_name: str, 
        tool_factories: Dict[str, Callable[[], Any]],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all tools in a category, creating them if needed.
        
        Args:
            cache_name (str): Name of the cache category
            tool_factories (Dict[str, Callable]): Dictionary of tool names to factory functions
            api_key (Optional[str]): API key for cache invalidation
            
        Returns:
            Dict[str, Any]: Dictionary of tool names to tool instances
        """
        tools = {}
        for tool_name, tool_factory in tool_factories.items():
            tools[tool_name] = self.get_or_create_tool(
                cache_name, tool_name, tool_factory, api_key
            )
        return tools
    
    def clear_cache(self, cache_name: Optional[str] = None):
        """
        Clear cache for a specific category or all caches.
        
        Args:
            cache_name (Optional[str]): Specific cache to clear, or None for all
        """
        with self._lock:
            if cache_name is None:
                self._caches.clear()
                self._cache_locks.clear()
            elif cache_name in self._caches:
                del self._caches[cache_name]
                if cache_name in self._cache_locks:
                    del self._cache_locks[cache_name]
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        info = {}
        for cache_name, cache_data in self._caches.items():
            info[cache_name] = {
                "cache_keys": list(cache_data.keys()),
                "total_tools": sum(len(tools) for tools in cache_data.values())
            }
        return info


# Global cache manager instance
_cache_manager = ToolCacheManager()


def get_cache_manager() -> ToolCacheManager:
    """Get the global cache manager instance."""
    return _cache_manager


def cached_tool(cache_name: str, tool_name: str):
    """
    Decorator for caching tool creation.
    
    Args:
        cache_name (str): Name of the cache category
        tool_name (str): Name of the specific tool
    """
    def decorator(tool_factory_func):
        @wraps(tool_factory_func)
        def wrapper(api_key: Optional[str] = None):
            return _cache_manager.get_or_create_tool(
                cache_name, tool_name, lambda: tool_factory_func(api_key), api_key
            )
        return wrapper
    return decorator


def clear_all_caches():
    """Clear all tool caches."""
    _cache_manager.clear_cache()


def clear_cache(cache_name: str):
    """Clear a specific tool cache."""
    _cache_manager.clear_cache(cache_name)


def get_cache_info() -> Dict[str, Any]:
    """Get information about current cache state."""
    return _cache_manager.get_cache_info()
