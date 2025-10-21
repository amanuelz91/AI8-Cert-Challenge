"""
Tools package for Tavily search functionality.
This package provides a hierarchical factory pattern for creating search tools.
"""

from .infrastructure.tool_registry import (
    get_tool_registry,
    get_tool,
    get_all_tools,
    list_available_tools,
    search_tools,
    get_tool_categories,
    get_tool_info
)

from .tavily.tavily_search_factory import (
    TavilySearchFactory,
    SearchConfig,
    create_tavily_tool
)

from .tavily.tavily_tool_builder import (
    TavilyToolBuilder,
    create_tool,
    create_tools_from_config
)

from .infrastructure.tool_cache_manager import (
    ToolCacheManager,
    get_cache_manager,
    cached_tool,
    clear_all_caches,
    clear_cache,
    get_cache_info
)

# Import tool implementations to trigger auto-registration
from .search import student_loan_tools, education_tools

__all__ = [
    # Registry functions
    "get_tool_registry",
    "get_tool", 
    "get_all_tools",
    "list_available_tools",
    "search_tools",
    "get_tool_categories",
    "get_tool_info",
    
    # Tavily factory classes and functions
    "TavilySearchFactory",
    "SearchConfig", 
    "create_tavily_tool",
    "TavilyToolBuilder",
    "create_tool",
    "create_tools_from_config",
    
    # Cache management
    "ToolCacheManager",
    "get_cache_manager",
    "cached_tool",
    "clear_all_caches",
    "clear_cache",
    "get_cache_info",
    
    # Tool implementations
    "student_loan_tools",
    "education_tools"
]
