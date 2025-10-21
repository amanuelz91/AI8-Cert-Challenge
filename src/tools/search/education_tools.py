"""
Education-specific search tools using the Tavily factory pattern.
This module demonstrates how to add new tool categories to the hierarchical cache system.
"""

from typing import Dict, Any, Optional
from ..tavily.tavily_tool_builder import create_tools_from_config
from ..infrastructure.tool_cache_manager import get_cache_manager, cached_tool


# Configuration templates for education tools
EDUCATION_SEARCH_CONFIGS = {
    "coursera": {
        "name": "Coursera",
        "description": "Search Coursera courses and specializations",
        "domains": ["coursera.org"],
        "max_results": 3,
        "search_depth": "advanced",
        "include_answer": True
    },
    "edx": {
        "name": "edX",
        "description": "Search edX courses and programs",
        "domains": ["edx.org"],
        "max_results": 3,
        "search_depth": "advanced",
        "include_answer": True
    },
    "khan_academy": {
        "name": "Khan Academy",
        "description": "Search Khan Academy educational content",
        "domains": ["khanacademy.org"],
        "max_results": 3,
        "search_depth": "basic",
        "include_answer": True
    }
}

# Cache category name for education tools
EDUCATION_CACHE_NAME = "education"


def _create_coursera_tool(api_key: Optional[str] = None) -> Any:
    """Internal function to create Coursera tool."""
    tools = create_tools_from_config(EDUCATION_SEARCH_CONFIGS, api_key)
    return tools["coursera"]


def _create_edx_tool(api_key: Optional[str] = None) -> Any:
    """Internal function to create edX tool."""
    tools = create_tools_from_config(EDUCATION_SEARCH_CONFIGS, api_key)
    return tools["edx"]


def _create_khan_academy_tool(api_key: Optional[str] = None) -> Any:
    """Internal function to create Khan Academy tool."""
    tools = create_tools_from_config(EDUCATION_SEARCH_CONFIGS, api_key)
    return tools["khan_academy"]


@cached_tool(EDUCATION_CACHE_NAME, "coursera")
def get_coursera_search_tool(api_key: Optional[str] = None) -> Any:
    """Get Coursera search tool (cached)."""
    return _create_coursera_tool(api_key)


@cached_tool(EDUCATION_CACHE_NAME, "edx")
def get_edx_search_tool(api_key: Optional[str] = None) -> Any:
    """Get edX search tool (cached)."""
    return _create_edx_tool(api_key)


@cached_tool(EDUCATION_CACHE_NAME, "khan_academy")
def get_khan_academy_search_tool(api_key: Optional[str] = None) -> Any:
    """Get Khan Academy search tool (cached)."""
    return _create_khan_academy_tool(api_key)


def get_all_education_tools(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Get all education-related search tools (cached)."""
    cache_manager = get_cache_manager()
    
    tool_factories = {
        "coursera": lambda: _create_coursera_tool(api_key),
        "edx": lambda: _create_edx_tool(api_key),
        "khan_academy": lambda: _create_khan_academy_tool(api_key)
    }
    
    return cache_manager.get_all_tools_in_category(
        EDUCATION_CACHE_NAME, tool_factories, api_key
    )


def create_education_tools_from_config(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Create education tools from configuration (always creates new instances)."""
    return create_tools_from_config(EDUCATION_SEARCH_CONFIGS, api_key)


def clear_education_cache():
    """Clear the education tools cache."""
    from .tool_cache_manager import clear_cache
    clear_cache(EDUCATION_CACHE_NAME)
