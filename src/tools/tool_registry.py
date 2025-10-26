"""
Tool registry and discovery system for LLM access.
This module provides a unified interface for LLMs to discover and access all available tools.
"""

from typing import Dict, Any, List, Optional, Callable
from .tool_cache_manager import get_cache_manager


class ToolRegistry:
    """Registry for all available tools across categories."""
    
    def __init__(self):
        self._tool_categories: Dict[str, Dict[str, Callable]] = {}
        self._tool_descriptions: Dict[str, str] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_category(
        self, 
        category_name: str, 
        tools: Dict[str, Callable], 
        descriptions: Dict[str, str],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Register a category of tools.
        
        Args:
            category_name (str): Name of the tool category
            tools (Dict[str, Callable]): Dictionary of tool names to tool functions
            descriptions (Dict[str, str]): Dictionary of tool descriptions
            metadata (Optional[Dict[str, Dict[str, Any]]]): Additional tool metadata
        """
        self._tool_categories[category_name] = tools
        self._tool_descriptions.update(descriptions)
        if metadata:
            self._tool_metadata.update(metadata)
    
    def get_tool(self, tool_name: str, api_key: Optional[str] = None) -> Any:
        """
        Get a specific tool by name.
        
        Args:
            tool_name (str): Name of the tool to retrieve
            api_key (Optional[str]): API key for tool creation
            
        Returns:
            Any: The requested tool
            
        Raises:
            KeyError: If tool is not found
        """
        for category_name, tools in self._tool_categories.items():
            if tool_name in tools:
                return tools[tool_name](api_key)
        
        raise KeyError(f"Tool '{tool_name}' not found in registry")
    
    def get_all_tools(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all available tools.
        
        Args:
            api_key (Optional[str]): API key for tool creation
            
        Returns:
            Dict[str, Any]: Dictionary of all tools
        """
        all_tools = {}
        for category_name, tools in self._tool_categories.items():
            for tool_name, tool_func in tools.items():
                all_tools[tool_name] = tool_func(api_key)
        return all_tools
    
    def get_tools_by_category(self, category_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all tools in a specific category.
        
        Args:
            category_name (str): Name of the category
            api_key (Optional[str]): API key for tool creation
            
        Returns:
            Dict[str, Any]: Dictionary of tools in the category
        """
        if category_name not in self._tool_categories:
            raise KeyError(f"Category '{category_name}' not found in registry")
        
        tools = {}
        for tool_name, tool_func in self._tool_categories[category_name].items():
            tools[tool_name] = tool_func(api_key)
        return tools
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools with their descriptions.
        
        Returns:
            List[Dict[str, Any]]: List of tool information
        """
        tools_info = []
        for category_name, tools in self._tool_categories.items():
            for tool_name in tools.keys():
                tool_info = {
                    "name": tool_name,
                    "category": category_name,
                    "description": self._tool_descriptions.get(tool_name, "No description available"),
                    "metadata": self._tool_metadata.get(tool_name, {})
                }
                tools_info.append(tool_info)
        return tools_info
    
    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tools by name or description.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict[str, Any]]: List of matching tools
        """
        query_lower = query.lower()
        matching_tools = []
        
        for tool_info in self.list_available_tools():
            if (query_lower in tool_info["name"].lower() or 
                query_lower in tool_info["description"].lower()):
                matching_tools.append(tool_info)
        
        return matching_tools
    
    def get_tool_categories(self) -> List[str]:
        """Get list of all available tool categories."""
        return list(self._tool_categories.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            Dict[str, Any]: Tool information
        """
        for category_name, tools in self._tool_categories.items():
            if tool_name in tools:
                return {
                    "name": tool_name,
                    "category": category_name,
                    "description": self._tool_descriptions.get(tool_name, "No description available"),
                    "metadata": self._tool_metadata.get(tool_name, {}),
                    "available": True
                }
        
        return {
            "name": tool_name,
            "category": None,
            "description": "Tool not found",
            "metadata": {},
            "available": False
        }


# Global tool registry instance
_tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _tool_registry


def register_tool_category(
    category_name: str,
    tools: Dict[str, Callable],
    descriptions: Dict[str, str],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Register a tool category in the global registry."""
    _tool_registry.register_category(category_name, tools, descriptions, metadata)


def get_tool(tool_name: str, api_key: Optional[str] = None) -> Any:
    """Get a tool by name from the global registry."""
    return _tool_registry.get_tool(tool_name, api_key)


def get_all_tools(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Get all tools from the global registry."""
    return _tool_registry.get_all_tools(api_key)


def list_available_tools() -> List[Dict[str, Any]]:
    """List all available tools."""
    return _tool_registry.list_available_tools()


def search_tools(query: str) -> List[Dict[str, Any]]:
    """Search for tools by name or description."""
    return _tool_registry.search_tools(query)


def get_tool_categories() -> List[str]:
    """Get all available tool categories."""
    return _tool_registry.get_tool_categories()


def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific tool."""
    return _tool_registry.get_tool_info(tool_name)
