"""
High-level builder for Tavily search tools.
This module provides a tool-agnostic factory for creating search tools.
"""

from typing import Dict, Any, List
from .tavily_search_factory import TavilySearchFactory, SearchConfig, create_tavily_tool


class TavilyToolBuilder:
    """Tool-agnostic builder class for creating Tavily search tools."""
    
    def __init__(self, api_key: str = None):
        """Initialize the builder with Tavily API key."""
        self.factory = TavilySearchFactory(api_key=api_key)
    
    def create_tool(self, name: str, description: str, domains: List[str], **kwargs) -> Any:
        """
        Create a custom search tool with specified parameters.
        
        Args:
            name (str): Tool name
            description (str): Tool description
            domains (List[str]): List of domains to search
            **kwargs: Additional configuration parameters
            
        Returns:
            Any: Configured search tool function
        """
        config = SearchConfig(
            name=name,
            description=description,
            domains=domains,
            **kwargs
        )
        return self.factory.create_search_tool(config)
    
    def create_tools_from_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create multiple tools from a list of configuration dictionaries.
        
        Args:
            configs (List[Dict[str, Any]]): List of tool configurations
            
        Returns:
            Dict[str, Any]: Dictionary mapping tool names to tool functions
        """
        tools = {}
        for config in configs:
            tool_name = config["name"].lower().replace(" ", "_").replace("-", "_")
            tools[tool_name] = self.create_tool(**config)
        return tools


def create_tool(name: str, description: str, domains: List[str], **kwargs) -> Any:
    """
    Convenience function to create a single search tool.
    
    Args:
        name (str): Tool name
        description (str): Tool description
        domains (List[str]): List of domains to search
        **kwargs: Additional configuration parameters
        
    Returns:
        Any: Configured search tool function
    """
    return create_tavily_tool(
        name=name,
        description=description,
        domains=domains,
        **kwargs
    )


def create_tools_from_config(configs: Dict[str, Dict], api_key: str = None) -> Dict[str, Any]:
    """
    Create multiple tools from a configuration dictionary.
    
    Args:
        configs (Dict[str, Dict]): Dictionary of tool configurations
        api_key (str, optional): Tavily API key
        
    Returns:
        Dict[str, Any]: Dictionary of tool functions
    """
    builder = TavilyToolBuilder(api_key=api_key)
    tools = {}
    
    for tool_name, config in configs.items():
        tools[tool_name] = builder.create_tool(**config)
    
    return tools
