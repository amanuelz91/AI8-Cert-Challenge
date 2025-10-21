"""
Student loan specific search tools using the Tavily factory pattern.
This module contains domain-specific implementations for student loan searches.
"""

from typing import Dict, Any, Optional
from ..tavily.tavily_tool_builder import create_tools_from_config
from ..infrastructure.tool_cache_manager import cached_tool
from ..infrastructure.tool_registry import register_tool_category


# Configuration templates for student loan tools
STUDENT_LOAN_SEARCH_CONFIGS = {
    "studentaid": {
        "name": "StudentAid.gov",
        "description": "Search only official resources on StudentAid.gov to retrieve authoritative federal information about FAFSA applications, loan forgiveness programs, repayment plans, and eligibility requirements.",
        "domains": ["studentaid.gov"],
        "max_results": 3,
        "search_depth": "advanced",
        "include_answer": True
    }
}

# Cache category name for student loan tools
STUDENT_LOAN_CACHE_NAME = "student_loan"


def _create_studentaid_tool(api_key: Optional[str] = None) -> Any:
    """Internal function to create StudentAid tool."""
    tools = create_tools_from_config(STUDENT_LOAN_SEARCH_CONFIGS, api_key)
    return tools["studentaid"]


@cached_tool(STUDENT_LOAN_CACHE_NAME, "studentaid")
def get_studentaid_search_tool(api_key: Optional[str] = None) -> Any:
    """Get StudentAid.gov search tool (cached)."""
    return _create_studentaid_tool(api_key)


def create_student_loan_tools_from_config(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Create student loan tools from configuration (always creates new instances)."""
    return create_tools_from_config(STUDENT_LOAN_SEARCH_CONFIGS, api_key)


def clear_student_loan_cache():
    """Clear the student loan tools cache."""
    from ..infrastructure.tool_cache_manager import clear_cache
    clear_cache(STUDENT_LOAN_CACHE_NAME)


# Register student loan tools with the global registry
def _register_student_loan_tools():
    """Register student loan tools with the global registry."""
    tools = {
        "studentaid_search": get_studentaid_search_tool
    }
    
    descriptions = {
        "studentaid_search": "Search StudentAid.gov for official federal student aid information including FAFSA, loan forgiveness, and repayment plans"
    }
    
    metadata = {
        "studentaid_search": {
            "domains": ["studentaid.gov"],
            "category": "student_loan",
            "official_source": True,
            "max_results": 3
        }
    }
    
    register_tool_category(
        category_name="student_loan",
        tools=tools,
        descriptions=descriptions,
        metadata=metadata
    )


# Auto-register tools when module is imported
_register_student_loan_tools()