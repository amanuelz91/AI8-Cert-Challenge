"""
Example usage of the hierarchical Tavily Search Factory pattern.
This demonstrates the umbrella cache system with multiple tool categories.
"""

import os
from ..tavily.tavily_search_factory import TavilySearchFactory, SearchConfig, create_tavily_tool
from ..tavily.tavily_tool_builder import TavilyToolBuilder, create_tool, create_tools_from_config
from ..infrastructure.tool_cache_manager import get_cache_manager, get_cache_info, clear_all_caches
from ..search.student_loan_tools import (
    get_studentaid_search_tool, 
    get_all_student_loan_tools,
    clear_student_loan_cache
)
from ..search.education_tools import (
    get_coursera_search_tool,
    get_edx_search_tool,
    get_khan_academy_search_tool,
    get_all_education_tools,
    clear_education_cache
)


def example_hierarchical_cache_system():
    """Example of the hierarchical cache system."""
    print("=== Hierarchical Cache System Example ===")
    
    # Show initial cache state
    print("Initial cache state:")
    print(get_cache_info())
    
    # Create tools from different categories
    print("\n1. Creating StudentAid tool...")
    studentaid_tool = get_studentaid_search_tool()
    print(f"   Created: {studentaid_tool.__name__}")
    
    print("\n2. Creating Coursera tool...")
    coursera_tool = get_coursera_search_tool()
    print(f"   Created: {coursera_tool.__name__}")
    
    print("\n3. Creating edX tool...")
    edx_tool = get_edx_search_tool()
    print(f"   Created: {edx_tool.__name__}")
    
    # Show cache state after creating tools
    print("\nCache state after creating tools:")
    cache_info = get_cache_info()
    for category, info in cache_info.items():
        print(f"  {category}: {info['total_tools']} tools, keys: {info['cache_keys']}")
    
    # Demonstrate caching (should be instant)
    print("\n4. Getting cached tools (should be instant)...")
    cached_studentaid = get_studentaid_search_tool()
    cached_coursera = get_coursera_search_tool()
    print(f"   Cached StudentAid: {cached_studentaid.__name__}")
    print(f"   Cached Coursera: {cached_coursera.__name__}")
    
    # Get all tools from categories
    print("\n5. Getting all tools from categories...")
    all_student_tools = get_all_student_loan_tools()
    all_education_tools = get_all_education_tools()
    print(f"   Student loan tools: {list(all_student_tools.keys())}")
    print(f"   Education tools: {list(all_education_tools.keys())}")
    
    # Show final cache state
    print("\nFinal cache state:")
    final_cache_info = get_cache_info()
    for category, info in final_cache_info.items():
        print(f"  {category}: {info['total_tools']} tools")


def example_cache_management():
    """Example of cache management operations."""
    print("\n=== Cache Management Example ===")
    
    # Clear specific cache
    print("1. Clearing student loan cache...")
    clear_student_loan_cache()
    print("   Student loan cache cleared")
    
    print("\nCache state after clearing student loan cache:")
    cache_info = get_cache_info()
    for category, info in cache_info.items():
        print(f"  {category}: {info['total_tools']} tools")
    
    # Clear all caches
    print("\n2. Clearing all caches...")
    clear_all_caches()
    print("   All caches cleared")
    
    print("\nCache state after clearing all caches:")
    cache_info = get_cache_info()
    if cache_info:
        for category, info in cache_info.items():
            print(f"  {category}: {info['total_tools']} tools")
    else:
        print("  No caches active")


def example_api_key_handling():
    """Example of API key handling in cache system."""
    print("\n=== API Key Handling Example ===")
    
    # Create tools with different API keys
    print("1. Creating tools with different API keys...")
    tool1 = get_studentaid_search_tool(api_key="key1")
    tool2 = get_studentaid_search_tool(api_key="key2")
    tool3 = get_studentaid_search_tool()  # Default key
    
    print(f"   Tool 1 (key1): {tool1.__name__}")
    print(f"   Tool 2 (key2): {tool2.__name__}")
    print(f"   Tool 3 (default): {tool3.__name__}")
    
    # Show cache state with multiple API keys
    print("\nCache state with multiple API keys:")
    cache_info = get_cache_info()
    for category, info in cache_info.items():
        print(f"  {category}: {info['total_tools']} tools, keys: {info['cache_keys']}")


def example_adding_new_tool_category():
    """Example of how easy it is to add new tool categories."""
    print("\n=== Adding New Tool Category Example ===")
    
    # This shows how easy it is to add new tool categories
    print("To add a new tool category (e.g., 'news_tools'):")
    print("1. Create news_tools.py with:")
    print("   - NEWS_SEARCH_CONFIGS")
    print("   - NEWS_CACHE_NAME = 'news'")
    print("   - Individual tool functions with @cached_tool decorator")
    print("   - get_all_news_tools() function")
    print("   - clear_news_cache() function")
    print("\n2. The hierarchical cache system automatically handles:")
    print("   - Separate cache compartments")
    print("   - Thread-safe operations")
    print("   - API key management")
    print("   - Cache invalidation")


def example_performance_benefits():
    """Example showing performance benefits of the cache system."""
    print("\n=== Performance Benefits Example ===")
    
    import time
    
    # First call (creates tool)
    start_time = time.time()
    tool1 = get_studentaid_search_tool()
    first_call_time = time.time() - start_time
    
    # Second call (uses cache)
    start_time = time.time()
    tool2 = get_studentaid_search_tool()
    second_call_time = time.time() - start_time
    
    print(f"First call (creation): {first_call_time:.6f} seconds")
    print(f"Second call (cached): {second_call_time:.6f} seconds")
    print(f"Performance improvement: {first_call_time/second_call_time:.1f}x faster")


if __name__ == "__main__":
    # Run examples (without actual API calls)
    try:
        example_hierarchical_cache_system()
        example_cache_management()
        example_api_key_handling()
        example_adding_new_tool_category()
        example_performance_benefits()
        
        print("\n✅ All hierarchical cache examples completed successfully!")
        print("\n🎯 Key Benefits of Hierarchical Cache System:")
        print("  - Umbrella cache manages multiple tool categories")
        print("  - Each category has its own cache compartment")
        print("  - Thread-safe operations")
        print("  - API key-aware caching")
        print("  - Easy to add new tool categories")
        print("  - Granular cache management")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Note: This is expected if TAVILY_API_KEY is not set")
