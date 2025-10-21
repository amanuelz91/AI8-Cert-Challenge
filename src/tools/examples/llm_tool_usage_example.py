"""
Example of how an LLM would discover and use tools through the registry system.
This demonstrates the unified interface for tool discovery and access.
"""

from ..infrastructure.tool_registry import (
    get_tool_registry, 
    get_tool, 
    get_all_tools, 
    list_available_tools, 
    search_tools,
    get_tool_categories,
    get_tool_info
)
from ..search.student_loan_tools import get_studentaid_search_tool  # This auto-registers


def example_llm_tool_discovery():
    """Example of how an LLM would discover available tools."""
    print("=== LLM Tool Discovery Example ===")
    
    # 1. List all available tools
    print("1. All available tools:")
    available_tools = list_available_tools()
    for tool in available_tools:
        print(f"   - {tool['name']} ({tool['category']}): {tool['description']}")
    
    # 2. Get tool categories
    print(f"\n2. Available categories: {get_tool_categories()}")
    
    # 3. Search for specific tools
    print("\n3. Searching for 'student' tools:")
    student_tools = search_tools("student")
    for tool in student_tools:
        print(f"   - {tool['name']}: {tool['description']}")
    
    # 4. Get detailed tool information
    print("\n4. Detailed tool information:")
    tool_info = get_tool_info("studentaid_search")
    print(f"   Tool: {tool_info['name']}")
    print(f"   Category: {tool_info['category']}")
    print(f"   Description: {tool_info['description']}")
    print(f"   Available: {tool_info['available']}")
    print(f"   Metadata: {tool_info['metadata']}")


def example_llm_tool_usage():
    """Example of how an LLM would use tools."""
    print("\n=== LLM Tool Usage Example ===")
    
    # 1. Get a specific tool
    print("1. Getting specific tool:")
    try:
        studentaid_tool = get_tool("studentaid_search")
        print(f"   Retrieved tool: {studentaid_tool.__name__}")
        print(f"   Tool description: {studentaid_tool.__doc__}")
    except KeyError as e:
        print(f"   Error: {e}")
    
    # 2. Get all tools
    print("\n2. Getting all tools:")
    all_tools = get_all_tools()
    print(f"   Retrieved {len(all_tools)} tools:")
    for tool_name, tool in all_tools.items():
        print(f"   - {tool_name}: {tool.__name__}")


def example_llm_workflow():
    """Example of a complete LLM workflow using the tool system."""
    print("\n=== LLM Workflow Example ===")
    
    # Simulate LLM decision making
    user_query = "I need help with student loan forgiveness"
    
    print(f"User query: '{user_query}'")
    print("\nLLM workflow:")
    
    # 1. Discover relevant tools
    print("1. Discovering relevant tools...")
    relevant_tools = search_tools("student loan")
    print(f"   Found {len(relevant_tools)} relevant tools")
    
    # 2. Get tool information
    print("2. Analyzing tool capabilities...")
    for tool_info in relevant_tools:
        print(f"   - {tool_info['name']}: {tool_info['description']}")
        print(f"     Domains: {tool_info['metadata'].get('domains', 'N/A')}")
        print(f"     Official source: {tool_info['metadata'].get('official_source', False)}")
    
    # 3. Select and use tool
    print("3. Selecting best tool...")
    if relevant_tools:
        best_tool_name = relevant_tools[0]['name']
        print(f"   Selected: {best_tool_name}")
        
        # Get the actual tool
        tool = get_tool(best_tool_name)
        print(f"   Tool retrieved: {tool.__name__}")
        
        # In real usage, LLM would call: result = tool("student loan forgiveness")
        print("   [In real usage, LLM would call tool with query]")


def example_adding_new_tool_category():
    """Example of how easy it is to add new tool categories."""
    print("\n=== Adding New Tool Category Example ===")
    
    # This shows how easy it is to add new tools
    print("To add a new tool category (e.g., 'news_tools'):")
    print("1. Create news_tools.py")
    print("2. Define tools with @cached_tool decorator")
    print("3. Register with registry:")
    print("""
    def _register_news_tools():
        tools = {
            "bbc_search": get_bbc_search_tool,
            "reuters_search": get_reuters_search_tool
        }
        descriptions = {
            "bbc_search": "Search BBC News for current events",
            "reuters_search": "Search Reuters for business news"
        }
        register_tool_category("news", tools, descriptions)
    
    _register_news_tools()  # Auto-register on import
    """)
    print("4. LLM automatically discovers new tools!")
    print("5. No changes needed to LLM code!")


def example_tool_registry_benefits():
    """Example showing benefits of the tool registry system."""
    print("\n=== Tool Registry Benefits ===")
    
    print("✅ Benefits for LLM:")
    print("  - Single interface to discover all tools")
    print("  - No need to know internal cache structure")
    print("  - Search and filter capabilities")
    print("  - Rich metadata for tool selection")
    print("  - Automatic tool discovery")
    
    print("\n✅ Benefits for Developers:")
    print("  - Easy to add new tool categories")
    print("  - Centralized tool management")
    print("  - Consistent tool interface")
    print("  - Rich metadata system")
    print("  - Auto-registration on import")
    
    print("\n✅ Benefits for System:")
    print("  - Hierarchical caching still works")
    print("  - Performance optimizations preserved")
    print("  - Thread-safe operations")
    print("  - API key management")
    print("  - Cache invalidation")


if __name__ == "__main__":
    # Run examples
    try:
        example_llm_tool_discovery()
        example_llm_tool_usage()
        example_llm_workflow()
        example_adding_new_tool_category()
        example_tool_registry_benefits()
        
        print("\n✅ All tool registry examples completed successfully!")
        print("\n🎯 Key Benefits:")
        print("  - LLM doesn't need to know cache structure")
        print("  - Unified tool discovery interface")
        print("  - Rich metadata for tool selection")
        print("  - Easy to add new tool categories")
        print("  - Automatic tool registration")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Note: This is expected if TAVILY_API_KEY is not set")
