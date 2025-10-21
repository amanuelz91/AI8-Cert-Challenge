"""
Example script demonstrating the production RAG system.

Shows how to initialize and use the RAG system with different retrieval methods.
"""

import os
from dotenv import load_dotenv
from src.core.system import create_production_rag_system
from src.tools.tavily.tavily_search_factory import TavilySearchFactory, SearchConfig
from src.utils.logging import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


def main():
    """Main example function."""
    try:
        logger.info("🚀 Starting RAG system example")
        
        # Option 1: Initialize RAG system with automatic Tavily tool detection
        rag_system = create_production_rag_system()
        
        # Option 2: Initialize RAG system with explicit Tavily tool
        # tavily_factory = TavilySearchFactory()
        # search_tool = tavily_factory.create_search_tool(
        #     SearchConfig(
        #         name="Student Aid Search",
        #         description="Search for federal student aid information",
        #         domains=["studentaid.gov", "ed.gov", "fafsa.gov"],
        #         max_results=5,
        #         search_depth="advanced"
        #     )
        # )
        # rag_system = create_production_rag_system(search_tool=search_tool)
        
        # Get system statistics
        stats = rag_system.get_system_stats()
        logger.info(f"📊 System stats: {stats}")
        
        # Perform health check
        health = rag_system.health_check()
        logger.info(f"🏥 Health status: {health}")
        
        # Example queries
        test_questions = [
            "What is FAFSA?",
            "How do I apply for student loans?",
            "What are the requirements for federal student aid?",
            "How can I get help with my student loan payments?"
        ]
        
        # Test different retrieval methods
        methods = ["naive", "semantic", "production"]
        
        for method in methods:
            logger.info(f"🔍 Testing {method} retrieval method")
            
            for question in test_questions[:2]:  # Test first 2 questions
                try:
                    result = rag_system.query(question, method=method)
                    logger.info(f"✅ {method} - Q: {question[:30]}...")
                    logger.info(f"📝 Response: {result['response'][:100]}...")
                    
                    if 'confidence' in result:
                        logger.info(f"🎯 Confidence: {result['confidence']}")
                    
                except Exception as e:
                    logger.error(f"❌ {method} failed for '{question}': {str(e)}")
        
        logger.info("✅ RAG system example completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
