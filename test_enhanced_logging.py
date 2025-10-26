#!/usr/bin/env python3
"""
Test script to verify enhanced logging in RAG system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.system.rag_system import create_production_rag_system
from src.utils.logging import setup_logging

def test_enhanced_logging():
    """Test the enhanced logging functionality."""
    print("🧪 Testing Enhanced RAG Logging")
    print("=" * 50)
    
    # Set up logging to see all the detailed logs
    logger = setup_logging("test_rag_logging", level="INFO", log_file="rag_test.log")
    
    try:
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag_system = create_production_rag_system()
        
        # Test query with detailed logging
        print("\n🔍 Testing query with enhanced logging...")
        test_question = "What are the requirements for student loans?"
        
        result = rag_system.query(
            question=test_question,
            method="naive",
            include_confidence=True
        )
        
        print(f"\n✅ Query completed!")
        print(f"📊 Result keys: {list(result.keys())}")
        if 'response' in result:
            print(f"📝 Response preview: {result['response'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_enhanced_logging()
    if success:
        print("\n🎉 Enhanced logging test completed successfully!")
    else:
        print("\n💥 Enhanced logging test failed!")
        sys.exit(1)
