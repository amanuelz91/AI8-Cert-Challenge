#!/usr/bin/env python3
"""
Debug script to check what's actually stored in the Qdrant vector store.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.vectorstore.qdrant_client import QdrantManager
from src.config.settings import get_config

def debug_vector_store():
    """Debug what's stored in the vector store."""
    print("🔍 Debugging Vector Store Contents")
    print("=" * 50)
    
    try:
        # Initialize Qdrant manager
        config = get_config()
        qdrant_manager = QdrantManager()
        
        # Get collection info
        info = qdrant_manager.get_collection_info()
        print(f"📊 Collection info: {info}")
        
        # Get a few sample points
        print("\n🔍 Getting sample points...")
        points = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )[0]
        
        print(f"📄 Found {len(points)} sample points:")
        for i, point in enumerate(points):
            print(f"\n  📄 Point {i+1}:")
            print(f"    ID: {point.id}")
            print(f"    Payload keys: {list(point.payload.keys())}")
            print(f"    Content field: {point.payload.get('content', 'NOT FOUND')[:100]}...")
            print(f"    Page content field: {point.payload.get('page_content', 'NOT FOUND')[:100]}...")
            print(f"    All payload: {point.payload}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = debug_vector_store()
    if success:
        print("\n🎉 Vector store debug completed!")
    else:
        print("\n💥 Vector store debug failed!")
        sys.exit(1)
