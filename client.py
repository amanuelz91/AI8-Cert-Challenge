"""
Client script for testing the RAG API.

Demonstrates how to interact with the FastAPI endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, List


class RAGAPIClient:
    """Client for interacting with the RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the RAG API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def get_available_methods(self) -> Dict[str, Any]:
        """Get available retrieval methods."""
        response = self.session.get(f"{self.base_url}/methods")
        response.raise_for_status()
        return response.json()
    
    def query(
        self, 
        question: str, 
        method: str = "production", 
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask
            method: Retrieval method to use
            include_confidence: Whether to include confidence scoring
            
        Returns:
            Query response
        """
        payload = {
            "question": question,
            "method": method,
            "include_confidence": include_confidence
        }
        
        response = self.session.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_query(
        self, 
        questions: List[str], 
        method: str = "production", 
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions to ask
            method: Retrieval method to use
            include_confidence: Whether to include confidence scoring
            
        Returns:
            Batch query response
        """
        payload = {
            "questions": questions,
            "method": method,
            "include_confidence": include_confidence
        }
        
        response = self.session.post(f"{self.base_url}/query/batch", json=payload)
        response.raise_for_status()
        return response.json()
    
    def reload_system(self) -> Dict[str, Any]:
        """Reload the RAG system."""
        response = self.session.post(f"{self.base_url}/reload")
        response.raise_for_status()
        return response.json()


def test_api():
    """Test the RAG API with various queries."""
    client = RAGAPIClient()
    
    print("🚀 Testing RAG API")
    print("=" * 50)
    
    try:
        # Health check
        print("1. Health Check")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Components: {list(health['components'].keys())}")
        print()
        
        # System stats
        print("2. System Statistics")
        stats = client.get_stats()
        print(f"   Documents: {stats['documents']}")
        print(f"   Retrievers: {list(stats['retrievers'].keys())}")
        print()
        
        # Available methods
        print("3. Available Methods")
        methods = client.get_available_methods()
        for method in methods['available_methods']:
            print(f"   - {method['name']}: {method['description']}")
        print()
        
        # Test queries
        test_questions = [
            "What is FAFSA?",
            "How do I apply for federal student loans?",
            "What are the requirements for student aid?"
        ]
        
        print("4. Testing Individual Queries")
        for i, question in enumerate(test_questions, 1):
            print(f"   Query {i}: {question}")
            try:
                result = client.query(question, method="production")
                print(f"   Answer: {result['answer'][:100]}...")
                print(f"   Sources: {result['sources']}")
                if result['confidence']:
                    print(f"   Confidence: {result['confidence']}")
                print()
            except Exception as e:
                print(f"   Error: {str(e)}")
                print()
        
        # Test batch query
        print("5. Testing Batch Query")
        try:
            batch_result = client.batch_query(test_questions, method="production")
            print(f"   Total Questions: {batch_result['total_questions']}")
            print(f"   Successful: {batch_result['successful_queries']}")
            print(f"   Failed: {batch_result['failed_queries']}")
            print(f"   Processing Time: {batch_result['processing_time']:.2f}s")
            print()
        except Exception as e:
            print(f"   Error: {str(e)}")
            print()
        
        print("✅ API testing completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")


def interactive_mode():
    """Interactive mode for testing queries."""
    client = RAGAPIClient()
    
    print("🎯 Interactive RAG API Client")
    print("=" * 50)
    print("Type 'quit' to exit, 'methods' to see available methods")
    print()
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'methods':
                methods = client.get_available_methods()
                for method in methods['available_methods']:
                    print(f"   - {method['name']}: {method['description']}")
                print()
                continue
            elif not question:
                continue
            
            # Ask for method
            method = input("Method (production): ").strip() or "production"
            
            print("Processing...")
            start_time = time.time()
            
            result = client.query(question, method=method)
            
            processing_time = time.time() - start_time
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {result['sources']}")
            print(f"Method: {result['method']}")
            print(f"Processing Time: {processing_time:.2f}s")
            
            if result['confidence']:
                print(f"Confidence: {result['confidence']}")
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        test_api()
