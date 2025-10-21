"""
Streaming client for testing RAG API streaming endpoints.

Demonstrates how to consume streaming responses from the API.
"""

import requests
import json
import time
from typing import Iterator, Dict, Any


class StreamingRAGClient:
    """Client for consuming streaming RAG responses."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the streaming client.
        
        Args:
            base_url: Base URL of the RAG API
        """
        self.base_url = base_url.rstrip('/')
    
    def stream_query(
        self, 
        question: str, 
        method: str = "production",
        include_confidence: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream a query response.
        
        Args:
            question: Question to ask
            method: Retrieval method
            include_confidence: Whether to include confidence
            
        Yields:
            Stream chunks
        """
        payload = {
            "question": question,
            "method": method,
            "include_confidence": include_confidence
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/stream/query",
                json=payload,
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                        yield chunk_data
                        
                        # Break on error or end
                        if chunk_data.get("chunk_type") in ["error", "end"]:
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            yield {
                "chunk_type": "error",
                "error": f"Request failed: {str(e)}",
                "timestamp": str(time.time())
            }
    
    def stream_query_with_llm(
        self, 
        question: str, 
        method: str = "production",
        include_confidence: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream a query response with real LLM streaming.
        
        Args:
            question: Question to ask
            method: Retrieval method
            include_confidence: Whether to include confidence
            
        Yields:
            Stream chunks
        """
        payload = {
            "question": question,
            "method": method,
            "include_confidence": include_confidence
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/stream/query/llm",
                json=payload,
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                        yield chunk_data
                        
                        # Break on error or end
                        if chunk_data.get("chunk_type") in ["error", "end"]:
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            yield {
                "chunk_type": "error",
                "error": f"Request failed: {str(e)}",
                "timestamp": str(time.time())
            }
    
    def stream_query_simple(self, question: str, method: str = "production") -> Iterator[Dict[str, Any]]:
        """
        Simple streaming query for testing.
        
        Args:
            question: Question to ask
            method: Retrieval method
            
        Yields:
            Stream chunks
        """
        try:
            response = requests.get(
                f"{self.base_url}/stream/query/simple",
                params={"question": question, "method": method},
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                        yield chunk_data
                        
                        # Break on error or end
                        if chunk_data.get("chunk_type") in ["error", "end"]:
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            yield {
                "chunk_type": "error",
                "error": f"Request failed: {str(e)}",
                "timestamp": str(time.time())
            }


def test_streaming():
    """Test streaming functionality."""
    client = StreamingRAGClient()
    
    print("🌊 Testing RAG Streaming API")
    print("=" * 50)
    
    test_questions = [
        "What is FAFSA?",
        "How do I apply for federal student loans?",
        "What are the requirements for student aid?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing stream for: {question}")
        print("-" * 40)
        
        try:
            full_response = ""
            chunk_count = 0
            
            for chunk in client.stream_query(question, method="production"):
                chunk_type = chunk.get("chunk_type", "unknown")
                
                if chunk_type == "start":
                    print(f"🚀 Started streaming for method: {chunk.get('method')}")
                    
                elif chunk_type == "content":
                    content = chunk.get("content", "")
                    full_response += content
                    chunk_count += 1
                    print(f"📝 Chunk {chunk_count}: {content}", end="", flush=True)
                    
                elif chunk_type == "end":
                    print(f"\n✅ Stream completed!")
                    print(f"   Total chunks: {chunk.get('total_chunks', 0)}")
                    print(f"   Sources: {chunk.get('sources', 0)}")
                    if chunk.get('confidence'):
                        print(f"   Confidence: {chunk.get('confidence')}")
                    
                elif chunk_type == "error":
                    print(f"\n❌ Stream error: {chunk.get('error')}")
                    break
                    
        except Exception as e:
            print(f"❌ Stream test failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Streaming tests completed!")


def test_llm_streaming():
    """Test LLM streaming functionality."""
    client = StreamingRAGClient()
    
    print("🤖 Testing LLM Streaming")
    print("=" * 50)
    
    question = "Explain the FAFSA application process step by step."
    print(f"Question: {question}")
    print("-" * 40)
    
    try:
        full_response = ""
        chunk_count = 0
        
        for chunk in client.stream_query_with_llm(question, method="production"):
            chunk_type = chunk.get("chunk_type", "unknown")
            
            if chunk_type == "start":
                print(f"🚀 Started LLM streaming for method: {chunk.get('method')}")
                
            elif chunk_type == "content":
                content = chunk.get("content", "")
                full_response += content
                chunk_count += 1
                print(f"🤖 LLM Chunk {chunk_count}: {content}", end="", flush=True)
                
            elif chunk_type == "end":
                print(f"\n✅ LLM Stream completed!")
                print(f"   Total chunks: {chunk.get('total_chunks', 0)}")
                print(f"   Sources: {chunk.get('sources', 0)}")
                print(f"   Response length: {len(full_response)}")
                
            elif chunk_type == "error":
                print(f"\n❌ LLM Stream error: {chunk.get('error')}")
                break
                
    except Exception as e:
        print(f"❌ LLM stream test failed: {str(e)}")


def interactive_streaming():
    """Interactive streaming mode."""
    client = StreamingRAGClient()
    
    print("🎯 Interactive Streaming RAG Client")
    print("=" * 50)
    print("Type 'quit' to exit, 'llm' for LLM streaming")
    print()
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'llm':
                test_llm_streaming()
                continue
            elif not question:
                continue
            
            # Ask for method
            method = input("Method (production): ").strip() or "production"
            
            print("Streaming...")
            print("-" * 40)
            
            full_response = ""
            chunk_count = 0
            
            for chunk in client.stream_query(question, method=method):
                chunk_type = chunk.get("chunk_type", "unknown")
                
                if chunk_type == "start":
                    print(f"🚀 Started streaming for method: {chunk.get('method')}")
                    
                elif chunk_type == "content":
                    content = chunk.get("content", "")
                    full_response += content
                    chunk_count += 1
                    print(f"📝 {content}", end="", flush=True)
                    
                elif chunk_type == "end":
                    print(f"\n✅ Stream completed!")
                    print(f"   Total chunks: {chunk.get('total_chunks', 0)}")
                    print(f"   Sources: {chunk.get('sources', 0)}")
                    break
                    
                elif chunk_type == "error":
                    print(f"\n❌ Stream error: {chunk.get('error')}")
                    break
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_streaming()
        elif sys.argv[1] == "llm":
            test_llm_streaming()
        else:
            test_streaming()
    else:
        test_streaming()
