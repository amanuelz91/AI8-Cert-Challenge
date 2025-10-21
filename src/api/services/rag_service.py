"""
RAG service for API operations.

Handles RAG system operations and business logic.
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from src.core.system import ProductionRAGSystem
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RAGService:
    """Service class for RAG operations."""
    
    def __init__(self, rag_system: Optional[ProductionRAGSystem] = None):
        """
        Initialize RAG service.
        
        Args:
            rag_system: RAG system instance
        """
        self.rag_system = rag_system
        logger.info("🔧 RAG service initialized")
    
    def set_rag_system(self, rag_system: ProductionRAGSystem) -> None:
        """Set the RAG system instance."""
        self.rag_system = rag_system
        logger.info("✅ RAG system set in service")
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self.rag_system is not None
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        if not self.is_ready():
            return {
                "overall": "unhealthy",
                "components": {"rag_system": False},
                "timestamp": str(datetime.now()),
                "error": "RAG system not initialized"
            }
        
        try:
            return self.rag_system.health_check()
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return {
                "overall": "unhealthy",
                "components": {"rag_system": False},
                "timestamp": str(datetime.now()),
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.is_ready():
            raise RuntimeError("RAG system not initialized")
        
        return self.rag_system.get_system_stats()
    
    def query(
        self,
        question: str,
        method: str = "production",
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single query.
        
        Args:
            question: Question to ask
            method: Retrieval method
            include_confidence: Whether to include confidence
            
        Returns:
            Query result
        """
        if not self.is_ready():
            raise RuntimeError("RAG system not initialized")
        
        logger.info(f"❓ Processing query: {question[:50]}...")
        
        result = self.rag_system.query(
            question=question,
            method=method,
            include_confidence=include_confidence
        )
        
        # Extract and format response data
        answer = result.get("response", "No response generated")
        confidence = result.get("confidence")
        metadata = result.get("metadata", {})
        
        # Count sources
        sources = self._count_sources(result)
        
        return {
            "answer": answer,
            "method": method,
            "confidence": confidence,
            "sources": sources,
            "metadata": metadata,
            "timestamp": str(datetime.now())
        }
    
    def batch_query(
        self,
        questions: List[str],
        method: str = "production",
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions
            method: Retrieval method
            include_confidence: Whether to include confidence
            
        Returns:
            Batch query results
        """
        if not self.is_ready():
            raise RuntimeError("RAG system not initialized")
        
        logger.info(f"📦 Processing batch of {len(questions)} queries")
        
        start_time = datetime.now()
        results = []
        successful = 0
        failed = 0
        
        for question in questions:
            try:
                result = self.query(question, method, include_confidence)
                results.append(result)
                successful += 1
            except Exception as e:
                logger.error(f"❌ Batch query failed for '{question}': {str(e)}")
                failed += 1
                
                # Add error result
                results.append({
                    "answer": f"Error processing query: {str(e)}",
                    "method": method,
                    "confidence": None,
                    "sources": 0,
                    "metadata": {"error": str(e)},
                    "timestamp": str(datetime.now())
                })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Batch processing complete: {successful} successful, {failed} failed")
        
        return {
            "results": results,
            "total_questions": len(questions),
            "successful_queries": successful,
            "failed_queries": failed,
            "processing_time": processing_time
        }
    
    def get_available_methods(self) -> Dict[str, Any]:
        """Get available retrieval methods."""
        return {
            "available_methods": [
                {
                    "name": "naive",
                    "description": "Fast cosine similarity search",
                    "use_case": "General purpose queries"
                },
                {
                    "name": "semantic", 
                    "description": "Semantic chunking for better boundaries",
                    "use_case": "Complex queries requiring semantic understanding"
                },
                {
                    "name": "tool",
                    "description": "Real-time web search (Tavily)",
                    "use_case": "Questions requiring current information"
                },
                {
                    "name": "hybrid",
                    "description": "Knowledge base + web search",
                    "use_case": "Comprehensive coverage"
                },
                {
                    "name": "production",
                    "description": "All three methods combined (recommended)",
                    "use_case": "Best overall performance"
                }
            ]
        }
    
    def reload_system(self) -> Dict[str, Any]:
        """Reload the RAG system."""
        logger.info("🔄 Reloading RAG system")
        
        try:
            from src.core.system import create_production_rag_system
            self.rag_system = create_production_rag_system()
            
            logger.info("✅ RAG system reloaded successfully")
            
            return {
                "message": "RAG system reloaded successfully",
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            logger.error(f"❌ Failed to reload RAG system: {str(e)}")
            raise RuntimeError(f"Failed to reload system: {str(e)}")
    
    def _count_sources(self, result: Dict[str, Any]) -> int:
        """Count sources from query result."""
        sources = 0
        
        if "contexts" in result:
            contexts = result["contexts"]
            if isinstance(contexts, dict):
                sources = sum(len(ctx) if isinstance(ctx, list) else 0 for ctx in contexts.values())
            elif isinstance(contexts, list):
                sources = len(contexts)
        
        return sources
    
    async def stream_query(
        self,
        question: str,
        method: str = "production",
        include_confidence: bool = True,
        chunk_size: int = 50
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a RAG query response.
        
        Args:
            question: Question to ask
            method: Retrieval method
            include_confidence: Whether to include confidence
            chunk_size: Size of content chunks
            
        Yields:
            Stream chunks
        """
        if not self.is_ready():
            yield {
                "chunk_type": "error",
                "error": "RAG system not initialized",
                "timestamp": str(datetime.now())
            }
            return
        
        try:
            logger.info(f"🌊 Starting stream for query: {question[:50]}...")
            
            # Send start event
            yield {
                "chunk_type": "start",
                "question": question,
                "method": method,
                "metadata": {"include_confidence": include_confidence},
                "timestamp": str(datetime.now())
            }
            
            # Process the query
            result = self.rag_system.query(
                question=question,
                method=method,
                include_confidence=include_confidence
            )
            
            # Extract response data
            answer = result.get("response", "No response generated")
            confidence = result.get("confidence")
            metadata = result.get("metadata", {})
            sources = self._count_sources(result)
            
            # Stream content in chunks
            chunk_index = 0
            for i in range(0, len(answer), chunk_size):
                chunk_content = answer[i:i + chunk_size]
                
                yield {
                    "chunk_type": "content",
                    "content": chunk_content,
                    "chunk_index": chunk_index,
                    "metadata": {"total_length": len(answer)},
                    "timestamp": str(datetime.now())
                }
                
                chunk_index += 1
                
                # Small delay to simulate streaming
                import asyncio
                await asyncio.sleep(0.05)
            
            # Send end event
            yield {
                "chunk_type": "end",
                "total_chunks": chunk_index,
                "sources": sources,
                "confidence": confidence,
                "metadata": metadata,
                "timestamp": str(datetime.now())
            }
            
            logger.info(f"✅ Stream completed for query: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Stream failed for query '{question[:50]}...': {str(e)}")
            yield {
                "chunk_type": "error",
                "error": str(e),
                "error_code": "STREAM_ERROR",
                "metadata": {"question": question, "method": method},
                "timestamp": str(datetime.now())
            }
    
    async def stream_query_with_llm(
        self,
        question: str,
        method: str = "production",
        include_confidence: bool = True,
        chunk_size: int = 50
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a RAG query with real LLM streaming.
        
        Args:
            question: Question to ask
            method: Retrieval method
            include_confidence: Whether to include confidence
            chunk_size: Size of content chunks
            
        Yields:
            Stream chunks
        """
        if not self.is_ready():
            yield {
                "chunk_type": "error",
                "error": "RAG system not initialized",
                "timestamp": str(datetime.now())
            }
            return
        
        try:
            logger.info(f"🌊 Starting LLM stream for query: {question[:50]}...")
            
            # Send start event
            yield {
                "chunk_type": "start",
                "question": question,
                "method": method,
                "metadata": {"include_confidence": include_confidence, "streaming": "llm"},
                "timestamp": str(datetime.now())
            }
            
            # Get retrieval results first
            retrieval_result = self.rag_system._retrieve_documents(
                question=question,
                method=method
            )
            
            # Count sources
            sources = self._count_sources({"contexts": retrieval_result})
            
            # Stream LLM response
            chunk_index = 0
            full_response = ""
            
            # Use the LLM's streaming capability if available
            if hasattr(self.rag_system.llm, 'astream'):
                async for chunk in self.rag_system.llm.astream(
                    self._format_prompt(question, retrieval_result)
                ):
                    if hasattr(chunk, 'content') and chunk.content:
                        content_chunk = chunk.content
                        full_response += content_chunk
                        
                        yield {
                            "chunk_type": "content",
                            "content": content_chunk,
                            "chunk_index": chunk_index,
                            "metadata": {"llm_chunk": True},
                            "timestamp": str(datetime.now())
                        }
                        
                        chunk_index += 1
                        
                        # Small delay for better streaming experience
                        import asyncio
                        await asyncio.sleep(0.02)
            else:
                # Fallback to regular query and chunk the response
                result = self.rag_system.query(
                    question=question,
                    method=method,
                    include_confidence=include_confidence
                )
                
                answer = result.get("response", "No response generated")
                full_response = answer
                
                # Stream content in chunks
                for i in range(0, len(answer), chunk_size):
                    chunk_content = answer[i:i + chunk_size]
                    
                    yield {
                        "chunk_type": "content",
                        "content": chunk_content,
                        "chunk_index": chunk_index,
                        "metadata": {"total_length": len(answer)},
                        "timestamp": str(datetime.now())
                    }
                    
                    chunk_index += 1
                    
                    # Small delay to simulate streaming
                    import asyncio
                    await asyncio.sleep(0.05)
            
            # Send end event
            yield {
                "chunk_type": "end",
                "total_chunks": chunk_index,
                "sources": sources,
                "confidence": result.get("confidence") if 'result' in locals() else None,
                "metadata": {"full_response_length": len(full_response)},
                "timestamp": str(datetime.now())
            }
            
            logger.info(f"✅ LLM stream completed for query: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ LLM stream failed for query '{question[:50]}...': {str(e)}")
            yield {
                "chunk_type": "error",
                "error": str(e),
                "error_code": "LLM_STREAM_ERROR",
                "metadata": {"question": question, "method": method},
                "timestamp": str(datetime.now())
            }
    
    def _format_prompt(self, question: str, contexts: Dict[str, Any]) -> str:
        """Format prompt for LLM streaming."""
        context_str = ""
        if isinstance(contexts, dict):
            for method, docs in contexts.items():
                if docs:
                    context_str += f"\n\n{method.upper()} Results:\n"
                    for doc in docs:
                        context_str += f"- {doc.page_content[:200]}...\n"
        elif isinstance(contexts, list):
            context_str = "\n\n".join([doc.page_content for doc in contexts])
        
        prompt = f"""You are a helpful assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Question: {question}

Context: {context_str}

Answer:"""
        
        return prompt


# Global service instance
rag_service = RAGService()
