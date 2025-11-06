"""
RAG service for API operations.

Handles RAG system operations and business logic.
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import asyncio
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
import cohere
from src.core.system import ProductionRAGSystem
from src.utils.logging import get_logger
from src.config.settings import get_config

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
        
        # Check for contexts key (old format)
        if "contexts" in result:
            contexts = result["contexts"]
            if isinstance(contexts, dict):
                sources = sum(len(ctx) if isinstance(ctx, list) else 0 for ctx in contexts.values())
            elif isinstance(contexts, list):
                sources = len(contexts)
        else:
            # Check for workflow format keys
            for key in ["naive_context", "semantic_context", "tool_context", "parent_context", "bm25_context", "combined_context"]:
                if key in result and isinstance(result[key], list):
                    sources += len(result[key])
        
        return sources
    
    def _extract_document_chunks(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and format document chunks from query result.
        
        Args:
            result: Query result dictionary
            
        Returns:
            List of formatted document chunks
        """
        chunks = []
        seen_ids = set()
        
        # Process combined_context first (most relevant)
        if "combined_context" in result and isinstance(result["combined_context"], list):
            for i, doc in enumerate(result["combined_context"]):
                # Check if doc has page_content - if not, try metadata['content']
                # Qdrant sometimes stores content in metadata['content'] instead of page_content
                content = None
                if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                    content = doc.page_content
                elif hasattr(doc, 'metadata') and doc.metadata:
                    # Try to get content from metadata (Qdrant stores it here sometimes)
                    content = doc.metadata.get('content', '')
                
                if not content or content.strip() == "":
                    logger.warning(f"⚠️ Document {i} has no content (page_content or metadata['content']). Metadata keys: {list(doc.metadata.keys()) if hasattr(doc, 'metadata') and doc.metadata else 'No metadata'}")
                    continue  # Skip documents without content
                
                content = content[:500]  # Truncate for display
                
                doc_id = doc.metadata.get("_id") or doc.metadata.get("point_id") if hasattr(doc, 'metadata') and doc.metadata else None
                if not doc_id:
                    # Use content hash as fallback ID
                    doc_id = hash(content[:100]) if content else None
                
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    chunks.append({
                        "content": content,
                        "metadata": {
                            "id": doc_id,
                            "page": doc.metadata.get("page") if hasattr(doc, 'metadata') and doc.metadata else None,
                            "source": (doc.metadata.get("source") or doc.metadata.get("file_path")) if hasattr(doc, 'metadata') and doc.metadata else None,
                            "document_id": doc.metadata.get("document_id") if hasattr(doc, 'metadata') and doc.metadata else None,
                            "retrieval_method": doc.metadata.get("retriever_method", "combined") if hasattr(doc, 'metadata') and doc.metadata else "combined"
                        }
                    })
        
        # Also include naive_context, semantic_context, parent_context, and bm25_context if not already included
        for context_key in ["naive_context", "semantic_context", "parent_context", "bm25_context"]:
            if context_key in result and isinstance(result[context_key], list):
                for doc in result[context_key]:
                    # Check if doc has page_content - if not, try metadata['content']
                    content = None
                    if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                        content = doc.page_content
                    elif hasattr(doc, 'metadata') and doc.metadata:
                        # Try to get content from metadata (Qdrant stores it here sometimes)
                        content = doc.metadata.get('content', '')
                    
                    if not content or content.strip() == "":
                        continue  # Skip documents without content
                    
                    content = content[:500]  # Truncate for display
                    
                    # Parent document retrievers may not have _id, use content hash as fallback
                    doc_id = doc.metadata.get("_id") or doc.metadata.get("point_id") if hasattr(doc, 'metadata') and doc.metadata else None
                    if not doc_id:
                        # Use content hash for parent documents or other retrievers without IDs
                        doc_id = hash(content[:100]) if content else None
                    
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        chunks.append({
                            "content": content,
                            "metadata": {
                                "id": doc_id,
                                "page": doc.metadata.get("page") if hasattr(doc, 'metadata') and doc.metadata else None,
                                "source": doc.metadata.get("source") or doc.metadata.get("file_path") if hasattr(doc, 'metadata') and doc.metadata else None,
                                "document_id": doc.metadata.get("document_id") if hasattr(doc, 'metadata') and doc.metadata else None,
                                "retrieval_method": context_key.replace("_context", "")
                            }
                        })
        
        # Include tool_context (web search results)
        if "tool_context" in result and isinstance(result["tool_context"], list):
            for doc in result["tool_context"]:
                # Tool results don't have _id, use content hash as identifier
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_ids:
                    seen_ids.add(content_hash)
                    chunks.append({
                        "content": doc.page_content[:500],
                        "metadata": {
                            "id": f"tool_{content_hash}",
                            "source": doc.metadata.get("source", "web_search"),
                            "title": doc.metadata.get("title"),
                            "url": doc.metadata.get("url"),
                            "retrieval_method": "tool"
                        }
                    })
        
        return chunks
    
    def _clean_metadata_for_serialization(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to remove non-JSON-serializable objects.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Cleaned metadata dictionary with only serializable values
        """
        from langchain_core.documents import Document
        
        cleaned = {}
        for key, value in metadata.items():
            # Skip Document objects and lists containing Documents
            if isinstance(value, Document):
                continue
            elif isinstance(value, list):
                # Check if list contains Documents
                if value and len(value) > 0 and isinstance(value[0], Document):
                    # Replace with count
                    cleaned[f"{key}_count"] = len(value)
                    continue
                elif not value:
                    # Empty list is fine
                    cleaned[key] = []
                else:
                    # Recursively clean list items
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, Document):
                            # Skip Document objects
                            continue
                        elif isinstance(item, dict):
                            cleaned_list.append(self._clean_metadata_for_serialization(item))
                        else:
                            try:
                                import json
                                json.dumps(item)
                                cleaned_list.append(item)
                            except (TypeError, ValueError):
                                cleaned_list.append(str(item))
                    cleaned[key] = cleaned_list
            elif isinstance(value, dict):
                # Recursively clean nested dictionaries
                cleaned[key] = self._clean_metadata_for_serialization(value)
            else:
                # Keep primitive types and strings
                try:
                    # Try to serialize to ensure it's JSON-compatible
                    import json
                    json.dumps(value)
                    cleaned[key] = value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    cleaned[key] = str(value)
        
        return cleaned
    
    def _log_production_workflow_retrieval(self, retrieval_result: Dict[str, Any], question: str):
        """
        Log retrieval details from production workflow.
        
        Args:
            retrieval_result: Retrieval result from production workflow
            question: Original query question
        """
        # Extract individual contexts
        naive_context = retrieval_result.get("naive_context", [])
        semantic_context = retrieval_result.get("semantic_context", [])
        tool_context = retrieval_result.get("tool_context", [])
        parent_context = retrieval_result.get("parent_context", [])
        bm25_context = retrieval_result.get("bm25_context", [])
        combined_context = retrieval_result.get("combined_context", [])
        
        # Determine which retrievers were used (check if lists have any documents)
        active_retrievers = []
        if naive_context and len(naive_context) > 0:
            active_retrievers.append("naive")
        if semantic_context and len(semantic_context) > 0:
            active_retrievers.append("semantic")
        if tool_context and len(tool_context) > 0:
            active_retrievers.append("tool")
        if parent_context and len(parent_context) > 0:
            active_retrievers.append("parent")
        if bm25_context and len(bm25_context) > 0:
            active_retrievers.append("bm25")
        
        # Extract scores from documents if available
        retriever_scores_log = {}
        
        def extract_scores_from_docs(docs, method_name):
            if not docs or len(docs) == 0:
                return None
            scores = []
            for doc in docs:
                # Try to get similarity score from metadata
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Check for various score fields
                    score = (doc.metadata.get('similarity_score') or 
                            doc.metadata.get('score') or
                            doc.metadata.get('relevance_score') or
                            doc.metadata.get('distance'))  # Distance is inverse of similarity
                    if score is not None:
                        # If it's a distance, convert to similarity (assume cosine distance)
                        if isinstance(score, (int, float)) and score > 1:
                            # Likely a distance, convert: similarity ≈ 1 - distance
                            score = max(0, 1 - score)
                        scores.append(score)
            return scores if scores else None
        
        if naive_context:
            scores = extract_scores_from_docs(naive_context, "naive")
            retriever_scores_log["naive"] = {
                "documents": len(naive_context),
                "scores": scores,
                "avg_score": sum(scores) / len(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "min_score": min(scores) if scores else None
            }
        
        if semantic_context:
            scores = extract_scores_from_docs(semantic_context, "semantic")
            retriever_scores_log["semantic"] = {
                "documents": len(semantic_context),
                "scores": scores,
                "avg_score": sum(scores) / len(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "min_score": min(scores) if scores else None
            }
        
        if tool_context:
            retriever_scores_log["tool"] = {
                "documents": len(tool_context),
                "scores": None,  # Tool-based doesn't have similarity scores
                "avg_score": None,
                "max_score": None,
                "min_score": None
            }
        
        if parent_context:
            scores = extract_scores_from_docs(parent_context, "parent")
            retriever_scores_log["parent"] = {
                "documents": len(parent_context),
                "scores": scores,
                "avg_score": sum(scores) / len(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "min_score": min(scores) if scores else None
            }
        
        if bm25_context:
            scores = extract_scores_from_docs(bm25_context, "bm25")
            retriever_scores_log["bm25"] = {
                "documents": len(bm25_context),
                "scores": scores,
                "avg_score": sum(scores) / len(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "min_score": min(scores) if scores else None
            }
        
        # Log detailed retrieval information
        logger.info("=" * 80)
        logger.info("📊 PRODUCTION WORKFLOW RETRIEVAL DETAILS")
        logger.info("=" * 80)
        logger.info(f"🔍 Retrievers Used: {', '.join(active_retrievers) if active_retrievers else 'None'}")
        logger.info(f"📋 Query: {question[:100]}{'...' if len(question) > 100 else ''}")
        logger.info("")
        
        for method_name, scores_info in retriever_scores_log.items():
            logger.info(f"📋 [{method_name.upper()}] Retriever:")
            logger.info(f"   Documents: {scores_info['documents']}")
            if scores_info['avg_score'] is not None:
                logger.info(f"   Avg Score: {scores_info['avg_score']:.4f}")
                logger.info(f"   Max Score: {scores_info['max_score']:.4f}")
                logger.info(f"   Min Score: {scores_info['min_score']:.4f}")
                if scores_info['scores']:
                    logger.info(f"   Scores: {[f'{s:.4f}' for s in scores_info['scores'][:5]]}{'...' if len(scores_info['scores']) > 5 else ''}")
            else:
                logger.info(f"   Scores: Not available")
            logger.info("")
        
        logger.info(f"✅ Combined Context: {len(combined_context)} unique documents")
        logger.info("=" * 80)
    
    async def _retrieve_from_multiple_methods(
        self,
        question: str,
        methods: List[str],
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Retrieve documents from multiple methods in parallel and rerank with Cohere.
        
        Args:
            question: Query question
            methods: List of method names to use
            top_k: Number of top documents to return after reranking
            
        Returns:
            Dictionary with combined_context and metadata
        """
        logger.info(f"🔄 Retrieving from {len(methods)} methods: {methods}")
        
        # Map methods to their retrievers/chains
        method_retrievers = {}
        
        # Get retrievers for each method
        if "naive" in methods:
            from src.core.retrieval import create_naive_retriever
            method_retrievers["naive"] = create_naive_retriever(
                self.rag_system.vector_store,
                k=self.rag_system.config.retrieval.default_k,
                similarity_threshold=self.rag_system.config.retrieval.similarity_threshold
            )
        
        if "semantic" in methods:
            from src.core.retrieval import create_semantic_retriever
            method_retrievers["semantic"] = create_semantic_retriever(
                self.rag_system.vector_store,
                self.rag_system.embeddings,
                k=self.rag_system.config.retrieval.default_k,
                similarity_threshold=self.rag_system.config.retrieval.similarity_threshold
            )
        
        if "bm25" in methods and "bm25_rag" in self.rag_system.chains:
            # Use BM25 chain's retriever if available
            # We'll need to extract it or create a new one
            from src.core.retrieval import create_bm25_retriever
            try:
                method_retrievers["bm25"] = create_bm25_retriever(
                    documents=self.rag_system.chunked_documents,
                    k=self.rag_system.config.retrieval.default_k
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not create BM25 retriever: {e}")
        
        if "tool" in methods and self.rag_system.search_tool:
            from src.core.retrieval import create_tool_based_retriever
            method_retrievers["tool"] = create_tool_based_retriever(
                self.rag_system.search_tool,
                k=self.rag_system.config.retrieval.default_k
            )
        
        # Run retrievers in parallel (using sync functions in executor)
        # Use retrieve_with_scores to get relevance scores
        async def retrieve_from_method(method_name: str, retriever):
            try:
                logger.info(f"🔍 [{method_name.upper()}] Retrieving documents with scores...")
                loop = asyncio.get_event_loop()
                
                # Try to get scores if available, otherwise fall back to regular retrieval
                if hasattr(retriever, 'retrieve_with_scores'):
                    docs_with_scores = await loop.run_in_executor(
                        None,
                        retriever.retrieve_with_scores,
                        question
                    )
                    # Validate documents have content (check both page_content and metadata['content'])
                    valid_docs = []
                    valid_scores = []
                    for doc, score in docs_with_scores:
                        # Check if document has content in page_content or metadata['content']
                        has_content = False
                        content_source = None
                        
                        if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                            has_content = True
                            content_source = "page_content"
                        elif hasattr(doc, 'metadata') and doc.metadata:
                            # Qdrant sometimes stores content in metadata['content']
                            content = doc.metadata.get('content', '')
                            if content and content.strip():
                                has_content = True
                                content_source = "metadata['content']"
                                # Populate page_content from metadata for consistency
                                doc.page_content = content
                        
                        if has_content:
                            valid_docs.append(doc)
                            valid_scores.append(score)
                        else:
                            # Debug: log what's actually in the document
                            metadata_keys = list(doc.metadata.keys()) if hasattr(doc, 'metadata') and doc.metadata else []
                            page_content_preview = doc.page_content[:50] if hasattr(doc, 'page_content') else "NO page_content attr"
                            logger.warning(f"⚠️ [{method_name.upper()}] Skipping document without content. page_content: '{page_content_preview}', metadata keys: {metadata_keys}")
                    documents = valid_docs
                    scores = valid_scores
                    logger.info(f"✅ [{method_name.upper()}] Retrieved {len(documents)} valid documents with scores")
                    return method_name, documents, scores
                else:
                    # Fallback to regular retrieval
                    documents = await loop.run_in_executor(
                        None,
                        retriever.retrieve_documents,
                        question
                    )
                    # Validate documents have content (check both page_content and metadata['content'])
                    valid_docs = []
                    for doc in documents:
                        # Check if document has content in page_content or metadata['content']
                        has_content = False
                        if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                            has_content = True
                        elif hasattr(doc, 'metadata') and doc.metadata:
                            # Qdrant sometimes stores content in metadata['content']
                            content = doc.metadata.get('content', '')
                            if content and content.strip():
                                has_content = True
                        
                        if has_content:
                            valid_docs.append(doc)
                        else:
                            logger.warning(f"⚠️ [{method_name.upper()}] Skipping document without content (page_content or metadata['content']) from retrieve_documents")
                    documents = valid_docs
                    # No scores available
                    scores = [None] * len(documents)
                    logger.info(f"✅ [{method_name.upper()}] Retrieved {len(documents)} valid documents (no scores available)")
                    return method_name, documents, scores
            except Exception as e:
                logger.error(f"❌ [{method_name.upper()}] Retrieval failed: {e}")
                return method_name, [], []
        
        # Run all retrievers concurrently
        tasks = [retrieve_from_method(method, retriever) 
                for method, retriever in method_retrievers.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all documents with scores and method tracking
        all_documents = []
        method_contexts = {}
        retriever_scores_log = {}  # Track scores from each retriever
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Retrieval task failed: {result}")
                continue
            method_name, documents, scores = result
            method_contexts[f"{method_name}_context"] = documents
            
            # Track documents with their retriever scores
            for i, (doc, score) in enumerate(zip(documents, scores)):
                # Validate document has content (check both page_content and metadata['content'])
                has_content = False
                if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                    has_content = True
                elif hasattr(doc, 'metadata') and doc.metadata:
                    content = doc.metadata.get('content', '')
                    if content and content.strip():
                        has_content = True
                
                if not has_content:
                    logger.warning(f"⚠️ [{method_name.upper()}] Document {i} missing content (page_content or metadata['content']), skipping")
                    continue
                
                # Add retriever info to document metadata (don't modify page_content)
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retriever_method'] = method_name
                doc.metadata['retriever_score'] = score
                doc.metadata['retriever_position'] = i + 1
                all_documents.append((doc, method_name, score))
        
        # Create detailed retriever scores log
        for method_name, retriever in method_retrievers.items():
            # Find matching results
            for result in results:
                if isinstance(result, Exception) or result[0] != method_name:
                    continue
                _, documents, scores = result
                retriever_scores_log[method_name] = {
                    "documents": len(documents),
                    "scores": scores,
                    "avg_score": sum(scores) / len(scores) if scores and all(s is not None for s in scores) else None,
                    "max_score": max(scores) if scores and all(s is not None for s in scores) else None,
                    "min_score": min(scores) if scores and all(s is not None for s in scores) else None
                }
        
        logger.info(f"📚 Collected {len(all_documents)} total documents from {len(method_retrievers)} methods")
        
        # Extract documents from tuples (doc, method, score)
        documents_with_metadata = all_documents
        
        # Deduplicate documents (based on content)
        seen_content = set()
        unique_documents = []
        unique_documents_with_metadata = []
        for doc, method, score in documents_with_metadata:
            # Ensure document has content (check both page_content and metadata['content'])
            content = None
            if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                content = doc.page_content
            elif hasattr(doc, 'metadata') and doc.metadata:
                content = doc.metadata.get('content', '')
            
            if not content or not content.strip():
                logger.warning(f"⚠️ Skipping document from {method} without content (page_content or metadata['content']) during deduplication")
                continue
            
            content_key = content[:100]  # First 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_documents.append(doc)
                unique_documents_with_metadata.append((doc, method, score))
        
        logger.info(f"🔄 Deduplicated to {len(unique_documents)} unique documents")
        
        # Rerank with Cohere if available and we have multiple documents
        reranked_documents = unique_documents
        reranking_applied = False
        cohere_scores_log = []
        
        if len(unique_documents) > 1:
            try:
                config = get_config()
                if config.cohere_api_key:
                    logger.info(f"🎯 Reranking {len(unique_documents)} documents with Cohere...")
                    
                    # Use Cohere API directly to get actual relevance scores
                    loop = asyncio.get_event_loop()
                    
                    # Prepare documents for Cohere API
                    # Validate that all documents have content before reranking
                    # Extract content from page_content or metadata['content']
                    doc_texts = []
                    valid_documents = []
                    for i, doc in enumerate(unique_documents):
                        # Get content from page_content or metadata['content']
                        content = None
                        if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                            content = doc.page_content
                        elif hasattr(doc, 'metadata') and doc.metadata:
                            content = doc.metadata.get('content', '')
                        
                        if content and content.strip():
                            doc_texts.append(content)
                            valid_documents.append(doc)
                        else:
                            logger.warning(f"⚠️ Skipping document {i} with empty content (page_content or metadata['content']) in unique_documents")
                    
                    if not doc_texts:
                        logger.warning("⚠️ No valid documents with content for Cohere reranking")
                        reranked_documents = unique_documents
                        reranking_applied = False
                    else:
                        # Update unique_documents to only include valid ones
                        unique_documents = valid_documents
                    
                    # Call Cohere rerank API to get scores (run in executor since it's synchronous)
                    def get_cohere_scores():
                        co = cohere.Client(api_key=config.cohere_api_key)
                        rerank_response = co.rerank(
                            model='rerank-v3.5',
                            query=question,
                            documents=doc_texts,
                            top_n=len(unique_documents)  # Get scores for all documents
                        )
                        return rerank_response
                    
                    rerank_response = await loop.run_in_executor(None, get_cohere_scores)
                    
                    # Reorder documents based on Cohere rerank results
                    # rerank_response.results is already sorted by relevance (highest first)
                    # Each result has: index (original position) and relevance_score
                    reranked_documents = []
                    for rank_position, result in enumerate(rerank_response.results):
                        original_doc = unique_documents[result.index]
                        
                        # Ensure document has page_content (should already have it, but double-check)
                        if not hasattr(original_doc, 'page_content') or not original_doc.page_content:
                            logger.warning(f"⚠️ Document at index {result.index} missing page_content after reranking, skipping")
                            continue
                        
                        reranked_documents.append(original_doc)
                        
                        # Get document source info
                        source = original_doc.metadata.get('source', 'Unknown')
                        page = original_doc.metadata.get('page', 'N/A')
                        
                        cohere_scores_log.append({
                            "position": rank_position + 1,
                            "cohere_score": result.relevance_score,
                            "content_preview": original_doc.page_content[:100] + "..." if len(original_doc.page_content) > 100 else original_doc.page_content,
                            "retriever_method": original_doc.metadata.get('retriever_method', 'unknown'),
                            "retriever_score": original_doc.metadata.get('retriever_score'),
                            "retriever_position": original_doc.metadata.get('retriever_position'),
                            "source": source,
                            "page": page
                        })
                    
                    reranking_applied = True
                    
                    logger.info(f"✅ Reranked to {len(reranked_documents)} documents with Cohere scores")
                else:
                    logger.warning("⚠️ Cohere API key not found, skipping reranking")
            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}, using original documents")
                reranked_documents = unique_documents
        
        # Limit to top_k
        if len(reranked_documents) > top_k:
            reranked_documents = reranked_documents[:top_k]
        
        # Build detailed retrieval log
        retrieval_log = {
            "retrievers_used": list(method_retrievers.keys()),
            "retriever_scores": retriever_scores_log,
            "cohere_reranking": {
                "applied": reranking_applied,
                "input_documents": len(unique_documents),
                "output_documents": len(reranked_documents),
                "reranked_documents": cohere_scores_log
            },
            "summary": {
                "total_documents_retrieved": len(all_documents),
                "unique_documents_after_dedup": len(unique_documents),
                "final_documents_after_rerank": len(reranked_documents)
            }
        }
        
        # Log detailed retrieval information
        logger.info("=" * 80)
        logger.info("📊 RETRIEVAL DETAILS LOG")
        logger.info("=" * 80)
        logger.info(f"🔍 Retrievers Used: {', '.join(retrieval_log['retrievers_used'])}")
        logger.info("")
        
        for method_name, scores_info in retrieval_log['retriever_scores'].items():
            logger.info(f"📋 [{method_name.upper()}] Retriever:")
            logger.info(f"   Documents: {scores_info['documents']}")
            if scores_info['avg_score'] is not None:
                logger.info(f"   Avg Score: {scores_info['avg_score']:.4f}")
                logger.info(f"   Max Score: {scores_info['max_score']:.4f}")
                logger.info(f"   Min Score: {scores_info['min_score']:.4f}")
                logger.info(f"   Scores: {[f'{s:.4f}' for s in scores_info['scores'][:5]]}{'...' if len(scores_info['scores']) > 5 else ''}")
            else:
                logger.info(f"   Scores: Not available")
            logger.info("")
        
        if reranking_applied:
            logger.info("🎯 Cohere Reranking Results:")
            logger.info(f"   Input: {len(unique_documents)} unique documents")
            logger.info(f"   Output: {len(reranked_documents)} reranked documents")
            logger.info("")
            logger.info("   Top Reranked Documents (by Cohere relevance):")
            for doc_info in cohere_scores_log[:10]:  # Show top 10
                cohere_score_str = f"{doc_info['cohere_score']:.4f}" if doc_info['cohere_score'] is not None else "N/A (position-based)"
                retriever_score_str = f"{doc_info['retriever_score']:.4f}" if doc_info['retriever_score'] is not None else "N/A"
                logger.info(f"   {doc_info['position']}. [{doc_info['retriever_method'].upper()}] "
                          f"Cohere Score: {cohere_score_str} | "
                          f"Retriever Score: {retriever_score_str} | "
                          f"Source: {doc_info['source']} | "
                          f"Page: {doc_info['page']}")
                logger.info(f"      Preview: {doc_info['content_preview']}")
            logger.info("")
        else:
            logger.info("⚠️ Cohere reranking not applied")
            logger.info("")
        
        logger.info(f"✅ Final Context: {len(reranked_documents)} documents selected")
        logger.info("=" * 80)
        
        # Build metadata
        config = get_config()
        metadata = {
            **method_contexts,
            "methods_used": list(method_retrievers.keys()),
            "total_documents": len(all_documents),
            "unique_documents": len(unique_documents),
            "reranked_documents": len(reranked_documents),
            "reranking_applied": reranking_applied,
            "retrieval_log": retrieval_log  # Include detailed log in metadata
        }
        
        return {
            "combined_context": reranked_documents,
            "metadata": metadata
        }
    
    async def stream_query(
        self,
        question: str,
        method: Optional[str] = None,
        methods: Optional[List[str]] = None,
        include_confidence: bool = True,
        chunk_size: int = 50
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a RAG query response.
        
        Args:
            question: Question to ask
            method: Retrieval method (deprecated, use methods)
            methods: List of retrieval methods to use
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
            
            # Handle multiple methods selection
            use_multiple_methods = False
            selected_methods = []
            
            if methods and len(methods) > 0:
                if "production" in methods:
                    # Production workflow combines all methods internally
                    method = "production"
                    use_multiple_methods = False
                elif len(methods) > 1:
                    # Multiple individual methods selected - combine and rerank
                    method = "production"  # Use as identifier, but we'll handle multi-method
                    use_multiple_methods = True
                    selected_methods = methods
                else:
                    # Single method selected
                    method = methods[0]
                    use_multiple_methods = False
            elif method:
                # Use provided method (backward compatibility)
                method = method
                use_multiple_methods = False
            else:
                # Default to production
                method = "production"
                use_multiple_methods = False
            
            # Send start event
            yield {
                "chunk_type": "start",
                "question": question,
                "method": method if not use_multiple_methods else ",".join(selected_methods),
                "metadata": {"include_confidence": include_confidence, "methods": selected_methods if use_multiple_methods else [method]},
                "timestamp": str(datetime.now())
            }
            
            # First, get the retrieval context (but don't generate response yet)
            # We'll stream the generation directly from the LLM
            logger.info("🔍 Retrieving context...")
            
            # Get retrieval context without generating response
            retrieval_result = None
            
            # Log which method was chosen and why
            logger.info("=" * 80)
            logger.info("🎯 METHOD SELECTION")
            logger.info("=" * 80)
            if use_multiple_methods:
                logger.info(f"✅ Selected: Multiple Methods with Cohere Reranking")
                logger.info(f"   Methods: {', '.join(selected_methods)}")
                logger.info(f"   Reason: Multiple individual retrieval methods explicitly selected")
                logger.info(f"   Strategy: Parallel retrieval → Deduplication → Cohere reranking")
            elif method == "production":
                logger.info(f"✅ Selected: Production Workflow")
                logger.info(f"   Methods: All available (naive, semantic, tool, parent, bm25)")
                logger.info(f"   Reason: {'Production method explicitly selected' if methods and 'production' in methods else 'Default method (no specific selection)'}")
                logger.info(f"   Strategy: Parallel retrieval → Context combination → LLM generation")
            else:
                logger.info(f"✅ Selected: Single Method")
                logger.info(f"   Method: {method}")
                logger.info(f"   Reason: Single retrieval method explicitly selected")
                logger.info(f"   Strategy: Direct retrieval → LLM generation")
            logger.info("=" * 80)
            logger.info("")
            
            # Handle multiple methods with reranking
            if use_multiple_methods:
                logger.info(f"🔄 Using multiple methods with Cohere reranking: {selected_methods}")
                retrieval_result = await self._retrieve_from_multiple_methods(
                    question=question,
                    methods=selected_methods,
                    top_k=20  # Get top 20 after reranking
                )
            elif method == "production":
                # Use workflow to get context, but we'll handle generation ourselves
                workflow = self.rag_system.workflows.get("production_rag")
                if workflow:
                    # Prepare input
                    input_data = {
                        "question": question,
                        "response": "",
                        "metadata": {},
                        "naive_context": [],
                        "semantic_context": [],
                        "tool_context": [],
                        "parent_context": [],
                        "bm25_context": [],
                        "combined_context": [],
                        "retrieval_results": {},
                        "knowledge_response": "",
                        "search_response": "",
                        "final_response": "",
                        "source_attribution": {},
                        "confidence_scores": {},
                        "quality_metrics": {},
                        "performance_metrics": {},
                        "error_handling": None,
                        "retrieval_method": []
                    }
                    
                    # Run workflow up to generation (we'll get combined_context)
                    # Use astream to get intermediate states - wait for combine_contexts node
                    combined_state = None
                    async for event in workflow.astream(input_data):
                        # Look for the combine_contexts node output
                        if "combine_contexts" in event:
                            combined_state = event["combine_contexts"]
                            # We have the combined context, but let's also collect individual contexts
                            # by checking earlier events
                            retrieval_result = {
                                "combined_context": combined_state.get("combined_context", []),
                                "metadata": combined_state.get("metadata", {}),
                                "naive_context": combined_state.get("naive_context", []),
                                "semantic_context": combined_state.get("semantic_context", []),
                                "tool_context": combined_state.get("tool_context", []),
                                "parent_context": combined_state.get("parent_context", []),
                                "bm25_context": combined_state.get("bm25_context", [])
                            }
                            
                            # Log production workflow retrieval details
                            self._log_production_workflow_retrieval(retrieval_result, question)
                            
                            # Continue to get all contexts, but break after combine_contexts
                            # We break here because we have everything we need for streaming
                            break
            else:
                # For other methods, use regular query but extract context
                result = self.rag_system.query(
                    question=question,
                    method=method,
                    include_confidence=include_confidence
                )
                retrieval_result = result
            
            # If we didn't get retrieval result, fall back to full query
            if not retrieval_result:
                logger.warning("⚠️ Could not get retrieval context, falling back to full query")
                result = self.rag_system.query(
                    question=question,
                    method=method,
                    include_confidence=include_confidence
                )
                answer = result.get("response", "No response generated")
                confidence = result.get("confidence")
                metadata = result.get("metadata", {})
                sources = self._count_sources(result)
                document_chunks = self._extract_document_chunks(result)
                
                # Stream the complete answer in chunks (fallback)
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
                    import asyncio
                    await asyncio.sleep(0.05)
            else:
                # Stream from LLM using the retrieved context
                logger.info("🌊 Streaming response from LLM...")
                
                # Format context
                context = retrieval_result.get("combined_context", [])
                context_text = "\n\n".join(doc.page_content for doc in context) if context else ""
                
                # Build prompt
                prompt = f"""You are a helpful assistant. Use the provided context to answer the question accurately.

Question: {question}

Context:
{context_text}

Please provide a helpful response based on the context above."""
                
                # Stream from LLM
                full_response = ""
                chunk_index = 0
                
                if hasattr(self.rag_system.llm, 'astream'):
                    async for chunk in self.rag_system.llm.astream(prompt):
                        if hasattr(chunk, 'content') and chunk.content:
                            content_chunk = chunk.content
                            full_response += content_chunk
                            
                            yield {
                                "chunk_type": "content",
                                "content": content_chunk,
                                "chunk_index": chunk_index,
                                "metadata": {"llm_streaming": True},
                                "timestamp": str(datetime.now())
                            }
                            chunk_index += 1
                else:
                    # Fallback if LLM doesn't support astream
                    logger.warning("⚠️ LLM doesn't support astream, using invoke")
                    response = self.rag_system.llm.invoke(prompt).content
                    full_response = response
                    
                    # Stream in chunks
                    for i in range(0, len(response), chunk_size):
                        chunk_content = response[i:i + chunk_size]
                        yield {
                            "chunk_type": "content",
                            "content": chunk_content,
                            "chunk_index": chunk_index,
                            "metadata": {"total_length": len(response)},
                            "timestamp": str(datetime.now())
                        }
                        chunk_index += 1
                        import asyncio
                        await asyncio.sleep(0.05)
                
                # Extract metadata from retrieval result
                confidence = None  # Will be calculated if needed
                raw_metadata = retrieval_result.get("metadata", {})
                sources = self._count_sources(retrieval_result)
                document_chunks = self._extract_document_chunks(retrieval_result)
                
                # Clean metadata to remove non-serializable objects (like Document objects)
                metadata = self._clean_metadata_for_serialization(raw_metadata)
                
                # If confidence is needed, we can calculate it from the response
                # For now, we'll skip it during streaming for performance
                if include_confidence:
                    # Calculate a simple confidence score based on context quality
                    context_count = len(context) if context else 0
                    confidence = min(0.95, 0.5 + (context_count / 20) * 0.45) if context_count > 0 else 0.3
            
            # Send document chunks
            if document_chunks:
                yield {
                    "chunk_type": "sources",
                    "chunks": document_chunks,
                    "timestamp": str(datetime.now())
                }
            
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
        method: Optional[str] = None,
        methods: Optional[List[str]] = None,
        include_confidence: bool = True,
        chunk_size: int = 50
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a RAG query with real LLM streaming.
        
        Args:
            question: Question to ask
            method: Retrieval method (deprecated, use methods)
            methods: List of retrieval methods to use
            include_confidence: Whether to include confidence
            chunk_size: Size of content chunks
            
        Yields:
            Stream chunks
        """
        # Handle backward compatibility and determine which method to use
        if methods and len(methods) > 0:
            # Use the first method from the list (or "production" if available)
            method = methods[0] if "production" not in methods else "production"
        elif method:
            # Use provided method (backward compatibility)
            method = method
        else:
            # Default to production
            method = "production"
        
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
