"""
Semantic chunking retriever implementation.

Production-grade retriever using semantic chunking for better document boundaries.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.utils.decorators import timing_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class SemanticRetriever(BaseRAGRetriever):
    """Semantic retriever using semantic chunking for better boundaries."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        breakpoint_threshold_type: str = "percentile"
    ):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            embeddings: Embedding model for semantic analysis
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
            breakpoint_threshold_type: Type of threshold for semantic breakpoints
        """
        super().__init__("semantic", k)
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold or get_config().retrieval.similarity_threshold
        self.breakpoint_threshold_type = breakpoint_threshold_type
        
        # Initialize semantic chunker
        self.semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type
        )
        
        logger.info(f"🧠 Initialized semantic retriever (k={k}, threshold={self.similarity_threshold})")
    
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using semantic similarity.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🧠 [SEMANTIC RETRIEVAL] Starting retrieval for query: '{query}'")
            logger.info(f"🧠 [SEMANTIC RETRIEVAL] Parameters: k={self.k}, threshold={self.similarity_threshold}")
            logger.info(f"🧠 [SEMANTIC RETRIEVAL] Breakpoint threshold type: {self.breakpoint_threshold_type}")
            
            # Use similarity search with score
            logger.debug(f"🧠 [SEMANTIC RETRIEVAL] Calling vector_store.similarity_search_with_score with k={self.k}")
            
            # Try to get documents with full payload using direct Qdrant client
            try:
                # Get query embedding using the embeddings from the retriever
                query_embedding = self.embeddings.embed_query(query)
                
                # Search using direct Qdrant client with full payload
                search_results = self.vector_store.client.search(
                    collection_name=self.vector_store.collection_name,
                    query_vector=query_embedding,
                    limit=self.k,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Convert Qdrant results to Document objects
                docs_with_scores = []
                for result in search_results:
                    # Extract content from payload
                    content = result.payload.get('content', '') or result.payload.get('page_content', '')
                    
                    # Create Document object
                    doc = Document(
                        page_content=content,
                        metadata={
                            '_id': result.id,
                            '_collection_name': self.vector_store.collection_name,
                            **result.payload
                        }
                    )
                    
                    # Convert distance to similarity score
                    distance = result.score
                    docs_with_scores.append((doc, distance))
                
                logger.info(f"🧠 [SEMANTIC RETRIEVAL] Direct Qdrant search returned {len(docs_with_scores)} documents")
                
            except Exception as e:
                logger.warning(f"⚠️ [SEMANTIC RETRIEVAL] Direct Qdrant search failed: {str(e)}")
                logger.info(f"🧠 [SEMANTIC RETRIEVAL] Falling back to LangChain similarity_search_with_score")
                
                # Fallback to LangChain method
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query, 
                    k=self.k
                )
            
            logger.info(f"🧠 [SEMANTIC RETRIEVAL] Vector store returned {len(docs_with_scores)} documents")
            
            # Filter by similarity threshold and enhance with semantic info
            filtered_docs = []
            rejected_docs = []
            
            logger.info(f"🧠 [SEMANTIC RETRIEVAL] Analyzing semantic similarity scores:")
            for i, (doc, score) in enumerate(docs_with_scores):
                # Convert distance to similarity (Qdrant returns distance)
                similarity = 1 - score
                
                # Fix: If page_content is empty but content exists in metadata, use it
                if not doc.page_content and doc.metadata and 'content' in doc.metadata:
                    doc.page_content = doc.metadata['content']
                    logger.info(f"  🔧 [SEMANTIC RETRIEVAL] Fixed empty page_content using metadata['content']")
                
                doc_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                
                logger.info(f"  📄 Doc {i+1}: similarity={similarity:.4f}, distance={score:.4f}")
                logger.info(f"      Content preview: {doc_preview}...")
                logger.info(f"      Metadata: {doc.metadata}")
                
                if similarity >= self.similarity_threshold:
                    # Add semantic metadata
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['similarity_score'] = similarity
                    doc.metadata['relevance_score'] = similarity
                    doc.metadata['chunking_method'] = 'semantic'
                    doc.metadata['breakpoint_threshold'] = self.breakpoint_threshold_type
                    doc.metadata['retrieval_method'] = 'semantic'
                    filtered_docs.append(doc)
                    logger.info(f"      ✅ ACCEPTED (above threshold {self.similarity_threshold})")
                else:
                    rejected_docs.append((doc, similarity))
                    logger.warning(f"      ❌ REJECTED (below threshold {self.similarity_threshold})")
            
            logger.info(f"📚 [SEMANTIC RETRIEVAL] Final results:")
            logger.info(f"  ✅ Accepted: {len(filtered_docs)} documents")
            logger.info(f"  ❌ Rejected: {len(rejected_docs)} documents")
            logger.info(f"  📊 Acceptance rate: {len(filtered_docs)/len(docs_with_scores)*100:.1f}%")
            
            if len(filtered_docs) == 0:
                logger.warning(f"⚠️ [SEMANTIC RETRIEVAL] NO DOCUMENTS PASSED SIMILARITY THRESHOLD!")
                logger.warning(f"⚠️ [SEMANTIC RETRIEVAL] Consider lowering threshold from {self.similarity_threshold}")
                if rejected_docs:
                    best_similarity = max(similarity for _, similarity in rejected_docs)
                    logger.warning(f"⚠️ [SEMANTIC RETRIEVAL] Best similarity score was: {best_similarity:.4f}")
            
            return filtered_docs
            
        except Exception as e:
            error_msg = f"Semantic retrieval failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"🧠 [SEMANTIC RETRIEVAL] Query that failed: '{query}'")
            logger.error(f"🧠 [SEMANTIC RETRIEVAL] Parameters: k={self.k}, threshold={self.similarity_threshold}")
            import traceback
            logger.error(f"🧠 [SEMANTIC RETRIEVAL] Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🧠 [Semantic] Retrieving documents with scores for: {query[:50]}...")
            
            # Use similarity search with score
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=self.k
            )
            
            # Convert distance to similarity and filter
            results = []
            for doc, distance in docs_with_scores:
                similarity = 1 - distance
                if similarity >= self.similarity_threshold:
                    # Add semantic metadata
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['similarity_score'] = similarity
                    doc.metadata['relevance_score'] = similarity
                    doc.metadata['chunking_method'] = 'semantic'
                    doc.metadata['breakpoint_threshold'] = self.breakpoint_threshold_type
                    results.append((doc, similarity))
            
            logger.info(f"📚 [Semantic] Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            error_msg = f"Semantic retrieval with scores failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def retrieve_with_result(self, query: str) -> RetrievalResult:
        """
        Retrieve documents and return as RetrievalResult.
        
        Args:
            query: Query string
            
        Returns:
            RetrievalResult with documents and metadata
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            docs_with_scores = self.retrieve_with_scores(query)
            
            if not docs_with_scores:
                return RetrievalResult(
                    documents=[],
                    scores=[],
                    retriever_name=self.name,
                    query=query,
                    metadata={"warning": "No documents found above similarity threshold"}
                )
            
            documents, scores = zip(*docs_with_scores)
            
            return RetrievalResult(
                documents=list(documents),
                scores=list(scores),
                retriever_name=self.name,
                query=query,
                metadata={
                    "similarity_threshold": self.similarity_threshold,
                    "avg_similarity": sum(scores) / len(scores),
                    "max_similarity": max(scores),
                    "min_similarity": min(scores),
                    "chunking_method": "semantic",
                    "breakpoint_threshold_type": self.breakpoint_threshold_type
                }
            )
            
        except Exception as e:
            error_msg = f"Semantic retrieval with result failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def get_retriever_stats(self) -> dict:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with retriever statistics
        """
        return {
            "name": self.name,
            "k": self.k,
            "similarity_threshold": self.similarity_threshold,
            "type": "semantic_similarity",
            "vector_store_type": "QdrantVectorStore",
            "chunking_method": "semantic",
            "breakpoint_threshold_type": self.breakpoint_threshold_type
        }


def create_semantic_retriever(
    vector_store: QdrantVectorStore,
    embeddings,
    k: int = 5,
    similarity_threshold: Optional[float] = None,
    breakpoint_threshold_type: str = "percentile"
) -> SemanticRetriever:
    """
    Create a semantic retriever instance.
    
    Args:
        vector_store: Qdrant vector store instance
        embeddings: Embedding model for semantic analysis
        k: Number of documents to retrieve
        similarity_threshold: Minimum similarity threshold
        breakpoint_threshold_type: Type of threshold for semantic breakpoints
        
    Returns:
        Semantic retriever instance
    """
    return SemanticRetriever(
        vector_store, 
        embeddings, 
        k, 
        similarity_threshold, 
        breakpoint_threshold_type
    )
