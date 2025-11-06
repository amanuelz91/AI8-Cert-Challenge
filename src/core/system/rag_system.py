"""
Main RAG system class that orchestrates all components.

Production-grade RAG system with naive, semantic, and tool-based retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import BaseTool

from src.core.data import (
    load_pdf_documents,
    preprocess_documents,
    split_documents
)
from src.core.embeddings import get_default_embedding_provider
from src.core.vectorstore import get_default_vector_store_manager
from src.core.retrieval import (
    NaiveRetriever,
    SemanticRetriever,
    ToolBasedRetriever,
    create_naive_retriever,
    create_semantic_retriever,
    create_tool_based_retriever
)
from src.chains.rag_chains import create_production_chains
from src.workflows.rag_workflow import create_production_workflows
from src.utils.logging import get_logger
from src.utils.exceptions import RAGException, ConfigurationError
from src.config.settings import validate_api_keys, get_config
from src.tools.tavily.tavily_search_factory import TavilySearchFactory, SearchConfig

logger = get_logger(__name__)


class ProductionRAGSystem:
    """Production-grade RAG system with multiple retrieval strategies."""
    
    def __init__(
        self,
        data_folder: Optional[str] = None,
        llm: Optional[ChatOpenAI] = None,
        embeddings: Optional[OpenAIEmbeddings] = None,
        search_tool: Optional[BaseTool] = None
    ):
        """
        Initialize production RAG system.
        
        Args:
            data_folder: Path to data folder
            llm: Language model instance
            embeddings: Embedding model instance
            search_tool: Search tool instance (e.g., Tavily)
            
        Raises:
            ConfigurationError: If configuration is invalid
            RAGException: If initialization fails
        """
        try:
            # Validate configuration
            validate_api_keys()
            
            # Initialize configuration
            self.config = get_config()
            self.data_folder = data_folder or self.config.data.data_folder
            
            # Initialize components
            self.llm = llm or ChatOpenAI(
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout,
                openai_api_key=self.config.openai_api_key
            )
            
            self.embeddings = embeddings or OpenAIEmbeddings(
                model=self.config.embedding.model_name,
                openai_api_key=self.config.openai_api_key
            )
            
            # Create default search tool if not provided
            if search_tool is None:
                try:
                    tavily_factory = TavilySearchFactory()
                    self.search_tool = tavily_factory.create_search_tool(
                        SearchConfig(
                            name="General Search",
                            description="General web search for real-time information",
                            domains=["studentaid.gov", "ed.gov", "fafsa.gov"],
                            max_results=5,
                            search_depth="advanced",
                            include_answer=True
                        )
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Could not create Tavily search tool: {str(e)}")
                    self.search_tool = None
            else:
                self.search_tool = search_tool
            
            # Initialize system components
            self._initialize_components()
            
            logger.info("🚀 Production RAG system initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize RAG system: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            logger.info("🔧 Initializing RAG system components")
            
            # Load and process documents
            self._load_documents()
            
            # Initialize vector store
            self._initialize_vector_store()
            
            # Create retrievers inline (not stored as attributes)
            # These are only used to build chains
            naive_retriever = create_naive_retriever(
                self.vector_store,
                k=self.config.retrieval.default_k,
                similarity_threshold=self.config.retrieval.similarity_threshold
            )
            
            semantic_retriever = create_semantic_retriever(
                self.vector_store,
                self.embeddings,
                k=self.config.retrieval.default_k,
                similarity_threshold=self.config.retrieval.similarity_threshold
            )
            
            if self.search_tool:
                tool_retriever = create_tool_based_retriever(
                    self.search_tool,
                    k=self.config.retrieval.default_k
                )
            else:
                logger.warning("⚠️ No search tool provided, tool-based retrieval disabled")
                tool_retriever = None
            
            # Create parent document retriever (optional)
            # Reuse the main cloud Qdrant instance with a different collection for parent document child chunks
            parent_document_retriever = None
            try:
                from src.core.retrieval import create_parent_document_retriever
                logger.info("📄 Creating parent document retriever...")
                logger.info("♻️ Reusing cloud Qdrant instance for parent document child chunks")
                # Use processed documents (before chunking) for parent document retriever
                # This allows hierarchical retrieval with parent documents
                # Pass the main vector_store to reuse the cloud Qdrant instance
                parent_document_retriever = create_parent_document_retriever(
                    parent_documents=self.processed_documents,  # Use full documents as parents
                    embeddings=self.embeddings,
                    k=self.config.retrieval.default_k,
                    child_chunk_size=750,
                    child_chunk_overlap=100,
                    vector_store=self.vector_store  # Reuse main cloud Qdrant instance
                )
                logger.info("✅ Parent document retriever created (using cloud Qdrant)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create parent document retriever: {str(e)}")
                logger.warning("⚠️ Continuing without parent document retrieval")
            
            # Create BM25 retriever (optional)
            bm25_retriever = None
            try:
                from src.core.retrieval import create_bm25_retriever
                logger.info("🔤 Creating BM25 retriever...")
                # Use chunked documents for BM25 (keyword-based search works best with chunks)
                bm25_retriever = create_bm25_retriever(
                    documents=self.chunked_documents,
                    k=self.config.retrieval.default_k
                )
                logger.info("✅ BM25 retriever created")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create BM25 retriever: {str(e)}")
                logger.warning("⚠️ Continuing without BM25 retrieval")
            
            logger.info("✅ Retrievers created for chains")
            
            # Initialize chains and workflows using the retrievers
            self._initialize_chains_and_workflows(
                naive_retriever, 
                semantic_retriever, 
                tool_retriever,
                parent_document_retriever,
                bm25_retriever
            )
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize components: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _load_documents(self) -> None:
        """Load and process documents."""
        try:
            logger.info("📚 Loading documents")
            
            # Load PDF documents
            self.raw_documents = load_pdf_documents(self.data_folder)
            logger.info(f"📄 Loaded {len(self.raw_documents)} raw documents")
            
            # Preprocess documents
            self.processed_documents = preprocess_documents(self.raw_documents)
            logger.info(f"🔧 Processed {len(self.processed_documents)} documents")
            
            # Split documents into chunks
            self.chunked_documents = split_documents(self.processed_documents)
            logger.info(f"✂️ Created {len(self.chunked_documents)} document chunks")
            
        except Exception as e:
            error_msg = f"Failed to load documents: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _initialize_vector_store(self) -> None:
        """Initialize vector store."""
        try:
            logger.info("🗃️ Initializing vector store")
            
            # Get vector store manager
            self.vector_store_manager = get_default_vector_store_manager()
            
            # Initialize vector store
            self.vector_store = self.vector_store_manager.initialize_vector_store(self.embeddings)
            
            # Check if documents already exist in the collection
            collection_stats = self.vector_store_manager.get_stats()
            existing_points = collection_stats.get("points_count", 0)
            
            if existing_points > 0:
                logger.info(f"📦 Collection already contains {existing_points} documents, skipping upload")
                logger.info("✅ Vector store initialized successfully (documents already loaded)")
            else:
                logger.info("📤 No existing documents found, uploading documents...")
                # Add documents to vector store
                embeddings_list = self.embeddings.embed_documents([doc.page_content for doc in self.chunked_documents])
                self.vector_store_manager.add_documents(self.chunked_documents, embeddings_list)
                logger.info("✅ Vector store initialized successfully with new documents")
            
        except Exception as e:
            error_msg = f"Failed to initialize vector store: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _initialize_chains_and_workflows(
        self,
        naive_retriever,
        semantic_retriever,
        tool_retriever,
        parent_document_retriever=None,
        bm25_retriever=None
    ) -> None:
        """Initialize LCEL chains and LangGraph workflows."""
        try:
            logger.info("🔗 Initializing chains and workflows")
            
            # Initialize chains
            logger.info(f"🔍 Creating chains with retrievers:")
            logger.info(f"  - naive_retriever: {type(naive_retriever)}")
            logger.info(f"  - semantic_retriever: {type(semantic_retriever)}")
            logger.info(f"  - tool_retriever: {type(tool_retriever)}")
            if parent_document_retriever:
                logger.info(f"  - parent_document_retriever: {type(parent_document_retriever)}")
            if bm25_retriever:
                logger.info(f"  - bm25_retriever: {type(bm25_retriever)}")
            logger.info(f"  - llm: {type(self.llm)}")
            
            # Create production chains - returns tuple (chains, retrievers)
            self.chains, _ = create_production_chains(
                naive_retriever,
                semantic_retriever,
                tool_retriever,
                self.llm,
                documents=self.chunked_documents  # Pass documents for naive_rag chain
            )
            
            logger.info(f"✅ Chains created: {list(self.chains.keys())}")
            for chain_name, chain_obj in self.chains.items():
                logger.info(f"  - {chain_name}: {type(chain_obj)}")
                logger.info(f"    Has invoke: {hasattr(chain_obj, 'invoke')}")
            
            # Initialize workflows
            logger.info(f"🔄 Creating workflows with retrievers:")
            self.workflows = create_production_workflows(
                naive_retriever,
                semantic_retriever,
                tool_retriever,
                self.llm,
                parent_document_retriever,
                bm25_retriever
            )
            
            logger.info(f"✅ Workflows created: {list(self.workflows.keys())}")
            for workflow_name, workflow_obj in self.workflows.items():
                logger.info(f"  - {workflow_name}: {type(workflow_obj)}")
                logger.info(f"    Has invoke: {hasattr(workflow_obj, 'invoke')}")
            
            logger.info("✅ Chains and workflows initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize chains and workflows: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RAGException(error_msg) from e
    
    def query(
        self,
        question: str,
        method: str = "production",
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            method: Query method ("naive", "semantic", "tool", "hybrid", "production")
            include_confidence: Whether to include confidence scoring
            
        Returns:
            Dictionary with response and metadata
            
        Raises:
            RAGException: If query fails
        """
        try:
            logger.info(f"🚀 [RAG QUERY] ==========================================")
            logger.info(f"🚀 [RAG QUERY] Starting RAG query processing")
            logger.info(f"🚀 [RAG QUERY] Question: '{question}'")
            logger.info(f"🚀 [RAG QUERY] Method: {method}")
            logger.info(f"🚀 [RAG QUERY] Include confidence: {include_confidence}")
            logger.info(f"🚀 [RAG QUERY] ==========================================")
            
            # Log system state
            logger.info(f"🔍 [RAG QUERY] System state:")
            logger.info(f"  📚 Available chains: {list(self.chains.keys())}")
            logger.info(f"  🔄 Available workflows: {list(self.workflows.keys())}")
            logger.info(f"  🤖 LLM: {type(self.llm)}")
            
            # Prepare input - ensure all required state fields are initialized for workflows
            if method == "production":
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
            else:
                input_data = {"question": question}
            
            logger.info(f"📝 [RAG QUERY] Input data prepared: {list(input_data.keys())}")
            
            # Execute based on method
            logger.info(f"⚡ [RAG QUERY] Executing query using method: {method}")
            
            if method == "naive":
                logger.info(f"🔗 [RAG QUERY] Using NAIVE chain")
                logger.info(f"🔗 [RAG QUERY] Chain type: {type(self.chains['naive_rag'])}")
                logger.info(f"🔗 [RAG QUERY] Chain has invoke method: {hasattr(self.chains['naive_rag'], 'invoke')}")
                logger.info(f"🔗 [RAG QUERY] Invoking naive_rag chain...")
                result = self.chains["naive_rag"].invoke(input_data)
                logger.info(f"✅ [RAG QUERY] Naive chain execution completed")
                
            elif method == "semantic":
                logger.info(f"🔗 [RAG QUERY] Using SEMANTIC chain")
                logger.info(f"🔗 [RAG QUERY] Chain type: {type(self.chains['semantic_rag'])}")
                logger.info(f"🔗 [RAG QUERY] Chain has invoke method: {hasattr(self.chains['semantic_rag'], 'invoke')}")
                logger.info(f"🔗 [RAG QUERY] Invoking semantic_rag chain...")
                result = self.chains["semantic_rag"].invoke(input_data)
                logger.info(f"✅ [RAG QUERY] Semantic chain execution completed")
                
            elif method == "tool" and self.search_tool:
                logger.info(f"🔗 [RAG QUERY] Using TOOL chain")
                logger.info(f"🔗 [RAG QUERY] Chain type: {type(self.chains['tool_search'])}")
                logger.info(f"🔗 [RAG QUERY] Chain has invoke method: {hasattr(self.chains['tool_search'], 'invoke')}")
                logger.info(f"🔗 [RAG QUERY] Invoking tool_search chain...")
                result = self.chains["tool_search"].invoke(input_data)
                logger.info(f"✅ [RAG QUERY] Tool chain execution completed")
                
            elif method == "hybrid" and self.search_tool:
                logger.info(f"🔗 [RAG QUERY] Using HYBRID chain")
                logger.info(f"🔗 [RAG QUERY] Chain type: {type(self.chains['hybrid_rag'])}")
                logger.info(f"🔗 [RAG QUERY] Chain has invoke method: {hasattr(self.chains['hybrid_rag'], 'invoke')}")
                logger.info(f"🔗 [RAG QUERY] Invoking hybrid_rag chain...")
                result = self.chains["hybrid_rag"].invoke(input_data)
                logger.info(f"✅ [RAG QUERY] Hybrid chain execution completed")
                
            elif method == "bm25" and "bm25_rag" in self.chains:
                logger.info(f"🔗 [RAG QUERY] Using BM25 chain")
                logger.info(f"🔗 [RAG QUERY] Chain type: {type(self.chains['bm25_rag'])}")
                logger.info(f"🔗 [RAG QUERY] Chain has invoke method: {hasattr(self.chains['bm25_rag'], 'invoke')}")
                logger.info(f"🔗 [RAG QUERY] Invoking bm25_rag chain...")
                result = self.chains["bm25_rag"].invoke(input_data)
                logger.info(f"✅ [RAG QUERY] BM25 chain execution completed")
                
            elif method == "production":
                logger.info(f"🔄 [RAG QUERY] Using PRODUCTION workflow")
                logger.info(f"🔄 [RAG QUERY] Workflow type: {type(self.workflows['production_rag'])}")
                logger.info(f"🔄 [RAG QUERY] Workflow has invoke method: {hasattr(self.workflows['production_rag'], 'invoke')}")
                logger.info(f"🔄 [RAG QUERY] Invoking production_rag workflow...")
                result = self.workflows["production_rag"].invoke(input_data)
                logger.info(f"✅ [RAG QUERY] Production workflow execution completed")
                
            else:
                error_msg = f"Invalid method: {method}"
                logger.error(f"❌ [RAG QUERY] {error_msg}")
                raise ValueError(error_msg)
            
            logger.info(f"📊 [RAG QUERY] Chain/workflow execution completed")
            logger.info(f"📊 [RAG QUERY] Result type: {type(result)}")
            logger.info(f"📊 [RAG QUERY] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Log result details
            if isinstance(result, dict):
                logger.info(f"📊 [RAG QUERY] Result details:")
                for key, value in result.items():
                    if key == "response":
                        logger.info(f"  📝 {key}: {str(value)[:200]}...")
                    elif key == "context":
                        logger.info(f"  📚 {key}: {len(value) if isinstance(value, list) else 'Not a list'} items")
                        if isinstance(value, list) and len(value) > 0:
                            logger.info(f"    First context item: {str(value[0])[:100]}...")
                    else:
                        logger.info(f"  🔑 {key}: {str(value)[:100]}...")
            
            # Add confidence scoring if requested
            if include_confidence and "response" in result:
                logger.info(f"🎯 [RAG QUERY] Adding confidence scoring...")
                logger.info(f"🎯 [RAG QUERY] Confidence chain type: {type(self.chains['confidence'])}")
                logger.info(f"🎯 [RAG QUERY] Confidence chain has invoke: {hasattr(self.chains['confidence'], 'invoke')}")
                
                confidence_input = {
                    "question": question,
                    "context": result.get("context", []),
                    "response": result["response"]
                }
                logger.info(f"🎯 [RAG QUERY] Confidence input prepared: {list(confidence_input.keys())}")
                
                confidence_result = self.chains["confidence"].invoke(confidence_input)
                result["confidence"] = confidence_result.get("confidence", {})
                
                logger.info(f"✅ [RAG QUERY] Confidence scoring completed")
                logger.info(f"🎯 [RAG QUERY] Confidence result: {result.get('confidence', {})}")
            
            # Add metadata
            result["metadata"] = {
                "method": method,
                "question": question,
                "timestamp": str(datetime.now()),
                "system_version": "1.0.0"
            }
            
            logger.info(f"🎉 [RAG QUERY] Query processed successfully using {method}")
            logger.info(f"🎉 [RAG QUERY] Final result keys: {list(result.keys())}")
            logger.info(f"🚀 [RAG QUERY] ==========================================")
            
            return result
            
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            logger.error(f"💥 [RAG QUERY] ==========================================")
            logger.error(f"💥 [RAG QUERY] QUERY FAILED!")
            logger.error(f"💥 [RAG QUERY] Error: {error_msg}")
            logger.error(f"💥 [RAG QUERY] Question: '{question}'")
            logger.error(f"💥 [RAG QUERY] Method: {method}")
            logger.error(f"💥 [RAG QUERY] Error type: {type(e)}")
            logger.error(f"💥 [RAG QUERY] Error args: {e.args}")
            import traceback
            logger.error(f"💥 [RAG QUERY] Full traceback: {traceback.format_exc()}")
            logger.error(f"💥 [RAG QUERY] ==========================================")
            raise RAGException(error_msg) from e
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "documents": {
                "raw_documents": len(self.raw_documents),
                "processed_documents": len(self.processed_documents),
                "chunked_documents": len(self.chunked_documents)
            },
            "chains": {
                "available": list(self.chains.keys()),
                "count": len(self.chains)
            },
            "vector_store": self.vector_store_manager.get_stats(),
            "configuration": {
                "llm_model": self.llm.model_name,
                "embedding_model": getattr(self.embeddings, 'model', self.config.embedding.model_name),
                "data_folder": self.data_folder
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check.
        
        Returns:
            Dictionary with health status
        """
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": str(datetime.now())
        }
        
        try:
            # Check vector store
            health_status["components"]["vector_store"] = self.vector_store_manager.qdrant_manager.health_check()
            
            # Check chains
            health_status["components"]["chains"] = {
                "count": len(self.chains),
                "available": list(self.chains.keys())
            }
            
            # Check LLM
            try:
                test_response = self.llm.invoke("Test")
                health_status["components"]["llm"] = True
            except:
                health_status["components"]["llm"] = False
                health_status["overall"] = "degraded"
            
        except Exception as e:
            health_status["overall"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status


def create_production_rag_system(
    data_folder: Optional[str] = None,
    llm: Optional[ChatOpenAI] = None,
    embeddings: Optional[OpenAIEmbeddings] = None,
    search_tool: Optional[BaseTool] = None
) -> ProductionRAGSystem:
    """
    Create a production RAG system instance.
    
    Args:
        data_folder: Path to data folder
        llm: Language model instance
        embeddings: Embedding model instance
        search_tool: Search tool instance
        
    Returns:
        Production RAG system instance
    """
    return ProductionRAGSystem(data_folder, llm, embeddings, search_tool)
