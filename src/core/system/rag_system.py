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
            
            # Initialize retrievers
            self._initialize_retrievers()
            
            # Initialize chains and workflows
            self._initialize_chains_and_workflows()
            
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
    
    def _initialize_retrievers(self) -> None:
        """Initialize retrieval components."""
        try:
            logger.info("🔍 Initializing retrievers")
            
            # Initialize production retrievers
            self.naive_retriever = create_naive_retriever(
                self.vector_store,
                k=self.config.retrieval.default_k,
                similarity_threshold=self.config.retrieval.similarity_threshold
            )
            
            self.semantic_retriever = create_semantic_retriever(
                self.vector_store,
                self.embeddings,
                k=self.config.retrieval.default_k,
                similarity_threshold=self.config.retrieval.similarity_threshold
            )
            
            if self.search_tool:
                self.tool_retriever = create_tool_based_retriever(
                    self.search_tool,
                    k=self.config.retrieval.default_k
                )
            else:
                logger.warning("⚠️ No search tool provided, tool-based retrieval disabled")
                self.tool_retriever = None
            
            logger.info("✅ Retrievers initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize retrievers: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _initialize_chains_and_workflows(self) -> None:
        """Initialize LCEL chains and LangGraph workflows."""
        try:
            logger.info("🔗 Initializing chains and workflows")
            
            # Initialize chains
            logger.info(f"🔍 Creating chains with retrievers:")
            logger.info(f"  - naive_retriever: {type(self.naive_retriever)}")
            logger.info(f"  - semantic_retriever: {type(self.semantic_retriever)}")
            logger.info(f"  - tool_retriever: {type(self.tool_retriever)}")
            logger.info(f"  - llm: {type(self.llm)}")
            
            self.chains = create_production_chains(
                self.naive_retriever,
                self.semantic_retriever,
                self.tool_retriever,
                self.llm
            )
            
            logger.info(f"✅ Chains created: {list(self.chains.keys())}")
            for chain_name, chain_obj in self.chains.items():
                logger.info(f"  - {chain_name}: {type(chain_obj)}")
                logger.info(f"    Has invoke: {hasattr(chain_obj, 'invoke')}")
            
            # Initialize workflows
            logger.info(f"🔄 Creating workflows with retrievers:")
            self.workflows = create_production_workflows(
                self.naive_retriever,
                self.semantic_retriever,
                self.tool_retriever,
                self.llm
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
            logger.info(f"❓ Processing query: {question[:50]}...")
            logger.info(f"🔍 Available chains: {list(self.chains.keys())}")
            logger.info(f"🔍 Available workflows: {list(self.workflows.keys())}")
            
            # Prepare input
            input_data = {"question": question}
            logger.info(f"📝 Input data: {input_data}")
            
            # Execute based on method
            if method == "naive":
                logger.info(f"🔗 Using naive chain: {type(self.chains['naive_rag'])}")
                logger.info(f"🔗 Chain has invoke method: {hasattr(self.chains['naive_rag'], 'invoke')}")
                result = self.chains["naive_rag"].invoke(input_data)
            elif method == "semantic":
                logger.info(f"🔗 Using semantic chain: {type(self.chains['semantic_rag'])}")
                logger.info(f"🔗 Chain has invoke method: {hasattr(self.chains['semantic_rag'], 'invoke')}")
                result = self.chains["semantic_rag"].invoke(input_data)
            elif method == "tool" and self.tool_retriever:
                logger.info(f"🔗 Using tool chain: {type(self.chains['tool_search'])}")
                logger.info(f"🔗 Chain has invoke method: {hasattr(self.chains['tool_search'], 'invoke')}")
                result = self.chains["tool_search"].invoke(input_data)
            elif method == "hybrid" and self.tool_retriever:
                logger.info(f"🔗 Using hybrid chain: {type(self.chains['hybrid_rag'])}")
                logger.info(f"🔗 Chain has invoke method: {hasattr(self.chains['hybrid_rag'], 'invoke')}")
                result = self.chains["hybrid_rag"].invoke(input_data)
            elif method == "production":
                logger.info(f"🔄 Using production workflow: {type(self.workflows['production_rag'])}")
                logger.info(f"🔄 Workflow has invoke method: {hasattr(self.workflows['production_rag'], 'invoke')}")
                result = self.workflows["production_rag"].invoke(input_data)
            else:
                raise ValueError(f"Invalid method: {method}")
            
            logger.info(f"✅ Chain/workflow execution completed, result type: {type(result)}")
            
            # Add confidence scoring if requested
            if include_confidence and "response" in result:
                logger.info(f"🎯 Adding confidence scoring")
                confidence_result = self.chains["confidence"].invoke({
                    "question": question,
                    "context": result.get("context", []),
                    "response": result["response"]
                })
                result["confidence"] = confidence_result.get("confidence", {})
            
            # Add metadata
            result["metadata"] = {
                "method": method,
                "question": question,
                "timestamp": str(datetime.now()),
                "system_version": "1.0.0"
            }
            
            logger.info(f"✅ Query processed successfully using {method}")
            return result
            
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"🔍 Error type: {type(e)}")
            logger.error(f"🔍 Error args: {e.args}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
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
            "retrievers": {
                "naive": self.naive_retriever.get_retriever_stats(),
                "semantic": self.semantic_retriever.get_retriever_stats(),
                "tool": self.tool_retriever.get_retriever_stats() if self.tool_retriever else None
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
            
            # Check retrievers
            health_status["components"]["retrievers"] = {
                "naive": True,
                "semantic": True,
                "tool": self.tool_retriever is not None
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
