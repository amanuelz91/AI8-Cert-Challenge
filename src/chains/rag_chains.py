"""
LCEL chain compositions for the RAG system.

Contains LangChain Expression Language chains for production RAG pipeline.
"""

from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from src.chains.prompts import (
    get_rag_prompt,
    get_confidence_prompt,
    get_tool_search_prompt,
    get_hybrid_prompt
)
from src.core.retrieval import (
    NaiveRetriever,
    SemanticRetriever,
    ToolBasedRetriever,
    RetrievalResult
)
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.config.settings import get_config

logger = get_logger(__name__)
config = get_config()


class RAGChainBuilder:
    """Builder for RAG chains using LCEL."""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize RAG chain builder.
        
        Args:
            llm: Language model to use
        """
        self.llm = llm or ChatOpenAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
            openai_api_key=config.openai_api_key
        )
        
        # Initialize prompts
        self.rag_prompt = get_rag_prompt()
        self.confidence_prompt = get_confidence_prompt()
        self.tool_search_prompt = get_tool_search_prompt()
        self.hybrid_prompt = get_hybrid_prompt()
        
        logger.info(f"🔗 Initialized RAG chain builder with {self.llm.model_name}")
    
    def create_naive_rag_chain(self, naive_retriever: NaiveRetriever):
        """
        Create naive RAG chain using LCEL.
        
        Args:
            naive_retriever: Naive retriever instance
            
        Returns:
            LCEL chain for naive RAG
        """
        try:
            logger.info("🔗 Creating naive RAG chain")
            logger.info(f"🔍 naive_retriever type: {type(naive_retriever)}")
            logger.info(f"🔍 naive_retriever has retrieve_documents: {hasattr(naive_retriever, 'retrieve_documents')}")
            logger.info(f"🔍 llm type: {type(self.llm)}")
            logger.info(f"🔍 rag_prompt type: {type(self.rag_prompt)}")
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(naive_retriever.retrieve_documents), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response
                | {"response": self.rag_prompt | self.llm | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created naive RAG chain: {type(chain)}")
            logger.info(f"🔍 Chain has invoke: {hasattr(chain, 'invoke')}")
            return chain
            
        except Exception as e:
            error_msg = f"Failed to create naive RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_semantic_rag_chain(self, semantic_retriever: SemanticRetriever):
        """
        Create semantic RAG chain using LCEL.
        
        Args:
            semantic_retriever: Semantic retriever instance
            
        Returns:
            LCEL chain for semantic RAG
        """
        try:
            logger.info("🔗 Creating semantic RAG chain")
            logger.info(f"🔍 semantic_retriever type: {type(semantic_retriever)}")
            logger.info(f"🔍 semantic_retriever has retrieve_documents: {hasattr(semantic_retriever, 'retrieve_documents')}")
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(semantic_retriever.retrieve_documents), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response
                | {"response": self.rag_prompt | self.llm | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created semantic RAG chain: {type(chain)}")
            logger.info(f"🔍 Chain has invoke: {hasattr(chain, 'invoke')}")
            return chain
            
        except Exception as e:
            error_msg = f"Failed to create semantic RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_tool_search_chain(self, tool_retriever: ToolBasedRetriever):
        """
        Create tool-based search chain using LCEL.
        
        Args:
            tool_retriever: Tool-based retriever instance
            
        Returns:
            LCEL chain for tool-based search
        """
        try:
            logger.info("🔗 Creating tool search chain")
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"search_results": itemgetter("question") | RunnableLambda(tool_retriever.retrieve_documents), 
                 "query": itemgetter("question")}
                # Pass through search results and format prompt
                | RunnablePassthrough.assign(search_results=itemgetter("search_results"))
                # Generate response
                | {"response": self.tool_search_prompt | self.llm | StrOutputParser(), 
                   "search_results": itemgetter("search_results")}
            )
            
            logger.info("✅ Created tool search chain")
            return chain
            
        except Exception as e:
            error_msg = f"Failed to create tool search chain: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def create_hybrid_rag_chain(
        self, 
        naive_retriever: NaiveRetriever, 
        tool_retriever: ToolBasedRetriever
    ):
        """
        Create hybrid RAG chain combining knowledge base and search results.
        
        Args:
            naive_retriever: Naive retriever for knowledge base
            tool_retriever: Tool retriever for search results
            
        Returns:
            LCEL chain for hybrid RAG
        """
        try:
            logger.info("🔗 Creating hybrid RAG chain")
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"knowledge_context": itemgetter("question") | RunnableLambda(naive_retriever.retrieve_documents),
                 "search_context": itemgetter("question") | RunnableLambda(tool_retriever.retrieve_documents),
                 "question": itemgetter("question")}
                # Pass through contexts and format prompt
                | RunnablePassthrough.assign(
                    knowledge_context=itemgetter("knowledge_context"),
                    search_context=itemgetter("search_context")
                )
                # Generate response
                | {"response": self.hybrid_prompt | self.llm | StrOutputParser(),
                   "knowledge_context": itemgetter("knowledge_context"),
                   "search_context": itemgetter("search_context")}
            )
            
            logger.info("✅ Created hybrid RAG chain")
            return chain
            
        except Exception as e:
            error_msg = f"Failed to create hybrid RAG chain: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def create_confidence_chain(self):
        """
        Create confidence scoring chain using LCEL.
        
        Returns:
            LCEL chain for confidence scoring
        """
        try:
            logger.info("🔗 Creating confidence scoring chain")
            
            # Create the chain using LCEL - fix the syntax
            chain = (
                # Input: {"question": "question", "context": "context", "response": "response"}
                RunnablePassthrough.assign(
                    confidence=self.confidence_prompt | self.llm | JsonOutputParser()
                )
            )
            
            logger.info(f"✅ Created confidence scoring chain: {type(chain)}")
            logger.info(f"🔍 Chain has invoke: {hasattr(chain, 'invoke')}")
            return chain
            
        except Exception as e:
            error_msg = f"Failed to create confidence chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_production_rag_chain(
        self,
        naive_retriever: NaiveRetriever,
        semantic_retriever: SemanticRetriever,
        tool_retriever: Optional[ToolBasedRetriever],
        use_hybrid: bool = True
    ):
        """
        Create production RAG chain with multiple retrieval methods.
        
        Args:
            naive_retriever: Naive retriever instance
            semantic_retriever: Semantic retriever instance
            tool_retriever: Tool retriever instance
            use_hybrid: Whether to use hybrid approach
            
        Returns:
            LCEL chain for production RAG
        """
        try:
            logger.info("🔗 Creating production RAG chain")
            
            if use_hybrid and tool_retriever:
                # Use hybrid approach combining all methods
                chain = (
                    # Input: {"question": "user question"}
                    {"naive_context": itemgetter("question") | RunnableLambda(naive_retriever.retrieve_documents),
                     "semantic_context": itemgetter("question") | RunnableLambda(semantic_retriever.retrieve_documents),
                     "search_context": itemgetter("question") | RunnableLambda(tool_retriever.retrieve_documents),
                     "question": itemgetter("question")}
                    # Combine contexts
                    | RunnablePassthrough.assign(
                        combined_context=lambda x: {
                            "naive": x["naive_context"],
                            "semantic": x["semantic_context"],
                            "search": x["search_context"]
                        }
                    )
                    # Generate response
                    | {"response": self.hybrid_prompt | self.llm | StrOutputParser(),
                       "contexts": itemgetter("combined_context")}
                )
            else:
                # Use semantic retriever as primary with tool fallback
                chain = (
                    # Input: {"question": "user question"}
                    {"context": itemgetter("question") | RunnableLambda(semantic_retriever.retrieve_documents),
                     "question": itemgetter("question")}
                    # Pass through context and format prompt
                    | RunnablePassthrough.assign(context=itemgetter("context"))
                    # Generate response
                    | {"response": self.rag_prompt | self.llm | StrOutputParser(),
                       "context": itemgetter("context")}
                )
            
            logger.info("✅ Created production RAG chain")
            return chain
            
        except Exception as e:
            error_msg = f"Failed to create production RAG chain: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e


def create_rag_chain_builder(llm: Optional[ChatOpenAI] = None) -> RAGChainBuilder:
    """
    Create a RAG chain builder instance.
    
    Args:
        llm: Language model to use
        
    Returns:
        RAG chain builder instance
    """
    return RAGChainBuilder(llm)


def create_production_chains(
    naive_retriever: NaiveRetriever,
    semantic_retriever: SemanticRetriever,
    tool_retriever: Optional[ToolBasedRetriever],
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Create all production RAG chains.
    
    Args:
        naive_retriever: Naive retriever instance
        semantic_retriever: Semantic retriever instance
        tool_retriever: Tool retriever instance (can be None)
        llm: Language model to use
        
    Returns:
        Dictionary containing all chains
    """
    builder = create_rag_chain_builder(llm)
    
    chains = {
        "naive_rag": builder.create_naive_rag_chain(naive_retriever),
        "semantic_rag": builder.create_semantic_rag_chain(semantic_retriever),
        "confidence": builder.create_confidence_chain()
    }
    
    # Add tool-based chains only if tool retriever is available
    if tool_retriever:
        chains.update({
            "tool_search": builder.create_tool_search_chain(tool_retriever),
            "hybrid_rag": builder.create_hybrid_rag_chain(naive_retriever, tool_retriever),
            "production_rag": builder.create_production_rag_chain(
                naive_retriever, semantic_retriever, tool_retriever
            )
        })
    else:
        # Create production chain without tool retriever
        chains["production_rag"] = builder.create_production_rag_chain(
            naive_retriever, semantic_retriever, None, use_hybrid=False
        )
    
    logger.info(f"✅ Created {len(chains)} production RAG chains")
    return chains
