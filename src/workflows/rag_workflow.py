"""
LangGraph workflows for complex RAG operations.

Contains workflow definitions for production RAG pipeline.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from src.workflows.state import (
    ProductionRAGState,
    HybridRAGState,
    MultiRetrievalState,
    EvaluationWorkflowState
)
from src.core.retrieval import (
    NaiveRetriever,
    SemanticRetriever,
    ToolBasedRetriever,
    RetrievalPipeline
)
from src.chains.rag_chains import RAGChainBuilder
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError
from src.config.settings import get_config

logger = get_logger(__name__)
config = get_config()


class RAGWorkflowBuilder:
    """Builder for LangGraph RAG workflows."""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize RAG workflow builder.
        
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
        
        self.chain_builder = RAGChainBuilder(self.llm)
        logger.info(f"🔄 Initialized RAG workflow builder with {self.llm.model_name}")
    
    def create_naive_retrieval_node(self, retriever: NaiveRetriever):
        """
        Create naive retrieval node.
        
        Args:
            retriever: Naive retriever instance
            
        Returns:
            Node function for naive retrieval
        """
        def naive_retrieve(state: ProductionRAGState) -> ProductionRAGState:
            """Naive retrieval node."""
            try:
                logger.info(f"🔍 [Naive] Retrieving for: {state['question'][:50]}...")
                
                # Retrieve documents
                documents = retriever.retrieve_documents(state["question"])
                
                # Update state
                state["naive_context"] = documents
                state["retrieval_method"] = "naive"
                state["metadata"]["naive_retrieval"] = {
                    "num_documents": len(documents),
                    "retriever_name": retriever.name
                }
                
                logger.info(f"📚 [Naive] Retrieved {len(documents)} documents")
                return state
                
            except Exception as e:
                logger.error(f"❌ Naive retrieval failed: {str(e)}")
                state["error_handling"] = f"Naive retrieval failed: {str(e)}"
                return state
        
        return naive_retrieve
    
    def create_semantic_retrieval_node(self, retriever: SemanticRetriever):
        """
        Create semantic retrieval node.
        
        Args:
            retriever: Semantic retriever instance
            
        Returns:
            Node function for semantic retrieval
        """
        def semantic_retrieve(state: ProductionRAGState) -> ProductionRAGState:
            """Semantic retrieval node."""
            try:
                logger.info(f"🧠 [Semantic] Retrieving for: {state['question'][:50]}...")
                
                # Retrieve documents
                documents = retriever.retrieve_documents(state["question"])
                
                # Update state
                state["semantic_context"] = documents
                state["retrieval_method"] = "semantic"
                state["metadata"]["semantic_retrieval"] = {
                    "num_documents": len(documents),
                    "retriever_name": retriever.name
                }
                
                logger.info(f"📚 [Semantic] Retrieved {len(documents)} documents")
                return state
                
            except Exception as e:
                logger.error(f"❌ Semantic retrieval failed: {str(e)}")
                state["error_handling"] = f"Semantic retrieval failed: {str(e)}"
                return state
        
        return semantic_retrieve
    
    def create_tool_retrieval_node(self, retriever: ToolBasedRetriever):
        """
        Create tool retrieval node.
        
        Args:
            retriever: Tool-based retriever instance
            
        Returns:
            Node function for tool retrieval
        """
        def tool_retrieve(state: ProductionRAGState) -> ProductionRAGState:
            """Tool retrieval node."""
            try:
                logger.info(f"🔧 [Tool] Retrieving for: {state['question'][:50]}...")
                
                # Retrieve documents
                documents = retriever.retrieve_documents(state["question"])
                
                # Update state
                state["tool_context"] = documents
                state["retrieval_method"] = "tool_based"
                state["metadata"]["tool_retrieval"] = {
                    "num_documents": len(documents),
                    "retriever_name": retriever.name
                }
                
                logger.info(f"📚 [Tool] Retrieved {len(documents)} documents")
                return state
                
            except Exception as e:
                logger.error(f"❌ Tool retrieval failed: {str(e)}")
                state["error_handling"] = f"Tool retrieval failed: {str(e)}"
                return state
        
        return tool_retrieve
    
    def create_context_combination_node(self):
        """
        Create context combination node.
        
        Returns:
            Node function for context combination
        """
        def combine_contexts(state: ProductionRAGState) -> ProductionRAGState:
            """Context combination node."""
            try:
                logger.info("🔄 Combining contexts from multiple retrievers")
                
                # Combine all contexts
                combined_context = []
                
                # Add naive context
                if "naive_context" in state and state["naive_context"]:
                    combined_context.extend(state["naive_context"])
                
                # Add semantic context
                if "semantic_context" in state and state["semantic_context"]:
                    combined_context.extend(state["semantic_context"])
                
                # Add tool context
                if "tool_context" in state and state["tool_context"]:
                    combined_context.extend(state["tool_context"])
                
                # Remove duplicates based on content
                seen_content = set()
                unique_context = []
                for doc in combined_context:
                    content_key = doc.page_content[:100]  # First 100 chars as key
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        unique_context.append(doc)
                
                # Update state
                state["combined_context"] = unique_context
                state["metadata"]["context_combination"] = {
                    "total_documents": len(combined_context),
                    "unique_documents": len(unique_context),
                    "deduplication_applied": True
                }
                
                logger.info(f"✅ Combined {len(unique_context)} unique documents")
                return state
                
            except Exception as e:
                logger.error(f"❌ Context combination failed: {str(e)}")
                state["error_handling"] = f"Context combination failed: {str(e)}"
                return state
        
        return combine_contexts
    
    def create_generation_node(self):
        """
        Create text generation node.
        
        Returns:
            Node function for text generation
        """
        def generate_response(state: ProductionRAGState) -> ProductionRAGState:
            """Text generation node."""
            try:
                logger.info("🤖 Generating response")
                
                # Get context
                context = state.get("combined_context", [])
                question = state["question"]
                
                if not context:
                    state["response"] = "I don't have enough context to answer your question."
                    state["metadata"]["generation"] = {"warning": "No context available"}
                    return state
                
                # Format context
                context_text = "\n\n".join(doc.page_content for doc in context)
                
                # Generate response using LLM
                prompt = f"""You are a helpful assistant. Use the provided context to answer the question accurately.

Question: {question}

Context:
{context_text}

Please provide a helpful response based on the context above."""
                
                response = self.llm.invoke(prompt).content
                
                # Update state
                state["response"] = response
                state["metadata"]["generation"] = {
                    "context_length": len(context_text),
                    "num_context_docs": len(context),
                    "response_length": len(response)
                }
                
                logger.info(f"✅ Generated response ({len(response)} chars)")
                return state
                
            except Exception as e:
                logger.error(f"❌ Generation failed: {str(e)}")
                state["error_handling"] = f"Generation failed: {str(e)}"
                state["response"] = "I encountered an error while generating a response."
                return state
        
        return generate_response
    
    def create_production_rag_workflow(
        self,
        naive_retriever: NaiveRetriever,
        semantic_retriever: SemanticRetriever,
        tool_retriever: ToolBasedRetriever
    ):
        """
        Create production RAG workflow.
        
        Args:
            naive_retriever: Naive retriever instance
            semantic_retriever: Semantic retriever instance
            tool_retriever: Tool retriever instance
            
        Returns:
            Compiled LangGraph workflow
        """
        try:
            logger.info("🔄 Creating production RAG workflow")
            
            # Create nodes
            naive_node = self.create_naive_retrieval_node(naive_retriever)
            semantic_node = self.create_semantic_retrieval_node(semantic_retriever)
            tool_node = self.create_tool_retrieval_node(tool_retriever)
            combine_node = self.create_context_combination_node()
            generate_node = self.create_generation_node()
            
            # Create workflow
            workflow = StateGraph(ProductionRAGState)
            
            # Add nodes
            workflow.add_node("naive_retrieve", naive_node)
            workflow.add_node("semantic_retrieve", semantic_node)
            workflow.add_node("tool_retrieve", tool_node)
            workflow.add_node("combine_contexts", combine_node)
            workflow.add_node("generate_response", generate_node)
            
            # Add edges
            workflow.add_edge(START, "naive_retrieve")
            workflow.add_edge(START, "semantic_retrieve")
            workflow.add_edge(START, "tool_retrieve")
            workflow.add_edge("naive_retrieve", "combine_contexts")
            workflow.add_edge("semantic_retrieve", "combine_contexts")
            workflow.add_edge("tool_retrieve", "combine_contexts")
            workflow.add_edge("combine_contexts", "generate_response")
            workflow.add_edge("generate_response", END)
            
            # Compile workflow
            compiled_workflow = workflow.compile()
            
            logger.info("✅ Created production RAG workflow")
            return compiled_workflow
            
        except Exception as e:
            error_msg = f"Failed to create production RAG workflow: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def create_hybrid_rag_workflow(
        self,
        naive_retriever: NaiveRetriever,
        tool_retriever: ToolBasedRetriever
    ):
        """
        Create hybrid RAG workflow combining knowledge base and search.
        
        Args:
            naive_retriever: Naive retriever for knowledge base
            tool_retriever: Tool retriever for search
            
        Returns:
            Compiled LangGraph workflow
        """
        try:
            logger.info("🔄 Creating hybrid RAG workflow")
            
            # Create nodes
            naive_node = self.create_naive_retrieval_node(naive_retriever)
            tool_node = self.create_tool_retrieval_node(tool_retriever)
            combine_node = self.create_context_combination_node()
            generate_node = self.create_generation_node()
            
            # Create workflow
            workflow = StateGraph(HybridRAGState)
            
            # Add nodes
            workflow.add_node("knowledge_retrieve", naive_node)
            workflow.add_node("search_retrieve", tool_node)
            workflow.add_node("combine_contexts", combine_node)
            workflow.add_node("generate_response", generate_node)
            
            # Add edges
            workflow.add_edge(START, "knowledge_retrieve")
            workflow.add_edge(START, "search_retrieve")
            workflow.add_edge("knowledge_retrieve", "combine_contexts")
            workflow.add_edge("search_retrieve", "combine_contexts")
            workflow.add_edge("combine_contexts", "generate_response")
            workflow.add_edge("generate_response", END)
            
            # Compile workflow
            compiled_workflow = workflow.compile()
            
            logger.info("✅ Created hybrid RAG workflow")
            return compiled_workflow
            
        except Exception as e:
            error_msg = f"Failed to create hybrid RAG workflow: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e


def create_rag_workflow_builder(llm: Optional[ChatOpenAI] = None) -> RAGWorkflowBuilder:
    """
    Create a RAG workflow builder instance.
    
    Args:
        llm: Language model to use
        
    Returns:
        RAG workflow builder instance
    """
    return RAGWorkflowBuilder(llm)


def create_production_workflows(
    naive_retriever: NaiveRetriever,
    semantic_retriever: SemanticRetriever,
    tool_retriever: Optional[ToolBasedRetriever],
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Create all production RAG workflows.
    
    Args:
        naive_retriever: Naive retriever instance
        semantic_retriever: Semantic retriever instance
        tool_retriever: Tool retriever instance (can be None)
        llm: Language model to use
        
    Returns:
        Dictionary containing all workflows
    """
    builder = create_rag_workflow_builder(llm)
    
    workflows = {}
    
    # Only create workflows that require tool retriever if it's available
    if tool_retriever:
        workflows.update({
            "production_rag": builder.create_production_rag_workflow(
                naive_retriever, semantic_retriever, tool_retriever
            ),
            "hybrid_rag": builder.create_hybrid_rag_workflow(
                naive_retriever, tool_retriever
            )
        })
    else:
        logger.warning("⚠️ No tool retriever available, skipping tool-based workflows")
    
    logger.info(f"✅ Created {len(workflows)} production RAG workflows")
    return workflows
