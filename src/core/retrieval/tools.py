"""
Tool-based search retriever implementation.

Production-grade retriever using external tools (Tavily) for real-time information.
"""

from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from src.core.retrieval.base import BaseRAGRetriever, RetrievalResult
from src.utils.logging import get_logger
from src.utils.exceptions import RetrievalError, APIKeyError
from src.utils.decorators import timing_decorator, retry_decorator
from src.config.settings import get_config

logger = get_logger(__name__)


class ToolBasedRetriever(BaseRAGRetriever):
    """Tool-based retriever using external search tools."""
    
    def __init__(
        self,
        search_tool: BaseTool,
        k: int = 5,
        max_results: int = 10,
        include_snippets: bool = True
    ):
        """
        Initialize tool-based retriever.
        
        Args:
            search_tool: Search tool instance (e.g., Tavily)
            k: Number of documents to retrieve
            max_results: Maximum results from search tool
            include_snippets: Whether to include search snippets
            
        Raises:
            APIKeyError: If required API keys are missing
        """
        super().__init__("tool_based", k)
        self.search_tool = search_tool
        self.max_results = max_results
        self.include_snippets = include_snippets
        
        # Validate API keys
        self._validate_api_keys()
        
        logger.info(f"🔧 Initialized tool-based retriever (k={k}, max_results={max_results})")
    
    def _validate_api_keys(self) -> None:
        """Validate that required API keys are present."""
        config = get_config()
        
        # Check for Tavily API key if using Tavily tool
        if hasattr(self.search_tool, 'name') and 'tavily' in self.search_tool.name.lower():
            if not config.tavily_api_key:
                raise APIKeyError("Tavily API key is required for tool-based search")
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using external search tool.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔧 [Tool-Based] Searching for: {query[:50]}...")
            
            # Use search tool to get results
            # Handle both BaseTool objects and function tools
            if hasattr(self.search_tool, 'invoke'):
                search_results = self.search_tool.invoke({"query": query})
            else:
                # Handle function-based tools (like Tavily factory tools)
                search_results = self.search_tool(query)
            
            # Convert search results to documents
            documents = self._convert_search_results_to_documents(search_results, query)
            
            # Limit to k documents
            if len(documents) > self.k:
                documents = documents[:self.k]
            
            logger.info(f"📚 [Tool-Based] Retrieved {len(documents)} documents from search")
            return documents
            
        except Exception as e:
            error_msg = f"Tool-based retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    @retry_decorator(max_retries=3, delay=1.0)
    @timing_decorator
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(f"🔧 [Tool-Based] Searching with scores for: {query[:50]}...")
            
            # Use search tool to get results
            # Handle both BaseTool objects and function tools
            if hasattr(self.search_tool, 'invoke'):
                search_results = self.search_tool.invoke({"query": query})
            else:
                # Handle function-based tools (like Tavily factory tools)
                search_results = self.search_tool(query)
            
            # Convert search results to documents with scores
            documents_with_scores = self._convert_search_results_with_scores(search_results, query)
            
            # Limit to k documents
            if len(documents_with_scores) > self.k:
                documents_with_scores = documents_with_scores[:self.k]
            
            logger.info(f"📚 [Tool-Based] Retrieved {len(documents_with_scores)} documents with scores")
            return documents_with_scores
            
        except Exception as e:
            error_msg = f"Tool-based retrieval with scores failed: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def _convert_search_results_to_documents(
        self, 
        search_results: Any, 
        query: str
    ) -> List[Document]:
        """
        Convert search tool results to Document objects.
        
        Args:
            search_results: Results from search tool
            query: Original query
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            # Handle different result formats
            if isinstance(search_results, list):
                results = search_results
            elif isinstance(search_results, dict) and 'results' in search_results:
                results = search_results['results']
            elif isinstance(search_results, dict) and 'answer' in search_results:
                # Single result format
                results = [search_results]
            else:
                # Fallback: treat as single result
                results = [search_results]
            
            for i, result in enumerate(results[:self.max_results]):
                # Extract content based on result format
                if isinstance(result, dict):
                    title = result.get('title', f'Search Result {i+1}')
                    content = result.get('content', '') or result.get('snippet', '') or result.get('answer', '')
                    url = result.get('url', '')
                    score = result.get('score', 1.0)
                else:
                    # String result
                    title = f'Search Result {i+1}'
                    content = str(result)
                    url = ''
                    score = 1.0
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata={
                        'title': title,
                        'url': url,
                        'source': 'tool_search',
                        'search_query': query,
                        'relevance_score': score,
                        'retrieval_method': 'tool_based',
                        'result_index': i
                    }
                )
                
                documents.append(doc)
                
        except Exception as e:
            logger.warning(f"⚠️ Error converting search results: {str(e)}")
            # Create fallback document
            fallback_doc = Document(
                page_content=f"Search results for: {query}",
                metadata={
                    'title': 'Search Results',
                    'source': 'tool_search',
                    'search_query': query,
                    'relevance_score': 0.5,
                    'retrieval_method': 'tool_based',
                    'error': str(e)
                }
            )
            documents.append(fallback_doc)
        
        return documents
    
    def _convert_search_results_with_scores(
        self, 
        search_results: Any, 
        query: str
    ) -> List[Tuple[Document, float]]:
        """
        Convert search tool results to (Document, score) tuples.
        
        Args:
            search_results: Results from search tool
            query: Original query
            
        Returns:
            List of (Document, float) tuples
        """
        documents = self._convert_search_results_to_documents(search_results, query)
        
        # Extract scores from documents
        results_with_scores = []
        for doc in documents:
            score = doc.metadata.get('relevance_score', 1.0)
            results_with_scores.append((doc, float(score)))
        
        return results_with_scores
    
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
                    metadata={"warning": "No search results found"}
                )
            
            documents, scores = zip(*docs_with_scores)
            
            # Extract additional metadata
            urls = [doc.metadata.get('url', '') for doc in documents]
            titles = [doc.metadata.get('title', '') for doc in documents]
            
            return RetrievalResult(
                documents=list(documents),
                scores=list(scores),
                retriever_name=self.name,
                query=query,
                metadata={
                    "avg_relevance": sum(scores) / len(scores),
                    "max_relevance": max(scores),
                    "min_relevance": min(scores),
                    "unique_urls": len(set(urls)),
                    "titles": titles,
                    "search_tool": self.search_tool.name if hasattr(self.search_tool, 'name') else 'unknown'
                }
            )
            
        except Exception as e:
            error_msg = f"Tool-based retrieval with result failed: {str(e)}"
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
            "max_results": self.max_results,
            "type": "tool_based_search",
            "search_tool": self.search_tool.name if hasattr(self.search_tool, 'name') else 'unknown',
            "include_snippets": self.include_snippets
        }


def create_tool_based_retriever(
    search_tool: BaseTool,
    k: int = 5,
    max_results: int = 10,
    include_snippets: bool = True
) -> ToolBasedRetriever:
    """
    Create a tool-based retriever instance.
    
    Args:
        search_tool: Search tool instance (e.g., Tavily)
        k: Number of documents to retrieve
        max_results: Maximum results from search tool
        include_snippets: Whether to include search snippets
        
    Returns:
        Tool-based retriever instance
    """
    return ToolBasedRetriever(search_tool, k, max_results, include_snippets)
