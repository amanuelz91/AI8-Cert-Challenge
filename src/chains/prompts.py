"""
Prompt templates for the RAG system.

Contains all prompt templates used in the RAG pipeline.
"""

from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import get_config

config = get_config()


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Get the main RAG prompt template.
    
    Returns:
        ChatPromptTemplate for RAG responses
    """
    RAG_TEMPLATE = """You are a helpful and knowledgeable assistant. Use the provided context to answer the question accurately and comprehensively.

Guidelines:
- Only use information from the provided context
- If the context doesn't contain enough information to answer the question, say so clearly
- Provide specific examples or details when available
- Be concise but thorough in your response
- If you're uncertain about any information, express that uncertainty

Question: {question}

Context:
{context}

Please provide a helpful response based on the context above."""

    return ChatPromptTemplate.from_template(RAG_TEMPLATE)


def get_confidence_prompt() -> ChatPromptTemplate:
    """
    Get the confidence scoring prompt template.
    
    Returns:
        ChatPromptTemplate for confidence scoring
    """
    CONFIDENCE_TEMPLATE = """Based on the provided context and your response, please provide a confidence score and reasoning.

Question: {question}
Context: {context}
Response: {response}

Please provide your assessment in the following JSON format:
{{
    "confidence_score": <score between 0.0 and 1.0>,
    "reasoning": "<explanation of the confidence score>",
    "context_utilization": <how well the context was used, 0.0 to 1.0>,
    "answer_completeness": <how complete the answer is, 0.0 to 1.0>
}}"""

    return ChatPromptTemplate.from_template(CONFIDENCE_TEMPLATE)


def get_tool_search_prompt() -> ChatPromptTemplate:
    """
    Get the tool search prompt template.
    
    Returns:
        ChatPromptTemplate for tool-based search
    """
    TOOL_SEARCH_TEMPLATE = """You are a research assistant. Use the search results to provide a comprehensive answer to the user's question.

Search Query: {query}
Search Results: {search_results}

Instructions:
- Synthesize information from multiple search results when relevant
- Cite sources when providing specific information
- If search results don't contain relevant information, state this clearly
- Provide a well-structured and informative response

Please provide your response based on the search results above."""

    return ChatPromptTemplate.from_template(TOOL_SEARCH_TEMPLATE)


def get_hybrid_prompt() -> ChatPromptTemplate:
    """
    Get the hybrid RAG prompt template (combining knowledge base and search results).
    
    Returns:
        ChatPromptTemplate for hybrid responses
    """
    HYBRID_TEMPLATE = """You are a comprehensive assistant that combines knowledge from multiple sources to provide the best possible answer.

Question: {question}

Knowledge Base Context:
{knowledge_context}

Search Results:
{search_context}

Instructions:
- Prioritize information from the knowledge base when it directly answers the question
- Use search results to supplement or provide additional context
- If there are conflicts between sources, note them and explain
- Provide a well-integrated response that draws from both sources
- Be clear about which information comes from which source

Please provide a comprehensive response that effectively combines both knowledge sources."""

    return ChatPromptTemplate.from_template(HYBRID_TEMPLATE)


def get_evaluation_prompt() -> ChatPromptTemplate:
    """
    Get the evaluation prompt template for RAGAS evaluation.
    
    Returns:
        ChatPromptTemplate for evaluation
    """
    EVALUATION_TEMPLATE = """You are an expert evaluator. Please assess the quality of the RAG response based on the given criteria.

Question: {question}
Context: {context}
Response: {response}

Please evaluate the response on the following criteria (provide scores from 0.0 to 1.0):

1. Faithfulness: Does the response accurately reflect the information in the context?
2. Answer Relevancy: How relevant is the response to the question?
3. Context Precision: How precise is the context used in the response?
4. Context Recall: How well does the context cover the information needed to answer the question?
5. Answer Correctness: How factually correct is the response?
6. Answer Completeness: How complete is the response?

Provide your evaluation in JSON format:
{{
    "faithfulness": <score>,
    "answer_relevancy": <score>,
    "context_precision": <score>,
    "context_recall": <score>,
    "answer_correctness": <score>,
    "answer_completeness": <score>,
    "overall_assessment": "<brief explanation of your evaluation>"
}}"""

    return ChatPromptTemplate.from_template(EVALUATION_TEMPLATE)


def get_query_expansion_prompt() -> ChatPromptTemplate:
    """
    Get the query expansion prompt template for multi-query retrieval.
    
    Returns:
        ChatPromptTemplate for query expansion
    """
    QUERY_EXPANSION_TEMPLATE = """You are an expert at expanding queries to improve information retrieval. Given a user's question, generate multiple alternative queries that could help find relevant information.

Original Question: {question}

Generate 3-5 alternative queries that:
- Use different terminology or synonyms
- Focus on different aspects of the question
- Vary in specificity (broader and narrower)
- Use different question formats

Provide your alternative queries as a JSON array:
{{
    "alternative_queries": [
        "<query 1>",
        "<query 2>",
        "<query 3>",
        "<query 4>",
        "<query 5>"
    ]
}}"""

    return ChatPromptTemplate.from_template(QUERY_EXPANSION_TEMPLATE)


def get_semantic_chunking_prompt() -> ChatPromptTemplate:
    """
    Get the semantic chunking prompt template for better document boundaries.
    
    Returns:
        ChatPromptTemplate for semantic chunking
    """
    SEMANTIC_CHUNKING_TEMPLATE = """You are an expert at identifying semantic boundaries in text. Analyze the following text and identify where natural semantic breaks occur.

Text: {text}

Identify semantic boundaries where:
- Topics or themes change
- New concepts are introduced
- Context shifts significantly
- Logical sections begin or end

Provide your analysis in JSON format:
{{
    "semantic_boundaries": [
        {{
            "position": <character position>,
            "reason": "<explanation of why this is a semantic boundary>",
            "confidence": <confidence score 0.0-1.0>
        }}
    ],
    "overall_structure": "<description of the overall text structure>"
}}"""

    return ChatPromptTemplate.from_template(SEMANTIC_CHUNKING_TEMPLATE)


# Export all prompts
__all__ = [
    "get_rag_prompt",
    "get_confidence_prompt", 
    "get_tool_search_prompt",
    "get_hybrid_prompt",
    "get_evaluation_prompt",
    "get_query_expansion_prompt",
    "get_semantic_chunking_prompt"
]
