"""
LCEL chain compositions for the RAG system.

Contains LangChain Expression Language chains for production RAG pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
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
        
        # # Initialize storage for retrievers
        # self.retrievers = {
        #     'naive': None,
        #     'bm25': None,
        #     'parent_document': None,
        #     'contextual_compression': None,
        #     'multi_query': None,
        #     'semantic_chunking': None
        # }
        
        # Initialize list to store all created retrievers
        self.retriever_list = []
        
        logger.info(f"🔗 Initialized RAG chain builder with {self.llm.model_name}")
    
    def _logged_llm_call(self, prompt_input):
        """Wrapper for LLM calls with detailed logging."""
        logger.info(f"🤖 [LLM GENERATION] Starting LLM generation")
        logger.info(f"🤖 [LLM GENERATION] LLM model: {self.llm.model_name}")
        logger.info(f"🤖 [LLM GENERATION] Prompt input type: {type(prompt_input)}")
        
        if isinstance(prompt_input, dict):
            logger.info(f"🤖 [LLM GENERATION] Prompt keys: {list(prompt_input.keys())}")
            if 'question' in prompt_input:
                logger.info(f"🤖 [LLM GENERATION] Question: '{prompt_input['question']}'")
            if 'context' in prompt_input:
                context = prompt_input['context']
                if isinstance(context, list):
                    logger.info(f"🤖 [LLM GENERATION] Context: {len(context)} documents")
                    for i, doc in enumerate(context):
                        logger.info(f"  📄 Context Doc {i+1}: {str(doc)[:100]}...")
                else:
                    logger.info(f"🤖 [LLM GENERATION] Context: {str(context)[:200]}...")
        
        logger.info(f"🤖 [LLM GENERATION] Calling LLM...")
        response = self.llm.invoke(prompt_input)
        logger.info(f"✅ [LLM GENERATION] LLM response received")
        logger.info(f"🤖 [LLM GENERATION] Response type: {type(response)}")
        logger.info(f"🤖 [LLM GENERATION] Response preview: {str(response)[:200]}...")
        
        return response
    
    def create_naive_rag_chain(self, documents: List[Document], embeddings: OpenAIEmbeddings, k: int = 10):
        """
        Create naive RAG chain using LCEL with simple Qdrant vectorstore.
        
        Args:
            documents: List of documents to create vectorstore from
            embeddings: Embeddings model to use
            k: Number of documents to retrieve (default: 10)
            
        Returns:
            Tuple of (LCEL chain for naive RAG, retriever)
        """
        try:
            logger.info("🔗 Creating naive RAG chain")
            logger.info(f"📚 Documents count: {len(documents)}")
            logger.info(f"🔍 Embeddings model: {embeddings.model}")
            logger.info(f"🔍 k value: {k}")
            
            # Create vectorstore from documents (in-memory)
            logger.info("🗃️ Creating vectorstore from documents...")
            vectorstore = QdrantVectorStore.from_documents(
                documents,
                embeddings,
                location=":memory:",
                collection_name="Naive_RAG_Collection"
            )
            logger.info("✅ Vectorstore created")
            
            # Create simple retriever
            logger.info(f"🔍 Creating retriever with k={k}")
            naive_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            logger.info("✅ Retriever created")
            
            # Store retriever at class level
            # self.retrievers['naive'] = naive_retriever
            self.retriever_list.append(naive_retriever)
            logger.info(f"📌 Added naive retriever to class retriever_list (total: {len(self.retriever_list)})")
            
            # Create the chain using LCEL with simple pattern
            chain = (
                # {"question": "<<user question>>"}
                {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
                # Pass context through
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response
                | {"response": self.rag_prompt | self.llm | StrOutputParser(), "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created naive RAG chain: {type(chain)}")
            return chain, naive_retriever
            
        except Exception as e:
            error_msg = f"Failed to create naive RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_semantic_rag_chain(self, semantic_retriever: SemanticRetriever):
        """
        Create semantic RAG chain using LCEL with semantic retriever.
        
        Args:
            semantic_retriever: Semantic retriever instance
            
        Returns:
            Tuple of (LCEL chain for semantic RAG, semantic_retriever)
        """
        try:
            logger.info("🔗 Creating semantic RAG chain")
            logger.info(f"🔍 semantic_retriever type: {type(semantic_retriever)}")
            
            # Create a wrapper function with detailed logging
            def logged_semantic_retrieve(question: str):
                logger.info(f"🧠 [SEMANTIC CHAIN] Starting retrieval for question: '{question}'")
                docs = semantic_retriever.retrieve_documents(question)
                logger.info(f"📚 [SEMANTIC CHAIN] Retrieved {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning(f"⚠️ [SEMANTIC CHAIN] NO DOCUMENTS RETRIEVED!")
                else:
                    logger.info(f"📚 [SEMANTIC CHAIN] Document details:")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                        logger.info(f"  📄 Doc {i+1}: {content_preview}...")
                        logger.info(f"      Metadata: {doc.metadata}")
                
                return docs
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(logged_semantic_retrieve), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response with logging
                | {"response": self.rag_prompt | self._logged_llm_call | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created semantic RAG chain: {type(chain)}")
            return chain, semantic_retriever
            
        except Exception as e:
            error_msg = f"Failed to create semantic RAG chain: {str(e)}"
            logger.error(error_msg)
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
            return chain, tool_retriever
            
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
            return chain, (naive_retriever, tool_retriever)
            
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
            return chain, None
            
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
            return chain, (naive_retriever, semantic_retriever, tool_retriever) if tool_retriever else (naive_retriever, semantic_retriever)
            
        except Exception as e:
            error_msg = f"Failed to create production RAG chain: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def create_bm25_rag_chain(self, documents: List[Any]):
        """
        Create BM25 RAG chain using LCEL.
        
        Args:
            documents: List of documents to create BM25 retriever from
            
        Returns:
            LCEL chain for BM25 RAG
        """
        try:
            logger.info("🔗 Creating BM25 RAG chain")
            logger.info(f"📚 Documents count: {len(documents)}")
            
            # Create BM25 retriever from documents
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 5  # Set number of documents to retrieve
            
            # Store retriever at class level
            # self.retrievers['bm25'] = bm25_retriever
            self.retriever_list.append(bm25_retriever)
            logger.info(f"📌 Added BM25 retriever to class retriever_list (total: {len(self.retriever_list)})")
            
            # Create a wrapper function with detailed logging
            def logged_bm25_retrieve(question: str):
                logger.info(f"🔍 [BM25 CHAIN] Starting BM25 retrieval for question: '{question}'")
                docs = bm25_retriever.invoke(question)
                logger.info(f"📚 [BM25 CHAIN] Retrieved {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning(f"⚠️ [BM25 CHAIN] NO DOCUMENTS RETRIEVED!")
                else:
                    logger.info(f"📚 [BM25 CHAIN] Document details:")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                        logger.info(f"  📄 Doc {i+1}: {content_preview}...")
                        logger.info(f"      Metadata: {doc.metadata}")
                
                return docs
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(logged_bm25_retrieve), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response with logging
                | {"response": self.rag_prompt | self._logged_llm_call | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created BM25 RAG chain: {type(chain)}")
            return chain, bm25_retriever
            
        except Exception as e:
            error_msg = f"Failed to create BM25 RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_contextual_compression_rag_chain(self, base_retriever: Any):
        """
        Create contextual compression RAG chain using Cohere reranking.
        
        Args:
            base_retriever: Base retriever to compress
            
        Returns:
            LCEL chain for contextual compression RAG
        """
        try:
            logger.info("🔗 Creating contextual compression RAG chain")
            
            # Create Cohere reranker
            compressor = CohereRerank(model="rerank-v3.5")
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=base_retriever
            )
            
            # Store retriever at class level
            # self.retrievers['contextual_compression'] = compression_retriever
            self.retriever_list.append(compression_retriever)
            logger.info(f"📌 Added contextual compression retriever to class retriever_list (total: {len(self.retriever_list)})")
            
            # Create a wrapper function with detailed logging
            def logged_compression_retrieve(question: str):
                logger.info(f"🧠 [COMPRESSION CHAIN] Starting compression retrieval for question: '{question}'")
                docs = compression_retriever.invoke(question)
                logger.info(f"📚 [COMPRESSION CHAIN] Retrieved {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning(f"⚠️ [COMPRESSION CHAIN] NO DOCUMENTS RETRIEVED!")
                else:
                    logger.info(f"📚 [COMPRESSION CHAIN] Document details:")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                        logger.info(f"  📄 Doc {i+1}: {content_preview}...")
                        logger.info(f"      Metadata: {doc.metadata}")
                
                return docs
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(logged_compression_retrieve), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response with logging
                | {"response": self.rag_prompt | self._logged_llm_call | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created contextual compression RAG chain: {type(chain)}")
            return chain, compression_retriever
            
        except Exception as e:
            error_msg = f"Failed to create contextual compression RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_multi_query_rag_chain(self, base_retriever: Any):
        """
        Create multi-query RAG chain using LCEL.
        
        Args:
            base_retriever: Base retriever to use for multi-query
            
        Returns:
            LCEL chain for multi-query RAG
        """
        try:
            logger.info("🔗 Creating multi-query RAG chain")
            
            # Create multi-query retriever
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever, 
                llm=self.llm
            )
            
            # Store retriever at class level
            # self.retrievers['multi_query'] = multi_query_retriever
            self.retriever_list.append(multi_query_retriever)
            logger.info(f"📌 Added multi-query retriever to class retriever_list (total: {len(self.retriever_list)})")
            
            # Create a wrapper function with detailed logging
            def logged_multi_query_retrieve(question: str):
                logger.info(f"🔍 [MULTI-QUERY CHAIN] Starting multi-query retrieval for question: '{question}'")
                docs = multi_query_retriever.invoke(question)
                logger.info(f"📚 [MULTI-QUERY CHAIN] Retrieved {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning(f"⚠️ [MULTI-QUERY CHAIN] NO DOCUMENTS RETRIEVED!")
                else:
                    logger.info(f"📚 [MULTI-QUERY CHAIN] Document details:")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                        logger.info(f"  📄 Doc {i+1}: {content_preview}...")
                        logger.info(f"      Metadata: {doc.metadata}")
                
                return docs
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(logged_multi_query_retrieve), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response with logging
                | {"response": self.rag_prompt | self._logged_llm_call | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created multi-query RAG chain: {type(chain)}")
            return chain, multi_query_retriever
            
        except Exception as e:
            error_msg = f"Failed to create multi-query RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_parent_document_rag_chain(self, documents: List[Any]):
        """
        Create parent document RAG chain using LCEL.
        
        Args:
            documents: List of documents to create parent document retriever from
            
        Returns:
            LCEL chain for parent document RAG
        """
        try:
            logger.info("🔗 Creating parent document RAG chain")
            logger.info(f"📚 Documents count: {len(documents)}")
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Create child splitter
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)
            
            # Create Qdrant client and collection
            client = QdrantClient(location=":memory:")
            client.create_collection(
                collection_name="full_documents",
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )
            
            # Create parent document vectorstore
            parent_document_vectorstore = QdrantVectorStore(
                collection_name="full_documents", 
                embedding=embeddings, 
                client=client
            )
            
            # Create store
            store = InMemoryStore()
            
            # Create parent document retriever
            parent_document_retriever = ParentDocumentRetriever(
                vectorstore=parent_document_vectorstore,
                docstore=store,
                child_splitter=child_splitter,
            )
            
            # Add documents
            parent_document_retriever.add_documents(documents, ids=None)
            
            # Store retriever at class level
            # self.retrievers['parent_document'] = parent_document_retriever
            self.retriever_list.append(parent_document_retriever)
            logger.info(f"📌 Added parent document retriever to class retriever_list (total: {len(self.retriever_list)})")
            
            # Create a wrapper function with detailed logging
            def logged_parent_document_retrieve(question: str):
                logger.info(f"🔍 [PARENT DOC CHAIN] Starting parent document retrieval for question: '{question}'")
                docs = parent_document_retriever.invoke(question)
                logger.info(f"📚 [PARENT DOC CHAIN] Retrieved {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning(f"⚠️ [PARENT DOC CHAIN] NO DOCUMENTS RETRIEVED!")
                else:
                    logger.info(f"📚 [PARENT DOC CHAIN] Document details:")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                        logger.info(f"  📄 Doc {i+1}: {content_preview}...")
                        logger.info(f"      Metadata: {doc.metadata}")
                
                return docs
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(logged_parent_document_retrieve), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response with logging
                | {"response": self.rag_prompt | self._logged_llm_call | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created parent document RAG chain: {type(chain)}")
            return chain, parent_document_retriever
            
        except Exception as e:
            error_msg = f"Failed to create parent document RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_ensemble_rag_chain(self, retriever_list: List[Any]):
        """
        Create ensemble RAG chain using LCEL.
        
        Args:
            retriever_list: List of retrievers to combine
            
        Returns:
            LCEL chain for ensemble RAG
        """
        try:
            logger.info("🔗 Creating ensemble RAG chain")
            logger.info(f"🔍 Number of retrievers: {len(retriever_list)}")
            
            # Create equal weighting
            equal_weighting = [1/len(retriever_list)] * len(retriever_list)
            
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=retriever_list, 
                weights=equal_weighting
            )
            
            # Create a wrapper function with detailed logging
            def logged_ensemble_retrieve(question: str):
                logger.info(f"🔍 [ENSEMBLE CHAIN] Starting ensemble retrieval for question: '{question}'")
                docs = ensemble_retriever.invoke(question)
                logger.info(f"📚 [ENSEMBLE CHAIN] Retrieved {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning(f"⚠️ [ENSEMBLE CHAIN] NO DOCUMENTS RETRIEVED!")
                else:
                    logger.info(f"📚 [ENSEMBLE CHAIN] Document details:")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                        logger.info(f"  📄 Doc {i+1}: {content_preview}...")
                        logger.info(f"      Metadata: {doc.metadata}")
                
                return docs
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(logged_ensemble_retrieve), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response with logging
                | {"response": self.rag_prompt | self._logged_llm_call | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created ensemble RAG chain: {type(chain)}")
            return chain, ensemble_retriever
            
        except Exception as e:
            error_msg = f"Failed to create ensemble RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_semantic_chunking_rag_chain(self, documents: List[Any]):
        """
        Create semantic chunking RAG chain using LCEL.
        
        Args:
            documents: List of documents to create semantic chunking retriever from
            
        Returns:
            LCEL chain for semantic chunking RAG
        """
        try:
            logger.info("🔗 Creating semantic chunking RAG chain")
            logger.info(f"📚 Documents count: {len(documents)}")
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Create semantic chunker
            semantic_chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile"
            )
            
            # Split documents semantically (limit to first 20 for performance)
            semantic_documents = semantic_chunker.split_documents(documents[:20])
            logger.info(f"📚 Created {len(semantic_documents)} semantic chunks")
            
            # Create semantic vectorstore
            from qdrant_client import QdrantClient
            client = QdrantClient(location=":memory:")
            
            semantic_vectorstore = QdrantVectorStore.from_documents(
                semantic_documents,
                embeddings,
                client=client,
                collection_name="Synthetic_Usecase_Data_Semantic_Chunks"
            )
            
            # Create semantic retriever
            semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})
            
            # Store retriever at class level
            # self.retrievers['semantic_chunking'] = semantic_retriever
            self.retriever_list.append(semantic_retriever)
            logger.info(f"📌 Added semantic chunking retriever to class retriever_list (total: {len(self.retriever_list)})")
            
            # Create a wrapper function with detailed logging
            def logged_semantic_chunking_retrieve(question: str):
                logger.info(f"🔍 [SEMANTIC CHUNKING CHAIN] Starting semantic chunking retrieval for question: '{question}'")
                docs = semantic_retriever.invoke(question)
                logger.info(f"📚 [SEMANTIC CHUNKING CHAIN] Retrieved {len(docs)} documents")
                
                if len(docs) == 0:
                    logger.warning(f"⚠️ [SEMANTIC CHUNKING CHAIN] NO DOCUMENTS RETRIEVED!")
                else:
                    logger.info(f"📚 [SEMANTIC CHUNKING CHAIN] Document details:")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else "NO CONTENT"
                        logger.info(f"  📄 Doc {i+1}: {content_preview}...")
                        logger.info(f"      Metadata: {doc.metadata}")
                
                return docs
            
            # Create the chain using LCEL
            chain = (
                # Input: {"question": "user question"}
                {"context": itemgetter("question") | RunnableLambda(logged_semantic_chunking_retrieve), 
                 "question": itemgetter("question")}
                # Pass through context and format prompt
                | RunnablePassthrough.assign(context=itemgetter("context"))
                # Generate response with logging
                | {"response": self.rag_prompt | self._logged_llm_call | StrOutputParser(), 
                   "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created semantic chunking RAG chain: {type(chain)}")
            return chain, semantic_retriever
            
        except Exception as e:
            error_msg = f"Failed to create semantic chunking RAG chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RetrievalError(error_msg) from e
    
    def create_comprehensive_ensemble_chain(self, documents: List[Document], embeddings: OpenAIEmbeddings):
        """
        Create a comprehensive ensemble chain using retrievers from class retriever_list.
        
        Args:
            documents: List of documents (optional, may be needed for some retrievers)
            embeddings: Embeddings model to use (optional)
            
        Returns:
            Tuple of (LCEL chain, ensemble_retriever)
        """
        try:
            logger.info("🔗 Creating comprehensive ensemble chain")
            logger.info(f"📋 Current retriever_list has {len(self.retriever_list)} retrievers")
            
            # Use the class-level retriever_list
            retriever_list = self.retriever_list.copy() if self.retriever_list else []
            
            if len(retriever_list) < 2:
                error_msg = f"Need at least 2 retrievers for ensemble (got {len(retriever_list)}). Create some chains first."
                logger.error(error_msg)
                logger.info("💡 Tip: Call create methods like create_naive_rag_chain, create_bm25_rag_chain, etc. to add retrievers")
                raise RetrievalError(error_msg)
            
            logger.info(f"🔗 Combining {len(retriever_list)} retrievers for ensemble")
            
            # Create equal weighting
            equal_weighting = [1/len(retriever_list)] * len(retriever_list)
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=retriever_list,
                weights=equal_weighting
            )
            logger.info("✅ Created ensemble retriever")
            
            # Create the chain using LCEL
            ensemble_chain = (
                {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
                | RunnablePassthrough.assign(context=itemgetter("context"))
                | {"response": self.rag_prompt | self.llm | StrOutputParser(), "context": itemgetter("context")}
            )
            
            logger.info(f"✅ Created comprehensive ensemble chain with {len(retriever_list)} retrievers")
            return ensemble_chain, ensemble_retriever
            
        except Exception as e:
            error_msg = f"Failed to create comprehensive ensemble chain: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
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
    llm: Optional[ChatOpenAI] = None,
    documents: Optional[List[Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create all production RAG chains including advanced retrieval methods.
    
    Args:
        naive_retriever: Naive retriever instance (kept for backward compatibility, but naive_rag chain now uses documents directly)
        semantic_retriever: Semantic retriever instance
        tool_retriever: Tool retriever instance (can be None)
        llm: Language model to use
        documents: List of documents for advanced retrievers (optional)
        
    Returns:
        Tuple of (chains dictionary, retrievers dictionary)
    """
    builder = create_rag_chain_builder(llm)
    
    # Get embeddings from config
    from src.core.embeddings import get_default_embedding_provider
    embeddings = get_default_embedding_provider()
    
    # Create chains - for naive_rag, use documents directly if available
    chains = {}
    retrievers = {}
    
    # Create semantic chain
    semantic_chain, semantic_chain_retriever = builder.create_semantic_rag_chain(semantic_retriever)
    chains["semantic_rag"] = semantic_chain
    retrievers["semantic_rag"] = semantic_chain_retriever
    
    # Create confidence chain
    confidence_chain, _ = builder.create_confidence_chain()
    chains["confidence"] = confidence_chain
    retrievers["confidence"] = None
    
    # Add naive_rag chain if documents are provided
    if documents:
        try:
            naive_chain, naive_retriever = builder.create_naive_rag_chain(documents, embeddings, k=config.retrieval.default_k)
            chains["naive_rag"] = naive_chain
            retrievers["naive_rag"] = naive_retriever
        except Exception as e:
            logger.warning(f"⚠️ Failed to create naive chain: {e}")
    else:
        logger.warning("⚠️ No documents provided, skipping naive_rag chain")
    
    # Add tool-based chains only if tool retriever is available
    if tool_retriever:
        tool_chain, tool_ret = builder.create_tool_search_chain(tool_retriever)
        chains["tool_search"] = tool_chain
        retrievers["tool_search"] = tool_ret
        
        hybrid_chain, hybrid_rets = builder.create_hybrid_rag_chain(naive_retriever, tool_retriever)
        chains["hybrid_rag"] = hybrid_chain
        retrievers["hybrid_rag"] = hybrid_rets
        
        prod_chain, prod_rets = builder.create_production_rag_chain(
            naive_retriever, semantic_retriever, tool_retriever
        )
        chains["production_rag"] = prod_chain
        retrievers["production_rag"] = prod_rets
    else:
        # Create production chain without tool retriever
        prod_chain, prod_rets = builder.create_production_rag_chain(
            naive_retriever, semantic_retriever, None, use_hybrid=False
        )
        chains["production_rag"] = prod_chain
        retrievers["production_rag"] = prod_rets
    
    # Add advanced retrieval chains if documents are provided
    if documents:
        logger.info("🔗 Adding advanced retrieval chains")
        
        # BM25 RAG chain
        try:
            bm25_chain, bm25_ret = builder.create_bm25_rag_chain(documents)
            chains["bm25_rag"] = bm25_chain
            retrievers["bm25_rag"] = bm25_ret
        except Exception as e:
            logger.warning(f"⚠️ Failed to create BM25 chain: {e}")
        
        # Contextual compression RAG chain
        try:
            cc_chain, cc_ret = builder.create_contextual_compression_rag_chain(naive_retriever)
            chains["contextual_compression_rag"] = cc_chain
            retrievers["contextual_compression_rag"] = cc_ret
        except Exception as e:
            logger.warning(f"⚠️ Failed to create contextual compression chain: {e}")
        
        # Multi-query RAG chain
        try:
            mq_chain, mq_ret = builder.create_multi_query_rag_chain(naive_retriever)
            chains["multi_query_rag"] = mq_chain
            retrievers["multi_query_rag"] = mq_ret
        except Exception as e:
            logger.warning(f"⚠️ Failed to create multi-query chain: {e}")
        
        # Parent document RAG chain
        try:
            pd_chain, pd_ret = builder.create_parent_document_rag_chain(documents)
            chains["parent_document_rag"] = pd_chain
            retrievers["parent_document_rag"] = pd_ret
        except Exception as e:
            logger.warning(f"⚠️ Failed to create parent document chain: {e}")
        
        # Semantic chunking RAG chain
        try:
            sc_chain, sc_ret = builder.create_semantic_chunking_rag_chain(documents)
            chains["semantic_chunking_rag"] = sc_chain
            retrievers["semantic_chunking_rag"] = sc_ret
        except Exception as e:
            logger.warning(f"⚠️ Failed to create semantic chunking chain: {e}")
        
        # Ensemble RAG chain (combine multiple retrievers)
        try:
            retriever_list = []
            # Add available retrievers to the list
            if hasattr(naive_retriever, 'invoke'):
                retriever_list.append(naive_retriever)
            if hasattr(semantic_retriever, 'invoke'):
                retriever_list.append(semantic_retriever)
            
            # Create BM25 retriever for ensemble
            try:
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 5
                retriever_list.append(bm25_retriever)
            except Exception as e:
                logger.warning(f"⚠️ Failed to create BM25 retriever for ensemble: {e}")
            
            if len(retriever_list) >= 2:
                ens_chain, ens_ret = builder.create_ensemble_rag_chain(retriever_list)
                chains["ensemble_rag"] = ens_chain
                retrievers["ensemble_rag"] = ens_ret
            else:
                logger.warning("⚠️ Not enough retrievers for ensemble chain")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to create ensemble chain: {e}")
    
    logger.info(f"✅ Created {len(chains)} production RAG chains")
    return chains, retrievers


def create_advanced_retrieval_chains(
    documents: List[Any],
    naive_retriever: Optional[NaiveRetriever] = None,
    llm: Optional[ChatOpenAI] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create all advanced retrieval chains for testing and evaluation.
    
    Args:
        documents: List of documents for advanced retrievers
        naive_retriever: Optional naive retriever for compression and multi-query
        llm: Language model to use
        
    Returns:
        Tuple of (chains dictionary, retrievers dictionary)
    """
    builder = create_rag_chain_builder(llm)
    chains = {}
    retrievers = {}
    
    logger.info("🔗 Creating advanced retrieval chains for testing and evaluation")
    
    # BM25 RAG chain
    try:
        bm25_chain, bm25_ret = builder.create_bm25_rag_chain(documents)
        chains["bm25_rag"] = bm25_chain
        retrievers["bm25_rag"] = bm25_ret
        logger.info("✅ Created BM25 RAG chain")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create BM25 chain: {e}")
    
    # Contextual compression RAG chain (requires base retriever)
    if naive_retriever:
        try:
            cc_chain, cc_ret = builder.create_contextual_compression_rag_chain(naive_retriever)
            chains["contextual_compression_rag"] = cc_chain
            retrievers["contextual_compression_rag"] = cc_ret
            logger.info("✅ Created contextual compression RAG chain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create contextual compression chain: {e}")
    
    # Multi-query RAG chain (requires base retriever)
    if naive_retriever:
        try:
            mq_chain, mq_ret = builder.create_multi_query_rag_chain(naive_retriever)
            chains["multi_query_rag"] = mq_chain
            retrievers["multi_query_rag"] = mq_ret
            logger.info("✅ Created multi-query RAG chain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create multi-query chain: {e}")
    
    # Parent document RAG chain
    try:
        pd_chain, pd_ret = builder.create_parent_document_rag_chain(documents)
        chains["parent_document_rag"] = pd_chain
        retrievers["parent_document_rag"] = pd_ret
        logger.info("✅ Created parent document RAG chain")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create parent document chain: {e}")
    
    # Semantic chunking RAG chain
    try:
        sc_chain, sc_ret = builder.create_semantic_chunking_rag_chain(documents)
        chains["semantic_chunking_rag"] = sc_chain
        retrievers["semantic_chunking_rag"] = sc_ret
        logger.info("✅ Created semantic chunking RAG chain")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create semantic chunking chain: {e}")
    
    # Ensemble RAG chain (combine multiple retrievers)
    try:
        retriever_list = []
        
        # Create BM25 retriever for ensemble
        try:
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 5
            retriever_list.append(bm25_retriever)
        except Exception as e:
            logger.warning(f"⚠️ Failed to create BM25 retriever for ensemble: {e}")
        
        # Add naive retriever if available
        if naive_retriever and hasattr(naive_retriever, 'invoke'):
            retriever_list.append(naive_retriever)
        
        if len(retriever_list) >= 2:
            ens_chain, ens_ret = builder.create_ensemble_rag_chain(retriever_list)
            chains["ensemble_rag"] = ens_chain
            retrievers["ensemble_rag"] = ens_ret
            logger.info("✅ Created ensemble RAG chain")
        else:
            logger.warning("⚠️ Not enough retrievers for ensemble chain")
            
    except Exception as e:
        logger.warning(f"⚠️ Failed to create ensemble chain: {e}")
    
    logger.info(f"✅ Created {len(chains)} advanced retrieval chains")
    return chains, retrievers
