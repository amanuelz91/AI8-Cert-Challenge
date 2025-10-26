#!/usr/bin/env python3
"""
Simple Naive RAG Demo

Run basic naive RAG queries using the existing naive RAG chain.

Usage:
    # Interactive mode
    python simple_naive_rag_demo.py
    
    # Single query
    python simple_naive_rag_demo.py --query "What is the Direct Loan Program?"
    
    # Use existing dataset for testing
    python simple_naive_rag_demo.py --use-dataset evaluation_results/evaluation_dataset_20251026_003352.csv
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ""

# Imports
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd

# Our imports
from src.chains.rag_chains import create_rag_chain_builder
from src.config.settings import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_documents(data_path: str = "src/data", max_docs: int = 50):
    """Load PDF documents from the data directory."""
    logger.info(f"📁 Loading documents from: {data_path}")
    
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    
    # Limit the number of documents for faster startup
    if len(docs) > max_docs:
        logger.info(f"⚠️ Limiting to first {max_docs} documents (had {len(docs)} total)")
        docs = docs[:max_docs]
    
    logger.info(f"✅ Loaded {len(docs)} documents")
    return docs


def create_naive_rag_chain(documents):
    """Create naive RAG chain."""
    logger.info("🔗 Creating naive RAG chain")
    
    config = get_config()
    llm = ChatOpenAI(
        model=config.llm.model_name,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
        openai_api_key=config.openai_api_key
    )
    
    embeddings = OpenAIEmbeddings(
        model=config.embedding.model_name,
        openai_api_key=config.openai_api_key
    )
    
    # Create builder
    builder = create_rag_chain_builder(llm)
    
    # Create naive RAG chain with smaller k for speed
    logger.info("📊 Creating Naive RAG chain...")
    chain, retriever = builder.create_naive_rag_chain(
        documents, embeddings, k=5  # Smaller k for faster responses
    )
    
    logger.info("✅ Naive RAG chain created")
    
    # Warmup query to avoid lazy initialization affecting first real query timing
    logger.info("🔥 Running warmup query...")
    try:
        import time
        warmup_start = time.time()
        _ = chain.invoke({"question": "test"})
        warmup_time = time.time() - warmup_start
        logger.info(f"✅ Warmup complete ({warmup_time:.2f}s)")
    except Exception as e:
        logger.warning(f"⚠️ Warmup query failed: {e}")
    
    return chain


def run_query(chain, question: str):
    """Run a single query through the chain - timing only the actual query processing."""
    import time
    
    logger.info(f"🔍 Question: {question}")
    logger.info("⏳ Processing query...")
    
    # Time ONLY the chain invocation - nothing else
    query_start = time.time()
    response = chain.invoke({"question": question})
    query_latency = time.time() - query_start
    
    # Extract results
    answer = response.get("response", "No response generated")
    context = response.get("context", [])
    
    logger.info(f"✅ Query complete - Latency: {query_latency:.2f}s")
    logger.info(f"📚 Retrieved {len(context)} context documents")
    
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "latency": query_latency
    }


def interactive_mode(chain):
    """Run in interactive mode."""
    logger.info("🎯 Interactive mode - Type 'exit' to quit")
    logger.info("=" * 80)
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'exit':
                logger.info("👋 Goodbye!")
                break
            
            result = run_query(chain, question)
            
            print("\n" + "=" * 80)
            print(f"✅ Answer ({result['latency']:.2f}s):")
            print("-" * 80)
            print(result['answer'])
            print("=" * 80)
            
            # Show context snippets
            if result['context']:
                print("\n📚 Context (first document):")
                print("-" * 80)
                first_doc = result['context'][0]
                if hasattr(first_doc, 'page_content'):
                    preview = first_doc.page_content[:300]
                    print(f"{preview}...")
                else:
                    print(str(first_doc)[:300])
                print("=" * 80)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")


def demo_dataset_queries(chain, dataset_path: str):
    """Run queries from a dataset."""
    logger.info(f"📊 Loading dataset from: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    questions = df['user_input'].tolist()
    
    logger.info(f"✅ Loaded {len(questions)} questions from dataset")
    logger.info("=" * 80)
    
    results = []
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] Processing question {i}")
        
        result = run_query(chain, question)
        results.append(result)
        
        print(f"\n✅ Answer ({result['latency']:.2f}s):")
        print("-" * 80)
        print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])
        print("=" * 80)
    
    # Summary
    avg_latency = sum(r['latency'] for r in results) / len(results)
    total_time = sum(r['latency'] for r in results)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 Summary:")
    logger.info(f"  Total queries: {len(results)}")
    logger.info(f"  Avg latency: {avg_latency:.2f}s")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run simple naive RAG queries"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a single query"
    )
    parser.add_argument(
        "--use-dataset",
        type=str,
        help="Run queries from a dataset CSV file"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50,
        help="Maximum number of documents to load (default: 50)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load documents
        logger.info("🚀 Starting Naive RAG Demo")
        docs = load_documents(max_docs=args.max_docs)
        
        # Create chain
        chain = create_naive_rag_chain(docs)
        
        # Run queries based on mode
        if args.use_dataset:
            demo_dataset_queries(chain, args.use_dataset)
        elif args.query:
            result = run_query(chain, args.query)
            print("\n" + "=" * 80)
            print(f"✅ Answer ({result['latency']:.2f}s):")
            print("-" * 80)
            print(result['answer'])
            print("=" * 80)
        else:
            interactive_mode(chain)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

