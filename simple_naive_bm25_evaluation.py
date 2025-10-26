#!/usr/bin/env python3
"""
Evaluation script for all retrieval chains (Naive, BM25, Contextual Compression, Multi-Query, Parent Document).

Usage:
    uv run python simple_naive_bm25_evaluation.py
    uv run python simple_naive_bm25_evaluation.py --dataset path/to/dataset.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Optional
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ""

# Data Preparation
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy, 
    ContextEntityRecall, 
    NoiseSensitivity
)
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.testset import TestsetGenerator

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Our RAG chain imports
from src.chains.rag_chains import create_rag_chain_builder
from src.utils.logging import get_logger
from src.config.settings import get_config

logger = get_logger(__name__)


def load_documents(data_path: str = "src/data", max_docs: int = 20):
    """Load PDF documents from the data directory."""
    logger.info(f"📁 Loading documents from: {data_path}")
    
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    
    # Limit the number of documents to avoid hanging on large datasets
    if len(docs) > max_docs:
        logger.info(f"⚠️ Limiting to first {max_docs} documents (had {len(docs)} total)")
        docs = docs[:max_docs]
    
    logger.info(f"✅ Loaded {len(docs)} documents")
    return docs


def create_all_retriever_chains(documents):
    """Create all retrieval chains (except ensemble)."""
    logger.info("🔗 Creating all retrieval chains")
    
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
    
    chains = {}
    retrievers = {}
    
    # Create Naive RAG chain
    logger.info("📊 Creating Naive RAG chain...")
    try:
        # Use smaller k for faster retrieval in production
        naive_chain, naive_ret = builder.create_naive_rag_chain(
            documents, embeddings, k=5  # Reduced from default for speed
        )
        chains["naive_rag"] = naive_chain
        retrievers["naive_rag"] = naive_ret
        logger.info("✅ Created Naive RAG chain")
    except Exception as e:
        logger.error(f"❌ Failed to create Naive RAG chain: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Create BM25 RAG chain
    logger.info("📊 Creating BM25 RAG chain...")
    try:
        bm25_chain, bm25_ret = builder.create_bm25_rag_chain(documents)
        chains["bm25_rag"] = bm25_chain
        retrievers["bm25_rag"] = bm25_ret
        logger.info("✅ Created BM25 RAG chain")
    except Exception as e:
        logger.error(f"❌ Failed to create BM25 RAG chain: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Create Contextual Compression chain (requires a base retriever)
    logger.info("📊 Creating Contextual Compression RAG chain...")
    try:
        # Use naive retriever as base
        if "naive_rag" in retrievers and retrievers["naive_rag"] is not None:
            comp_chain, comp_ret = builder.create_contextual_compression_rag_chain(retrievers["naive_rag"])
            chains["contextual_compression_rag"] = comp_chain
            retrievers["contextual_compression_rag"] = comp_ret
            logger.info("✅ Created Contextual Compression RAG chain")
        else:
            logger.warning("⚠️ Skipping contextual compression (requires naive retriever)")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create Contextual Compression chain: {e}")
    
    # Create Multi-Query chain (requires a base retriever)
    logger.info("📊 Creating Multi-Query RAG chain...")
    try:
        # Use naive retriever as base
        if "naive_rag" in retrievers and retrievers["naive_rag"] is not None:
            mq_chain, mq_ret = builder.create_multi_query_rag_chain(retrievers["naive_rag"])
            chains["multi_query_rag"] = mq_chain
            retrievers["multi_query_rag"] = mq_ret
            logger.info("✅ Created Multi-Query RAG chain")
        else:
            logger.warning("⚠️ Skipping multi-query (requires naive retriever)")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create Multi-Query chain: {e}")
    
    # Create Parent Document chain
    logger.info("📊 Creating Parent Document RAG chain...")
    try:
        pd_chain, pd_ret = builder.create_parent_document_rag_chain(documents)
        chains["parent_document_rag"] = pd_chain
        retrievers["parent_document_rag"] = pd_ret
        logger.info("✅ Created Parent Document RAG chain")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create Parent Document chain: {e}")
    
    logger.info(f"✅ Created {len(chains)} chains: {list(chains.keys())}")
    return chains, retrievers


def generate_test_dataset(docs, testset_size: int = 10):
    """Generate test dataset using RAGAS TestsetGenerator."""
    logger.info(f"🎯 Generating test dataset with {testset_size} samples")
    
    # Setup generator
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    
    # Generate dataset
    logger.info("⏳ This may take a few minutes...")
    dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    
    logger.info(f"✅ Generated test dataset with {len(dataset)} samples")
    return dataset


def load_test_dataset(file_path: str):
    """Load test dataset from CSV file."""
    logger.info(f"📁 Loading test dataset from: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Parse list columns that were stored as strings
    import ast
    for col in df.columns:
        if col in ['reference_contexts', 'retrieved_contexts', 'retrieved_document_ids']:
            # Check if the first value looks like a string representation of a list
            if len(df) > 0 and isinstance(df[col].iloc[0], str) and df[col].iloc[0].strip().startswith('['):
                logger.info(f"🔧 Parsing {col} column as list")
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    dataset = EvaluationDataset.from_pandas(df)
    
    logger.info(f"✅ Loaded test dataset with {len(dataset)} samples")
    return dataset


def evaluate_chain(chain_name: str, chain: Any, dataset: EvaluationDataset):
    """Evaluate a specific chain."""
    logger.info(f"🎯 Evaluating chain: {chain_name}")
    
    import time
    latencies = []  # Track response times
    
    # Generate responses for this chain
    for i, test_row in enumerate(dataset):
        try:
            question = test_row.user_input
            
            # Measure latency for this query
            start_time = time.time()
            response = chain.invoke({"question": question})
            latency = time.time() - start_time
            latencies.append(latency)
            
            logger.info(f"⏱️ Query {i+1}/{len(dataset)}: {latency:.2f}s")
            
            # Extract response and context
            response_text = response.get("response", "")
            context = response.get("context", [])
            
            # Convert context to strings
            context_strings = []
            if isinstance(context, list):
                for doc in context:
                    if hasattr(doc, 'page_content'):
                        context_strings.append(doc.page_content)
                    else:
                        context_strings.append(str(doc))
            else:
                context_strings = [str(context)]
            
            # Update the test row
            test_row.response = response_text
            test_row.retrieved_contexts = context_strings
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {chain_name} on query {i+1}: {str(e)}")
            test_row.response = ""
            test_row.retrieved_contexts = []
            latencies.append(0)  # Failed request counted as 0 latency
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total_latency = sum(latencies)
    logger.info(f"✅ Generated responses for {chain_name}")
    logger.info(f"⏱️ Avg latency: {avg_latency:.2f}s | Total: {total_latency:.2f}s")
    
    return dataset, latencies


def run_ragas_evaluation(dataset: EvaluationDataset, method_name: str):
    """Run RAGAS evaluation on the dataset."""
    logger.info(f"📊 Running RAGAS evaluation for {method_name}")
    
    # Setup evaluator LLM
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    
    # Setup metrics
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ]
    
    # Setup run config with timeout
    custom_run_config = RunConfig(timeout=360)
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    
    logger.info(f"✅ RAGAS evaluation completed for {method_name}")
    return result


def print_evaluation_results(results: dict):
    """Print evaluation results in a readable format."""
    print("\n" + "="*80)
    print("🎉 RAG EVALUATION RESULTS - All Retrievers")
    print("="*80)
    
    for method_name, result in results.items():
        print(f"\n📊 {method_name.upper()}")
        print("-" * 50)
        
        if isinstance(result, pd.DataFrame):
            df = result
        elif hasattr(result, 'to_pandas'):
            df = result.to_pandas()
        else:
            continue
            
        # Calculate average scores (exclude latency from quality metrics)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        quality_cols = [col for col in numeric_cols if col != 'latency']
        avg_scores = df[quality_cols].mean()
        
        print(f"📈 Average Scores:")
        for metric, score in avg_scores.items():
            print(f"  {metric}: {score:.3f}")
        
        # Overall score
        overall_score = avg_scores.mean()
        print(f"\n🏆 Overall Score: {overall_score:.3f}")
        
        # Latency information
        if 'latency' in df.columns:
            avg_latency = df['latency'].mean()
            total_latency = df['latency'].sum()
            print(f"⏱️ Average Latency: {avg_latency:.2f}s")
            print(f"⏱️ Total Time: {total_latency:.2f}s")


def main(dataset_path: Optional[str] = None, no_generation: bool = False, force_regenerate: bool = False):
    """Main evaluation function."""
    logger.info("🚀 Starting All Retriever Evaluation")
    
    try:
        # Step 1: Load documents (limit to 20 to avoid hanging)
        docs = load_documents(max_docs=50)
        
        # Step 2: Ensure evaluation_results directory exists
        os.makedirs("evaluation_results", exist_ok=True)
        
        # Step 3: Try to load existing dataset or generate new one
        try:
            # Use provided dataset if specified
            if dataset_path:
                logger.info(f"📁 Using provided dataset: {dataset_path}")
                dataset = load_test_dataset(dataset_path)
            else:
                # Try to find existing dataset
                import glob
                dataset_files = glob.glob("evaluation_results/evaluation_dataset_*.csv")
                
                # Check if we should use existing dataset or generate new one
                if dataset_files and not force_regenerate:
                    # Use existing dataset
                    latest_dataset = max(dataset_files, key=os.path.getctime)
                    logger.info(f"📁 Found existing dataset: {latest_dataset}")
                    logger.info("✅ Using existing dataset to avoid long generation time")
                    dataset = load_test_dataset(latest_dataset)
                else:
                    # Need to generate new dataset
                    if force_regenerate:
                        logger.info("🔄 --force-regenerate flag set, generating new dataset")
                    
                    if no_generation and not force_regenerate:
                        logger.error("❌ No existing dataset found and --no-generation flag is set")
                        logger.info("💡 Available datasets:")
                        for f in glob.glob("evaluation_results/*.csv"):
                            logger.info(f"   - {f}")
                        raise FileNotFoundError("No evaluation dataset available")
                    
                    if not force_regenerate:
                        logger.warning("⚠️ No existing dataset found")
                        logger.warning("⚠️ Dataset generation can take 5-10 minutes and may hang")
                        logger.warning("⚠️ Consider using an existing dataset or pressing Ctrl+C to cancel")
                    
                    logger.info("🎲 Generating new test dataset...")
                    dataset = generate_test_dataset(docs, testset_size=10)
                    # Save the dataset
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"evaluation_results/evaluation_dataset_{timestamp}.csv"
                    dataset.to_pandas().to_csv(save_path, index=False)
                    logger.info(f"💾 Dataset saved to: {save_path}")
        except KeyboardInterrupt:
            logger.error("❌ Dataset generation was cancelled")
            logger.info("💡 Tip: Use an existing dataset from evaluation_results/ folder")
            raise
        except Exception as e:
            logger.error(f"❌ Dataset generation failed: {e}")
            logger.info("💡 Tip: Use an existing dataset from evaluation_results/ folder")
            raise
        
        # Step 4: Create all retrieval chains
        chains, retrievers = create_all_retriever_chains(docs)
        
        if not chains:
            logger.error("❌ No chains were created successfully!")
            return
        
        # Step 5: Evaluate each chain
        evaluation_results = {}
        latency_data = {}  # Store latencies separately
        
        for chain_name, chain in chains.items():
            logger.info(f"🔄 Processing method: {chain_name}")
            
            # Create a copy of the dataset for this method
            method_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
            
            # Evaluate the method (returns dataset and latencies)
            method_dataset, latencies = evaluate_chain(chain_name, chain, method_dataset)
            latency_data[chain_name] = latencies
            
            # Run RAGAS evaluation
            ragas_result = run_ragas_evaluation(method_dataset, chain_name)
            
            # Add latency to RAGAS results
            if hasattr(ragas_result, 'to_pandas'):
                df = ragas_result.to_pandas()
                if len(latencies) == len(df):
                    df['latency'] = latencies
                result = df
            else:
                result = ragas_result
            
            evaluation_results[chain_name] = result
        
        # Step 6: Print results
        print_evaluation_results(evaluation_results)
        
        # Step 7: Save results
        results_df = pd.DataFrame()
        for method_name, result in evaluation_results.items():
            if result is not None:
                # Check if result is already a DataFrame
                if isinstance(result, pd.DataFrame):
                    df = result
                elif hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    # Add latency if we have it stored separately
                    if method_name in latency_data:
                        df['latency'] = latency_data[method_name]
                else:
                    continue
                
                df['method'] = method_name
                results_df = pd.concat([results_df, df], ignore_index=True)
        
        # Save detailed results to CSV (one row per question)
        if not results_df.empty:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"evaluation_results/all_retriever_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            logger.info(f"💾 Detailed results saved to: {results_file}")
            
            # Also save average scores summary
            summary_rows = []
            for method_name, result in evaluation_results.items():
                if result is not None:
                    # Handle both DataFrame and RAGAS result objects
                    if isinstance(result, pd.DataFrame):
                        df = result
                    elif hasattr(result, 'to_pandas'):
                        df = result.to_pandas()
                        # Add latency if we have it
                        if method_name in latency_data:
                            df['latency'] = latency_data[method_name]
                    else:
                        continue
                    
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    
                    # Separate latency from quality metrics
                    quality_metrics = [col for col in numeric_cols if col != 'latency']
                    avg_scores = df[quality_metrics].mean()
                    
                    summary_row = {'method': method_name}
                    for metric, score in avg_scores.items():
                        summary_row[metric] = score
                    
                    # Add latency metrics
                    if 'latency' in df.columns:
                        summary_row['avg_latency_seconds'] = df['latency'].mean()
                        summary_row['total_time_seconds'] = df['latency'].sum()
                        summary_row['min_latency_seconds'] = df['latency'].min()
                        summary_row['max_latency_seconds'] = df['latency'].max()
                    
                    summary_rows.append(summary_row)
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_file = f"evaluation_results/all_retriever_summary_{timestamp}.csv"
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"💾 Average scores summary saved to: {summary_file}")
        
        logger.info("🎉 All retriever evaluation completed!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all retrieval RAG chains using RAGAS metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to existing evaluation dataset CSV file"
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip dataset generation (fail if no dataset exists)"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of test dataset (ignores existing datasets)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # If a specific dataset is provided, use it
    if args.dataset:
        logger.info(f"📁 Using provided dataset: {args.dataset}")
        # We'll need to modify main() to accept dataset path
        sys.argv = [sys.argv[0]]  # Clear other args for now
    
    main(dataset_path=args.dataset, no_generation=args.no_generation, force_regenerate=args.force_regenerate)

