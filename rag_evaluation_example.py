#!/usr/bin/env python3
"""
RAG Evaluation Example

Complete example demonstrating RAG evaluation using RAGAS.
This script follows the coding practices specified in the requirements.

Usage:
    # Run with default settings (generates new dataset, enables LangSmith)
    python rag_evaluation_example.py
    
    # Run with existing dataset, disable LangSmith tracing
    python rag_evaluation_example.py --dataset evaluation_results/evaluation_dataset_20251025_132815.csv --no-langsmith
    
    # Run with custom test size, disable LangSmith
    python rag_evaluation_example.py --testset-size 5 --no-langsmith
    
    # Run with specific methods only
    python rag_evaluation_example.py --methods naive semantic --no-langsmith
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy, 
    ContextEntityRecall, 
    NoiseSensitivity
)
from ragas import RunConfig

from src.core.system.rag_system import create_production_rag_system
from src.utils.logging import get_logger
from src.evaluation.dataset_generator import DatasetGenerator
from src.evaluation.config import EvaluationConfig

logger = get_logger(__name__)


def load_documents(data_path="src/data"):
    """Load PDF documents using the specified approach."""
    logger.info(f"📚 Loading PDF documents from: {data_path}")
    
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    
    logger.info(f"✅ Loaded {len(docs)} documents")
    return docs




def generate_golden_dataset(docs, testset_size=5):
    """Generate golden dataset using DatasetGenerator."""
    logger.info(f"🎯 Generating golden dataset with {testset_size} samples...")
    
    # Create evaluation config
    config = EvaluationConfig()
    config.testset_size = testset_size
    config.langsmith_tracing = False  # Disable LangSmith for dataset generation
    
    # Create dataset generator
    generator = DatasetGenerator(config)
    
    # Generate dataset
    df = generator.generate_dataset(docs, testset_size=testset_size)
    
    # Convert to RAGAS dataset format
    from ragas import EvaluationDataset
    dataset = EvaluationDataset.from_pandas(df)
    
    logger.info(f"✅ Generated dataset with {len(df)} samples")
    logger.info(f"📊 Dataset columns: {list(df.columns)}")
    
    return dataset, df


def evaluate_rag_method(rag_system, method, dataset, evaluator_llm):
    """Evaluate specific RAG method."""
    logger.info(f"🔍 Evaluating RAG method: {method}")
    
    # Generate responses for this method
    responses = []
    contexts = []
    
    for sample in dataset.samples:
        question = sample.user_input
        
        # Query RAG system with specific method
        result = rag_system.query(
            question=question,
            method=method,
            include_confidence=False
        )
        
        response = result.get("response", "")
        context = result.get("context", [])
        
        # Convert Document objects to strings for RAGAS
        context_strings = []
        for doc in context:
            if hasattr(doc, 'page_content'):
                context_strings.append(doc.page_content)
            else:
                context_strings.append(str(doc))
        
        responses.append(response)
        contexts.append(context_strings)
    
    # Create evaluation dataset
    from ragas import EvaluationDataset
    import pandas as pd
    
    # Debug: Check dataset structure
    logger.info(f"🔍 Dataset samples count: {len(dataset.samples)}")
    if len(dataset.samples) > 0:
        sample = dataset.samples[0]
        logger.info(f"🔍 Sample attributes: {dir(sample)}")
        logger.info(f"🔍 Sample user_input: {getattr(sample, 'user_input', 'NO USER_INPUT ATTR')}")
        logger.info(f"🔍 Sample reference: {getattr(sample, 'reference', 'NO REFERENCE ATTR')}")
    
    # Create DataFrame for RAGAS evaluation with correct column names
    eval_data = {
        "user_input": [s.user_input for s in dataset.samples],
        "reference": [s.reference for s in dataset.samples],
        "retrieved_contexts": contexts,
        "response": responses,
        "reference_contexts": [s.reference_contexts for s in dataset.samples]
    }
    
    eval_df = pd.DataFrame(eval_data)
    method_dataset = EvaluationDataset.from_pandas(eval_df)
    
    # Run evaluation
    custom_run_config = RunConfig(timeout=360)
    
    baseline_result = evaluate(
        dataset=method_dataset,
        metrics=[
            LLMContextRecall(), 
            Faithfulness(), 
            FactualCorrectness(), 
            ResponseRelevancy(), 
            ContextEntityRecall(), 
            NoiseSensitivity()
        ],
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    
    logger.info(f"✅ Evaluation completed for method: {method}")
    return baseline_result


def print_evaluation_results(method, result):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS FOR: {method.upper()}")
    print(f"{'='*60}")
    
    if hasattr(result, 'to_pandas'):
        df = result.to_pandas()
        
        print(f"📈 Sample Response:")
        if len(df) > 0:
            sample_response = df['response'].iloc[0]
            print(f"   {sample_response[:200]}...")
        
        print(f"\n📊 Metric Scores:")
        for col in df.columns:
            if col not in ['question', 'response']:
                score = df[col].mean()
                print(f"   {col}: {score:.3f}")
        
        # Calculate overall score
        metric_cols = [col for col in df.columns if col not in ['question', 'response']]
        overall_score = df[metric_cols].mean().mean()
        print(f"\n🎯 Overall Score: {overall_score:.3f}")
    
    print(f"{'='*60}")


def print_summary(results):
    """Print evaluation summary."""
    print(f"\n{'='*80}")
    print(f"🎉 RAG EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    failed_results = {k: v for k, v in results.items() if "error" in v}
    
    print(f"📊 Total methods evaluated: {len(results)}")
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(failed_results)}")
    
    if successful_results:
        print(f"\n🏆 METHOD RANKINGS:")
        print(f"{'-'*40}")
        
        # Calculate overall scores and rank methods
        method_scores = {}
        for method, result in successful_results.items():
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                metric_cols = [col for col in df.columns if col not in ['question', 'response']]
                overall_score = df[metric_cols].mean().mean()
                method_scores[method] = overall_score
        
        # Sort by score
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (method, score) in enumerate(sorted_methods, 1):
            print(f"{i}. {method}: {score:.3f}")
        
        if sorted_methods:
            print(f"\n🥇 Best method: {sorted_methods[0][0]} ({sorted_methods[0][1]:.3f})")
    
    if failed_results:
        print(f"\n❌ FAILED METHODS:")
        print(f"{'-'*40}")
        for method, result in failed_results.items():
            print(f"• {method}: {result['error']}")
    
    print(f"{'='*80}")


def load_existing_dataset(dataset_path):
    """Load existing dataset from CSV file."""
    logger.info(f"📂 Loading existing dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load CSV
    df = pd.read_csv(dataset_path)
    logger.info(f"✅ Loaded dataset with {len(df)} samples")
    logger.info(f"📊 Dataset columns: {list(df.columns)}")
    
    # Convert to RAGAS format
    from ragas import EvaluationDataset
    import ast
    
    # Parse reference_contexts if it's stored as string
    if 'reference_contexts' in df.columns:
        df['reference_contexts'] = df['reference_contexts'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
        )
    
    # Create RAGAS dataset - use the correct column names
    dataset_df = pd.DataFrame({
        "user_input": df['user_input'].tolist(),
        "reference": df['reference'].tolist(),
        "reference_contexts": df['reference_contexts'].tolist(),
    })
    
    dataset = EvaluationDataset.from_pandas(dataset_df)
    
    # Debug: Check the dataset structure
    logger.info(f"🔍 Dataset samples count: {len(dataset.samples)}")
    if len(dataset.samples) > 0:
        sample = dataset.samples[0]
        logger.info(f"🔍 Sample user_input: {sample.user_input}")
        logger.info(f"🔍 Sample reference: {sample.reference}")
    logger.info(f"✅ Converted to RAGAS dataset format")
    
    return dataset, df


def setup_environment(disable_langsmith=False):
    """Setup environment variables."""
    if disable_langsmith:
        logger.info("🚫 Disabling LangSmith tracing...")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_PROJECT"] = ""
    else:
        logger.info("🔍 LangSmith tracing enabled")


def main():
    """Main evaluation workflow with CLI support."""
    parser = argparse.ArgumentParser(description="RAG Evaluation using RAGAS")
    parser.add_argument(
        "--dataset", 
        help="Path to existing CSV dataset (skips generation)"
    )
    parser.add_argument(
        "--testset-size", 
        type=int, 
        default=5,
        help="Number of test samples to generate (default: 5)"
    )
    parser.add_argument(
        "--methods", 
        nargs="+", 
        default=["naive", "semantic", "production"],
        choices=["naive", "semantic", "tool", "hybrid", "production"],
        help="RAG methods to evaluate"
    )
    parser.add_argument(
        "--no-langsmith", 
        action="store_true",
        help="Disable LangSmith tracing"
    )
    parser.add_argument(
        "--data-path",
        default="src/data",
        help="Path to PDF documents directory (default: src/data)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting RAG evaluation workflow...")
        logger.info(f"📋 Methods to evaluate: {args.methods}")
        logger.info(f"🚫 LangSmith tracing: {'Disabled' if args.no_langsmith else 'Enabled'}")
        
        # Setup environment
        setup_environment(args.no_langsmith)
        
        # Load or generate dataset
        if args.dataset:
            dataset, df = load_existing_dataset(args.dataset)
        else:
            logger.info("📚 Generating new dataset...")
            docs = load_documents(args.data_path)
            dataset, df = generate_golden_dataset(docs, args.testset_size)
        
        # Initialize RAG system
        logger.info("🔧 Initializing RAG system...")
        rag_system = create_production_rag_system()
        
        # Setup evaluator LLM
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        
        # Evaluate specified methods
        results = {}
        for method in args.methods:
            try:
                logger.info(f"🔄 Evaluating method: {method}")
                result = evaluate_rag_method(rag_system, method, dataset, evaluator_llm)
                results[method] = result
                print_evaluation_results(method, result)
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate method {method}: {str(e)}")
                print(f"\n❌ ERROR evaluating {method}: {str(e)}")
                results[method] = {"error": str(e)}
        
        # Print summary
        print_summary(results)
        logger.info("🎉 RAG evaluation workflow completed!")
        
    except Exception as e:
        logger.error(f"💥 Evaluation workflow failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
