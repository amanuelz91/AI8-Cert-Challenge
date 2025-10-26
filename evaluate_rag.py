#!/usr/bin/env python3
"""
RAG Evaluation Script

Comprehensive evaluation of RAG retrieval methods using RAGAS and LangSmith.
This is the main evaluation script that handles all evaluation workflows:
- Generate golden datasets from PDF documents
- Evaluate existing datasets
- Run evaluations with different methods
- Create LangSmith datasets and experiments

Usage Examples:
    # Complete workflow: generate dataset and evaluate
    uv run python evaluate_rag.py --data-path ./src/data --testset-size 10
    
    # Generate dataset only
    uv run python evaluate_rag.py --data-path ./src/data --testset-size 10 --dataset-only
    
    # Evaluate existing local dataset
    uv run python evaluate_rag.py --evaluate-only --dataset-path ./evaluation_results/dataset.csv
    
    # Evaluate using LangSmith dataset
    uv run python evaluate_rag.py --evaluate-only --langsmith-dataset my_dataset_name
    
    # List available LangSmith datasets
    uv run python evaluate_rag.py --list-langsmith-datasets
    
    # Evaluate specific methods only
    uv run python evaluate_rag.py --methods naive semantic production
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.system.rag_system import create_production_rag_system
from src.evaluation import (
    DatasetGenerator,
    RAGEvaluator,
    EvaluationConfig,
    EvaluationMetrics
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def setup_environment() -> None:
    """Setup environment variables and paths."""
    try:
        # Set up paths
        project_root = Path(__file__).parent
        os.chdir(project_root)
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        logger.info("🔧 Environment setup complete")
        
    except Exception as e:
        logger.error(f"Failed to setup environment: {str(e)}")
        raise


def generate_dataset(
    data_path: str,
    testset_size: int = 10,
    config: Optional[EvaluationConfig] = None
) -> str:
    """
    Generate evaluation dataset.
    
    Args:
        data_path: Path to PDF documents
        testset_size: Number of test samples
        config: Evaluation configuration
        
    Returns:
        Path to generated dataset file
    """
    try:
        logger.info("🎯 Starting dataset generation")
        logger.info(f"📁 Data path: {data_path}")
        logger.info(f"📊 Testset size: {testset_size}")
        
        # Initialize dataset generator
        generator = DatasetGenerator(config)
        
        # Generate and save dataset
        result = generator.generate_and_save_dataset(
            data_path=data_path,
            testset_size=testset_size
        )
        
        logger.info("✅ Dataset generation completed")
        logger.info(f"📄 Dataset saved to: {result['filepath']}")
        logger.info(f"📊 Generated {result['num_samples']} samples")
        
        return result['filepath']
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        raise


def evaluate_rag_system(
    dataset_source: tuple,
    methods: Optional[List[str]] = None,
    config: Optional[EvaluationConfig] = None
) -> str:
    """
    Evaluate RAG system with different methods.
    
    Args:
        dataset_source: Tuple of (source_type, source_value) where source_type is "file" or "langsmith"
        methods: List of methods to evaluate
        config: Evaluation configuration
        
    Returns:
        Path to evaluation results file
    """
    try:
        source_type, source_value = dataset_source
        logger.info("🚀 Starting RAG system evaluation")
        logger.info(f"📊 Dataset source: {source_type} - {source_value}")
        logger.info(f"🔍 Methods: {methods or 'All available'}")
        
        # Initialize RAG system
        logger.info("🔧 Initializing RAG system...")
        rag_system = create_production_rag_system()
        
        # Initialize evaluator
        evaluator = RAGEvaluator(config, rag_system)
        
        # Load dataset based on source type
        if source_type == "langsmith":
            dataset = evaluator.load_langsmith_dataset(source_value)
        else:  # file
            dataset = source_value
        
        # Create LangSmith dataset if tracing is enabled and using file source
        langsmith_dataset_name = None
        if config.langsmith_tracing and source_type == "file":
            logger.info("📊 Creating LangSmith dataset...")
            try:
                langsmith_dataset_name = evaluator.create_langsmith_dataset(
                    dataset=dataset,
                    dataset_name=f"rag_eval_{Path(dataset).stem}"
                )
                logger.info(f"✅ Created LangSmith dataset: {langsmith_dataset_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create LangSmith dataset: {str(e)}")
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_all_methods(
            dataset=dataset,
            methods=methods
        )
        
        # Save results
        results_path = evaluator.save_results(results)
        
        logger.info("✅ RAG system evaluation completed")
        logger.info(f"📄 Results saved to: {results_path}")
        
        # Print summary
        print_evaluation_summary(results)
        
        return results_path
        
    except Exception as e:
        logger.error(f"Failed to evaluate RAG system: {str(e)}")
        raise


def print_evaluation_summary(results: dict) -> None:
    """Print evaluation summary."""
    try:
        print("\n" + "="*60)
        print("🎉 RAG EVALUATION SUMMARY")
        print("="*60)
        
        individual_results = results.get("individual_results", {})
        comparison = results.get("comparison", {})
        
        print(f"📊 Methods evaluated: {len(individual_results)}")
        print(f"🏆 Best method: {comparison.get('best_method', 'N/A')}")
        
        print("\n📈 Individual Results:")
        print("-" * 40)
        
        for method, result in individual_results.items():
            if "error" in result:
                print(f"❌ {method}: ERROR - {result['error']}")
            else:
                score = result.get("overall_score", 0)
                print(f"✅ {method}: {score:.3f}")
                
                # Print individual metrics
                metric_scores = result.get("metric_scores", {})
                for metric, score in metric_scores.items():
                    print(f"   📊 {metric}: {score:.3f}")
        
        print("\n🏆 Method Rankings:")
        print("-" * 40)
        rankings = comparison.get("method_rankings", [])
        for i, (method, score) in enumerate(rankings, 1):
            print(f"{i}. {method}: {score:.3f}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to print summary: {str(e)}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="RAG System Evaluation")
    parser.add_argument(
        "--data-path", 
        default="./src/data",
        help="Path to PDF documents directory"
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=10,
        help="Number of test samples to generate"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["naive", "semantic", "tool", "hybrid", "production"],
        help="Specific methods to evaluate"
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only generate dataset, don't evaluate"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true", 
        help="Only evaluate, don't generate dataset"
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to existing dataset for evaluation-only mode"
    )
    parser.add_argument(
        "--langsmith-dataset",
        help="Name of existing LangSmith dataset to use for evaluation"
    )
    parser.add_argument(
        "--list-langsmith-datasets",
        action="store_true",
        help="List all available LangSmith datasets and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup environment
        setup_environment()
        
        # Initialize configuration
        config = EvaluationConfig()
        
        # Handle list datasets option
        if args.list_langsmith_datasets:
            logger.info("📊 Listing LangSmith datasets...")
            evaluator = RAGEvaluator(config)
            datasets = evaluator.list_langsmith_datasets()
            
            print("\n" + "="*60)
            print("📊 AVAILABLE LANGSMITH DATASETS")
            print("="*60)
            
            if not datasets:
                print("No datasets found in LangSmith project.")
            else:
                for i, dataset in enumerate(datasets, 1):
                    print(f"{i}. {dataset['name']}")
                    print(f"   Description: {dataset['description'] or 'No description'}")
                    print(f"   Examples: {dataset['example_count']}")
                    print(f"   Created: {dataset['created_at']}")
                    print()
            
            print("="*60)
            print("Usage: uv run python evaluate_rag.py --evaluate-only --langsmith-dataset <dataset_name>")
            return
        
        dataset_path = None
        
        # Determine dataset source
        dataset_source = None
        if args.langsmith_dataset:
            dataset_source = ("langsmith", args.langsmith_dataset)
            logger.info(f"📊 Using LangSmith dataset: {args.langsmith_dataset}")
        elif not args.evaluate_only:
            logger.info("🎯 Generating evaluation dataset...")
            dataset_path = generate_dataset(
                data_path=args.data_path,
                testset_size=args.testset_size,
                config=config
            )
            dataset_source = ("file", dataset_path)
        elif args.dataset_path:
            dataset_source = ("file", args.dataset_path)
        else:
            raise ValueError("Must provide --dataset-path or --langsmith-dataset when using --evaluate-only")
        
        # Evaluate if not dataset-only
        if not args.dataset_only:
            logger.info("🚀 Starting RAG system evaluation...")
            results_path = evaluate_rag_system(
                dataset_source=dataset_source,
                methods=args.methods,
                config=config
            )
            
            logger.info(f"🎉 Evaluation completed! Results saved to: {results_path}")
        else:
            logger.info("✅ Dataset generation completed!")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
