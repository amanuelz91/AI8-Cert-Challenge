#!/usr/bin/env python3
"""
Quick RAG Evaluation Test

Simple test script to verify the evaluation system works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.system.rag_system import create_production_rag_system
from src.evaluation import DatasetGenerator, RAGEvaluator, EvaluationConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def test_dataset_generation():
    """Test dataset generation."""
    try:
        logger.info("🧪 Testing dataset generation...")
        
        # Initialize generator
        config = EvaluationConfig()
        config.testset_size = 3  # Small test size
        generator = DatasetGenerator(config)
        
        # Load documents
        data_path = "./src/data"
        documents = generator.load_documents(data_path)
        
        logger.info(f"✅ Loaded {len(documents)} documents")
        
        # Generate small dataset
        dataset = generator.generate_dataset(documents, testset_size=3)
        
        logger.info(f"✅ Generated dataset with {len(dataset)} samples")
        logger.info(f"📊 Dataset columns: {list(dataset.columns)}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Dataset generation test failed: {str(e)}")
        raise


def test_rag_evaluation(dataset):
    """Test RAG evaluation."""
    try:
        logger.info("🧪 Testing RAG evaluation...")
        
        # Initialize RAG system
        logger.info("🔧 Initializing RAG system...")
        rag_system = create_production_rag_system()
        
        # Initialize evaluator
        config = EvaluationConfig()
        evaluator = RAGEvaluator(config, rag_system)
        
        # Test single method evaluation
        logger.info("🔍 Testing naive method evaluation...")
        result = evaluator.evaluate_method("naive", dataset)
        
        logger.info(f"✅ Naive method evaluation completed")
        logger.info(f"📊 Overall score: {result.get('overall_score', 'N/A')}")
        
        # Test all methods evaluation
        logger.info("🔍 Testing all methods evaluation...")
        all_results = evaluator.evaluate_all_methods(
            dataset=dataset,
            methods=["naive", "semantic"]  # Test only available methods
        )
        
        logger.info(f"✅ All methods evaluation completed")
        logger.info(f"📊 Methods evaluated: {len(all_results.get('individual_results', {}))}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ RAG evaluation test failed: {str(e)}")
        raise


def main():
    """Run evaluation tests."""
    try:
        logger.info("🚀 Starting RAG evaluation tests...")
        
        # Test dataset generation
        dataset = test_dataset_generation()
        
        # Test RAG evaluation
        results = test_rag_evaluation(dataset)
        
        # Print summary
        print("\n" + "="*50)
        print("🎉 EVALUATION TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        individual_results = results.get("individual_results", {})
        for method, result in individual_results.items():
            if "error" not in result:
                score = result.get("overall_score", 0)
                print(f"✅ {method}: {score:.3f}")
            else:
                print(f"❌ {method}: ERROR")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"💥 Evaluation tests failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
