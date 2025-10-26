#!/usr/bin/env python3
"""
LangSmith Experiment Demo

This script demonstrates how to:
1. Run RAGAS evaluation
2. Create a LangSmith dataset
3. Run LangSmith experiments
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.evaluation import DatasetGenerator, RAGEvaluator, EvaluationConfig
from src.core.system.rag_system import create_production_rag_system
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Run LangSmith experiment demo."""
    print("🚀 LangSmith Experiment Demo")
    print("=" * 50)
    
    try:
        # Initialize components
        config = EvaluationConfig()
        evaluator = RAGEvaluator(config)
        rag_system = create_production_rag_system()
        evaluator.set_rag_system(rag_system)
        
        print("✅ Components initialized")
        
        # Load existing dataset
        dataset_path = "evaluation_results/evaluation_dataset_20251023_182510.csv"
        print(f"📊 Loading dataset: {dataset_path}")
        
        # Create LangSmith dataset
        print("\n🔬 Creating LangSmith dataset...")
        dataset_name = evaluator.create_langsmith_dataset(
            dataset=dataset_path,
            dataset_name="rag_eval_demo_dataset"
        )
        print(f"✅ Created LangSmith dataset: {dataset_name}")
        
        # Run LangSmith experiment
        print("\n🚀 Running LangSmith experiment...")
        experiment_results = evaluator.run_langsmith_experiment(
            dataset_name=dataset_name,
            method="naive",
            experiment_prefix="rag-eval-demo",
            max_concurrency=2
        )
        
        print(f"✅ Experiment completed!")
        print(f"🔗 View results at: {experiment_results}")
        
        print("\n🎉 LangSmith experiment demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



