"""
RAG evaluation system using RAGAS and LangSmith.

Evaluates different RAG retrieval methods against golden datasets.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas import EvaluationDataset
from langsmith import Client

from src.core.system.rag_system import ProductionRAGSystem
from src.utils.logging import get_logger
from src.utils.exceptions import RAGException
from .config import EvaluationConfig
from .metrics import EvaluationMetrics
from .dataset_generator import DatasetGenerator

logger = get_logger(__name__)


class RAGEvaluator:
    """Evaluate RAG systems using RAGAS metrics."""
    
    def __init__(
        self, 
        config: Optional[EvaluationConfig] = None,
        rag_system: Optional[ProductionRAGSystem] = None
    ):
        """
        Initialize RAG evaluator.
        
        Args:
            config: Evaluation configuration
            rag_system: RAG system to evaluate
        """
        self.config = config or EvaluationConfig()
        self.rag_system = rag_system
        self._setup_components()
        
    def _setup_components(self) -> None:
        """Setup evaluation components."""
        try:
            logger.info("🔧 Setting up RAG evaluation components")
            
            # Setup evaluator LLM
            evaluator_llm = ChatOpenAI(
                model=self.config.evaluator_model,
                temperature=self.config.evaluator_temperature
            )
            self.evaluator_llm = LangchainLLMWrapper(evaluator_llm)
            
            # Setup metrics
            self.metrics_manager = EvaluationMetrics(self.config)
            
            # Setup dataset generator
            self.dataset_generator = DatasetGenerator(self.config)
            
            # Setup LangSmith if enabled
            if self.config.langsmith_tracing:
                self._setup_langsmith()
            
            logger.info("✅ RAG evaluation components setup complete")
            
        except Exception as e:
            error_msg = f"Failed to setup evaluation components: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _setup_langsmith(self) -> None:
        """Setup LangSmith tracing and client."""
        try:
            logger.info("🔍 Setting up LangSmith tracing")
            
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_PROJECT"] = self.config.langsmith_project
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            
            # Initialize LangSmith client
            self.langsmith_client = Client()
            
            logger.info(f"✅ LangSmith tracing enabled for project: {self.config.langsmith_project}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup LangSmith tracing: {str(e)}")
            self.langsmith_client = None
    
    def set_rag_system(self, rag_system: ProductionRAGSystem) -> None:
        """
        Set RAG system to evaluate.
        
        Args:
            rag_system: RAG system instance
        """
        self.rag_system = rag_system
        logger.info("✅ RAG system set for evaluation")
    
    def prepare_dataset(self, dataset: Union[pd.DataFrame, str, EvaluationDataset]) -> EvaluationDataset:
        """
        Prepare dataset for evaluation.
        
        Args:
            dataset: DataFrame, path to dataset file, or RAGAS EvaluationDataset
            
        Returns:
            RAGAS Dataset object
        """
        try:
            logger.info("📊 Preparing dataset for evaluation")
            
            # If already a RAGAS dataset, return as-is
            if isinstance(dataset, EvaluationDataset):
                logger.info(f"✅ Dataset already in RAGAS format with {len(dataset)} samples")
                return dataset
            
            # Load dataset if path provided
            if isinstance(dataset, str):
                df = self.dataset_generator.load_dataset(dataset)
            else:
                df = dataset
            
            logger.info(f"📈 Dataset shape: {df.shape}")
            logger.info(f"📋 Dataset columns: {list(df.columns)}")
            
            # Convert to RAGAS Dataset
            ragas_dataset = EvaluationDataset.from_pandas(df)
            
            logger.info(f"✅ Dataset prepared with {len(ragas_dataset)} samples")
            
            return ragas_dataset
            
        except Exception as e:
            error_msg = f"Failed to prepare dataset: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def evaluate_method(
        self, 
        method: str, 
        dataset: Union[pd.DataFrame, str, EvaluationDataset],
        custom_run_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate specific RAG method.
        
        Args:
            method: RAG method to evaluate ("naive", "semantic", "tool", "hybrid", "production")
            dataset: Evaluation dataset
            custom_run_config: Custom RAGAS run configuration
            
        Returns:
            Evaluation results
        """
        try:
            if not self.rag_system:
                raise ValueError("RAG system not set. Use set_rag_system() first.")
            
            logger.info(f"🎯 Evaluating RAG method: {method}")
            
            # Prepare dataset
            if not isinstance(dataset, EvaluationDataset):
                ragas_dataset = self.prepare_dataset(dataset)
            else:
                ragas_dataset = dataset
            
            # Create custom dataset with method-specific responses
            logger.info(f"🔄 Generating responses for method: {method}")
            method_dataset = self._generate_method_responses(method, ragas_dataset)
            
            # Run evaluation
            logger.info(f"📊 Running RAGAS evaluation for method: {method}")
            run_config = custom_run_config or self.metrics_manager.get_run_config()
            
            result = evaluate(
                dataset=method_dataset,
                metrics=self.metrics_manager.get_metrics(),
                llm=self.evaluator_llm,
                run_config=run_config
            )
            
            # Process results
            evaluation_results = self._process_results(result, method)
            
            logger.info(f"✅ Evaluation completed for method: {method}")
            logger.info(f"📈 Overall score: {evaluation_results.get('overall_score', 'N/A')}")
            
            return evaluation_results
            
        except Exception as e:
            error_msg = f"Failed to evaluate method {method}: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _generate_method_responses(
        self, 
        method: str, 
        dataset: EvaluationDataset
    ) -> EvaluationDataset:
        """
        Generate responses for specific method.
        
        Args:
            method: RAG method
            dataset: Original dataset
            
        Returns:
            Dataset with method-specific responses
        """
        try:
            logger.info(f"🔄 Generating responses using method: {method}")
            
            responses = []
            retrieved_contexts = []
            user_inputs = []
            reference_contexts_list = []
            references = []
            
            for sample in dataset.samples:
                # Get user input from the sample
                question = sample.user_input
                user_inputs.append(question)
                
                # Get reference data
                ref_contexts = sample.reference_contexts or []
                reference_contexts_list.append(ref_contexts)
                references.append(sample.reference)
                
                # Query RAG system
                result = self.rag_system.query(
                    question=question,
                    method=method,
                    include_confidence=False
                )
                
                response = result.get("response", "")
                # Extract context from the result
                context = result.get("context", [])
                # Convert Document objects to strings if needed
                context_strings = [
                    doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    for doc in context
                ]
                
                responses.append(response)
                retrieved_contexts.append(context_strings)
                
                logger.info(f"  📝 Generated response for: {question[:50]}...")
            
            # Create new dataset with responses using RAGAS schema
            # Convert to pandas DataFrame format expected by RAGAS
            import pandas as pd
            dataset_data = []
            for i in range(len(user_inputs)):
                dataset_data.append({
                    "user_input": user_inputs[i],
                    "reference_contexts": reference_contexts_list[i],
                    "reference": references[i],
                    "retrieved_contexts": retrieved_contexts[i],
                    "response": responses[i]
                })
            
            df = pd.DataFrame(dataset_data)
            method_dataset = EvaluationDataset.from_pandas(df)
            
            logger.info(f"✅ Generated {len(responses)} responses for method: {method}")
            
            return method_dataset
            
        except Exception as e:
            error_msg = f"Failed to generate responses for method {method}: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _process_results(self, result: Any, method: str) -> Dict[str, Any]:
        """
        Process RAGAS evaluation results.
        
        Args:
            result: RAGAS evaluation result
            method: Method that was evaluated
            
        Returns:
            Processed results dictionary
        """
        try:
            logger.info(f"📊 Processing evaluation results for method: {method}")
            
            # Extract scores
            scores = {}
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for col in df.columns:
                    if col != 'question':
                        scores[col] = df[col].mean()
            
            # Calculate overall score
            overall_score = sum(scores.values()) / len(scores) if scores else 0
            
            processed_results = {
                "method": method,
                "timestamp": str(datetime.now()),
                "overall_score": overall_score,
                "metric_scores": scores,
                "raw_result": result,
                "num_samples": len(result.to_pandas()) if hasattr(result, 'to_pandas') else 0
            }
            
            logger.info(f"📈 Processed results for {method}:")
            logger.info(f"  🎯 Overall score: {overall_score:.3f}")
            for metric, score in scores.items():
                logger.info(f"  📊 {metric}: {score:.3f}")
            
            return processed_results
            
        except Exception as e:
            error_msg = f"Failed to process results for method {method}: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def evaluate_all_methods(
        self, 
        dataset: Union[pd.DataFrame, str, EvaluationDataset],
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all RAG methods.
        
        Args:
            dataset: Evaluation dataset
            methods: List of methods to evaluate (default: all available)
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            if not self.rag_system:
                raise ValueError("RAG system not set. Use set_rag_system() first.")
            
            methods = methods or ["naive", "semantic", "production"]
            if self.rag_system.tool_retriever:
                methods.extend(["tool", "hybrid"])
            
            logger.info(f"🚀 Starting comprehensive evaluation of {len(methods)} methods")
            logger.info(f"📋 Methods to evaluate: {methods}")
            
            all_results = {}
            
            for method in methods:
                try:
                    logger.info(f"🔄 Evaluating method: {method}")
                    result = self.evaluate_method(method, dataset)
                    all_results[method] = result
                    
                except Exception as e:
                    logger.error(f"❌ Failed to evaluate method {method}: {str(e)}")
                    all_results[method] = {
                        "error": str(e),
                        "method": method,
                        "timestamp": str(datetime.now())
                    }
            
            # Generate comparison summary
            comparison = self._generate_comparison(all_results)
            
            comprehensive_results = {
                "individual_results": all_results,
                "comparison": comparison,
                "timestamp": str(datetime.now()),
                "methods_evaluated": methods,
                "total_methods": len(methods)
            }
            
            logger.info("🎉 Comprehensive evaluation completed")
            logger.info(f"📊 Results summary:")
            for method, result in all_results.items():
                if "error" not in result:
                    score = result.get("overall_score", "N/A")
                    logger.info(f"  {method}: {score}")
            
            return comprehensive_results
            
        except Exception as e:
            error_msg = f"Failed to evaluate all methods: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def _generate_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison summary of all methods.
        
        Args:
            results: Individual method results
            
        Returns:
            Comparison summary
        """
        try:
            logger.info("📊 Generating comparison summary")
            
            comparison = {
                "method_rankings": [],
                "best_method": None,
                "score_differences": {},
                "metric_comparisons": {}
            }
            
            # Extract scores and rank methods
            method_scores = {}
            for method, result in results.items():
                if "error" not in result:
                    score = result.get("overall_score", 0)
                    method_scores[method] = score
            
            # Sort by score
            sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
            comparison["method_rankings"] = sorted_methods
            
            if sorted_methods:
                comparison["best_method"] = sorted_methods[0][0]
            
            # Calculate score differences
            if len(sorted_methods) > 1:
                best_score = sorted_methods[0][1]
                for method, score in sorted_methods[1:]:
                    comparison["score_differences"][method] = best_score - score
            
            logger.info(f"🏆 Best method: {comparison['best_method']}")
            logger.info(f"📈 Method rankings: {comparison['method_rankings']}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to generate comparison: {str(e)}")
            return {"error": str(e)}
    
    def list_langsmith_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets in the LangSmith project.
        
        Returns:
            List of dataset information dictionaries
        """
        try:
            if not hasattr(self, 'langsmith_client') or self.langsmith_client is None:
                raise ValueError("LangSmith client not initialized. Enable LangSmith tracing in config.")
            
            logger.info("📊 Listing LangSmith datasets...")
            
            # Get all datasets in the project
            datasets = self.langsmith_client.list_datasets()
            
            dataset_info = []
            for dataset in datasets:
                info = {
                    "name": dataset.name,
                    "description": dataset.description,
                    "created_at": dataset.created_at,
                    "example_count": len(dataset.examples) if hasattr(dataset, 'examples') else 0
                }
                dataset_info.append(info)
            
            logger.info(f"✅ Found {len(dataset_info)} datasets in LangSmith project")
            for info in dataset_info:
                logger.info(f"  📊 {info['name']} ({info['example_count']} examples)")
            
            return dataset_info
            
        except Exception as e:
            error_msg = f"Failed to list LangSmith datasets: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def load_langsmith_dataset(self, dataset_name: str) -> EvaluationDataset:
        """
        Load a dataset from LangSmith.
        
        Args:
            dataset_name: Name of the LangSmith dataset
            
        Returns:
            RAGAS EvaluationDataset object
        """
        try:
            if not hasattr(self, 'langsmith_client') or self.langsmith_client is None:
                raise ValueError("LangSmith client not initialized. Enable LangSmith tracing in config.")
            
            logger.info(f"📊 Loading LangSmith dataset: {dataset_name}")
            
            # Get dataset from LangSmith
            dataset = self.langsmith_client.read_dataset(dataset_name=dataset_name)
            
            # Convert LangSmith examples to RAGAS format
            examples = []
            for example in dataset.examples:
                # Extract data from LangSmith example format
                inputs = example.inputs
                outputs = example.outputs
                metadata = example.metadata or {}
                
                # Convert to RAGAS format
                ragas_example = {
                    "user_input": inputs.get("question", ""),
                    "reference": outputs.get("reference", ""),
                    "reference_contexts": metadata.get("reference_contexts", [])
                }
                examples.append(ragas_example)
            
            # Create RAGAS dataset
            import pandas as pd
            df = pd.DataFrame(examples)
            ragas_dataset = EvaluationDataset.from_pandas(df)
            
            logger.info(f"✅ Loaded LangSmith dataset: {dataset_name}")
            logger.info(f"📊 Dataset contains {len(ragas_dataset)} samples")
            
            return ragas_dataset
            
        except Exception as e:
            error_msg = f"Failed to load LangSmith dataset {dataset_name}: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def create_langsmith_dataset(
        self, 
        dataset: Union[pd.DataFrame, str, EvaluationDataset],
        dataset_name: Optional[str] = None
    ) -> str:
        """
        Create a dataset in LangSmith.
        
        Args:
            dataset: Evaluation dataset
            dataset_name: Optional name for the LangSmith dataset
            
        Returns:
            Dataset name in LangSmith
        """
        try:
            if not hasattr(self, 'langsmith_client') or self.langsmith_client is None:
                raise ValueError("LangSmith client not initialized. Enable LangSmith tracing in config.")
            
            logger.info("📊 Creating LangSmith dataset")
            
            # Prepare dataset
            if not isinstance(dataset, EvaluationDataset):
                ragas_dataset = self.prepare_dataset(dataset)
            else:
                ragas_dataset = dataset
            
            # Generate dataset name if not provided
            if dataset_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"rag_eval_dataset_{timestamp}"
            
            # Convert to LangSmith format
            examples = []
            for sample in ragas_dataset.samples:
                example = {
                    "inputs": {"question": sample.user_input},
                    "outputs": {"reference": sample.reference},
                    "metadata": {
                        "reference_contexts": sample.reference_contexts,
                    }
                }
                examples.append(example)
            
            # Create dataset in LangSmith
            self.langsmith_client.create_dataset(
                dataset_name=dataset_name,
                description=f"RAG evaluation dataset created on {datetime.now().isoformat()}"
            )
            
            # Add examples to dataset
            for example in examples:
                self.langsmith_client.create_example(
                    inputs=example["inputs"],
                    outputs=example["outputs"],
                    metadata=example["metadata"],
                    dataset_name=dataset_name
                )
            
            logger.info(f"✅ Created LangSmith dataset: {dataset_name}")
            logger.info(f"📊 Added {len(examples)} examples to dataset")
            
            return dataset_name
            
        except Exception as e:
            error_msg = f"Failed to create LangSmith dataset: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def run_langsmith_experiment(
        self,
        dataset_name: str,
        method: str,
        experiment_prefix: Optional[str] = None,
        max_concurrency: int = 2
    ) -> Any:
        """
        Run a LangSmith experiment using RAGAS evaluators.
        
        Args:
            dataset_name: Name of the LangSmith dataset
            method: RAG method to evaluate
            experiment_prefix: Prefix for the experiment name
            max_concurrency: Maximum concurrent evaluations
            
        Returns:
            Experiment results from LangSmith
        """
        try:
            if not hasattr(self, 'langsmith_client') or self.langsmith_client is None:
                raise ValueError("LangSmith client not initialized. Enable LangSmith tracing in config.")
            
            if not self.rag_system:
                raise ValueError("RAG system not set. Use set_rag_system() first.")
            
            logger.info(f"🚀 Running LangSmith experiment for method: {method}")
            logger.info(f"📊 Dataset: {dataset_name}")
            
            # Define the target function (what we're evaluating)
            def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
                """Target function for LangSmith evaluation."""
                question = inputs.get("question", "")
                
                # Query RAG system
                result = self.rag_system.query(
                    question=question,
                    method=method,
                    include_confidence=False
                )
                
                return {
                    "response": result.get("response", ""),
                    "context": result.get("context", [])
                }
            
            # Create RAGAS evaluators for LangSmith
            from langsmith.evaluation import LangChainStringEvaluator
            
            evaluators = []
            
            # You can add custom evaluators here based on RAGAS metrics
            # For now, we'll use a simple correctness evaluator
            try:
                correctness_evaluator = LangChainStringEvaluator(
                    "qa",
                    config={"llm": ChatOpenAI(model=self.config.evaluator_model)}
                )
                evaluators.append(correctness_evaluator)
            except Exception as e:
                logger.warning(f"Failed to create correctness evaluator: {e}")
            
            # Generate experiment name
            if experiment_prefix is None:
                experiment_prefix = f"rag-eval-{method}"
            
            # Run the experiment
            logger.info(f"🔬 Starting experiment: {experiment_prefix}")
            experiment_results = self.langsmith_client.evaluate(
                target,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=experiment_prefix,
                max_concurrency=max_concurrency,
            )
            
            logger.info("✅ LangSmith experiment completed!")
            logger.info(f"🔗 View results at: {experiment_results}")
            
            return experiment_results
            
        except Exception as e:
            error_msg = f"Failed to run LangSmith experiment: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            # Create results directory
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evaluation_results_{timestamp}.json"
            
            filepath = results_dir / filename
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Evaluation results saved to: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
