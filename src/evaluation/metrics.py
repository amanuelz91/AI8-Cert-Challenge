"""
RAGAS evaluation metrics configuration and management.
"""

from typing import List, Dict, Any
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness, 
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)
from ragas import RunConfig

from src.utils.logging import get_logger
from .config import EvaluationConfig

logger = get_logger(__name__)


class EvaluationMetrics:
    """Manage RAGAS evaluation metrics."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation metrics.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics = self._setup_metrics()
        self.run_config = self._setup_run_config()
        
    def _setup_metrics(self) -> List:
        """Setup RAGAS evaluation metrics."""
        try:
            logger.info("📊 Setting up RAGAS evaluation metrics")
            
            metrics = [
                LLMContextRecall(),
                Faithfulness(),
                FactualCorrectness(), 
                ResponseRelevancy(),
                ContextEntityRecall(),
                NoiseSensitivity()
            ]
            
            logger.info(f"✅ Configured {len(metrics)} evaluation metrics:")
            for metric in metrics:
                logger.info(f"  📈 {metric.__class__.__name__}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to setup metrics: {str(e)}")
            raise
    
    def _setup_run_config(self) -> RunConfig:
        """Setup RAGAS run configuration."""
        try:
            logger.info("⚙️ Setting up RAGAS run configuration")
            
            run_config = RunConfig(timeout=self.config.timeout)
            
            logger.info(f"✅ Run config created with timeout: {self.config.timeout}s")
            
            return run_config
            
        except Exception as e:
            logger.error(f"Failed to setup run config: {str(e)}")
            raise
    
    def get_metrics(self) -> List:
        """Get configured metrics."""
        return self.metrics
    
    def get_run_config(self) -> RunConfig:
        """Get run configuration."""
        return self.run_config
    
    def get_metric_names(self) -> List[str]:
        """Get names of configured metrics."""
        return [metric.__class__.__name__ for metric in self.metrics]
    
    def add_custom_metric(self, metric) -> None:
        """
        Add custom metric to evaluation.
        
        Args:
            metric: RAGAS metric instance
        """
        try:
            self.metrics.append(metric)
            logger.info(f"✅ Added custom metric: {metric.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to add custom metric: {str(e)}")
            raise
    
    def remove_metric(self, metric_name: str) -> None:
        """
        Remove metric by name.
        
        Args:
            metric_name: Name of metric to remove
        """
        try:
            original_count = len(self.metrics)
            self.metrics = [m for m in self.metrics if m.__class__.__name__ != metric_name]
            
            if len(self.metrics) < original_count:
                logger.info(f"✅ Removed metric: {metric_name}")
            else:
                logger.warning(f"⚠️ Metric not found: {metric_name}")
                
        except Exception as e:
            logger.error(f"Failed to remove metric: {str(e)}")
            raise
