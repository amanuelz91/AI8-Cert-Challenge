"""
Dataset generation using RAGAS TestsetGenerator.

Creates golden datasets from PDF documents for RAG evaluation.
"""

import os
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

from src.utils.logging import get_logger
from src.utils.exceptions import RAGException
from .config import EvaluationConfig

logger = get_logger(__name__)


class DatasetGenerator:
    """Generate evaluation datasets using RAGAS."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize dataset generator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self._setup_components()
        
    def _setup_components(self) -> None:
        """Setup LLM and embedding components."""
        try:
            logger.info("🔧 Setting up dataset generation components")
            
            # Setup generator LLM
            generator_llm = ChatOpenAI(
                model=self.config.generator_model,
                temperature=self.config.generator_temperature,
                n=3  # Generate multiple variations
            )
            self.generator_llm = LangchainLLMWrapper(generator_llm)
            
            # Setup embeddings
            generator_embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model
            )
            self.generator_embeddings = LangchainEmbeddingsWrapper(generator_embeddings)
            
            # Create testset generator
            self.generator = TestsetGenerator(
                llm=self.generator_llm,
                embedding_model=self.generator_embeddings
            )
            
            logger.info("✅ Dataset generation components setup complete")
            
        except Exception as e:
            error_msg = f"Failed to setup dataset generation components: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def load_documents(self, data_path: str) -> List[Document]:
        """
        Load PDF documents from directory.
        
        Args:
            data_path: Path to directory containing PDF files
            
        Returns:
            List of loaded documents
            
        Raises:
            RAGException: If document loading fails
        """
        try:
            logger.info(f"📚 Loading documents from: {data_path}")
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path does not exist: {data_path}")
            
            # Load PDF documents
            loader = DirectoryLoader(
                data_path, 
                glob="*.pdf", 
                loader_cls=PyMuPDFLoader
            )
            docs = loader.load()
            
            logger.info(f"✅ Loaded {len(docs)} documents")
            
            # Log document details
            for i, doc in enumerate(docs):
                logger.info(f"  📄 Document {i+1}: {len(doc.page_content)} characters")
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    logger.info(f"    Source: {doc.metadata['source']}")
            
            return docs
            
        except Exception as e:
            error_msg = f"Failed to load documents: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def generate_dataset(
        self, 
        documents: List[Document], 
        testset_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate evaluation dataset from documents.
        
        Args:
            documents: List of documents to generate questions from
            testset_size: Number of test samples to generate
            
        Returns:
            DataFrame containing the generated dataset
            
        Raises:
            RAGException: If dataset generation fails
        """
        try:
            testset_size = testset_size or self.config.testset_size
            
            logger.info(f"🎯 Generating evaluation dataset")
            logger.info(f"  📊 Target size: {testset_size} samples")
            logger.info(f"  📚 Source documents: {len(documents)}")
            
            # Generate dataset using RAGAS
            logger.info("🔄 Generating testset with RAGAS...")
            logger.info("⚠️  Note: You may see 'headlines' transformation warnings - this is a known RAGAS issue and can be safely ignored")
            
            try:
                dataset = self.generator.generate_with_langchain_docs(
                    documents, 
                    testset_size=testset_size
                )
            except Exception as e:
                # Sometimes RAGAS transformations fail but dataset generation still works
                logger.warning(f"⚠️  RAGAS transformation error (this is often safe to ignore): {str(e)}")
                logger.info("🔄 Retrying with error handling...")
                dataset = self.generator.generate_with_langchain_docs(
                    documents, 
                    testset_size=testset_size
                )
            
            # Convert to pandas DataFrame
            df = dataset.to_pandas()
            
            logger.info(f"✅ Generated dataset with {len(df)} samples")
            logger.info(f"📊 Dataset columns: {list(df.columns)}")
            
            # Log sample data
            if len(df) > 0:
                logger.info("📝 Sample generated data:")
                for col in df.columns:
                    sample_value = str(df[col].iloc[0])[:100] + "..." if len(str(df[col].iloc[0])) > 100 else str(df[col].iloc[0])
                    logger.info(f"  {col}: {sample_value}")
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to generate dataset: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def save_dataset(self, dataset: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset DataFrame to save
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
                filename = f"evaluation_dataset_{timestamp}.csv"
            
            filepath = results_dir / filename
            
            # Save dataset
            dataset.to_csv(filepath, index=False)
            
            logger.info(f"💾 Dataset saved to: {filepath}")
            logger.info(f"📊 Dataset shape: {dataset.shape}")
            
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Failed to save dataset: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            Loaded dataset DataFrame
        """
        try:
            import ast
            
            logger.info(f"📂 Loading dataset from: {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            
            # Parse string representations of lists back to actual lists
            # This is needed because CSV stores lists as strings
            if 'reference_contexts' in df.columns:
                logger.info("🔧 Parsing reference_contexts from string to list")
                df['reference_contexts'] = df['reference_contexts'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
                )
            
            # Also handle ground_truths if present
            if 'ground_truths' in df.columns:
                logger.info("🔧 Parsing ground_truths from string to list")
                df['ground_truths'] = df['ground_truths'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
                )
            
            logger.info(f"✅ Loaded dataset with {len(df)} samples")
            logger.info(f"📊 Dataset columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
    
    def generate_and_save_dataset(
        self, 
        data_path: str, 
        testset_size: Optional[int] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate and save evaluation dataset in one step.
        
        Args:
            data_path: Path to directory containing PDF files
            testset_size: Number of test samples to generate
            filename: Optional custom filename for saving
            
        Returns:
            Dictionary with dataset info and filepath
        """
        try:
            logger.info("🚀 Starting complete dataset generation workflow")
            
            # Load documents
            documents = self.load_documents(data_path)
            
            # Generate dataset
            dataset = self.generate_dataset(documents, testset_size)
            
            # Save dataset
            filepath = self.save_dataset(dataset, filename)
            
            result = {
                "dataset": dataset,
                "filepath": filepath,
                "num_samples": len(dataset),
                "columns": list(dataset.columns),
                "documents_processed": len(documents)
            }
            
            logger.info("🎉 Dataset generation workflow completed successfully")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate and save dataset: {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg) from e
