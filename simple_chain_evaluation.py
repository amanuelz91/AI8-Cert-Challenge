#!/usr/bin/env python3
"""
Simplified RAG Chain Evaluation Script using RAGAS

This script:
1. Loads PDF documents
2. Creates chains directly from rag_chains.py
3. Generates or loads test dataset
4. Evaluates all chains using RAGAS metrics

Usage:
    # Generate new dataset and evaluate
    python simple_chain_evaluation.py
    
    # Load existing dataset and evaluate
    python simple_chain_evaluation.py --load-dataset dataset.csv
    
    # Generate dataset and save, but don't evaluate
    python simple_chain_evaluation.py --dataset-only
    
    # Load dataset, skip generation
    python simple_chain_evaluation.py --skip-generation --load-dataset dataset.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
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
from ragas.testset import TestsetGenerator
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy, 
    ContextEntityRecall, 
    NoiseSensitivity
)

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Our RAG chain imports
from src.chains.rag_chains import RAGChainBuilder, create_rag_chain_builder
from src.utils.logging import get_logger
from src.config.settings import get_config

logger = get_logger(__name__)


def load_documents(data_path: str = "src/data"):
    """Load PDF documents from the data directory."""
    logger.info(f"📁 Loading documents from: {data_path}")
    
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    
    logger.info(f"✅ Loaded {len(docs)} documents")
    return docs


def create_in_memory_vectorstore(documents: List, embeddings: OpenAIEmbeddings):
    """Create in-memory Qdrant vector store."""
    logger.info("🗃️ Creating in-memory Qdrant vector store")
    
    # Create in-memory Qdrant client
    client = QdrantClient(location=":memory:")
    
    # Create vector store
    vectorstore = QdrantVectorStore.from_documents(
        documents,
        embeddings,
        client=client,
        collection_name="rag_evaluation"
    )
    
    logger.info("✅ In-memory vector store created")
    return vectorstore


def create_chains_with_retrievers(documents: List):
    """Create all chains directly using rag_chains.py."""
    logger.info("🔗 Creating RAG chains")
    
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
    
    logger.info("✅ Builder created")
    
    # Create basic chains using simplified approach
    chains = {}
    retrievers = {}
    
    naive_chain, naive_ret = builder.create_naive_rag_chain(documents, embeddings, k=config.retrieval.default_k)
    chains["naive_rag"] = naive_chain
    retrievers["naive_rag"] = naive_ret
    
    confidence_chain, _ = builder.create_confidence_chain()
    chains["confidence"] = confidence_chain
    retrievers["confidence"] = None
    
    # Add advanced retrieval chains
    logger.info("🔗 Adding advanced retrieval chains")
    
    # Skip semantic chunking chain - it hangs during document splitting
    # The SemanticChunker makes many slow LLM API calls to determine chunk boundaries
    # logger.info("⚠️ Skipping semantic chunking chain (known to hang/freeze)")
    # try:
    #     sc_chain, sc_ret = builder.create_semantic_chunking_rag_chain(documents)
    #     chains["semantic_chunking_rag"] = sc_chain
    #     retrievers["semantic_chunking_rag"] = sc_ret
    #     logger.info("✅ Created semantic chunking chain")
    # except Exception as e:
    #     logger.warning(f"⚠️ Failed to create semantic chunking chain: {e}")
    
    try:
        bm25_chain, bm25_ret = builder.create_bm25_rag_chain(documents)
        chains["bm25_rag"] = bm25_chain
        retrievers["bm25_rag"] = bm25_ret
        logger.info("✅ Created BM25 chain")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create BM25 chain: {e}")
    
    # Skipping contextual compression and multi-query chains as they require retrievers
    # Commented out for simplified approach
    # try:
    #     chains["contextual_compression_rag"] = builder.create_contextual_compression_rag_chain(naive_retriever)
    #     logger.info("✅ Created contextual compression chain")
    # except Exception as e:
    #     logger.warning(f"⚠️ Failed to create contextual compression chain: {e}")
    
    # try:
    #     chains["multi_query_rag"] = builder.create_multi_query_rag_chain(naive_retriever)
    #     logger.info("✅ Created multi-query chain")
    # except Exception as e:
    #     logger.warning(f"⚠️ Failed to create multi-query chain: {e}")
    
    # Skip parent document chain - it hangs during document indexing
    # The ParentDocumentRetriever creates a complex graph structure that can freeze
    # logger.info("⚠️ Skipping parent document chain (known to hang/freeze)")
    # try:
    #     pd_chain, pd_ret = builder.create_parent_document_rag_chain(documents)
    #     chains["parent_document_rag"] = pd_chain
    #     retrievers["parent_document_rag"] = pd_ret
    #     logger.info("✅ Created parent document chain")
    # except Exception as e:
    #     logger.warning(f"⚠️ Failed to create parent document chain: {e}")
    
    # Skipping ensemble chain as it requires retrievers
    # Commented out for simplified approach
    # try:
    #     from langchain_community.retrievers import BM25Retriever
    #     bm25_retriever = BM25Retriever.from_documents(documents)
    #     bm25_retriever.k = 5
    #     
    #     from langchain.retrievers import EnsembleRetriever
    #     ensemble_retriever = EnsembleRetriever(
    #         retrievers=[naive_retriever, semantic_retriever, bm25_retriever],
    #         weights=[1/3, 1/3, 1/3]
    #     )
    #     
    #     chains["ensemble_rag"] = builder.create_ensemble_rag_chain([naive_retriever, semantic_retriever, bm25_retriever])
    #     logger.info("✅ Created ensemble chain")
    # except Exception as e:
    #     logger.warning(f"⚠️ Failed to create ensemble chain: {e}")
    
    logger.info(f"✅ Created {len(chains)} chains: {list(chains.keys())}")
    return chains


def generate_test_dataset(docs: List, testset_size: int = 10, save_path: Optional[str] = None):
    """Generate test dataset using RAGAS TestsetGenerator."""
    logger.info(f"🎯 Generating test dataset with {testset_size} samples")
    
    # Setup generator LLM and embeddings
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Create testset generator
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    
    # Generate dataset
    logger.info("⏳ This may take a few minutes...")
    dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    
    logger.info(f"✅ Generated test dataset with {len(dataset)} samples")
    
    # Save dataset if path provided
    if save_path:
        save_path = Path(save_path)
        dataset.to_pandas().to_csv(save_path, index=False)
        logger.info(f"💾 Dataset saved to: {save_path}")
    
    return dataset


def load_test_dataset(file_path: str):
    """Load test dataset from CSV file."""
    logger.info(f"📁 Loading test dataset from: {file_path}")
    
    df = pd.read_csv(file_path)
    dataset = EvaluationDataset.from_pandas(df)
    
    logger.info(f"✅ Loaded test dataset with {len(dataset)} samples")
    return dataset


def evaluate_chain(chain_name: str, chain: Any, dataset: EvaluationDataset):
    """Evaluate a specific chain."""
    logger.info(f"🎯 Evaluating chain: {chain_name}")
    
    # Generate responses for this chain
    for test_row in dataset:
        try:
            question = test_row.user_input
            
            # Invoke the chain
            response = chain.invoke({"question": question})
            
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
            logger.error(f"❌ Error evaluating {chain_name}: {str(e)}")
            test_row.response = ""
            test_row.retrieved_contexts = []
    
    logger.info(f"✅ Generated responses for {chain_name}")
    return dataset


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
    
    # Setup run config
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


def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results in a readable format."""
    print("\n" + "="*80)
    print("🎉 RAG EVALUATION RESULTS")
    print("="*80)
    
    for method_name, result in results.items():
        print(f"\n📊 {method_name.upper()}")
        print("-" * 50)
        
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            
            # Calculate average scores
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            avg_scores = df[numeric_cols].mean()
            
            print(f"📈 Average Scores:")
            for metric, score in avg_scores.items():
                print(f"  {metric}: {score:.3f}")
            
            # Overall score
            overall_score = avg_scores.mean()
            print(f"\n🏆 Overall Score: {overall_score:.3f}")
        else:
            print(f"❌ No results available for {method_name}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG chains using RAGAS metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--load-dataset",
        type=str,
        help="Load existing dataset from CSV file"
    )
    
    parser.add_argument(
        "--save-dataset",
        type=str,
        help="Save generated dataset to CSV file"
    )
    
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation (must provide --load-dataset)"
    )
    
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Generate and save dataset only, skip evaluation"
    )
    
    parser.add_argument(
        "--testset-size",
        type=int,
        default=10,
        help="Number of test samples to generate (default: 10)"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    try:
        logger.info("🚀 Starting Simplified Chain Evaluation")
        
        # Step 1: Load or generate test dataset
        if args.skip_generation:
            if not args.load_dataset:
                logger.error("❌ --skip-generation requires --load-dataset")
                sys.exit(1)
            docs = []  # Not needed when loading dataset
            dataset = load_test_dataset(args.load_dataset)
        elif args.load_dataset:
            # Load existing dataset and generate it if it doesn't exist
            dataset_path = Path(args.load_dataset)
            if dataset_path.exists():
                logger.info("📁 Existing dataset found, loading...")
                dataset = load_test_dataset(str(dataset_path))
                docs = []  # Not needed
            else:
                logger.info("📁 Dataset not found, generating new one...")
                docs = load_documents()
                dataset = generate_test_dataset(docs, testset_size=args.testset_size, save_path=args.load_dataset)
        else:
            # Generate new dataset
            docs = load_documents()
            save_path = args.save_dataset or f"evaluation_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            dataset = generate_test_dataset(docs, testset_size=args.testset_size, save_path=save_path)
        
        # If dataset-only mode, exit after saving
        if args.dataset_only:
            logger.info("🎉 Dataset generation completed!")
            return
        
        # Step 2: Load documents if not already loaded
        if not docs:
            docs = load_documents()
        
        # Step 3: Create chains with retrievers
        chains = create_chains_with_retrievers(docs)
        
        # Step 4: Evaluate each chain
        evaluation_results = {}
        
        for chain_name, chain in chains.items():
            try:
                # Skip confidence chain for now as it's not a retrieval method
                if chain_name == "confidence":
                    logger.info(f"⏭️ Skipping {chain_name} (not a retrieval method)")
                    continue
                
                logger.info(f"🔄 Processing method: {chain_name}")
                
                # Create a copy of the dataset for this method
                method_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
                
                # Evaluate the method
                method_dataset = evaluate_chain(chain_name, chain, method_dataset)
                
                # Run RAGAS evaluation
                result = run_ragas_evaluation(method_dataset, chain_name)
                
                evaluation_results[chain_name] = result
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {chain_name}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                evaluation_results[chain_name] = None
        
        # Step 5: Print results
        print_evaluation_results(evaluation_results)
        
        # Step 6: Save results
        results_df = pd.DataFrame()
        for method_name, result in evaluation_results.items():
            if result is not None and hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                df['method'] = method_name
                results_df = pd.concat([results_df, df], ignore_index=True)
        
        # Save to CSV
        if not results_df.empty:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"simple_evaluation_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            
            logger.info(f"💾 Results saved to: {results_file}")
        
        logger.info("🎉 Simplified chain evaluation completed!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()

