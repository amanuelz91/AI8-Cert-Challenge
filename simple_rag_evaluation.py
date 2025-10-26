#!/usr/bin/env python3
"""
Simple RAG Evaluation Script using RAGAS

This script follows a straightforward approach:
1. Load PDF documents from data directory
2. Generate test dataset using RAGAS TestsetGenerator
3. Evaluate all retrieval methods defined in rag_chains.py
4. Compare results using RAGAS metrics

Usage:
    python simple_rag_evaluation.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

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

# Our RAG system imports
from src.chains.rag_chains import create_rag_chain_builder
from src.config.settings import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_documents(data_path: str = "src/data", max_docs: int = 50):
    """Load PDF documents from the data directory."""
    logger.info(f"📁 Loading documents from: {data_path}")
    
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    
    # # Limit the number of documents to speed up processing
    # if len(docs) > max_docs:
    #     logger.info(f"📄 Limiting to {max_docs} documents (from {len(docs)}) for faster processing")
    #     docs = docs[:max_docs]
    
    logger.info(f"✅ Loaded {len(docs)} documents")
    return docs


def generate_test_dataset(docs: List, testset_size: int = 10):
    """Generate test dataset using RAGAS TestsetGenerator."""
    logger.info(f"🎯 Generating test dataset with {testset_size} samples")
    logger.info(f"📄 Processing {len(docs)} documents...")
    
    # Setup generator LLM and embeddings
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Create testset generator
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    
    # Generate dataset with smaller size for faster processing
    logger.info("⏳ This may take a few minutes...")
    logger.info("⚠️  Note: You may see 'headlines' transformation warnings - this is a known RAGAS issue and can be safely ignored")
    
    try:
        dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    except Exception as e:
        # Sometimes RAGAS transformations fail but dataset generation still works
        logger.warning(f"⚠️  RAGAS transformation error (this is often safe to ignore): {str(e)}")
        logger.info("🔄 Retrying with error handling...")
        dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    
    logger.info(f"✅ Generated test dataset with {len(dataset)} samples")
    return dataset


def setup_rag_chains(documents: List[Any]):
    """Setup all RAG chains for evaluation."""
    logger.info("🔧 Setting up RAG chains")
    
    # Get config and create LLM
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
    
    # Create chains using the builder directly
    builder = create_rag_chain_builder(llm)
    chains = {}
    retrievers = {}
    
    logger.info("🔗 Creating basic RAG chains")
    
    # Create naive chain if documents are provided
    if documents:
        try:
            naive_chain, naive_ret = builder.create_naive_rag_chain(documents, embeddings, k=config.retrieval.default_k)
            chains["naive_rag"] = naive_chain
            retrievers["naive_rag"] = naive_ret
            logger.info("✅ Created naive chain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create naive chain: {e}")
        
        # Add advanced retrieval chains
        logger.info("🔗 Adding advanced retrieval chains")
        try:
            bm25_chain, bm25_ret = builder.create_bm25_rag_chain(documents)
            chains["bm25_rag"] = bm25_chain
            retrievers["bm25_rag"] = bm25_ret
            logger.info("✅ Created BM25 chain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create BM25 chain: {e}")
        
        try:
            pd_chain, pd_ret = builder.create_parent_document_rag_chain(documents)
            chains["parent_document_rag"] = pd_chain
            retrievers["parent_document_rag"] = pd_ret
            logger.info("✅ Created parent document chain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create parent document chain: {e}")
        
        try:
            sc_chain, sc_ret = builder.create_semantic_chunking_rag_chain(documents)
            chains["semantic_chunking_rag"] = sc_chain
            retrievers["semantic_chunking_rag"] = sc_ret
            logger.info("✅ Created semantic chunking chain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create semantic chunking chain: {e}")
        
        # Create comprehensive ensemble chain using all retrievers
        try:
            logger.info("🔗 Creating comprehensive ensemble chain...")
            ensemble_chain, ensemble_ret = builder.create_comprehensive_ensemble_chain(documents, embeddings)
            chains["comprehensive_ensemble_rag"] = ensemble_chain
            retrievers["comprehensive_ensemble_rag"] = ensemble_ret
            logger.info("✅ Created comprehensive ensemble chain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create comprehensive ensemble chain: {e}")
        
        # Note: Compression and multi-query need base retrievers, which we no longer store
        # These would need to be recreated if needed
        logger.warning("⚠️ Contextual compression and multi-query chains require retrievers (not available)")
    
    logger.info(f"✅ Final chains: {list(chains.keys())}")
    
    return chains


def evaluate_rag_method(chain_name: str, chain: Any, dataset: EvaluationDataset):
    """Evaluate a specific RAG method."""
    logger.info(f"🎯 Evaluating method: {chain_name}")
    
    # Generate responses for this method
    responses = []
    retrieved_contexts = []
    
    for test_row in dataset:
        try:
            # Get the question from the test row - RAGAS uses different structure
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
            
            responses.append(response_text)
            retrieved_contexts.append(context_strings)
            
            # Update the test row with our response - RAGAS uses different structure
            test_row.response = response_text
            test_row.retrieved_contexts = context_strings
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {chain_name}: {str(e)}")
            responses.append("")
            retrieved_contexts.append([])
            
            # Update with empty response on error
            test_row.response = ""
            test_row.retrieved_contexts = []
    
    logger.info(f"✅ Generated {len(responses)} responses for {chain_name}")
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
            
            # Show sample results
            print(f"\n📋 Sample Results:")
            for i, row in df.head(3).iterrows():
                print(f"  Question {i+1}: {row.get('question', 'N/A')[:50]}...")
                print(f"  Response: {row.get('response', 'N/A')[:100]}...")
                print()
        else:
            print(f"❌ No results available for {method_name}")


def main():
    """Main evaluation function."""
    try:
        # Disable LangSmith tracing to avoid rate limits
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_PROJECT"] = ""
        
        logger.info("🚀 Starting Simple RAG Evaluation")
        
        # Step 1: Load documents - just 1 for fast testing
        docs = load_documents(max_docs=50)
        
        # Step 2: Generate test dataset - just 2 samples for fast testing
        dataset = generate_test_dataset(docs, testset_size=10)
        
        # Step 3: Setup RAG chains (pass documents to enable advanced retrievers)
        chains = setup_rag_chains(docs)
        
        # Step 4: Evaluate each method
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
                method_dataset = evaluate_rag_method(chain_name, chain, method_dataset)
                
                # Run RAGAS evaluation
                result = run_ragas_evaluation(method_dataset, chain_name)
                
                evaluation_results[chain_name] = result
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {chain_name}: {str(e)}")
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
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simple_evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"💾 Results saved to: {results_file}")
        logger.info("🎉 Simple RAG evaluation completed!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
