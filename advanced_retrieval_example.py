#!/usr/bin/env python3
"""
Example script demonstrating advanced retrieval chains for testing and evaluation.

This script shows how to use the new retrieval methods added to rag_chains.py:
- BM25 retrieval
- Contextual compression with Cohere reranking
- Multi-query retrieval
- Parent document retrieval
- Ensemble retrieval
- Semantic chunking retrieval
"""

import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.chains.rag_chains import create_advanced_retrieval_chains, create_rag_chain_builder
from src.core.retrieval import NaiveRetriever
from src.config.settings import get_config

# Set up environment
config = get_config()

def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    sample_texts = [
        "Student loans are financial aid that must be repaid with interest. They help students pay for college expenses including tuition, fees, room and board, books, and supplies.",
        "Federal student loans typically have lower interest rates and more flexible repayment options compared to private student loans. They include Direct Subsidized Loans, Direct Unsubsidized Loans, and Direct PLUS Loans.",
        "The Free Application for Federal Student Aid (FAFSA) is required to apply for federal student loans. It determines your eligibility for financial aid based on your family's financial situation.",
        "Student loan repayment plans include Standard Repayment, Graduated Repayment, Extended Repayment, and Income-Driven Repayment plans. Income-driven plans base payments on your income and family size.",
        "Public Service Loan Forgiveness (PSLF) is available for borrowers who work in qualifying public service jobs and make 120 qualifying payments under an income-driven repayment plan.",
        "Student loan consolidation allows you to combine multiple federal student loans into a single loan with one monthly payment. It can simplify repayment but may not always reduce your interest rate.",
        "Defaulting on student loans can have serious consequences including wage garnishment, tax refund offset, and damage to your credit score. It's important to contact your loan servicer if you're having trouble making payments.",
        "Student loan refinancing involves taking out a new loan with a private lender to pay off existing student loans. This can potentially lower your interest rate but you'll lose federal loan benefits.",
        "The grace period is the time after you graduate, leave school, or drop below half-time enrollment before you must begin repaying your student loans. Most federal loans have a six-month grace period.",
        "Student loan interest may be tax deductible depending on your income and filing status. The maximum deduction is $2,500 per year for qualified student loan interest."
    ]
    
    documents = []
    for i, text in enumerate(sample_texts):
        doc = Document(
            page_content=text,
            metadata={
                "source": f"student_loan_guide_{i+1}",
                "topic": "student_loans",
                "section": f"section_{i+1}"
            }
        )
        documents.append(doc)
    
    return documents

def test_advanced_retrieval_chains():
    """Test all advanced retrieval chains."""
    print("🚀 Testing Advanced Retrieval Chains")
    print("=" * 50)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"📚 Created {len(documents)} sample documents")
    
    # Create a simple naive retriever for testing
    try:
        naive_retriever = NaiveRetriever()
        print("✅ Created naive retriever")
    except Exception as e:
        print(f"⚠️ Failed to create naive retriever: {e}")
        naive_retriever = None
    
    # Create advanced retrieval chains
    try:
        chains = create_advanced_retrieval_chains(
            documents=documents,
            naive_retriever=naive_retriever
        )
        print(f"✅ Created {len(chains)} advanced retrieval chains")
        
        # List available chains
        print("\n📋 Available chains:")
        for chain_name in chains.keys():
            print(f"  - {chain_name}")
        
    except Exception as e:
        print(f"❌ Failed to create advanced retrieval chains: {e}")
        return
    
    # Test each chain with a sample question
    test_question = "What are the different types of student loan repayment plans?"
    print(f"\n🔍 Testing with question: '{test_question}'")
    print("=" * 50)
    
    for chain_name, chain in chains.items():
        try:
            print(f"\n🧪 Testing {chain_name}...")
            
            # Invoke the chain
            result = chain.invoke({"question": test_question})
            
            # Extract response and context
            response = result.get("response", "No response")
            context = result.get("context", [])
            
            print(f"✅ {chain_name} completed successfully")
            print(f"📄 Retrieved {len(context)} documents")
            print(f"💬 Response preview: {response[:200]}...")
            
            # Show context details
            if context:
                print(f"📚 Context sources:")
                for i, doc in enumerate(context[:3]):  # Show first 3 docs
                    source = doc.metadata.get("source", "Unknown")
                    print(f"  {i+1}. {source}")
            
        except Exception as e:
            print(f"❌ {chain_name} failed: {e}")
    
    print("\n🎉 Advanced retrieval chain testing completed!")

def test_individual_chains():
    """Test individual chain creation."""
    print("\n🔧 Testing Individual Chain Creation")
    print("=" * 50)
    
    documents = create_sample_documents()
    builder = create_rag_chain_builder()
    
    # Test BM25 chain
    try:
        bm25_chain = builder.create_bm25_rag_chain(documents)
        print("✅ BM25 chain created successfully")
        
        # Test the chain
        result = bm25_chain.invoke({"question": "What is FAFSA?"})
        print(f"📄 BM25 retrieved {len(result.get('context', []))} documents")
        
    except Exception as e:
        print(f"❌ BM25 chain failed: {e}")
    
    # Test semantic chunking chain
    try:
        semantic_chunking_chain = builder.create_semantic_chunking_rag_chain(documents)
        print("✅ Semantic chunking chain created successfully")
        
        # Test the chain
        result = semantic_chunking_chain.invoke({"question": "What is FAFSA?"})
        print(f"📄 Semantic chunking retrieved {len(result.get('context', []))} documents")
        
    except Exception as e:
        print(f"❌ Semantic chunking chain failed: {e}")

if __name__ == "__main__":
    print("🚀 Advanced Retrieval Chains Example")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set. Some chains may not work properly.")
    
    # Run tests
    test_advanced_retrieval_chains()
    test_individual_chains()
    
    print("\n✨ Example completed!")
    print("\n💡 Usage Tips:")
    print("  - Each chain can be invoked using .invoke({'question': 'your question'})")
    print("  - All chains return {'response': '...', 'context': [...]}")
    print("  - Use these chains for A/B testing different retrieval methods")
    print("  - Compare performance using evaluation metrics")
