# RAG Evaluation System

Comprehensive evaluation framework for RAG retrieval methods using RAGAS and LangSmith.

## Features

- **Golden Dataset Generation**: Uses RAGAS TestsetGenerator to create evaluation datasets from PDF documents
- **Multi-Method Evaluation**: Evaluates naive, semantic, tool-based, hybrid, and production RAG methods
- **Comprehensive Metrics**: Uses RAGAS metrics including Context Recall, Faithfulness, Factual Correctness, Response Relevancy, Context Entity Recall, and Noise Sensitivity
- **LangSmith Integration**: Optional tracing and monitoring with LangSmith
- **Configurable**: Flexible configuration through environment variables
- **Results Management**: Automatic saving and comparison of evaluation results

## Quick Start

The evaluation system has been simplified to use a single main script (`evaluate_rag.py`) that handles all evaluation workflows. This eliminates confusion about which script to use.

### 1. Install Dependencies

The evaluation system requires RAGAS and LangSmith. These are already included in the project dependencies:

```bash
# Install project dependencies (includes RAGAS and LangSmith)
uv sync
```

### 2. Set Environment Variables

Create a `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional (for LangSmith tracing)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-evaluation

# Optional (for Tavily search)
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Run Evaluation

#### Complete Evaluation Workflow

```bash
# Generate dataset and evaluate all methods
uv run python evaluate_rag.py --data-path ./src/data --testset-size 10
```

#### Generate Dataset Only

```bash
# Generate evaluation dataset
uv run python evaluate_rag.py --data-path ./src/data --testset-size 10 --dataset-only
```

#### Evaluate Existing Dataset

```bash
# Evaluate using existing local dataset
uv run python evaluate_rag.py --evaluate-only --dataset-path ./evaluation_results/dataset.csv

# Evaluate using LangSmith dataset
uv run python evaluate_rag.py --evaluate-only --langsmith-dataset my_dataset_name

# Or evaluate using the most recent local dataset automatically
uv run python evaluate_rag.py --evaluate-only
```

#### Evaluate Specific Methods

```bash
# Evaluate only specific methods
uv run python evaluate_rag.py --methods naive semantic production
```

### 4. Run Example Script

The example script demonstrates the exact coding practices specified:

```bash
uv run python rag_evaluation_example.py
```

## Configuration

### Environment Variables

| Variable               | Default                | Description                        |
| ---------------------- | ---------------------- | ---------------------------------- |
| `EVAL_TESTSET_SIZE`    | 10                     | Number of test samples to generate |
| `EVAL_GENERATOR_MODEL` | gpt-4o                 | Model for dataset generation       |
| `EVAL_EVALUATOR_MODEL` | gpt-4o-mini            | Model for evaluation               |
| `EVAL_EMBEDDING_MODEL` | text-embedding-3-small | Embedding model                    |
| `EVAL_TIMEOUT`         | 360                    | Evaluation timeout in seconds      |
| `LANGSMITH_PROJECT`    | rag-evaluation         | LangSmith project name             |
| `EVAL_RESULTS_DIR`     | ./evaluation_results   | Results directory                  |

## Evaluation Metrics

The system uses RAGAS metrics to evaluate RAG performance:

- **LLMContextRecall**: Measures how well the retrieved context covers the ground truth
- **Faithfulness**: Evaluates if the response is faithful to the retrieved context
- **FactualCorrectness**: Checks factual accuracy of responses
- **ResponseRelevancy**: Measures relevance of response to the question
- **ContextEntityRecall**: Evaluates entity-level recall from context
- **NoiseSensitivity**: Tests robustness to noisy inputs

## RAG Methods Evaluated

1. **Naive RAG**: Basic vector similarity retrieval
2. **Semantic RAG**: Enhanced semantic retrieval with reranking
3. **Tool-based RAG**: Web search integration (if Tavily available)
4. **Hybrid RAG**: Combination of semantic and tool-based retrieval
5. **Production RAG**: Advanced workflow with multiple strategies

## Results

Evaluation results are saved in JSON format with:

- Individual method scores
- Metric breakdowns
- Method rankings
- Comparison analysis
- Timestamps and metadata

## Example Usage

### Programmatic Usage

```python
from src.evaluation import DatasetGenerator, RAGEvaluator, EvaluationConfig
from src.core.system.rag_system import create_production_rag_system

# Initialize components
config = EvaluationConfig()
rag_system = create_production_rag_system()

# Generate dataset
generator = DatasetGenerator(config)
dataset_result = generator.generate_and_save_dataset(
    data_path="./src/data",
    testset_size=10
)

# Evaluate system
evaluator = RAGEvaluator(config, rag_system)
results = evaluator.evaluate_all_methods(
    dataset=dataset_result['filepath'],
    methods=["naive", "semantic", "production"]
)

# Save results
results_path = evaluator.save_results(results)
print(f"Results saved to: {results_path}")
```

### Following Specified Coding Practices

The system follows the exact coding practices specified in the requirements:

```python
# Document loading
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader

path = "data/"
loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

# RAGAS setup
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", n=3))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from ragas.testset import TestsetGenerator
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

# Evaluation
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas import RunConfig

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
custom_run_config = RunConfig(timeout=360)

baseline_result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=custom_run_config
)
```

## Testing

Run the test script to verify the evaluation system:

```bash
uv run python test_evaluation.py
```

This will:

1. Test dataset generation with a small sample
2. Test RAG evaluation with available methods
3. Print a summary of results

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env`
2. **Memory Issues**: Reduce `testset_size` for large document collections
3. **Timeout Errors**: Increase `EVAL_TIMEOUT` for complex evaluations
4. **Import Errors**: Ensure all dependencies are installed with `uv sync`
5. **RAGAS Transformation Warnings**: You may see warnings like "unable to apply transformation: 'headlines' property not found in this node" - this is a known RAGAS issue and can be safely ignored. The evaluation will still complete successfully.

### Logs

The system provides detailed logging. Check logs for:

- Document loading progress
- Dataset generation status
- Evaluation progress
- Error details

## Architecture

```
src/evaluation/
├── __init__.py          # Module exports
├── config.py            # Configuration management
├── dataset_generator.py # RAGAS dataset generation
├── evaluator.py         # RAG evaluation framework
└── metrics.py           # RAGAS metrics management

evaluate_rag.py          # Main evaluation script (handles all workflows)
rag_evaluation_example.py # Example following specified practices
langsmith_demo.py        # LangSmith integration demo
test_evaluation.py       # Test script
```

The evaluation system integrates seamlessly with the existing RAG system architecture and provides comprehensive evaluation capabilities for all implemented retrieval methods.
