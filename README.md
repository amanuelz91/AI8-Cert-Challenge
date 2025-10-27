# RAG System for Student Loan Customer Service

## Task 1: Problem Statement

### 1. Write a succinct 1-sentence description of the problem

**Answer:** Obtaining precise, reliable information about student loans and repayment options poses significant challenges across all stakeholders—from current and prospective borrowers to customer service representatives at federal student loan organizations such as MOHELA and Federal Student Aid.

### 2. Write 1-2 paragraphs on why this is a problem for your specific user

**Answer:** Millions of Americans face substantial challenges due to unclear, inconsistent, and difficult-to-access information about student loans as they navigate the repayment process. With payments resuming and regulations in constant flux, borrowers increasingly rely on servicers like MOHELA and Federal Student Aid for guidance. However, even the customer service representatives within these institutions find it difficult to deliver accurate responses, hampered by legacy technology infrastructure, disconnected information systems, and ever-changing federal regulations.

These information barriers result in extended call wait times and communication breakdowns, which exacerbate borrowers' financial stress and erode trust in the system. The scale of this issue is immense: as of Q1 2025, total student loan debt stands at $1.777 trillion, with $1.693 trillion held by the federal government, directly impacting approximately 42.7 million individuals across the United States. Demand for support services is projected to increase significantly in 2025, as more borrowers seek assistance with understanding repayment plans, forgiveness opportunities, and interest accrual.

### 3. Success

**Answer:** 🧪 **Hypothesis:** Existing tools for searching, retrieving, and generating helpful responses fail to significantly enhance an agent's productivity or ability to manage high volumes of customer inquiries on a daily basis.

### 4. Audience

**Answer:** Our target users are customer service representatives working for federal student loan organizations, including MOHELA and Federal Student Aid. These professionals handle thousands of daily inquiries from borrowers seeking assistance with repayment strategies, forgiveness programs, and regulatory updates. Our objective is to provide them with more efficient, precise tools for information retrieval and generation—minimizing the time they spend navigating disconnected databases and ensuring they can deliver clear, authoritative responses to borrowers in need.

---

## Task 2: Propose a Solution

### 1. Write 1-2 paragraphs on your proposed solution. How will it look and feel to the user?

**Answer:** Our solution is a production-grade RAG (Retrieval-Augmented Generation) system that provides customer service agents with an intuitive, fast, and reliable interface for answering student loan questions. The system feels like having an expert assistant by their side—agents simply type their question in a clean interface, and within seconds they receive a comprehensive answer backed by official policy documents, complete with source citations. The API supports both synchronous query responses and real-time streaming, so agents can see answers generate progressively for longer queries. Beyond simple Q&A, the system intelligently selects from multiple retrieval strategies—from fast semantic search over knowledge base documents to real-time web search via Tavily for questions not in the knowledge base. The interface provides confidence scores, source documents, and detailed metadata about which retrieval method was used, empowering agents to verify information and respond with full transparency to borrowers.

### 2. Describe the tools you plan to use in each part of your stack. Write one sentence on why you made each tooling choice.

#### 1. LLM

**Answer:** We use **GPT-4o-mini** as our primary LLM because it offers the best balance of cost, speed, and reasoning quality for customer support use cases, enabling us to handle high query volumes without compromising response accuracy.

#### 2. Embedding Model

**Answer:** We use **OpenAI's text-embedding-3-small** because it provides high-quality semantic representations with 1536 dimensions at a fraction of the cost of larger models, perfectly matching our vector database configuration and throughput requirements.

#### 3. Orchestration

**Answer:** We use **LangChain LCEL** for declarative chain composition and **LangGraph** for complex multi-step workflows because they provide clean abstractions for combining retrieval, generation, and tool calls while maintaining production-grade error handling and observability.

#### 4. Vector Database

**Answer:** We use **Qdrant** as our vector database because it offers production-ready performance, efficient similarity search at scale, and cloud deployment options that match enterprise requirements for reliability and compliance.

#### 5. Monitoring

**Answer:** We use **LangSmith** for observability because it provides comprehensive tracing of the entire RAG pipeline, from retrieval through generation, enabling us to debug issues, optimize performance, and ensure response quality in production.

#### 6. Evaluation

**Answer:** We use **RAGAS** (Retrieval-Augmented Generation Assessment) for evaluation because it provides automated, metrics-driven assessment of context recall, faithfulness, and answer relevancy without requiring manual labeling for every test case.

#### 7. User Interface

**Answer:** We use **FastAPI** for our REST API because it provides automatic OpenAPI documentation, native async/await support for concurrent queries, built-in validation with Pydantic, and easy streaming support for real-time responses.

#### 8. (Optional) Serving & Inference

**Answer:** We use **Uvicorn** as our ASGI server because it provides excellent performance for async workloads, native HTTP/2 support, and easy production deployment with process management and reload capabilities.

### 3. Where will you use an agent or agents? What will you use "agentic reasoning" for in your app?

**Answer:** We do not use explicit agentic frameworks or autonomous agents in our application. Instead, our system uses **tool-based retrieval** where the system intelligently selects and invokes external tools (like Tavily web search) when questions fall outside our knowledge base. The "reasoning" happens through our hybrid retrieval strategy—when a query comes in, the system retrieves from both the vector database and optional web search, then synthesizes these different information sources into a coherent answer. If we were to add agentic reasoning in the future, we would use it to dynamically route between retrieval methods based on question complexity (e.g., using a router agent to decide whether a query needs web search vs. knowledge base lookup), and to handle multi-step follow-up questions where the agent maintains conversation context and can ask clarifying questions before providing final answers.

#### We want to build a system that can intelligently answer real-world questions like these:

| **User Type**                        | **Scenario / Example Question**                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Current borrowers (in repayment)** | A borrower says they've fallen three months behind on payments and can't afford to catch up. What's the most effective course of action?        |
| **Current borrowers (in repayment)** | A borrower insists they already submitted their Income-Driven Repayment (IDR) renewal form, but no record exists. How should the agent proceed? |
| **Current borrowers (in repayment)** | A borrower argues, "I shouldn't have to repay these loans — this is unconstitutional." How can the agent respond professionally and accurately? |
| **Prospective borrowers**            | Is taking out a federal student loan in 2025 a financially smart decision?                                                                      |
| **Prospective borrowers**            | How much can a student currently borrow from the government to attend college? Is there a maximum limit?                                        |
| **Prospective borrowers**            | What grants and scholarships are available that don't need to be repaid?                                                                        |

---

## Task 3: Dealing with the Data

### 1. Describe all of your data sources and external APIs, and describe what you'll use them for.

**For our initial RAG prototype**, we'll use a curated selection of **official policy documents** from the [Federal Student Aid Handbook (2025–2026)](https://fsapartners.ed.gov/knowledge-center/fsa-handbook/pdf/2025-2026). These resources provide the foundational rules and procedures that govern federal student aid programs:

- [**Application and Verification Guide**](https://fsapartners.ed.gov/sites/default/files/2025-2026/2025-2026_Federal_Student_Aid_Handbook/_knowledge-center_fsa-handbook_2025-2026_application-and-verification-guide.pdf) — covers application processes, data verification, and eligibility documentation.
- [**Volume 3: Academic Calendars, Cost of Attendance, and Packaging**](https://fsapartners.ed.gov/sites/default/files/2025-2026/2025-2026_Federal_Student_Aid_Handbook/_knowledge-center_fsa-handbook_2025-2026_vol3.pdf) — defines institutional calendars, cost calculations, and aid packaging standards.
- [**Volume 7: The Federal Pell Grant Program**](https://fsapartners.ed.gov/sites/default/files/2025-2026/2025-2026_Federal_Student_Aid_Handbook/_knowledge-center_fsa-handbook_2025-2026_vol7.pdf) — outlines rules, calculations, and disbursement procedures for Pell Grants.
- [**Volume 8: The Direct Loan Program**](https://fsapartners.ed.gov/sites/default/files/2025-2026/2025-2026_Federal_Student_Aid_Handbook/_knowledge-center_fsa-handbook_2025-2026_vol8.pdf) — details policies and requirements for federal Direct Loans, including repayment and servicing.

Additionally, we integrate **Tavily Search API** as an external tool for real-time web search. This allows our system to retrieve up-to-date information that may not be in our knowledge base—particularly useful for questions about recent policy changes, current loan limits, or emerging programs that may not yet be reflected in the static handbook documents.

### 2. Describe the default chunking strategy that you will use. Why did you make this decision?

**Answer:** We use **RecursiveCharacterTextSplitter** with a chunk size of **750 characters** and a chunk overlap of **100 characters**. This strategy splits documents hierarchically across natural boundaries (paragraphs, sentences, words) to preserve semantic coherence. We chose 750 characters because it's large enough to capture meaningful context (typically 2-4 sentences) while remaining small enough to fit multiple chunks within LLM context windows for hybrid retrieval. The 100-character overlap ensures continuity between chunks, preventing important information from being split across chunk boundaries—critical for policy documents where a single sentence might span a chunk edge. For more advanced use cases, we also support **SemanticChunker** which identifies semantic breakpoints rather than fixed character counts, but we default to RecursiveCharacterTextSplitter for its speed and reliability in production.

### 3. [Optional] Will you need specific data for any other part of your application? If so, explain.

**Answer:** Beyond the initial policy documents, we plan to incorporate **student loan complaint data** from the Consumer Financial Protection Bureau (CFPB) as a validation and training corpus. This would help us understand common borrower pain points and ensure our system addresses real-world scenarios. We may also ingest **historical FSA policy updates** to enable temporal reasoning (e.g., "What were the rules last year vs. this year?"), though this requires more sophisticated metadata management. For evaluation purposes, we maintain a test suite of Q&A pairs derived from actual customer service transcripts to measure retrieval quality, response accuracy, and confidence calibration across different question types.

## Task 5: Creating a Golden Test Data Set

### 1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevancy, context precision, and context recall. Provide a table of your output results.

**Answer:** We evaluated five different retrieval strategies on our test dataset using the RAGAS framework. The following table summarizes the key metrics for each method:

| Method                         | Context Recall | Faithfulness | Factual Correctness | Answer Relevancy | Context Entity Recall | Noise Sensitivity | Avg Latency (s) |
| ------------------------------ | -------------- | ------------ | ------------------- | ---------------- | --------------------- | ----------------- | --------------- |
| **Naive RAG**                  | 0.911          | 0.937        | 0.671               | 0.870            | 0.402                 | 0.334             | 10.25           |
| **BM25 RAG**                   | 0.942          | 0.900        | 0.635               | 0.797            | 0.363                 | 0.268             | 8.58            |
| **Contextual Compression RAG** | 0.886          | 0.964        | 0.530               | 0.793            | 0.405                 | 0.354             | 7.10            |
| **Multi-Query RAG**            | 0.928          | 0.915        | 0.618               | 0.874            | 0.403                 | 0.203             | 12.04           |
| **Parent Document RAG**        | 0.862          | 0.954        | 0.622               | 0.878            | 0.366                 | 0.341             | 9.02            |

### 2. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

**Answer:** Several conclusions emerge from our evaluation:

**Strengths:**

- **Faithfulness**: All methods achieve high faithfulness scores (0.900-0.964), indicating that generated answers remain grounded in retrieved context. Contextual Compression RAG leads with 0.964.
- **Context Recall**: BM25 and Multi-Query RAG achieve the highest context recall (0.942 and 0.928), demonstrating success at retrieving relevant information.
- **Answer Relevancy**: Parent Document and Multi-Query RAG score highest (0.878 and 0.874), generating answers that directly address user queries.

**Weaknesses:**

- **Factual Correctness**: Lower across methods (0.530-0.671). Naive RAG leads at 0.671, suggesting retrieval quality and LLM synthesis need improvement.
- **Latency**: Wide range (7.10-12.04 seconds). Contextual Compression RAG is fastest (7.10s); Multi-Query RAG slowest (12.04s).
- **Context Entity Recall**: Low (0.36-0.41), indicating suboptimal extraction of named entities and important terms.

**Recommendations:**

- For production: BM25 RAG offers the best balance of context recall and latency.
- For higher faithfulness: use Contextual Compression RAG.
- For higher answer relevance: use Parent Document RAG.

---

## Task 6: The Benefits of Advanced Retrieval

### 1. Describe the retrieval techniques that you plan to try and to assess in your application. Write one sentence on why you believe each technique will be useful for your use case.

**Answer:**

1. **BM25 RAG**: Keyword-aware, fast retrieval suited to student loan questions that rely on precise terms, such as "Direct PLUS Loan" and "FAFSA form".
2. **Contextual Compression RAG**: Reduces noise by extracting only relevant passages, improving faithfulness, especially in verbose policy documents with tangential info.
3. **Multi-Query RAG**: Generates reformulations to broaden retrieval for paraphrased borrower questions.
4. **Parent Document RAG**: Avoids fragment loss by returning parent chunks, critical for multi-part policy requirements.

### 2. Test a host of advanced retrieval techniques on your application.

**Answer:** We implemented and evaluated all four techniques. Results are in the table in Task 5. Multi-Query and BM25 RAG show the highest context recall (0.928 and 0.942), while Contextual Compression RAG achieves the highest faithfulness (0.964).

---

## Task 7: Assessing Performance

### How does the performance compare to your original RAG application? Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements. Provide results in a table.

**Answer:** As noted in Task 5, advanced retrieval improves select metrics vs naive RAG:

- BM25 RAG: +3.4% context recall (0.942 vs 0.911), lower latency (8.58s vs 10.25s)
- Contextual Compression RAG: +2.9% faithfulness (0.964 vs 0.937), lower latency (7.10s vs 10.25s)
- Multi-Query RAG: +1.9% context recall (0.928 vs 0.911), lower noise sensitivity (0.203 vs 0.334)

Tradeoffs:

- Factual correctness is lower in advanced methods.
- Parent Document RAG has lower context recall (0.862).
- Multi-Query RAG increases latency (12.04s vs 10.25s).

**Recommendation:** BM25 RAG offers the best balance for our use case—higher context recall and lower latency, with slightly lower faithfulness.

### Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?

**Answer:** Planned changes:

1. **Hybrid Retrieval Pipeline**: Combine BM25 (recall), Contextual Compression (faithfulness), and naive semantic search with weighted fusion.
2. **Improved Entity Extraction**: Fix low entity recall (0.36-0.41) with named-entity recognition (NER) and coreference resolution.
3. **Response Verification Loop**: Add a post-retrieval verification step to improve factual correctness.
4. **Conversational Memory**: Implement conversation context management for multi-turn interactions.
5. **Confidence Calibration**: Provide per-answer confidence to help agents decide when to consult additional sources.
6. **Fine-tuned Embeddings**: Domain finetuning to improve retrieval of financial and policy concepts.
7. **Streaming + Caching**: Real-time streaming with semantic caching to reduce latency.
8. **A/B Testing**: Test retrieval strategies with real agents.
