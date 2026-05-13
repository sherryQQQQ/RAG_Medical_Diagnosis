# Medical Diagnosis with Agentic GraphRAG

This project is a medical question-answering system that evolves a baseline
Retrieval-Augmented Generation demo into an agentic GraphRAG architecture.

The core idea is simple: a language model should not answer medical questions
from memory alone. It should first retrieve grounded clinical evidence, combine
semantic retrieval with structured medical relationships, and then generate an
answer that can be evaluated for correctness and grounding.

## What This Project Does

Given a patient-style clinical question, the system is designed to:

1. Retrieve relevant guideline evidence.
2. Compare a simple vector-only RAG baseline with a graph-informed hybrid method.
3. Use structured medical terms from the original dataset to improve retrieval.
4. Provide measurable evaluation using MRR, Top-1 accuracy, and Top-3 accuracy.
5. Support a future agent workflow with LangGraph, Gemini, Neo4j, and LangSmith.

The current fully runnable path is the synthetic retrieval benchmark, which does
not require external APIs. The full LLM/Neo4j agent path requires credentials.

## Architecture

```mermaid
flowchart TD
    A["Original Dataset<br/>final/medical_generalization.csv"]:::data
    B["Synthetic Data Builder<br/>extract terms from prompt + answer"]:::process
    C["Synthetic Retrieval Dataset<br/>graphrag/eval/data/*.json"]:::data

    C --> D["Vector-Only Baseline<br/>lexical answer-document retrieval"]:::baseline
    C --> E["Graph-Term Retriever<br/>prompt/answer medical term matching"]:::graph

    D --> F["Vector Ranking"]:::rank
    E --> G["Graph Ranking"]:::rank
    F --> H["Hybrid Fusion<br/>Reciprocal Rank Fusion"]:::fusion
    G --> H

    D --> I["Evaluation<br/>MRR, Top-1, Top-3"]:::eval
    H --> I

    I --> J["Result<br/>Hybrid retrieval beats vector-only baseline"]:::result

    subgraph Agentic_Expansion["Agentic GraphRAG Expansion"]
        K["LangGraph ReAct Agent"]:::agent
        L["FAISS Vector Retriever"]:::agent
        M["Neo4j Knowledge Graph"]:::agent
        N["Gemini Generator"]:::agent
        O["Reflector / Validator"]:::agent
        K --> L
        K --> M
        L --> N
        M --> N
        N --> O
        O --> K
    end

    J -. "validated retrieval layer" .-> Agentic_Expansion

    classDef data fill:#E8F3FF,stroke:#2563EB,stroke-width:2px,color:#0F172A;
    classDef process fill:#F0FDFA,stroke:#0D9488,stroke-width:2px,color:#0F172A;
    classDef baseline fill:#FFF7ED,stroke:#EA580C,stroke-width:2px,color:#0F172A;
    classDef graph fill:#F5F3FF,stroke:#7C3AED,stroke-width:2px,color:#0F172A;
    classDef rank fill:#F8FAFC,stroke:#64748B,stroke-width:1.5px,color:#0F172A;
    classDef fusion fill:#ECFDF5,stroke:#16A34A,stroke-width:2px,color:#0F172A;
    classDef eval fill:#FEFCE8,stroke:#CA8A04,stroke-width:2px,color:#0F172A;
    classDef result fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#052E16;
    classDef agent fill:#FDF2F8,stroke:#DB2777,stroke-width:1.5px,color:#0F172A;
```

## How Synthetic Data Is Built

The synthetic benchmark is derived from the original dataset:

```text
final/medical_generalization.csv
```

Each original row contains:

- `prompt`: the original patient-style clinical question
- `answer_before`: older or weaker baseline answer
- `answer`: updated guideline-grounded answer

The synthetic data builder turns each row into a retrieval case:

```text
original QA row
-> case_id, source prompt, source answer
-> extracted prompt terms
-> extracted answer terms
-> synthetic retrieval queries
-> gold expected case_id
```

Term extraction is deterministic. It pulls:

- medical abbreviations, such as `DASI`, `EGFR`, `NSCLC`
- numeric clinical details, such as `14 days`, `180/110`, `3-4 days`
- unigram, bigram, and trigram medical phrases from prompts and answers

The generated JSON dataset is saved at:

```text
graphrag/eval/data/synthetic_medical_retrieval.json
```

## Compared Methods

### 1. Vector-Only Baseline

This is a lightweight stand-in for vanilla RAG retrieval.

It ranks cases using lexical overlap between the synthetic query and the
answer-style document. The original prompt is not placed directly into the
vector document, so the baseline cannot simply memorize the question text.

### 2. Hybrid Graph + Vector Retrieval

This method keeps the vector baseline but adds graph-style medical term
matching. Prompt and answer terms act like a lightweight synthetic graph:

```text
clinical terms -> original case -> updated guideline answer
```

The vector ranking and graph-term ranking are combined with Reciprocal Rank
Fusion (RRF):

```text
score = sum(1 / (k + rank))
```

This mirrors the intended GraphRAG design: vector search captures semantic text
similarity, while graph retrieval captures structured medical relationships.

## Results

Run:

```bash
.venv/bin/python -m graphrag.main generate-synthetic-data
.venv/bin/python -m graphrag.main synthetic-benchmark
```

Current result:

```text
Queries: 375

Method                    MRR     Top-1   Top-3
Vector-only baseline      0.598   0.541   0.648
Hybrid graph+vector       0.853   0.781   0.901
```

Interpretation:

The hybrid method performs better because many clinical queries contain
specific medical terms, abbreviations, durations, thresholds, or treatment
phrases. A pure vector/lexical baseline may miss the correct row when wording is
indirect. The graph-informed method can use structured prompt/answer terms to
recover the right retrieval target.

## Run Tests

```bash
.venv/bin/python -m unittest graphrag.eval.test_synthetic_compare
```

Expected result:

```text
Ran 4 tests
OK
```

## Full Agentic GraphRAG Path

The repository also contains a more complete agentic architecture:

- `graphrag/retrieval/vector.py`: FAISS vector retriever
- `graphrag/retrieval/graph.py`: Neo4j graph retriever
- `graphrag/retrieval/hybrid.py`: RRF hybrid retriever
- `graphrag/agent/react_agent.py`: LangGraph ReAct-style agent
- `graphrag/kg/builder.py`: Gemini-based knowledge graph extraction
- `graphrag/eval/benchmark.py`: benchmark scaffold

The intended full workflow is:

```text
user query
-> planner / ReAct agent
-> vector retrieval from FAISS
-> graph retrieval from Neo4j
-> hybrid context fusion
-> Gemini answer generation
-> reflector validates grounding and safety
-> final medical QA answer
```

This path requires external services.

## Environment Variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
chmod 600 .env
```

Fill in:

```bash
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

LANGSMITH_API_KEY=
LANGSMITH_PROJECT=medical-graphrag
```

Do not commit `.env`. It is ignored by git.

## Repository Layout

```text
final/
  prepare.py                  baseline chunking + embedding setup
  retrieve.py                 baseline retriever
  generate.py                 Gemini answer generation
  medical_generalization.csv  original evaluation dataset

graphrag/
  main.py                     CLI entry point
  eval/
    synthetic_compare.py      synthetic data builder + method comparison
    test_synthetic_compare.py tests for synthetic benchmark
    data/
      synthetic_medical_retrieval.json
  retrieval/
    vector.py                 FAISS retriever
    graph.py                  Neo4j retriever
    hybrid.py                 RRF hybrid retriever
  agent/
    react_agent.py            LangGraph agent
  kg/
    builder.py                Gemini-to-Neo4j graph builder
```

## Interview Summary

This project demonstrates how a medical QA system can move from vanilla RAG to
agentic GraphRAG. The baseline retrieves answer documents with vector-style
matching. The improved method adds graph-informed clinical term retrieval and
fuses rankings with RRF. On a synthetic benchmark derived from the original
medical dataset, hybrid retrieval improves MRR from `0.598` to `0.853` and
Top-3 accuracy from `0.648` to `0.901`.

The main lesson: RAG gives grounding, graph retrieval gives structure, and an
agent workflow gives controllable tool use and validation.
