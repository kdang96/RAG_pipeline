
# Title
Evaluation-Driven Retrieval for Document-Grounded Assistants

# Overview
This repository explores the design and evaluation of a document-grounded retrieval system intended for use in assistant-style autonomous agents. The focus is on evaluation-driven development, where retrieval quality is assessed independently of large language model (LLM) generation to ensure clear attribution of system behaviour.

The project demonstrates:
- structured ingestion of heterogeneous .docx documents
- heading-aware text chunking
- vector-based semantic retrieval
- offline retrieval evaluation (Recall@k, MRR)
- integration of retrieval into a minimal agentic assistant with tool-use

## Setup

\`\`\`bash
git clone https://github.com/kdang96/RAG_pipeline.git
cd RAG_pipeline

pip install -r requirements.txt
pip install -e .

# Motivation

In assistant systems, failures are often difficult to diagnose because retrieval, reasoning, and generation are tightly coupled. This project deliberately decouples retrieval evaluation from generation, allowing retrieval quality to be measured deterministically before introducing an LLM.

This mirrors evaluation-driven research workflows commonly used in applied ML systems.

# System Architecture

1. Ingestion

Raw .docx documents are parsed and normalised.

Document structure is preserved where possible (headings, sections).

2. Chunking

A simple, explicit chunking strategy is used.

Chunk size trade-offs are intentionally explored rather than over-optimised.

3. Vector Store

Chunks are embedded using a sentence-embedding model.

Embeddings are stored in Milvus for similarity search.

4. Retrieval

Similarity search retrieves top-k chunks per query.

Retrieval operates independently of any LLM.

5. Evaluation

Retrieval quality is measured using Recall@k and MRR against a hand-constructed test set.

Evaluation is fully deterministic and reproducible.

6. Agentic Demo

A minimal agent uses tool selection (tool_choice="auto") to decide when to retrieve.

The agent is provided for qualitative exploration only and is not part of the evaluation loop.

# Evaluation Methodology

Retrieval is evaluated using:

Recall@k: measures whether relevant chunks appear in the top-k results

Mean Reciprocal Rank (MRR): captures ranking quality

The evaluation dataset consists of:
- natural-language queries
- references to relevant document sections
- explicit relevance ordering where applicable

Importantly, evaluation is performed without an LLM to avoid confounding retrieval performance with generation quality.

# Documents

**The documents' heading structure has been amended for ease of parsing.**

State Treaty on the modernisation of media legislation in Germany (N-2020-0026-000-EN.DOCX)
https://technical-regulation-information-system.ec.europa.eu/en/notification/15957/text/D/EN

Report from the Commission to the European Parliament, the Council and the European Economic and Social Committee on the operation of the single market transparency directive from 2016 το 202020 (1_EN_ACT_part1_v4.docx)
https://secure.ipex.eu/IPEXL-WEB/download/file/082d29088354edb301836a5c43790652

Report from the Commission to the European Parliament, the Council and the European Economic and Social Committee on the operation of directive (EU) 2015/1535 FROM 2014 ΤΟ 2015 (1_EN_ACT_part1_v5.docx)
https://secure.ipex.eu/IPEXL-WEB/download/file/082dbcc5618c772b01618ff34350045d

Although several chunks exceed the effective context window of the embedding model, this setup is retained to illustrate realistic failure modes of retrieval in long, densely structured documents rather than to maximise benchmark performance.

Despite exceeding the embedding model’s optimal context length, the oversized chunk is often retrieved correctly due to reduced intra-document competition, illustrating how corpus-level dynamics can offset suboptimal chunk granularity in small collections.


# Design Decisions & Trade-offs

Chunk size: Some chunks intentionally exceed the embedding model’s nominal context window. In small corpora, reduced intra-document competition can still yield strong retrieval performance. This behaviour is documented rather than hidden.

Minimal agent design: The agent does not use reflection or multi-step planning. The goal is to demonstrate tool-use and autonomy without introducing unnecessary complexity.

No framework abstraction: Chunking and retrieval logic are implemented explicitly rather than via higher-level frameworks to keep behaviour transparent and debuggable.

# Limitations

The document set is small and domain-specific.

Retrieval performance may not generalise to large, noisy corpora.

The agentic demo is qualitative and not evaluated quantitatively.

These limitations are intentional to keep the system focused and interpretable.