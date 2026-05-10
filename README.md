# MemGraph
## Personal AI Memory + Knowledge Graph
Graph-RAG powered memory system built with Neo4j, pgvector, and hybrid retrieval.

## Overview

MemGraph ingests documents, notes, emails, and web content, builds a living knowledge graph, and enables natural-language querying using Graph-RAG.

Unlike traditional RAG systems, MemGraph combines:

* Semantic vector retrieval
* Multi-hop graph traversal
* Entity relationships
* Context-aware retrieval
* Persistent memory systems

This enables queries that flat vector search struggles to answer reliably.

## Features

| Feature                | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| Hybrid Graph-RAG       | Combines pgvector semantic search with Neo4j traversal |
| Knowledge Graph        | Entity + relationship extraction using LLMs            |
| Multi-source Ingestion | PDF, DOCX, TXT, Markdown, URLs                         |
| Memory Decay           | Importance scores decay over time                      |
| Async Pipeline         | FastAPI + Celery + Redis ingestion architecture        |
| Community Detection    | Louvain clustering for related concepts                |
| Local Embeddings       | sentence-transformers embedding pipeline               |
| Docker Stack           | Production-ready containerized services                |

## Architecture

```text
Documents / URLs
       │
       ▼
┌────────────────────┐
│  Ingestion Layer   │
│ FastAPI + Celery   │
└─────────┬──────────┘
          │
          ├──────────────▶ PostgreSQL + pgvector
          │
          └──────────────▶ Neo4j Knowledge Graph
                                  │
                                  ▼
                         Hybrid Graph-RAG
                                  │
                                  ▼
                             LLM Answers
```

## Quickstart

### 1. Clone Repository

```bash
git clone https://github.com/yourname/memgraph
cd memgraph
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_key_here
```

Get a free key from [https://openrouter.ai](https://openrouter.ai)

### 3. Start Services

```bash
docker compose up --build
```

This launches:

* FastAPI
* Neo4j
* PostgreSQL + pgvector
* Redis
* Celery worker

### 4. Ingest Content

**Ingest a URL**

```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://en.wikipedia.org/wiki/Knowledge_graph"}'
```

**Check Job Status**

```bash
curl http://localhost:8000/ingest/status/<job_id>
```

### 5. Query the System

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is a knowledge graph used for?"}'
```

### 6. Explore the Graph

**Neo4j Browser:** [http://localhost:7474](http://localhost:7474)

**Credentials**
