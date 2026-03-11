# LangChain RAG Chatbot

A production-grade Retrieval-Augmented Generation system with hybrid search, multi-stage reranking, knowledge graph entity matching, and LangGraph orchestration. Built for querying technical documentation with source-cited answers.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  LangGraph State Machine (RAGState)                     │
│                                                         │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐  │
│  │  Search   │───▶│ Build Context │───▶│   Generate   │  │
│  └──────────┘    └───────────────┘    └──────────────┘  │
│       │                  │                    │          │
│  BM25 + kNN +      Table-title          LLM answer      │
│  Graph search      linking +            with source      │
│  → RRF fusion      adjacent chunk       citations        │
│  → CrossEncoder    expansion                             │
│    + ColBERT                                             │
│    reranking                                             │
└─────────────────────────────────────────────────────────┘
     │
     ▼
  Answer + Sources + Confidence
```

### Pipeline Flow

1. **Hybrid Retrieval** — Queries Elasticsearch with three parallel strategies: BM25 (lexical), kNN (semantic via BGE embeddings), and knowledge graph entity matching. Results are fused using Reciprocal Rank Fusion (RRF).
2. **Multi-stage Reranking** — A CrossEncoder (`ms-marco-MiniLM-L-6-v2`) scores query-document relevance, followed by a secondary bi-encoder (`bge-base-en-v1.5`) for semantic diversity. Cross-referenced chunks from top results are fetched for completeness.
3. **Context Building** — Filters low-scoring chunks, expands with adjacent pages, links table titles to their table content via keyword matching, and injects section summaries. Truncates to a configurable character limit.
4. **Generation** — Passes the assembled context, conversation history, and question to the LLM with a system prompt requiring source citations (file name, page, section).

## Project Structure

```
.
├── config.py                     # All configuration (models, retrieval params, patterns)
├── rag_chatbot/                  # Local pipeline (GPU, self-hosted models)
│   ├── pipeline.py               # RAGPipeline class + LangGraph state machine
│   └── utils/
│       ├── types.py              # RAGState, RAGConfig, DocSearchArgs dataclasses
│       ├── model_loading.py      # Cached model loaders (LLM, embeddings, rerankers)
│       ├── knowledge_graph.py    # NetworkX-based hierarchical KG (Doc→Section→Chunk→Entity)
│       ├── table_context.py      # Context builder with table-title linking
│       ├── conversation_history.py  # Sliding-window conversation memory
│       └── optimized_retrieval.py   # Hybrid search + RRF + reranking logic
├── rag_chatbot_cloud/            # Cloud pipeline (OpenRouter API, Cohere reranking)
│   ├── cloud_rag.py              # Cloud RAGPipeline using OpenRouter (DeepSeek V3)
│   └── requirements.txt
├── webapp/
│   ├── backend/
│   │   └── api.py                # FastAPI server (/query, /feedback, /health)
│   └── frontend/
│       └── src/
│           └── App.vue           # Vue 3 chat interface with markdown + feedback
├── Dockerfile.backend            # Python 3.12 + PyTorch + LangChain + CUDA
├── Dockerfile.frontend           # Node 20 + Vite dev server
├── Dockerfile.nginx              # Reverse proxy (frontend + API routing)
├── nginx.conf                    # Proxy config with SSE streaming support
├── setup-network-access.sh       # LAN access setup (WSL2/Windows port forwarding)
├── toggle-access.sh              # Enable/disable network exposure
└── check-prerequisites.sh        # Dependency verification script
```

## Key Components

### Knowledge Graph

A hierarchical directed graph built with NetworkX, following the structure **Document → Section → Chunk → Entity**. Entities are extracted via regex patterns for industry standards (BS EN, IEC, ISO), specification references, and part numbers. The graph enables entity-based retrieval where user queries mentioning a standard like "IEC 60815" directly resolve to all chunks referencing it.

### Retrieval Strategy

| Stage | Method | Default K |
|-------|--------|-----------|
| BM25 (lexical) | Elasticsearch full-text | 50 |
| kNN (semantic) | BGE-large-en-v1.5 embeddings | 30 |
| Graph (entity) | Knowledge graph entity lookup | 20 |
| RRF Fusion | k=60 | — |
| CrossEncoder rerank | ms-marco-MiniLM-L-6-v2 | top 30 |
| ColBERT rerank | bge-base-en-v1.5 (weight 0.3) | — |
| Final selection | Score-sorted | top 10 |

### Table-Title Linking

When a body chunk references a table (e.g. "Table 3.2 — Insulation clearances"), the context builder scans all retrieved chunks for matching table content on the same page using keyword overlap, then injects the table data alongside the referencing text.

### Dual Mode

The system supports two runtime modes via `USAGE_MODE` in `config.py`:

- **Local** — Self-hosted models on GPU: Qwen2.5-3B-Instruct for generation, BGE-large for embeddings, CrossEncoder + ColBERT for reranking.
- **Cloud** — OpenRouter API (DeepSeek V3) for generation, optional Cohere reranking (`rerank-v3.5`), larger retrieval windows (100 BM25 / 100 kNN / top 20 final).

## Tech Stack

**Backend:** Python 3.12, FastAPI, LangChain, LangGraph, LangSmith, Elasticsearch 8, PyTorch, sentence-transformers, NetworkX

**Frontend:** Vue 3 (Composition API), Vite, vue-markdown-render

**Infrastructure:** Docker (multi-container), nginx reverse proxy, WSL2 LAN access scripts

**Models (local mode):**
- LLM: `Qwen/Qwen2.5-3B-Instruct` (FP16, GPU)
- Embeddings: `BAAI/bge-large-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Secondary reranker: `BAAI/bge-base-en-v1.5`

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA drivers (local mode)
- Elasticsearch 8.x instance
- Python 3.12+ (for development outside Docker)

Run the included check script to verify:

```bash
./check-prerequisites.sh
```

## Setup

### 1. Elasticsearch

Ensure Elasticsearch is running and accessible. Configure the connection in `config.py` or via environment variables:

```bash
export ES_URL="http://localhost:9200"
export ES_INDEX="rag_documents"
```

### 2. Knowledge Graph

Build the graph from your document extraction JSON:

```bash
python -m rag_chatbot.utils.knowledge_graph extraction.json graph.pkl
```

Place the output at the path specified by `GRAPH_PATH` in `config.py`.

### 3. LangSmith (Optional)

For tracing and observability, set your LangSmith API key in `config.py`:

```python
LANGCHAIN_TRACING_V2 = True
LANGSMITH_API_KEY = "your-key-here"
```

### 4. Run with Docker

```bash
docker build -f Dockerfile.backend -t rag-backend .
docker build -f Dockerfile.frontend -t rag-frontend .
docker build -f Dockerfile.nginx -t rag-nginx .
```

The backend runs on port 8000, the frontend on 5173, and nginx exposes port 80 with `/api/` routing to the backend.

### 5. LAN Access (WSL2)

To expose the app on your local network:

```bash
./setup-network-access.sh
./toggle-access.sh
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Submit a question, returns answer + sources + metadata |
| `/feedback` | POST | Log thumbs-up/down feedback with full context |
| `/health` | GET | Health check returning index name |

**Query request:**

```json
{ "question": "What are the insulation requirements for 132kV?" }
```

**Query response:**

```json
{
  "answer": "According to BS EN 60060-1, the insulation...",
  "sources": [
    { "file": "spec_132kv.pdf", "page": 14, "score": 1.234, "content_type": "body", "section": "3.2.1" }
  ],
  "metadata": { "num_results": 10, "confidence": "HIGH" }
}
```

## Configuration Reference

All tunable parameters live in `config.py`. Key settings:

| Parameter | Local | Cloud | Description |
|-----------|-------|-------|-------------|
| `BM25_K` | 50 | 100 | BM25 candidates |
| `VECTOR_K` | 30 | 100 | kNN candidates |
| `GRAPH_K` | 20 | — | Knowledge graph candidates |
| `RERANK_TOP_N` | 30 | 50 | Candidates passed to reranker |
| `FINAL_TOP_K` | 10 | 20 | Final chunks for context |
| `RRF_K` | 60 | 60 | RRF fusion constant |
| `MAX_CONTEXT_CHARS` | 10,000 | 30,000 | Context window character limit |
| `COLBERT_WEIGHT` | 0.3 | — | Secondary reranker weight |

## License

This project does not include a license file. All rights reserved unless otherwise specified.
