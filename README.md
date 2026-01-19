# RAG Query & Retrieval System

Production RAG pipeline with hybrid search, knowledge graph enhancement, and LangGraph orchestration.

## Architecture

```
User Query
    ↓
[Knowledge Graph] → Extract entities (BS-EN-60060, IEC standards, etc.)
    ↓
[Hybrid Search] → BM25 + kNN + Graph Matching
    ↓
[RRF Fusion] → Combine rankings (BM25:Vector:Graph = 1:1:1.5)
    ↓
[Cross-Encoder Reranking] → ms-marco-MiniLM-L-6-v2
    ↓
[Context Builder] → Adjacent pages + table linking
    ↓
[LLM Generation] → Qwen2.5-3B-Instruct
    ↓
Answer + Sources
```

## Features

### Hybrid Retrieval (`optimized_retrieval.py`)
- **Triple Search**: BM25 (50) + kNN (30) + Graph (20)
- **RRF Fusion**: Reciprocal rank with 1.5x boost for entity matches
- **Cross-Encoder Reranking**: Top-30 candidates, final score = 0.3*RRF + 0.7*rerank
- **Smart Filtering**: By file, content type, sections

### Knowledge Graph (`knowledge_graph.py`)
- **Entity Extraction**: Standards (BS-EN, IEC, ISO), specs, abbreviations
- **Hierarchical Structure**: Document → Section → Chunk → Entity
- **Graph Queries**: Find chunks by entity, section, or related chunks
- **NetworkX Backend**: Efficient traversal and relationship tracking

### Context Building (`table_context.py`)
- **Table-Title Linking**: Matches "Table 3 - Requirements" to actual table content
- **Adjacent Pages**: Includes ±1 page chunks for context continuity
- **Synonym Expansion**: "legislation" → {regulations, act, law}
- **Score Filtering**: Keep chunks ≥ 10% of top score

### LangGraph Pipeline (`rag_pipeline.py`)
- **State Machine**: search → build_context → generate
- **Confidence Scoring**: HIGH if top_score > 1.0
- **LangSmith Integration**: Automatic tracing with `LANGCHAIN_TRACING_V2=true`
- **Conversation Memory**: Last 10 turns with entity tracking

### Memory Management (`memory_hooks.py`)
- **Q&A History**: Last N exchanges with topics
- **Topic Graph**: Co-occurrence tracking (future: small-world navigation)
- **Auto-Summarization**: After 5 exchanges, compress old context
- **Hooks for**: Hierarchical memory, long-term summaries

## Installation

```bash
pip install elasticsearch langchain langchain-huggingface langgraph
pip install sentence-transformers networkx
```

## Configuration

```bash
# Models
export LLM_MODEL="Qwen/Qwen2.5-3B-Instruct"
export EMBED_MODEL="sentence-transformers/all-mpnet-base-v2"

# LangSmith (optional)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-key"
export LANGCHAIN_PROJECT="rag-system"

# Elasticsearch
export ES_HOST="http://localhost:9200"
export ES_INDEX="rag_documents"
```

## Usage

### Basic Query

```python
from elasticsearch import Elasticsearch
from rag_pipeline import RAGPipeline, preload_models
from conversation_history import ConversationHistory
from knowledge_graph import KnowledgeGraph

# Preload models (cache on GPU)
preload_models()

# Initialize
es = Elasticsearch("http://localhost:9200")
kg = KnowledgeGraph()
kg.load("knowledge_graph.pkl")

pipeline = RAGPipeline(es, "rag_documents", knowledge_graph=kg)
conversation = ConversationHistory()

# Query
response = pipeline.query("What are the requirements for BS EN 60060?", conversation)

print(response["answer"])
print(f"Confidence: {response['metadata']['confidence']}")
print(f"Sources: {len(response['sources'])}")
```

### With Filters

```python
from optimized_retrieval import search_documents
from model_loading import embed_query, get_reranker

results = search_documents(
    es, "rag_documents", 
    "insulator specifications",
    embed_query, 
    get_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    filters={"content_type": "table", "file_name": "specs/HTM_guide.pdf"},
    final_k=10
)
```

### Build Knowledge Graph

```python
from knowledge_graph import build_graph_from_json

# From extraction output
kg = build_graph_from_json("output.json", "kg.pkl")

# Stats
print(kg.stats())
# {"nodes": 15420, "edges": 23150, "entities": 487, "sections": 201}

# Entity lookup
chunks = kg.find_chunks_by_entity("BS-EN-60060-1")
docs = kg.find_documents_by_entity("IEC-60815")
```

## LangSmith Integration

Enable tracing to monitor:
- Retrieval latency and relevance
- Context quality and token usage
- LLM generation metrics
- End-to-end pipeline performance

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "rag-production"

# Queries automatically traced
response = pipeline.query("your question", conversation)
```

View traces at https://smith.langchain.com

## Pipeline Components

### 1. Search Node
```python
def _search_node(state):
    # Extract entities from query via KG
    # Run hybrid search: BM25 + kNN + Graph
    # Return top-K candidates
```

### 2. Context Node
```python
def _build_context_node(state):
    # Filter by score threshold
    # Add adjacent pages
    # Link table titles to content
    # Format with source attribution
```

### 3. Generation Node
```python
def _generate_node(state):
    # Invoke LLM with context + question
    # Calculate confidence score
    # Return answer with metadata
```

## Performance Tuning

### Retrieval Parameters

```python
# optimized_retrieval.py
BM25_K = 50          # BM25 candidates
VECTOR_K = 30        # kNN candidates  
GRAPH_K = 20         # Graph-matched chunks
RERANK_TOP_N = 30    # Chunks to rerank
FINAL_TOP_K = 10     # Return to user
RRF_K = 60           # RRF constant
```

### Context Limits

```python
# table_context.py
MAX_CONTEXT_CHARS = 10000  # ~4k tokens for LongT5
MIN_SCORE_RATIO = 0.1      # Keep if score ≥ 10% of top
```

### Memory

```python
# conversation_history.py
MAX_MESSAGES = 10    # Conversation window

# memory_hooks.py  
SUMMARY_THRESHOLD = 5  # Summarize after N exchanges
```

## Advanced Features

### Graph-Enhanced Retrieval

Knowledge graph provides:
- **Entity-based expansion**: "BS EN 60060" → all related chunks
- **Section navigation**: Find all chunks in Section 4.2
- **Related chunk discovery**: Via shared entities or sections
- **Document filtering**: Which docs mention standard X?

### Table Intelligence

Context builder:
1. Finds tables referenced by title (regex: `Table \d+ - Title`)
2. Scores tables using keyword matching + synonyms
3. Adds best-matching table if score > 0
4. Converts tables to natural language enumerations

### Conversation Continuity

```python
conversation = ConversationHistory()

# Entities tracked across turns
conversation.add_entities(["BS-EN-60060", "insulator"])

# Recent context for follow-ups
context = conversation.get_recent_context(n=3)
```

## Monitoring & Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Inspect Search Results

```python
results = search_documents(...)

for r in results[:3]:
    print(f"Score: {r['final_score']:.3f}")
    print(f"  BM25: {r.get('bm25_rank', 'N/A')}")
    print(f"  Vector: {r.get('vector_rank', 'N/A')}")
    print(f"  Graph: {r.get('graph_rank', 'N/A')}")
    print(f"  RRF: {r['rrf_score']:.3f}")
    print(f"  Rerank: {r['rerank_score']:.3f}")
```

### Graph Statistics

```python
kg.stats()
# {
#   "nodes": 15420,
#   "edges": 23150,
#   "types": {"document": 42, "section": 201, "chunk": 14690, "entity": 487},
#   "entities": 487,
#   "sections": 201
# }

# Top entities
top = sorted(kg.entity_to_chunks.items(), key=lambda x: len(x[1]), reverse=True)[:10]
```

## File Structure

```
rag_query/
├── rag_pipeline.py           # LangGraph orchestration
├── optimized_retrieval.py    # Hybrid search + RRF + reranking
├── knowledge_graph.py        # Entity extraction + graph queries
├── table_context.py          # Context building + table linking
├── model_loading.py          # LangChain HuggingFace loaders
├── conversation_history.py   # Conversation memory
└── memory_hooks.py           # Future: hierarchical/long-term memory
```

## Roadmap

- [x] Hybrid BM25 + kNN search
- [x] Knowledge graph integration
- [x] Cross-encoder reranking
- [x] Table-title linking
- [x] LangSmith tracing
- [ ] Multi-hop reasoning
- [ ] Query expansion with synonyms
- [ ] Confidence calibration
- [ ] Streaming responses
- [ ] Evaluation framework (RAGAS)

## Citation

```bibtex
@software{rag_query_system,
  title = {Hybrid RAG Query System with Knowledge Graph},
  author = {Your Name},
  year = {2025}
}
```

## License

MIT
