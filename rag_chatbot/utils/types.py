"""Shared types for RAG pipeline."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TypedDict
from elasticsearch import Elasticsearch

from config import FINAL_TOP_K, ES_INDEX


@dataclass
class Message:
    """A single conversation message."""
    role: str
    content: str
    sources: Optional[List[Dict]] = None


@dataclass
class DocSearchArgs:
    """Arguments for document search."""
    es: Elasticsearch
    index_name: str
    query: str
    embedder: Any = None
    reranker: Any = None
    colbert: Any = None
    filters_clause_values: Optional[Dict] = None
    knowledge_graph: Any = None

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    es: Elasticsearch
    knowledge_graph: Any
    index_name: str = ES_INDEX
    use_colbert: bool = True

class RAGState(TypedDict):
    """State for RAG pipeline graph."""
    question: str
    history: str
    results: List[Dict]
    context: str
    answer: str
    sources: List[Dict]
    confidence: str
    start_time: float
