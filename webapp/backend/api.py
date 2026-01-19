"""RAG API"""
from pathlib import Path
import sys
import os
RAG_CHATBOT_DIR = Path(__file__).resolve().parent.parent.parent / "rag_chatbot"
sys.path.insert(0, str(RAG_CHATBOT_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
import logging

from utils.rag_pipeline import RAGPipeline, preload_models
from utils.conversation_history import ConversationHistory
from utils.knowledge_graph import KnowledgeGraph

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("RAG_INDEX", "sse_specs")
GRAPH_PATH = os.getenv("GRAPH_PATH", "")

es = Elasticsearch(ES_URL, request_timeout=60)
preload_models()
conversation = ConversationHistory()

kg = None
if GRAPH_PATH and os.path.exists(GRAPH_PATH):
    kg = KnowledgeGraph()
    kg.load(GRAPH_PATH)
    logger.info(f"Knowledge graph loaded: {kg.stats()}")

pipeline = RAGPipeline(es, INDEX_NAME, knowledge_graph=kg)


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    page: int
    score: float
    content_type: Optional[str]
    section: Optional[str]
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    metadata: Dict


@app.post("/query")
def query(req: QueryRequest):
    result = pipeline.query(req.question, conversation)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metadata": result.get("metadata", {})
    }


@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_NAME}
