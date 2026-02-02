"""RAG API"""
from pathlib import Path
from datetime import datetime
import sys
import os
import json
RAG_CHATBOT_DIR = Path(__file__).resolve().parent.parent.parent / "rag_chatbot"
sys.path.insert(0, str(RAG_CHATBOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
import logging

from rag_chatbot.pipeline import RAGPipeline, RAGConfig, preload_models 
from rag_chatbot.utils.conversation_history import ConversationHistory
from rag_chatbot.utils.knowledge_graph import KnowledgeGraph
from config import FEEDBACK_DIR, ES_URL, ES_INDEX, GRAPH_PATH

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try: 
    es = Elasticsearch(ES_URL, request_timeout=60)
except Exception as e:
    logger.error(f"Error connecting to Elasticsearch: {e}")
    sys.exit(1)

preload_models()
conversation_history = ConversationHistory()

kg = None
if GRAPH_PATH and os.path.exists(GRAPH_PATH):
    kg = KnowledgeGraph()
    kg.load(GRAPH_PATH)
    logger.info(f"Knowledge graph loaded: {kg.stats()}")
else:
    logger.info(f"Knowledge graph not found: {GRAPH_PATH}")
    sys.exit(1)

config = RAGConfig(
    es=es,
    index_name=ES_INDEX,
    knowledge_graph=kg
)
pipeline = RAGPipeline(config)

class QueryRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    context: str
    metadata: Dict
    rating: str


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
def query(req: QueryRequest) -> Dict:
    result = pipeline.query(req.question, conversation_history)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "context": result.get("context", ""),
        "metadata": result.get("metadata", {})
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": req.question,
        "answer": req.answer,
        "sources": req.sources,
        "context": req.context,
        "metadata": req.metadata
    }
    filename = "feedback_good.json" if req.rating == "good" else "feedback_bad.json"
    filepath = FEEDBACK_DIR / filename
    data = []
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
    data.append(entry)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Feedback logged: {filepath}")
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "index": ES_INDEX}
