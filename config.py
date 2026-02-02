"""General RAG Configuration."""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "eval" / ".env")

USAGE_MODE = "local" # "local" or "cloud"
#------------------------------------------------------------------------------
# Run_eval.py configs 
#------------------------------------------------------------------------------
DATASETS_DIR = Path(__file__).resolve().parent / "eval" / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "eval" / "results"

#------------------------------------------------------------------------------
# General Webpage configs 
#------------------------------------------------------------------------------
# Backend
FEEDBACK_DIR = Path(__file__).resolve().parent / "dataprocessing" / "feedback"
GRAPH_PATH = Path(__file__).resolve().parent / "dataprocessing" / "graph.pkl"

#------------------------------------------------------------------------------
# General rag configs 
#------------------------------------------------------------------------------
# LangSmith
LANGCHAIN_TRACING_V2=True
LANGSMITH_API_KEY=""

# Elasticsearch
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_documents")

# --- Models ---
EMBED_MODEL = "BAAI/bge-large-en-v1.5"


if USAGE_MODE == "local":
    #------------------------------------------------------------------------------
    # Local rag configs 
    #------------------------------------------------------------------------------
    # --- LangSmith ---
    LANGSMITH_PROJECT="local_rag_chatbot"

    # --- Models ---
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval ---
    BM25_K = 50
    VECTOR_K = 30
    GRAPH_K = 20
    RERANK_TOP_N = 30
    FINAL_TOP_K = 10
    RRF_K = 60

    SOURCE_FIELDS = ["chunk_text", "file_name", "page_number", "content_type", "section",
                  "cross_references", "section_summary", "section_path", "section_context",
                  "resolved_refs", "related_definitions", "standards"]

    # --- Context ---
    MAX_CONTEXT_CHARS = 10000
    MIN_SCORE_RATIO = 0.1
    MAX_MESSAGES = 10

    # --- Knowledge Graph ---
    PATTERNS = {
        "standard": re.compile(r'(?:BS\s*EN|IEC|ISO|ENA(?:\s+(?:ER|TS))?)\s*(?:\w*\d+[-\d]*)', re.I),
        "spec_ref": re.compile(r'[SPF][RPOA][TG]-NET-[A-Z]{3}-\d+', re.I),
        "part_number": re.compile(r'^([A-Z](?:\.\d+)+|\d+(?:\.\d+)*)\s*'),
    }

    # --- Second-stage reranker (bi-encoder for semantic matching) ---
    COLBERT_MODEL = "BAAI/bge-base-en-v1.5"
    COLBERT_WEIGHT = 0.3  # Weight in final score


else:
    #------------------------------------------------------------------------------
    # Cloud rag configs
    #------------------------------------------------------------------------------
    # --- LangSmith ---
    LANGSMITH_PROJECT="cloud_rag_chatbot"

    # API Keys
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Models
    OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324"
    RERANKER_LOCAL = 0  # 0 = Cohere, 1 = local
    RERANK_MODEL_LOCAL = "BAAI/bge-reranker-large"
    RERANK_MODEL_COHERE = "rerank-v3.5"

    # Retrieval
    BM25_K = 100
    VECTOR_K = 100
    RERANK_TOP_N = 50
    FINAL_TOP_K = 20
    RRF_K = 60

    # Context
    MAX_CONTEXT_CHARS = 30000
    MIN_SCORE_RATIO = 0.1
    MAX_MESSAGES = 10

    # Query expansion
    SYNONYMS = {}
    DEFINITION_KEYWORDS = ["what is", "define", "definition", "meaning of"]
    TABLE_KEYWORDS = ["table", "list", "values", "specifications"]

    # Knowledge Graph
    PATTERNS = {
        "standard": re.compile(r'\b(BS\s*EN\s*\d+[-\d]*|IEC[/\s]*\d+[-\d]*|ISO\s*\d+[-\d]*)\b', re.I),
        "spec_ref": re.compile(r'\b([SPF][RPOA][-_]?NET[-_]?[A-Z]{3}[-_]?\d{3})\b', re.I),
        "part_number": re.compile(r'\b([A-Z]{2,4}\d{2,}[-_]?\d*)\b'),
    }
