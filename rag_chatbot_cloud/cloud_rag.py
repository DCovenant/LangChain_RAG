"""Cloud RAG Pipeline - OpenRouter."""
import time
import logging
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAX_CONTEXT_CHARS, FINAL_TOP_K, MIN_SCORE_RATIO,
    OPENROUTER_API_KEY, OPENROUTER_MODEL
)
from rag_chatbot.utils.conversation_history import ConversationHistory
from rag_chatbot.utils.types import RAGConfig
from .retrieval import search_documents

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technical assistant. Answer using the provided context. Cite sources with file name, page, section."""


class RAGPipeline:
    def __init__(self, config: RAGConfig, model: str = None):
        self.es = config.es
        self.index_name = config.index_name
        self.kg = config.knowledge_graph
        self.api_key = OPENROUTER_API_KEY
        self.api_url = 'https://openrouter.ai/api/v1/chat/completions'
        self.model = model or OPENROUTER_MODEL

    def query(self, question: str, conversation: Optional[ConversationHistory] = None) -> Dict:
        start = time.time()

        results = search_documents(
            self.es, self.index_name, question,
            knowledge_graph=self.kg, final_k=FINAL_TOP_K
        )

        if not results:
            return {"answer": "No relevant documents found.", "sources": [], "context": "", "query_time": time.time() - start, "tokens": {"in": 0, "out": 0}}

        context = self._build_context(results)
        history = (conversation.get_recent_context(4) + "\n\n") if conversation and conversation.messages else ""
        answer, tokens_in, tokens_out = self._generate(question, context, history)

        sources = [
            {"file": r.get("file_name"), "page": r.get("page_number"),
            "score": round(r.get("score", 0), 4), "content_type": r.get("content_type"),
            "section": r.get("section"), "preview": r.get("chunk_text", "")[:150] + "..."}
            for r in results
        ]

        if conversation:
            conversation.add_message("user", question)
            conversation.add_message("assistant", answer, sources=sources)

        top_score = max((r.get("score", 0) for r in results), default=0)
        confidence = "HIGH" if top_score > 0.5 else "LOW"

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "query_time": time.time() - start,
            "metadata": {"num_results": len(results), "confidence": confidence},
            "tokens": {"in": tokens_in, "out": tokens_out}
        }

    def _build_context(self, results: List[Dict]) -> str:
        if not results:
            return ""

        top_score = max(r.get("score", 0) for r in results)
        threshold = top_score * MIN_SCORE_RATIO
        filtered = [r for r in results if r.get("score", 0) >= threshold]

        seen = set()
        parts = []
        chars = 0

        for r in filtered:
            text = r.get("chunk_text", "").strip()
            if not text:
                continue

            sig = text[:100]
            if sig in seen:
                continue
            seen.add(sig)

            doc = r.get("file_name", "unknown")
            page = r.get("page_number", 0)
            section = r.get("section", "")
            section_str = f" | {section}" if section else ""
            header = f"[{doc} | Page {page}{section_str}]"
            entry = f"{header}\n{text}\n"

            if chars + len(entry) > MAX_CONTEXT_CHARS:
                break

            parts.append(entry)
            chars += len(entry)

        return "\n".join(parts)

    def _generate(self, question: str, context: str, history: str) -> tuple:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\n{history}Question: {question}"

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, json=data, headers=headers, timeout=120)
            if response.status_code != 200:
                return f"API error: {response.status_code} - {response.text[:300]}", 0, 0
            result = response.json()
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            return result['choices'][0]['message']['content'].strip(), prompt_tokens, completion_tokens
        except Exception as e:
            logger.error(f"API failed: {e}")
            return f"API error: {e}", 0, 0
