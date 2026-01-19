"""RAG Pipeline with LangGraph state machine."""
import time
import logging
from typing import List, Dict, Optional, TypedDict
from elasticsearch import Elasticsearch

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from .optimized_retrieval import search_documents
from .conversation_history import ConversationHistory
from .model_loading import get_reranker, get_llm, embed_query, get_embeddings
from .table_context import build_context
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

FINAL_TOP_K = 10
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a technical assistant. Answer using the provided context. Cite sources with file name, page, section."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])


class RAGState(TypedDict):
    question: str
    results: List[Dict]
    context: str
    answer: str
    sources: List[Dict]
    confidence: str
    start_time: float


def preload_models():
    logger.info("Preloading models...")
    get_embeddings()
    get_reranker(RERANK_MODEL)
    get_llm()
    logger.info("Models ready")


class RAGPipeline:
    def __init__(self, es: Elasticsearch, index_name: str,
                 knowledge_graph: Optional[KnowledgeGraph] = None):
        self.es = es
        self.index_name = index_name
        self.reranker = get_reranker(RERANK_MODEL)
        self.kg = knowledge_graph
        self.chain = PROMPT | get_llm()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(RAGState)

        graph.add_node("search", self._search_node, run_name="Search")
        graph.add_node("build_context", self._build_context_node, run_name="Build Context")
        graph.add_node("generate", self._generate_node, run_name="Generate")

        graph.add_edge(START, "search")
        graph.add_edge("search", "build_context")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def _search_node(self, state: RAGState) -> Dict:
        results = search_documents(
            self.es, self.index_name, state["question"],
            embed_query, self.reranker, final_k=FINAL_TOP_K,
            knowledge_graph=self.kg
        )
        return {"results": results}

    def _build_context_node(self, state: RAGState) -> Dict:
        if not state["results"]:
            return {"context": "", "sources": []}

        context = build_context(state["results"], all_results=state["results"])
        sources = [
            {"file": c.get('file_name'), "page": c.get('page_number'),
             "score": round(c.get('final_score', 0), 3), "content_type": c.get('content_type'),
             "section": c.get('section'), "preview": c.get('chunk_text', '')[:150] + "..."}
            for c in state["results"]
        ]
        return {"context": context, "sources": sources}

    def _generate_node(self, state: RAGState) -> Dict:
        if not state["context"]:
            return {"answer": "No relevant documents found.", "confidence": "NONE"}

        response = self.chain.invoke({"context": state["context"], "question": state["question"]})
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        top_score = max((c.get("final_score", 0) for c in state["results"]), default=0)
        confidence = "HIGH" if top_score > 1.0 else "LOW"

        return {"answer": answer, "confidence": confidence}

    def query(self, question: str, conversation: ConversationHistory) -> Dict:
        start = time.time()

        initial_state: RAGState = {
            "question": question,
            "results": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": "",
            "start_time": start
        }

        final_state = self.graph.invoke(initial_state)

        conversation.add_message("user", question)
        conversation.add_message("assistant", final_state["answer"], sources=final_state["sources"])

        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "query_time": time.time() - start,
            "metadata": {"num_results": len(final_state["results"]), "confidence": final_state["confidence"]}
        }
