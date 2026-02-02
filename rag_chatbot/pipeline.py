"""RAG Pipeline with LangGraph state machine."""
import time
import logging
from typing import Dict

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from rag_chatbot.utils.optimized_retrieval import search_documents, rerank_results, fetch_cross_references
from rag_chatbot.utils.conversation_history import ConversationHistory
from rag_chatbot.utils.model_loading import get_reranker, get_llm, get_embeddings, get_colbert
from rag_chatbot.utils.table_context import build_context
from rag_chatbot.utils.types import DocSearchArgs, RAGConfig, RAGState
from config import FINAL_TOP_K

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a technical assistant. Answer using the provided context. Cite sources with file name, page, section."),
    ("human", "Context:\n{context}\n\n{history}Question: {question}")
])


def preload_models():
    logger.info("Preloading models...")
    get_embeddings()
    get_reranker()
    get_llm()
    get_colbert()
    logger.info("Models ready")


class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.es = config.es
        self.index_name = config.index_name
        self.reranker = get_reranker()
        self.colbert = get_colbert()
        self.embedder = get_embeddings()
        self.kg = config.knowledge_graph
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
        """
        Search node:
        1. Search for relevant documents, bm25 + kNN + graph
        2. Rerank results, CrossEncoder + ColBERT
        3. Fetch cross-referenced chunks from top results
        """
        config = DocSearchArgs(
            es=self.es,
            index_name=self.index_name,
            query=state["question"],
            embedder=self.embedder,
            reranker=self.reranker,
            colbert=self.colbert,
            knowledge_graph=self.kg,
        )
        results = search_documents(config)
        reranked_results = rerank_results(config, results)
        sorted_results = sorted(reranked_results, key=lambda x: x["final_score"], reverse=True)[:FINAL_TOP_K]
        xref_results = fetch_cross_references(config, sorted_results)   # Fetch cross-referenced chunks from top results
        return {"results": xref_results}

    def _build_context_node(self, state: RAGState) -> Dict:
        """
        Build context node:
        """
        if not state["results"]:
            return {"context": "", "sources": []}

        context = build_context(state["results"], all_results=state["results"])
        sources = [
            {"file": c.get('file_name'), "page": c.get('page_number'),
             "score": round(c.get('final_score', 0), 3), "content_type": c.get('content_type'),
             "section": c.get('section_path') or c.get('section')}
            for c in state["results"]
        ]
        return {"context": context, "sources": sources}

    def _generate_node(self, state: RAGState) -> Dict:
        if not state["context"]:
            return {"answer": "No relevant documents found.", "confidence": "NONE"}

        response = self.chain.invoke({"context": state["context"], "history": state["history"], "question": state["question"]})
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        top_score = max((c.get("final_score", 0) for c in state["results"]), default=0)
        confidence = "HIGH" if top_score > 1.0 else "LOW"

        return {"answer": answer, "confidence": confidence}

    def query(self, question: str, conversation_history: ConversationHistory) -> Dict:
        start = time.time()

        initial_state: RAGState = {
            "question": question,
            "history": (conversation_history.get_recent_context(4) + "\n\n") if conversation_history.messages else "",
            "results": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": "",
            "start_time": start
        }

        final_state = self.graph.invoke(initial_state)

        conversation_history.add_message("user", question)
        conversation_history.add_message("assistant", final_state["answer"], sources=final_state["sources"])

        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "context": final_state["context"],
            "query_time": time.time() - start,
            "metadata": {"num_results": len(final_state["results"]), "confidence": final_state["confidence"]}
        }
