"""
Model loading - LangChain HuggingFace for LLM + Embeddings, GPU only.
"""
#TODO:
"""No, it's not ColBERT. Current approach is a bi-encoder (single vector per text).

ColBERT	Current (BGE bi-encoder)
Matching	Token-to-token (late interaction)	Single vector cosine
Strength	Exact term matching + semantic	Semantic only
Your error case	Would match "ceramic" token directly	May miss if overall embedding differs
For your "ceramic insulation" → "ceramic insulator" problem:

ColBERT: ✅ Would likely match (token "ceramic" matches exactly)
BGE: ⚠️ Relies on query expansion working
Is it still useful? Yes - adds semantic diversity to CrossEncoder. But for definition lookups, the query expansion + section boost will help more than the second-stage reranker.

Want true ColBERT? Need colbert-ai with custom index (more complex setup, ~2GB VRAM extra).  """

import logging
from typing import List
from functools import lru_cache
import torch

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import LLM_MODEL, EMBED_MODEL, COLBERT_MODEL, RERANK_MODEL

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Get cached HuggingFace embeddings on GPU."""
    logger.info(f"Loading embeddings: {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatHuggingFace:
    """Get cached ChatHuggingFace with GPU."""
    logger.info(f"Loading LLM: {LLM_MODEL}")
    pipe = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL,
        task="text-generation",
        device=0,
        pipeline_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.1,
            "do_sample": True,
            "return_full_text": False,
        },
        model_kwargs={
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
    )
    return ChatHuggingFace(llm=pipe)


@lru_cache(maxsize=1)
def get_reranker():
    """Get cached reranker on GPU."""
    logger.info(f"Loading reranker: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def get_colbert():
    """Get cached colbert on GPU."""
    logger.info(f"Loading embeddings: {COLBERT_MODEL}")
    return ColBERTReranker(COLBERT_MODEL)


class ColBERTReranker:
    """Simple ColBERT-style reranker using token-level maxsim."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.to("cuda")

    def rerank(self, query: str, documents: List[str], k: int = 10) -> List[dict]:
        """Rerank documents using late interaction scoring."""
        q_emb = self.model.encode(query, convert_to_tensor=True)
        d_embs = self.model.encode(documents, convert_to_tensor=True)

        # Cosine similarity (simplified maxsim for single vector models)
        scores = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), d_embs)
        scores = scores.cpu().tolist()

        results = [{"content": doc, "score": score} for doc, score in zip(documents, scores)]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


