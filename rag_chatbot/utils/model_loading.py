"""
Model loading - LangChain HuggingFace for LLM + Embeddings, GPU only.
"""
import os
import logging
from typing import List
from functools import lru_cache

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")

_reranker_cache = {}


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


def get_reranker(model_name: str) -> CrossEncoder:
    """Get cached CrossEncoder reranker."""
    if model_name not in _reranker_cache:
        logger.info(f"Loading reranker: {model_name}")
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


def embed_query(text: str) -> List[float]:
    """Embed single query text."""
    return get_embeddings().embed_query(text)
