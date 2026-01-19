#!/usr/bin/env python3
"""CLI for RAG Pipeline."""
import os
import sys
import argparse
import logging
from elasticsearch import Elasticsearch

from utils.rag_pipeline import RAGPipeline, preload_models
from utils.conversation_history import ConversationHistory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ES_URL = os.getenv("ES_URL", "http://localhost:9200")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path")
    parser.add_argument("query")
    args = parser.parse_args()

    index_name = os.path.basename(os.path.normpath(args.folder_path)).lower().replace(" ", "_").replace("-", "_")
    if index_name[0] in "_-+":
        index_name = "idx_" + index_name

    es = Elasticsearch(ES_URL, request_timeout=60)
    if not es.ping():
        logger.error("ES connection failed")
        sys.exit(1)
    if not es.indices.exists(index=index_name):
        logger.error(f"Index '{index_name}' not found")
        sys.exit(1)

    preload_models()
    pipeline = RAGPipeline(es, index_name)
    result = pipeline.query(args.query, ConversationHistory())

    print(f"\n{'='*60}\nANSWER:\n{'='*60}\n{result['answer']}")
    print(f"\n{'='*60}\nSOURCES ({len(result['sources'])}):\n{'='*60}")
    for i, s in enumerate(result['sources'], 1):
        section = f" | {s['section']}" if s.get('section') else ""
        print(f"[{i}] {s['file']} - Page {s['page']} | Score: {s['score']:.3f}{section}")
    print(f"\n{result['query_time']:.2f}s | {result['metadata']['confidence']} confidence")


if __name__ == "__main__":
    main()