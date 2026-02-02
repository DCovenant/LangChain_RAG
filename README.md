# MULTI RAG CHATBOT

This project has the following workflow.

text extraction -> text chunking, embedding and ElasticSearch indexing

User query -> LLM -> bm25, semantic search, knowledge graph -> reranker -> answer

In order to have the code be more manegable langchain/langgraph/langsmith was used.

# Code
The first method the backend calls is RAGPipeline.query from pipeline.py

method:query -> initial_state of the state machine of RAGstate LangGraph -> final_state populated by _build_graph() which returns compiled version of the graph -> logs the query and the answer in conversation_history -> returns dict with answer, sources, context, sources