# ai_analyst/rag/__init__.py
# RAG (Retrieval-Augmented Generation) package for Thynk rule lookup.
#
# Components:
#   embedder   — Gemini text-embedding-004 wrapper
#   rule_store — pgvector table operations (upsert + similarity search)
#   retriever  — high-level query(text, top_k) → formatted rule chunks
#   ingest     — one-time script: parse Thynk JSON → embed → store
