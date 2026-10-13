"""
ai_analyst/rag/rule_store.py
----------------------------
ChromaDB-backed Thynk rule store.

Replaces the earlier pgvector implementation.  ChromaDB runs as a
local persistent store — no PostgreSQL extension needed, no extra
infrastructure.

Responsibilities:
  - Initialise a persistent ChromaDB client + collection
  - Upsert rule chunks with pre-computed embedding vectors
  - Run cosine-similarity nearest-neighbour queries
  - Report rule count

Dependencies:
    pip install chromadb

The ChromaDB data directory defaults to:
    <project_root>/ai_analyst/rag/chroma_store/

Override via the CHROMA_STORE_PATH environment variable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Store path
# ------------------------------------------------------------------ #

_DEFAULT_STORE = Path(__file__).resolve().parent / "chroma_store"
_STORE_PATH    = Path(os.getenv("CHROMA_STORE_PATH", str(_DEFAULT_STORE)))
_COLLECTION    = "thynk_rules"

# ------------------------------------------------------------------ #
# Module-level client + collection (initialised lazily)
# ------------------------------------------------------------------ #

_client:     Optional[Any] = None
_collection: Optional[Any] = None


def _get_collection():
    """Return (and lazily create) the ChromaDB collection."""
    global _client, _collection

    if _collection is not None:
        return _collection

    import chromadb

    _STORE_PATH.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(_STORE_PATH))

    # Delete stale collection if it exists with wrong dimension
    try:
        existing = _client.get_collection(name=_COLLECTION)
        # Probe dimension by checking metadata
        sample = existing.peek(limit=1)
        if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
            dim = len(sample["embeddings"][0])
            from .embedder import EMBEDDING_DIM
            if dim != EMBEDDING_DIM:
                _client.delete_collection(name=_COLLECTION)
                existing = None
    except Exception:
        existing = None

    # cosine distance — matches what the embedder optimises for
    _collection = _client.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        "[RuleStore] ChromaDB collection '%s' ready at %s",
        _COLLECTION, _STORE_PATH,
    )
    return _collection


# ------------------------------------------------------------------ #
# Init (called at startup — idempotent)
# ------------------------------------------------------------------ #

def init_rule_store() -> None:
    """
    Ensure the ChromaDB store and collection exist.
    Call once at application startup.
    """
    global _client, _collection
    # Reset cache so _get_collection() reinitialises fresh
    _client = None
    _collection = None
    _get_collection()
    logger.info("[RuleStore] ChromaDB rule store initialised.")


# ------------------------------------------------------------------ #
# Upsert
# ------------------------------------------------------------------ #

def upsert_rules_batch(rules: List[Dict[str, Any]]) -> int:
    """
    Upsert a batch of rule dicts into ChromaDB.

    Each dict must have:
        rule_id, package_name, severity, object_type,
        title, short_message, long_message, chunk_text, embedding

    Returns the number of rules upserted.
    ChromaDB upsert is idempotent on the `rule_id` (used as the document id).
    """
    col = _get_collection()

    ids          = []
    embeddings   = []
    documents    = []
    metadatas    = []

    for rule in rules:
        ids.append(rule["rule_id"])
        embeddings.append(rule["embedding"])
        documents.append(rule["chunk_text"])
        metadatas.append({
            "package_name":  rule["package_name"],
            "severity":      rule["severity"],
            "object_type":   rule["object_type"],
            "title":         rule["title"],
            "short_message": rule["short_message"],
            "long_message":  rule["long_message"],
        })

    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    count = len(ids)
    logger.info("[RuleStore] Upserted %d rules into ChromaDB.", count)
    return count


# ------------------------------------------------------------------ #
# Similarity search
# ------------------------------------------------------------------ #

def similarity_search(
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return the top-k most similar rules to a query embedding.

    ChromaDB returns distances in [0, 2] for cosine space.
    We convert to similarity = 1 - (distance / 2) → [0, 1].

    Returns a list of dicts with rule metadata + similarity score.
    Returns an empty list if the collection has no documents or
    an error occurs.
    """
    col = _get_collection()

    try:
        n_results = min(top_k, col.count())
        if n_results == 0:
            return []

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        ids        = results["ids"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        output = []
        for rule_id, meta, dist in zip(ids, metadatas, distances):
            # cosine distance ∈ [0, 2] → similarity ∈ [0, 1]
            similarity = round(1.0 - dist / 2.0, 4)
            output.append({
                "rule_id":       rule_id,
                "package_name":  meta.get("package_name", ""),
                "severity":      meta.get("severity", ""),
                "object_type":   meta.get("object_type", ""),
                "title":         meta.get("title", ""),
                "short_message": meta.get("short_message", ""),
                "long_message":  meta.get("long_message", ""),
                "similarity":    similarity,
            })

        return output

    except Exception as exc:
        logger.warning("[RuleStore] Similarity search failed: %s", exc)
        return []


# ------------------------------------------------------------------ #
# Utility
# ------------------------------------------------------------------ #

def count_rules() -> int:
    """Return the number of rules currently stored in ChromaDB."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
