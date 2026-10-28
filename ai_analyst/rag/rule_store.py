"""
ai_analyst/rag/rule_store.py
----------------------------
ChromaDB-backed Thynk rule store.

Region-aware implementation that maintains separate ChromaDB collections
for each healthcare region (UAE/CHI and KSA/CCHI).

Replaces the earlier pgvector implementation.  ChromaDB runs as a
local persistent store — no PostgreSQL extension needed, no extra
infrastructure.

Responsibilities:
  - Initialise region-specific persistent ChromaDB clients + collections
  - Upsert rule chunks with pre-computed embedding vectors
  - Run cosine-similarity nearest-neighbour queries
  - Report rule count per region

Dependencies:
    pip install chromadb

The ChromaDB data directories default to:
    <project_root>/ai_analyst/rag/chroma_store_uae/
    <project_root>/ai_analyst/rag/chroma_store_ksa/

Override via environment variables:
    CHROMA_STORE_PATH_UAE
    CHROMA_STORE_PATH_KSA
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Store paths per region
# ------------------------------------------------------------------ #

_PROJECT_ROOT = Path(__file__).resolve().parent

_STORE_PATHS = {
    "uae": Path(os.getenv("CHROMA_STORE_PATH_UAE", str(_PROJECT_ROOT / "chroma_store_uae"))),
    "ksa": Path(os.getenv("CHROMA_STORE_PATH_KSA", str(_PROJECT_ROOT / "chroma_store_ksa"))),
}

_COLLECTION_NAMES = {
    "uae": "uae_rules",
    "ksa": "chi_rules",
}

# ------------------------------------------------------------------ #
# Module-level client + collection cache (per region)
# ------------------------------------------------------------------ #

_clients: Dict[str, Any] = {}
_collections: Dict[str, Any] = {}


def _get_collection(region: str = "uae"):
    """
    Return (and lazily create) the ChromaDB collection for a region.
    
    Args:
        region: 'uae' or 'ksa'
    """
    global _clients, _collections
    
    region = region.lower()
    if region not in _STORE_PATHS:
        raise ValueError(f"Unsupported region: {region}. Must be 'uae' or 'ksa'.")

    if region in _collections:
        return _collections[region]

    import chromadb

    store_path = _STORE_PATHS[region]
    collection_name = _COLLECTION_NAMES[region]
    
    store_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_path))
    _clients[region] = client

    # Delete stale collection if it exists with wrong dimension
    try:
        existing = client.get_collection(name=collection_name)
        # Probe dimension by checking metadata
        sample = existing.peek(limit=1)
        if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
            dim = len(sample["embeddings"][0])
            from .embedder import EMBEDDING_DIM
            if dim != EMBEDDING_DIM:
                client.delete_collection(name=collection_name)
                existing = None
    except Exception:
        existing = None

    # cosine distance — matches what the embedder optimises for
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    _collections[region] = collection

    logger.info(
        "[RuleStore] ChromaDB collection '%s' ready for region '%s' at %s",
        collection_name, region.upper(), store_path,
    )
    return collection


# ------------------------------------------------------------------ #
# Init (called at startup — idempotent)
# ------------------------------------------------------------------ #

def init_rule_store(region: str = "uae") -> None:
    """
    Ensure the ChromaDB store and collection exist for a region.
    Call once at application startup per region.
    
    Args:
        region: 'uae' or 'ksa'
    """
    global _clients, _collections
    region = region.lower()
    # Reset cache for this region so _get_collection() reinitialises fresh
    _clients.pop(region, None)
    _collections.pop(region, None)
    _get_collection(region)
    logger.info("[RuleStore] ChromaDB rule store initialised for region '%s'.", region.upper())


# ------------------------------------------------------------------ #
# Upsert
# ------------------------------------------------------------------ #

def upsert_rules_batch(rules: List[Dict[str, Any]], region: str = "uae") -> int:
    """
    Upsert a batch of rule dicts into ChromaDB for a specific region.

    Each dict must have:
        rule_id, package_name, severity, object_type,
        title, short_message, long_message, chunk_text, embedding
    
    Args:
        rules: List of rule dictionaries
        region: 'uae' or 'ksa'

    Returns the number of rules upserted.
    ChromaDB upsert is idempotent on the `rule_id` (used as the document id).
    """
    col = _get_collection(region)

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
    logger.info("[RuleStore] Upserted %d rules into ChromaDB for region '%s'.", count, region.upper())
    return count


# ------------------------------------------------------------------ #
# Similarity search
# ------------------------------------------------------------------ #

def similarity_search(
    query_embedding: List[float],
    region: str = "uae",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return the top-k most similar rules to a query embedding for a region.

    ChromaDB returns distances in [0, 2] for cosine space.
    We convert to similarity = 1 - (distance / 2) → [0, 1].
    
    Args:
        query_embedding: Query vector
        region: 'uae' or 'ksa'
        top_k: Maximum number of results

    Returns a list of dicts with rule metadata + similarity score.
    Returns an empty list if the collection has no documents or
    an error occurs.
    """
    col = _get_collection(region)

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

def count_rules(region: str = "uae") -> int:
    """
    Return the number of rules currently stored in ChromaDB for a region.
    
    Args:
        region: 'uae' or 'ksa'
    """
    try:
        return _get_collection(region).count()
    except Exception:
        return 0
