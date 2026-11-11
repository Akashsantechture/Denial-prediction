"""
ai_analyst/rag/claim_store.py
------------------------------
ChromaDB-backed claim summary store.

Stores natural-language summaries of evaluated claims so the AI
analyst can answer questions about historical claim patterns, payer
behaviour, and similar past cases.

One document per claim submission, keyed by "{region}_{claim_id}".
The document text is a structured natural-language chunk — no PHI,
no patient identifiers, no raw SHAP values.

Separate collections per region, same pattern as rule_store.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent

_STORE_PATHS = {
    "uae": Path(os.getenv("CHROMA_CLAIM_STORE_PATH_UAE",
                           str(_PROJECT_ROOT / "chroma_store_claims_uae"))),
    "ksa": Path(os.getenv("CHROMA_CLAIM_STORE_PATH_KSA",
                           str(_PROJECT_ROOT / "chroma_store_claims_ksa"))),
}

_COLLECTION_NAMES = {
    "uae": "uae_claims",
    "ksa": "ksa_claims",
}

_clients:     Dict[str, Any] = {}
_collections: Dict[str, Any] = {}


def _get_claim_collection(region: str = "uae"):
    global _clients, _collections

    region = region.lower()
    if region not in _STORE_PATHS:
        raise ValueError(f"Unsupported region: {region}")

    if region in _collections:
        return _collections[region]

    import chromadb
    from .embedder import EMBEDDING_DIM

    store_path      = _STORE_PATHS[region]
    collection_name = _COLLECTION_NAMES[region]

    store_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_path))
    _clients[region] = client

    # Drop collection if embedding dimension changed
    try:
        existing = client.get_collection(name=collection_name)
        sample   = existing.peek(limit=1)
        if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
            if len(sample["embeddings"][0]) != EMBEDDING_DIM:
                client.delete_collection(name=collection_name)
                existing = None
    except Exception:
        existing = None

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    _collections[region] = collection

    logger.info(
        "[ClaimStore] Collection '%s' ready for region '%s' at %s",
        collection_name, region.upper(), store_path,
    )
    return collection


def upsert_claim(
    claim_id:   str,
    chunk_text: str,
    embedding:  List[float],
    metadata:   Dict[str, Any],
    region:     str = "uae",
) -> None:
    """
    Upsert one claim summary into ChromaDB.
    Idempotent — same claim_id overwrites the previous entry.
    """
    col = _get_claim_collection(region)
    # Sanitise metadata — ChromaDB requires str/int/float/bool only
    safe_meta = {
        k: str(v) if v is not None else ""
        for k, v in metadata.items()
    }
    col.upsert(
        ids=[claim_id],
        embeddings=[embedding],
        documents=[chunk_text],
        metadatas=[safe_meta],
    )
    logger.info("[ClaimStore] Upserted claim '%s' (region: %s).", claim_id, region.upper())


def search_claims(
    query_embedding: List[float],
    region:          str = "uae",
    top_k:           int = 5,
    min_similarity:  float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Semantic search over past claim summaries.
    Lower threshold than rules (0.35 vs 0.40) because claim language
    is more varied than rule text.
    """
    col = _get_claim_collection(region)

    try:
        n = min(top_k, col.count())
        if n == 0:
            return []

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["metadatas", "documents", "distances"],
        )

        ids        = results["ids"][0]
        docs       = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        output = []
        for cid, doc, meta, dist in zip(ids, docs, metadatas, distances):
            similarity = round(1.0 - dist / 2.0, 4)
            if similarity >= min_similarity:
                output.append({
                    "claim_id":   cid,
                    "chunk_text": doc,
                    "similarity": similarity,
                    "risk_level": meta.get("risk_level", ""),
                    "payer":      meta.get("payer", ""),
                    "date":       meta.get("date", ""),
                    "region":     meta.get("region", region),
                })

        return output

    except Exception as exc:
        logger.warning("[ClaimStore] Search failed silently: %s", exc)
        return []


def count_claims(region: str = "uae") -> int:
    try:
        return _get_claim_collection(region).count()
    except Exception:
        return 0
