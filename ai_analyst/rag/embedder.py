"""
ai_analyst/rag/embedder.py
--------------------------
Gemini text-embedding-004 wrapper.

Produces 768-dimensional dense vectors for any text input.
Used by both ingest (rule indexing) and retriever (query embedding).

Usage:
    from ai_analyst.rag.embedder import Embedder
    emb = Embedder()
    vector = emb.embed("Medical necessity check failed for R51 + 80061")
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

# Load .env from ai_analyst/
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_BASE_DIR / ".env", override=False)

# Gemini embedding model — 3072 dimensions
_EMBEDDING_MODEL = "models/gemini-embedding-001"

# Dimension produced by gemini-embedding-001
EMBEDDING_DIM = 3072


class Embedder:
    """
    Thin wrapper around the Gemini Embeddings API.

    Re-uses a single genai.Client instance per Embedder object.
    Instantiate once at startup and share across the application.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API is not set. Cannot initialise Embedder."
            )
        self._client = genai.Client(api_key=api_key)
        logger.info(
            "[Embedder] Initialised with model %s (%d-dim)",
            _EMBEDDING_MODEL, EMBEDDING_DIM,
        )

    # ------------------------------------------------------------------ #
    # Single text
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> List[float]:
        """
        Embed a single string.

        Returns:
            List[float] of length EMBEDDING_DIM (768).
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty string.")

        result = self._client.models.embed_content(
            model=_EMBEDDING_MODEL,
            contents=text.strip(),
        )

        # google-genai returns EmbedContentResponse with .embeddings list
        return result.embeddings[0].values

    # ------------------------------------------------------------------ #
    # Batch texts (sequential — Gemini free-tier has no batch endpoint)
    # ------------------------------------------------------------------ #

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings one by one.

        Returns:
            List of embedding vectors, same order as input.
        """
        vectors = []
        for i, text in enumerate(texts):
            try:
                vectors.append(self.embed(text))
                if (i + 1) % 10 == 0:
                    logger.info("[Embedder] Embedded %d / %d", i + 1, len(texts))
            except Exception as exc:
                logger.warning(
                    "[Embedder] Failed to embed item %d: %s — using zero vector.",
                    i, exc,
                )
                vectors.append([0.0] * EMBEDDING_DIM)
        return vectors
