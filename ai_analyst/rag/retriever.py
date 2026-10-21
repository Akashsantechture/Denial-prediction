"""
ai_analyst/rag/retriever.py
---------------------------
High-level RAG retriever.

Takes a plain-text query, embeds it with Gemini, runs cosine
similarity search against ChromaDB, and returns a formatted
string block ready to be injected into the LLM prompt.

Usage:
    from ai_analyst.rag.retriever import Retriever

    retriever = Retriever()                    # init once at startup
    rules_block = retriever.query(
        "medical necessity diagnosis mismatch lipid panel",
        top_k=4,
    )
    # → formatted string for LLM prompt injection
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .embedder import Embedder
from .rule_store import similarity_search, count_rules

logger = logging.getLogger(__name__)

# Minimum similarity threshold — below this rules are too off-topic
_MIN_SIMILARITY = 0.40


class Retriever:
    """
    Embed a query and retrieve semantically similar Thynk rules.

    Region-aware: searches within the specified region's rule collection.
    Each retriever instance is bound to a specific region (UAE or KSA).
    """

    def __init__(self, region: str = "uae") -> None:
        """
        Initialize retriever for a specific region.
        
        Args:
            region: 'uae' or 'ksa' - determines which rule collection to search
        """
        self._embedder = Embedder()
        self._region = region.lower()
        
        total = count_rules(region=self._region)
        logger.info(
            "[Retriever] Ready for region '%s'. Rule store contains %d indexed rules.",
            self._region.upper(), total
        )

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        min_similarity: float = _MIN_SIMILARITY,
    ) -> Optional[str]:
        """
        Retrieve relevant Thynk rules for a query string.

        Args:
            query_text:     The user question or claim context to search against.
            top_k:          Maximum number of rules to retrieve.
            min_similarity: Filter out rules below this cosine similarity.

        Returns:
            A formatted string block for LLM prompt injection,
            or None if no relevant rules are found.
        """
        if not query_text or not query_text.strip():
            return None

        try:
            # Embed the query
            query_vec = self._embedder.embed(query_text)

            # Search ChromaDB (region-specific collection)
            results = similarity_search(query_vec, region=self._region, top_k=top_k)

            # Filter by minimum similarity
            relevant = [r for r in results if r["similarity"] >= min_similarity]

            if not relevant:
                logger.info(
                    "[Retriever] No rules above threshold %.2f for query: %s",
                    min_similarity, query_text[:80],
                )
                return None

            logger.info(
                "[Retriever] Retrieved %d relevant rules (top sim=%.3f).",
                len(relevant), relevant[0]["similarity"],
            )

            return self._format(relevant)

        except Exception as exc:
            logger.warning("[Retriever] Query failed silently: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Build query string from claim context + user question
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_query(user_message: str, claim_intelligence: dict) -> str:
        """
        Construct a rich query string from the user's message combined
        with key claim context (denied activity codes, diagnosis codes,
        fired Thynk severity) so the semantic search is claim-aware.
        """
        parts = [user_message.strip()]

        # Add denied activity codes
        activities = claim_intelligence.get("activity_predictions", [])
        denied = [
            a["activity_code"]
            for a in activities
            if a.get("predicted_denial")
        ]
        if denied:
            parts.append("Denied activity codes: " + ", ".join(denied))

        # Add primary diagnosis
        claim_info = claim_intelligence.get("claim", {})
        dx = claim_info.get("primary_diagnosis_code", "")
        if dx:
            parts.append(f"Primary diagnosis: {dx}")

        # Add Thynk fired severities as hints
        thynk = claim_intelligence.get("thynk_validation")
        if thynk and thynk.get("found"):
            summary = thynk.get("summary", {})
            if summary.get("critical", 0):
                parts.append("Thynk flagged CRITICAL compliance issues")
            if summary.get("severe", 0):
                parts.append("Thynk flagged SEVERE coding or medical necessity issues")
            # Add short messages of fired rules as semantic anchors
            for item in thynk.get("all", [])[:5]:
                msg = item.get("shortMessage", "")
                if msg:
                    parts.append(msg)

        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    # Format results for LLM prompt
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format(rules: List[dict]) -> str:
        lines = [
            "============================================================",
            "RETRIEVED THYNK RULEBOOK CONTEXT",
            "============================================================",
            "The following CHI rules are semantically relevant to the",
            "current question and claim context. Use them to provide",
            "detailed rule-level explanations and fix recommendations.",
            "Only cite rule IDs that appear below.",
            "",
        ]

        for i, rule in enumerate(rules, 1):
            lines += [
                f"--- Rule {i} (similarity {rule['similarity']:.2f}) ---",
                f"Rule ID      : {rule['rule_id']}",
                f"Package      : {rule['package_name']}",
                f"Severity     : {rule['severity']}",
                f"Applies to   : {rule['object_type']}",
                f"Issue        : {rule['short_message']}",
                f"Detail       : {rule['long_message']}",
                "",
            ]

        lines.append("============================================================")
        return "\n".join(lines)
