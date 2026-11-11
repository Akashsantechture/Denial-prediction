"""
ai_analyst/rag/orchestrator.py
-------------------------------
Routes analyst queries to the rule store, the claim store, or both,
then merges the results into a single context block for the LLM.

Intent classification is keyword-based (no LLM call) — fast and
deterministic. When intent is ambiguous, both stores are searched.

Usage:
    from ai_analyst.rag.orchestrator import RAGOrchestrator

    orch = RAGOrchestrator(region="uae")
    context = orch.query(
        user_message="Have we seen similar iron panel denials before?",
        claim_intelligence=claim_intelligence,
    )
    # context → merged string block injected into LLM prompt
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .retriever      import Retriever
from .claim_retriever import ClaimRetriever

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Intent keywords
# ------------------------------------------------------------------ #

_CLAIM_KEYWORDS = {
    "claim", "previous", "past", "history", "last time", "before",
    "earlier", "processed", "submitted", "pattern", "trend",
    "how many", "which claims", "similar", "seen before", "common",
    "usually", "typically", "often", "benchmark", "compare",
    "other claims", "last week", "last month", "frequent",
}

_RULE_KEYWORDS = {
    "rule", "regulation", "cchi", "chi", "nphies", "thynk",
    "policy", "guideline", "requirement", "medical necessity",
    "authorization", "prior auth", "eligibility", "crosswalk",
    "bundling", "coding", "mandatory", "compliance", "approval",
    "documentation", "timely filing", "coverage", "duplicate",
}


class RAGOrchestrator:
    """
    Single entry point for all RAG retrieval.
    Replaces direct Retriever calls in analyst_service.py.
    """

    def __init__(self, region: str = "uae") -> None:
        self._region          = region.lower()
        self._rule_retriever  = Retriever(region=self._region)
        self._claim_retriever = ClaimRetriever(region=self._region)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def query(
        self,
        user_message:       str,
        claim_intelligence: Dict[str, Any],
        top_k_rules:        int = 4,
        top_k_claims:       int = 4,
    ) -> Optional[str]:
        """
        Route the query and return a merged context block.
        Returns None when neither store returns relevant results.
        """
        intent = self._classify_intent(user_message)
        logger.info("[Orchestrator] Intent: '%s' for query: %s", intent, user_message[:80])

        rule_context  = None
        claim_context = None

        # ── Rules retrieval ────────────────────────────────────
        if intent in ("rules", "both"):
            query_text = self._rule_retriever.build_query(
                user_message=user_message,
                claim_intelligence=claim_intelligence,
            )
            rule_context = self._rule_retriever.query(query_text, top_k=top_k_rules)
            if rule_context:
                logger.info("[Orchestrator] Rule context retrieved.")

        # ── Claims retrieval ───────────────────────────────────
        if intent in ("claims", "both"):
            # Build a richer query for claim search
            claims_query = self._build_claims_query(user_message, claim_intelligence)
            claim_context = self._claim_retriever.search(claims_query, top_k=top_k_claims)
            if claim_context:
                logger.info("[Orchestrator] Claim context retrieved.")

        return self._merge(rule_context, claim_context)

    # ------------------------------------------------------------------ #
    # Build query for initial summary (used by analyst_service)
    # ------------------------------------------------------------------ #

    def build_summary_query(self, claim_intelligence: Dict[str, Any]) -> str:
        """
        Build the query string used for RAG retrieval during initial
        summary generation. Searches both stores.
        """
        return self._rule_retriever.build_query(
            user_message="Claim overview medical necessity payer rules denial",
            claim_intelligence=claim_intelligence,
        )

    def query_for_summary(
        self,
        claim_intelligence: Dict[str, Any],
        top_k_rules:  int = 8,
        top_k_claims: int = 3,
    ) -> Optional[str]:
        """
        Retrieve context for the initial auto-summary.
        Always searches both stores.
        """
        query_text = self.build_summary_query(claim_intelligence)

        rule_context  = self._rule_retriever.query(query_text, top_k=top_k_rules)

        claims_query  = self._build_claims_query(
            "Similar claim pattern denial overview", claim_intelligence
        )
        claim_context = self._claim_retriever.search(claims_query, top_k=top_k_claims)

        return self._merge(rule_context, claim_context)

    # ------------------------------------------------------------------ #
    # Intent classifier
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_intent(message: str) -> str:
        """
        Keyword-based intent classification — no LLM call.
        Returns 'rules', 'claims', or 'both'.
        """
        msg        = message.lower()
        has_claim  = any(k in msg for k in _CLAIM_KEYWORDS)
        has_rule   = any(k in msg for k in _RULE_KEYWORDS)

        if has_claim and has_rule:  return "both"
        if has_claim:               return "claims"
        if has_rule:                return "rules"
        return "both"  # default: search both when ambiguous

    # ------------------------------------------------------------------ #
    # Build claims search query
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_claims_query(
        user_message: str,
        claim_intelligence: Dict[str, Any],
    ) -> str:
        """
        Enrich the user message with claim context for better claim search.
        """
        parts = [user_message.strip()]

        pred  = claim_intelligence.get("prediction", {})
        claim = claim_intelligence.get("claim", {})

        risk = pred.get("claim_risk_level", "")
        if risk:
            parts.append(f"{risk} risk claim")

        dx = claim.get("primary_diagnosis_code", "")
        if dx:
            parts.append(f"diagnosis {dx}")

        denied = [
            a.get("activity_code", "")
            for a in pred.get("activity_predictions", [])
            if a.get("predicted_denial")
        ]
        if denied:
            parts.append(f"denied activities {' '.join(denied)}")

        payer = claim.get("payer_name") or claim.get("payer_classification", "")
        if payer:
            parts.append(f"payer {payer}")

        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    # Merge rule and claim contexts
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge(
        rule_context:  Optional[str],
        claim_context: Optional[str],
    ) -> Optional[str]:
        parts = []
        if rule_context:
            parts.append(rule_context)
        if claim_context:
            parts.append(claim_context)
        return "\n\n".join(parts) if parts else None
