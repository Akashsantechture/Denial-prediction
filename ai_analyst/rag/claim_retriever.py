"""
ai_analyst/rag/claim_retriever.py
----------------------------------
Builds claim summary chunks and searches past claims semantically.

The claim chunk is plain natural-language text optimised for
semantic search — no PHI, no patient names, no raw SHAP numbers.
Feature names and denial reasons are included; numeric scores are not.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .embedder import Embedder
from .claim_store import search_claims, count_claims

logger = logging.getLogger(__name__)


class ClaimRetriever:
    """
    Embeds claim intelligence into a searchable chunk and stores it.
    Also searches past claims by semantic similarity.
    """

    def __init__(self, region: str = "uae") -> None:
        self._embedder = Embedder()
        self._region   = region.lower()
        total = count_claims(region=self._region)
        logger.info(
            "[ClaimRetriever] Ready for region '%s'. %d claims indexed.",
            self._region.upper(), total,
        )

    # ------------------------------------------------------------------ #
    # Build chunk from claim intelligence
    # ------------------------------------------------------------------ #

    def build_claim_chunk(self, claim_intelligence: Dict[str, Any]) -> str:
        """
        Convert claim intelligence to a natural-language chunk.
        Called once per session creation.

        Deliberately excludes:
          - Patient name / DOB (PHI)
          - Raw SHAP numeric values (model-version-specific)
          - Claim gross / net amounts (sensitive financial data)

        Includes:
          - Encounter demographics (age, gender, encounter type)
          - Clinical context (diagnosis category, CPT category)
          - Payer classification
          - Prediction outcome (risk level, denial reasons)
          - Feature names driving the denial (not values)
          - Thynk rule IDs that fired
        """
        pred  = claim_intelligence.get("prediction", {})
        claim = claim_intelligence.get("claim", {})
        thynk = claim_intelligence.get("thynk_validation") or {}
        shap  = claim_intelligence.get("top_shap_contributions", [])

        # Denied activity codes + reasons
        denied_activities = []
        for a in pred.get("activity_predictions", []):
            if a.get("predicted_denial"):
                code   = a.get("activity_code", "?")
                reason = a.get("predicted_denial_reason") or "Unknown"
                cat    = a.get("cpt_category", "")
                denied_activities.append(
                    f"{code} ({cat}) — {reason}" if cat else f"{code} — {reason}"
                )

        # Top SHAP feature names (no values)
        top_drivers = [
            d.get("feature", "unknown")
            for d in shap[:5]
            if (d.get("shap_value") or d.get("value") or 0) > 0
        ]

        # Thynk rules that fired
        thynk_rules = []
        if thynk.get("found"):
            for item in thynk.get("all", [])[:5]:
                rule_id  = item.get("ruleId", "")
                severity = item.get("severity", "")
                msg      = item.get("shortMessage", "")
                if rule_id:
                    thynk_rules.append(f"{rule_id} [{severity}] — {msg}")

        # Secondary diagnosis categories present
        sec_cats = []
        for activity in claim.get("activities", []):
            for dx in (activity.get("diagnoses") or []):
                if dx.get("diagnosis_type", "").lower() not in ("principal", "primary"):
                    code = dx.get("diagnosis_code", "")
                    if code:
                        from .ingest import _build_chunk  # reuse ICD category logic
                        pass  # just collect the code
            # We use the secondary_dx_* flags from prediction features instead

        # Risk level
        risk_level    = pred.get("claim_risk_level", "UNKNOWN")
        denial_pct    = pred.get("claim_denial_probability_pct", 0)
        activity_count = pred.get("activity_count", 0)
        denied_count   = len(denied_activities)

        # Payer — prefer payer_name, fall back to payer_classification
        payer = (
            claim.get("payer_name")
            or claim.get("payer_classification")
            or claim.get("policy_name")
            or "Unknown Payer"
        )
        # Remove null strings
        if not payer or payer.lower() in ("none", "null", ""):
            payer = "Unknown Payer"

        chunk = f"""Claim evaluated on {datetime.now().strftime('%Y-%m-%d')}
Region: {self._region.upper()}
Encounter: {claim.get('patient_age', '?')}yr {claim.get('gender', '?')}, {claim.get('encounter_type', '?')} encounter
Facility: {claim.get('facility_type', claim.get('facility', 'Unknown'))}
Primary Diagnosis: {claim.get('primary_diagnosis_code', '?')} — Category: {claim.get('icd_category', '?')}
Payer: {payer}
Risk Level: {risk_level} — Overall denial probability {denial_pct:.1f}%
Activities: {activity_count} submitted, {denied_count} predicted denied
Denied: {'; '.join(denied_activities) if denied_activities else 'None predicted'}
Top denial drivers: {', '.join(top_drivers) if top_drivers else 'None identified'}
Thynk rules fired: {'; '.join(thynk_rules) if thynk_rules else 'None'}"""

        return chunk.strip()

    # ------------------------------------------------------------------ #
    # Store claim in ChromaDB
    # ------------------------------------------------------------------ #

    def index_claim(
        self,
        claim_id:         str,
        claim_intelligence: Dict[str, Any],
    ) -> None:
        """
        Build chunk, embed it, and store in the claims vector store.
        Called from analyst_service.create_session() — non-blocking.
        """
        from .claim_store import upsert_claim

        chunk  = self.build_claim_chunk(claim_intelligence)
        vector = self._embedder.embed(chunk)

        pred  = claim_intelligence.get("prediction", {})
        claim = claim_intelligence.get("claim", {})
        payer = (
            claim.get("payer_name")
            or claim.get("payer_classification")
            or claim.get("policy_name")
            or ""
        )

        upsert_claim(
            claim_id   = f"{self._region}_{claim_id}",
            chunk_text = chunk,
            embedding  = vector,
            metadata   = {
                "region":     self._region,
                "risk_level": pred.get("claim_risk_level", ""),
                "denial_pct": str(round(pred.get("claim_denial_probability_pct", 0), 1)),
                "payer":      str(payer)[:200],
                "encounter":  claim.get("encounter_type", ""),
                "date":       datetime.now().isoformat(),
            },
            region=self._region,
        )

    # ------------------------------------------------------------------ #
    # Search past claims
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_text: str,
        top_k:      int = 4,
    ) -> Optional[str]:
        """
        Search past claims by semantic similarity.
        Returns a formatted string block for LLM prompt injection, or None.
        """
        if not query_text or not query_text.strip():
            return None

        try:
            query_vec = self._embedder.embed(query_text)
            results   = search_claims(query_vec, region=self._region, top_k=top_k)

            if not results:
                return None

            logger.info(
                "[ClaimRetriever] Found %d similar past claims (top sim=%.3f).",
                len(results), results[0]["similarity"],
            )
            return self._format(results)

        except Exception as exc:
            logger.warning("[ClaimRetriever] Search failed silently: %s", exc)
            return None

    @staticmethod
    def _format(claims: List[Dict[str, Any]]) -> str:
        lines = [
            "============================================================",
            "SIMILAR PAST CLAIMS",
            "============================================================",
            "These are semantically similar claims previously processed.",
            "Use them to identify patterns, benchmark risk, and explain",
            "whether this type of claim is commonly denied.",
            "Do NOT cite specific patient details — reference patterns only.",
            "",
        ]
        for i, c in enumerate(claims, 1):
            lines += [
                f"--- Past Claim {i} (similarity {c['similarity']:.2f}) ---",
                c["chunk_text"],
                "",
            ]
        lines.append("============================================================")
        return "\n".join(lines)
