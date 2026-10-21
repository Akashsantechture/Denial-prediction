"""
ai_analyst/rag/ingest.py
------------------------
One-time (or on-demand) ingestion script.

Reads from TWO sources and combines them into the ChromaDB store:

  1. Thynk/thynk-claimlookup.json  — rules extracted from real claim
     responses (deduplicated by ruleId across all claims)

  2. Thynk/thynk-rag-rules.json    — standalone RAG knowledge base,
     rules added directly without needing a claim entry. This is the
     file to edit when you want to add new CHI rules to the LLM's
     knowledge base.

Both sources use the same ruleId deduplication — a rule in the RAG
file takes precedence over the same ruleId in the claim lookup.

Run via:
    python -m ai_analyst.rag.ingest
    # or via the API endpoint POST /analyst/ingest-rules

Safe to re-run — ChromaDB upsert is idempotent on rule_id.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Project root — two levels up from this file
_PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
_LOOKUP_PATH     = _PROJECT_ROOT / "Thynk" / "thynk-claimlookup.json"
_RAG_RULES_PATH  = _PROJECT_ROOT / "Thynk" / "thynk-rag-rules.json"


# ------------------------------------------------------------------ #
# Rule extraction + deduplication
# ------------------------------------------------------------------ #

def _extract_rules(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract unique rules from claim-lookup entries.
    Deduplicated by ruleId.
    """
    seen: Dict[str, Dict[str, Any]] = {}

    for claim_entry in data:
        for item in claim_entry.get("thynk_response", []):
            rule_id = item.get("ruleId", "").strip()
            if not rule_id or rule_id in seen:
                continue

            seen[rule_id] = {
                "rule_id":       rule_id,
                "package_name":  item.get("packageName", "unknown"),
                "severity":      item.get("severity", "INFO").upper(),
                "object_type":   item.get("objectType", "claim"),
                "short_message": item.get("shortMessage", ""),
                "long_message":  item.get("longMessage", ""),
                "code":          item.get("code") or "",
                "activity_type": item.get("activityType") or "",
            }

    return list(seen.values())


def _load_rag_rules(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Load rules directly from thynk-rag-rules.json.

    These are flat rule objects (not wrapped in claim entries).
    ruleId is still the unique key — RAG file rules take precedence
    over claim-lookup rules with the same ruleId.
    """
    rules = []
    for item in data:
        rule_id = item.get("ruleId", "").strip()
        if not rule_id:
            continue
        rules.append({
            "rule_id":       rule_id,
            "package_name":  item.get("packageName", "unknown"),
            "severity":      item.get("severity", "INFO").upper(),
            "object_type":   item.get("objectType", "claim"),
            "short_message": item.get("shortMessage", ""),
            "long_message":  item.get("longMessage", ""),
            "code":          item.get("code") or "",
            "activity_type": item.get("activityType") or "",
        })
    return rules


# ------------------------------------------------------------------ #
# Chunk builder — what gets embedded & searched
# ------------------------------------------------------------------ #

def _build_chunk(rule: Dict[str, Any]) -> str:
    """
    Build a rich text chunk for embedding.

    Combines structured metadata with natural language so the
    semantic search matches both technical queries ("R51 80061")
    and analyst language ("medical necessity mismatch lipid panel").
    """
    parts = [
        f"Rule ID: {rule['rule_id']}",
        f"Package: {rule['package_name']}",
        f"Severity: {rule['severity']}",
        f"Applies to: {rule['object_type']}",
    ]

    if rule["code"]:
        parts.append(f"Code: {rule['code']}")

    if rule["activity_type"]:
        parts.append(f"Activity type: {rule['activity_type']}")

    parts += [
        f"Issue: {rule['short_message']}",
        f"Detail: {rule['long_message']}",
    ]

    return "\n".join(parts)


# ------------------------------------------------------------------ #
# Main ingest function
# ------------------------------------------------------------------ #

def run_ingest(region: str = "uae") -> Dict[str, Any]:
    """
    Execute full ingest pipeline for a specific region.
    
    Args:
        region: 'uae' or 'ksa'

    Returns a summary dict with counts and status.
    """
    region = region.lower()
    if region not in ("uae", "ksa"):
        raise ValueError(f"Invalid region: {region}. Must be 'uae' or 'ksa'.")
    
    # Late import so this module can be loaded without DB connection
    from .embedder import Embedder
    from .rule_store import upsert_rules_batch, count_rules

    # Update paths based on region
    _LOOKUP_PATH     = _PROJECT_ROOT / "Thynk" / region / ("thynk-claimlookup.json" if region == "uae" else "nphies-claimlookup.json")
    _RAG_RULES_PATH  = _PROJECT_ROOT / "Thynk" / region / ("thynk-rag-rules.json" if region == "uae" else "nphies-rag-rules.json")

    # ── Load claim lookup JSON ─────────────────────────────────
    claim_rules: List[Dict[str, Any]] = []
    if _LOOKUP_PATH.exists():
        with open(_LOOKUP_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            claim_rules = _extract_rules(raw)
            logger.info(
                "[Ingest] Extracted %d unique rules from claim lookup.", len(claim_rules)
            )
        else:
            logger.warning("[Ingest] Unexpected format in claim lookup — skipping.")
    else:
        logger.warning("[Ingest] Claim lookup file not found: %s", _LOOKUP_PATH)

    # ── Load RAG-specific rules JSON ───────────────────────────
    rag_rules: List[Dict[str, Any]] = []
    if _RAG_RULES_PATH.exists():
        with open(_RAG_RULES_PATH, "r", encoding="utf-8") as fh:
            raw_rag = json.load(fh)
        if isinstance(raw_rag, list):
            rag_rules = _load_rag_rules(raw_rag)
            logger.info(
                "[Ingest] Loaded %d rules from RAG rules file.", len(rag_rules)
            )
        else:
            logger.warning("[Ingest] Unexpected format in RAG rules file — skipping.")
    else:
        logger.info("[Ingest] No RAG rules file found at %s — skipping.", _RAG_RULES_PATH)

    # ── Merge — RAG file takes precedence on ruleId conflicts ──
    merged: Dict[str, Dict[str, Any]] = {}
    for rule in claim_rules:
        merged[rule["rule_id"]] = rule
    for rule in rag_rules:                    # overwrites on conflict
        merged[rule["rule_id"]] = rule

    rules = list(merged.values())
    logger.info(
        "[Ingest] Total unique rules after merge: %d "
        "(claim lookup: %d, RAG file: %d)",
        len(rules), len(claim_rules), len(rag_rules),
    )

    if not rules:
        return {"status": "ok", "detail": "No rules found in either source.", "ingested": 0}

    # ── Embed ──────────────────────────────────────────────────
    embedder = Embedder()
    chunks   = [_build_chunk(r) for r in rules]

    logger.info("[Ingest] Embedding %d rule chunks…", len(chunks))
    vectors = embedder.embed_batch(chunks)

    # ── Prepare rows ───────────────────────────────────────────
    rows = []
    for rule, chunk, vector in zip(rules, chunks, vectors):
        rows.append({
            "rule_id":       rule["rule_id"],
            "package_name":  rule["package_name"],
            "severity":      rule["severity"],
            "object_type":   rule["object_type"],
            "title":         rule["short_message"][:120],   # short title
            "short_message": rule["short_message"],
            "long_message":  rule["long_message"],
            "chunk_text":    chunk,
            "embedding":     vector,
        })

    # ── Upsert ─────────────────────────────────────────────────
    count = upsert_rules_batch(rows, region=region)
    total = count_rules(region=region)

    logger.info(
        "[Ingest] Done for region '%s'. Upserted %d rules. Total in store: %d.",
        region.upper(), count, total
    )

    return {
        "status":          "ok",
        "region":          region,
        "ingested":        count,
        "total_in_store":  total,
        "from_claim_lookup": len(claim_rules),
        "from_rag_file":     len(rag_rules),
    }


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    # Ensure .env is loaded when run directly
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ingest rules into ChromaDB for a region")
    parser.add_argument(
        "--region",
        type=str,
        default="uae",
        choices=["uae", "ksa"],
        help="Region to ingest rules for (default: uae)"
    )
    args = parser.parse_args()

    result = run_ingest(region=args.region)
    print("\n" + "=" * 60)
    print(f"INGEST RESULT FOR {args.region.upper()}")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    sys.exit(0 if result["status"] == "ok" else 1)
