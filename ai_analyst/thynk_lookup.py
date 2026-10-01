"""
ai_analyst/thynk_lookup.py
--------------------------
Backend Thynk validation lookup.

Loads Thynk/thynk-claimlookup.json (relative to the project root),
normalises the response into claim / activity / diagnosis buckets,
and exposes a single find(claim_id) function.

Returns a dict when a match is found, or None when the claim_id is
not present or if the file cannot be loaded.  All errors are silent
so the AI analyst continues to work normally without Thynk data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Severity ordering — used for sorting within each bucket
# ============================================================

_SEVERITY_ORDER = {
    "CRITICAL": 0,
    "SEVERE":   1,
    "WARNING":  2,
    "INFO":     3,
}

# ============================================================
# Locate the JSON file
# ============================================================
# This file lives at:
#   <project_root>/Thynk/thynk-claimlookup.json
#
# ai_analyst/ is one level below the project root, so we go
# up two parents: thynk_lookup.py → ai_analyst/ → project root

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_LOOKUP_PATH   = _PROJECT_ROOT / "Thynk" / "thynk-claimlookup.json"

# ============================================================
# Module-level cache — loaded once at first call
# ============================================================

_cache: Optional[List[Dict[str, Any]]] = None


def _load() -> List[Dict[str, Any]]:
    """Load and cache the lookup JSON.  Returns [] on any error."""
    global _cache
    if _cache is not None:
        return _cache

    try:
        with open(_LOOKUP_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if not isinstance(raw, list):
            logger.warning("[ThynkLookup] Expected a JSON array — got %s", type(raw).__name__)
            _cache = []
            return _cache

        _cache = raw
        logger.info("[ThynkLookup] Loaded %d claim entries from %s", len(_cache), _LOOKUP_PATH)

    except FileNotFoundError:
        logger.warning("[ThynkLookup] Lookup file not found: %s — Thynk disabled.", _LOOKUP_PATH)
        _cache = []

    except Exception as exc:
        logger.warning("[ThynkLookup] Failed to load lookup file: %s — Thynk disabled.", exc)
        _cache = []

    return _cache


# ============================================================
# Normalise helper
# ============================================================

def _sort_key(item: Dict[str, Any]) -> int:
    return _SEVERITY_ORDER.get(
        str(item.get("severity", "")).upper(), 99
    )


def _normalise(claim_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split items into level buckets and compute summary counts."""

    claim_level     = sorted(
        [i for i in items if i.get("objectType") == "claim"],
        key=_sort_key,
    )
    activity_level  = sorted(
        [i for i in items if i.get("objectType") == "activity"],
        key=_sort_key,
    )
    diagnosis_level = sorted(
        [i for i in items if i.get("objectType") == "diagnosis"],
        key=_sort_key,
    )
    all_items = sorted(items, key=_sort_key)

    # Severity counts across all items
    counts: Dict[str, int] = {"CRITICAL": 0, "SEVERE": 0, "WARNING": 0, "INFO": 0}
    for item in items:
        sev = str(item.get("severity", "")).upper()
        if sev in counts:
            counts[sev] += 1

    return {
        "found":            True,
        "claim_id":         claim_id,
        "claim_level":      claim_level,
        "activity_level":   activity_level,
        "diagnosis_level":  diagnosis_level,
        "all":              all_items,
        "summary": {
            "total":    len(items),
            "critical": counts["CRITICAL"],
            "severe":   counts["SEVERE"],
            "warning":  counts["WARNING"],
            "info":     counts["INFO"],
        },
    }


# ============================================================
# Public API
# ============================================================

def find(claim_id: str) -> Optional[Dict[str, Any]]:
    """
    Look up Thynk validation data for a claim_id.

    Returns:
        A normalised dict  { found, claim_id, claim_level,
                             activity_level, diagnosis_level,
                             all, summary }
        when the claim is found,
        or None when not found or on any error.
    """
    if not claim_id:
        return None

    try:
        entries = _load()
        for entry in entries:
            if str(entry.get("claim_id", "")).strip() == str(claim_id).strip():
                items = entry.get("thynk_response", [])
                result = _normalise(claim_id, items)
                logger.info(
                    "[ThynkLookup] Found %d items for claim %s "
                    "(CRITICAL=%d SEVERE=%d WARNING=%d INFO=%d)",
                    result["summary"]["total"], claim_id,
                    result["summary"]["critical"],
                    result["summary"]["severe"],
                    result["summary"]["warning"],
                    result["summary"]["info"],
                )
                return result

        logger.info("[ThynkLookup] claim_id=%s not found in lookup — no Thynk data.", claim_id)
        return None

    except Exception as exc:
        logger.warning("[ThynkLookup] Unexpected error during lookup: %s", exc)
        return None
