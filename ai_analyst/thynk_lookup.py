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
# ============================================================
# Locate the JSON files per region
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _get_lookup_path(region: str) -> Path:
    region = (region or "uae").lower()
    if region == "ksa":
        ksa_path = _PROJECT_ROOT / "Thynk" / "ksa" / "ksa-claimlookup.json"
        if ksa_path.exists():
            return ksa_path
        nphies_path = _PROJECT_ROOT / "Thynk" / "ksa" / "nphies-claimlookup.json"
        if nphies_path.exists():
            return nphies_path
        return ksa_path
    else:
        uae_path = _PROJECT_ROOT / "Thynk" / "uae" / "thynk-claimlookup.json"
        if uae_path.exists():
            return uae_path
        return _PROJECT_ROOT / "Thynk" / "thynk-claimlookup.json"

# ============================================================
# Module-level cache — loaded once per region at first call
# ============================================================

_caches: Dict[str, List[Dict[str, Any]]] = {}


def _load(region: str = "uae") -> List[Dict[str, Any]]:
    """Load and cache the lookup JSON for a given region. Returns [] on any error."""
    reg = (region or "uae").lower()
    if reg in _caches:
        return _caches[reg]

    lookup_path = _get_lookup_path(reg)

    try:
        with open(lookup_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if not isinstance(raw, list):
            logger.warning("[ThynkLookup] Expected a JSON array — got %s", type(raw).__name__)
            _caches[reg] = []
            return _caches[reg]

        _caches[reg] = raw
        logger.info("[ThynkLookup] Loaded %d claim entries for %s from %s", len(_caches[reg]), reg, lookup_path)

    except FileNotFoundError:
        logger.warning("[ThynkLookup] Lookup file not found: %s — Thynk disabled for %s.", lookup_path, reg)
        _caches[reg] = []

    except Exception as exc:
        logger.warning("[ThynkLookup] Failed to load lookup file for %s: %s — Thynk disabled.", reg, exc)
        _caches[reg] = []

    return _caches[reg]


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

def find(claim_id: str, region: str = "uae") -> Optional[Dict[str, Any]]:
    """
    Look up Thynk validation data for a claim_id within a given region.

    Args:
        claim_id: The claim identifier string.
        region: 'uae' or 'ksa' (defaults to 'uae').

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
        entries = _load(region=region)
        for entry in entries:
            if str(entry.get("claim_id", "")).strip() == str(claim_id).strip():
                items = entry.get("thynk_response", [])
                result = _normalise(claim_id, items)
                logger.info(
                    "[ThynkLookup] Found %d items for claim %s (%s) "
                    "(CRITICAL=%d SEVERE=%d WARNING=%d INFO=%d)",
                    result["summary"]["total"], claim_id, region,
                    result["summary"]["critical"],
                    result["summary"]["severe"],
                    result["summary"]["warning"],
                    result["summary"]["info"],
                )
                return result

        logger.info("[ThynkLookup] claim_id=%s not found in %s lookup — no Thynk data.", claim_id, region)
        return None

    except Exception as exc:
        logger.warning("[ThynkLookup] Unexpected error during lookup: %s", exc)
        return None
