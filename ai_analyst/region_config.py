"""
ai_analyst/region_config.py
----------------------------
Multi-region configuration for UAE (CHI) and KSA (CCHI/Nphies).

This module provides region-specific settings for:
- Model paths and versions
- RAG knowledge base collections
- Rule source paths
- Regulatory metadata
- Display formatting

Usage:
    from ai_analyst.region_config import Region, get_region_config
    
    config = get_region_config(Region.UAE)
    print(config.regulator_name)  # "CHI"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Region(str, Enum):
    """Supported healthcare regions."""
    UAE = "uae"
    KSA = "ksa"


@dataclass
class RegionConfig:
    """Configuration for a specific healthcare region."""
    
    # Region identity
    region: Region
    regulator_name: str          # "CHI" or "CCHI"
    regulator_full_name: str
    country_name: str
    
    # Model configuration
    model_base_path: str         # e.g., "models/uae"
    model_version: str           # e.g., "v3"
    
    # RAG configuration
    chroma_store_path: str       # e.g., "ai_analyst/rag/chroma_store_uae"
    chroma_collection_name: str  # e.g., "chi_rules"
    rule_source_dir: str         # e.g., "Thynk/uae"
    claim_lookup_file: str       # e.g., "thynk-claimlookup.json"
    rag_rules_file: str          # e.g., "thynk-rag-rules.json"
    
    # Display configuration
    currency: str                # "AED" or "SAR"
    currency_symbol: str         # "د.إ" or "ر.س"
    date_format: str             # "%d/%m/%Y"
    
    # Regulatory metadata
    mandatory_fields: List[str]
    claim_system_name: str       # "Thynk" or "Nphies"
    
    # Feature flags
    is_production_ready: bool
    
    def get_model_path(self, model_type: str) -> Path:
        """Get the full path to a model file."""
        return _PROJECT_ROOT / self.model_base_path / model_type
    
    def get_rule_source_path(self) -> Path:
        """Get the full path to the rule source directory."""
        return _PROJECT_ROOT / self.rule_source_dir
    
    def get_chroma_store_path(self) -> Path:
        """Get the full path to the ChromaDB store."""
        return _PROJECT_ROOT / self.chroma_store_path


# ================================================================
# Region-specific configurations
# ================================================================

UAE_CONFIG = RegionConfig(
    # Identity
    region=Region.UAE,
    regulator_name="DoH/DHA",
    regulator_full_name="Department of Health Abu Dhabi / Dubai Health Authority",
    country_name="United Arab Emirates",
    
    # Models
    model_base_path="models/uae",
    model_version="v3",
    
    # RAG
    chroma_store_path="ai_analyst/rag/chroma_store_uae",
    chroma_collection_name="uae_rules",
    rule_source_dir="Thynk/uae",
    claim_lookup_file="thynk-claimlookup.json",
    rag_rules_file="thynk-rag-rules.json",
    
    # Display
    currency="AED",
    currency_symbol="د.إ",
    date_format="%d/%m/%Y",
    
    # Regulatory
    mandatory_fields=[
        "encounter_status",
        "coverage_class_type",
        "clinician_profession",
        "clinician_category",
        "primary_diagnosis",
    ],
    claim_system_name="Thynk",
    
    # Status
    is_production_ready=True,
)

KSA_CONFIG = RegionConfig(
    # Identity
    region=Region.KSA,
    regulator_name="CHI",
    regulator_full_name="Council of Health Insurance (مجلس الضمان الصحي)",
    country_name="Kingdom of Saudi Arabia",
    
    # Models
    model_base_path="models/ksa",
    model_version="v1",
    
    # RAG
    chroma_store_path="ai_analyst/rag/chroma_store_ksa",
    chroma_collection_name="chi_rules",
    rule_source_dir="Thynk/ksa",
    claim_lookup_file="ksa-claimlookup.json",
    rag_rules_file="ksa-rag-rules.json",
    
    # Display
    currency="SAR",
    currency_symbol="ر.س",
    date_format="%d/%m/%Y",
    
    # Regulatory
    mandatory_fields=[
        "approval_number",
        "policy_number",
        "member_id",
        "primary_diagnosis",
        "clinician_license",
    ],
    claim_system_name="Nphies",
    
    # Status
    is_production_ready=False,  # KSA models in training
)


# Registry mapping
_REGION_REGISTRY = {
    Region.UAE: UAE_CONFIG,
    Region.KSA: KSA_CONFIG,
}


# ================================================================
# Public API
# ================================================================

def get_region_config(region: Region) -> RegionConfig:
    """
    Get the configuration for a specific region.
    
    Args:
        region: The target region (UAE or KSA)
    
    Returns:
        RegionConfig instance for the specified region
    
    Raises:
        ValueError: If region is not supported
    """
    if region not in _REGION_REGISTRY:
        raise ValueError(f"Unsupported region: {region}")
    return _REGION_REGISTRY[region]


def get_default_region() -> Region:
    """Get the default region (UAE for now)."""
    return Region.UAE


def list_regions() -> List[Region]:
    """List all supported regions."""
    return list(_REGION_REGISTRY.keys())


def is_region_production_ready(region: Region) -> bool:
    """Check if a region is production-ready."""
    config = get_region_config(region)
    return config.is_production_ready
