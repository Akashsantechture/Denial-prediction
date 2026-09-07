"""
categorical_conversion.py

Final version for claim_activity_denial_model_v3
"""

from __future__ import annotations

import re
from typing import Any, List, Dict, Optional


# ============================================================
# ICD CATEGORY MAPPING
# ============================================================

ICD_CATEGORY_MAP = {
    "A": "Infectious",
    "B": "Infectious",
    "C": "Oncology",
    "D": "Oncology_Hematology",
    "E": "Endocrinology",
    "F": "Psychiatry",
    "G": "Neurology",
    "H": "Eye_Ear",
    "I": "Cardiology",
    "J": "Pulmonology",
    "K": "Gastroenterology_Dental",
    "L": "Dermatology",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
    "O": "OBGYN",
    "P": "Pediatrics",
    "Q": "Congenital",
    "R": "General_Symptoms",
    "S": "Trauma_Burns_Poisoning",
    "T": "Trauma_Burns_Poisoning",
    "V": "External_Causes",
    "W": "External_Causes",
    "X": "External_Causes",
    "Y": "External_Causes",
    "Z": "Factors_Influencing_Health_Status",
}


# ============================================================
# SECONDARY DIAGNOSIS FEATURE COLUMNS
# ============================================================

SECONDARY_COLUMNS = [
    "secondary_infectious",
    "secondary_oncology",
    "secondary_oncology_hematology",
    "secondary_endocrinology",
    "secondary_psychiatry",
    "secondary_neurology",
    "secondary_eye_ear",
    "secondary_cardiology",
    "secondary_pulmonology",
    "secondary_gastroenterology_dental",
    "secondary_dermatology",
    "secondary_musculoskeletal",
    "secondary_genitourinary",
    "secondary_obgyn",
    "secondary_pediatrics",
    "secondary_congenital",
    "secondary_general_symptoms",
    "secondary_trauma_burns_poisoning",
    "secondary_external_causes",
    "secondary_factors_influencing_health_status",
    "secondary_unknown_icd_category",
]


ICD_TO_SECONDARY_COLUMN = {
    "Infectious": "secondary_infectious",
    "Oncology": "secondary_oncology",
    "Oncology_Hematology": "secondary_oncology_hematology",
    "Endocrinology": "secondary_endocrinology",
    "Psychiatry": "secondary_psychiatry",
    "Neurology": "secondary_neurology",
    "Eye_Ear": "secondary_eye_ear",
    "Cardiology": "secondary_cardiology",
    "Pulmonology": "secondary_pulmonology",
    "Gastroenterology_Dental": "secondary_gastroenterology_dental",
    "Dermatology": "secondary_dermatology",
    "Musculoskeletal": "secondary_musculoskeletal",
    "Genitourinary": "secondary_genitourinary",
    "OBGYN": "secondary_obgyn",
    "Pediatrics": "secondary_pediatrics",
    "Congenital": "secondary_congenital",
    "General_Symptoms": "secondary_general_symptoms",
    "Trauma_Burns_Poisoning": "secondary_trauma_burns_poisoning",
    "External_Causes": "secondary_external_causes",
    "Factors_Influencing_Health_Status":
        "secondary_factors_influencing_health_status",
    "Unknown_ICD": "secondary_unknown_icd_category",
}


# ============================================================
# HELPERS
# ============================================================

def _is_numeric_code(value: Any) -> bool:

    if value is None:
        return False

    return bool(
        re.fullmatch(
            r"\d+",
            str(value).strip()
        )
    )


def _safe_numeric_code(value: Any) -> Optional[int]:

    if not _is_numeric_code(value):
        return None

    try:
        return int(str(value).strip())
    except Exception:
        return None


# ============================================================
# ICD CATEGORY
# ============================================================

def convert_icd_category(
    diagnosis_code: Any
) -> str:

    if diagnosis_code is None:
        return "Unknown_ICD"

    code = str(diagnosis_code).strip().upper()

    if not code:
        return "Unknown_ICD"

    return ICD_CATEGORY_MAP.get(
        code[0],
        "Unknown_ICD"
    )


# ============================================================
# CPT CATEGORY
# ============================================================

def convert_cpt_category(
    activity_code: Any
) -> str:

    if activity_code is None:
        return "Unknown_CPT"

    code = str(activity_code).strip()

    if not code:
        return "Unknown_CPT"

    numeric_code = _safe_numeric_code(code)

    # HCPCS / alphanumeric
    if numeric_code is None:
        return "HCPCS_Supplies"

    if 100 <= numeric_code <= 1999:
        return "Anesthesia"

    if 10000 <= numeric_code <= 69999:
        return "Surgery"

    if 70000 <= numeric_code <= 79999:
        return "Radiology"

    if 80000 <= numeric_code <= 89999:
        return "Pathology_Laboratory"

    if 99202 <= numeric_code <= 99499:
        return "Evaluation_Management"

    if numeric_code >= 90000:
        return "Medicine"

    return "Unknown_CPT"


# ============================================================
# PRIMARY DIAGNOSIS FEATURES
# ============================================================

def build_primary_features(
    primary_diagnosis_code: str
) -> Dict:

    return {
        "primary_diagnosis_code":
            primary_diagnosis_code,

        "primary_diagnosis_category":
            convert_icd_category(
                primary_diagnosis_code
            )
    }


# ============================================================
# SECONDARY DIAGNOSIS FEATURES
# ============================================================

def build_secondary_features(
    diagnosis_codes: List[str]
) -> Dict:

    result = {
        "secondary_dx_count":
            len(diagnosis_codes)
    }

    for col in SECONDARY_COLUMNS:
        result[col] = 0

    seen_categories = set()

    for dx in diagnosis_codes:

        category = convert_icd_category(dx)

        if category in seen_categories:
            continue

        seen_categories.add(category)

        column = ICD_TO_SECONDARY_COLUMN.get(
            category,
            "secondary_unknown_icd_category"
        )

        result[column] = 1

    return result


# ============================================================
# COMPLETE FEATURE GENERATION
# ============================================================

def generate_diagnosis_features(
    primary_diagnosis_code: str,
    secondary_diagnosis_codes: List[str]
) -> Dict:

    features = {}

    features.update(
        build_primary_features(
            primary_diagnosis_code
        )
    )

    features.update(
        build_secondary_features(
            secondary_diagnosis_codes
        )
    )

    return features


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    primary_dx = "E78.5"

    secondary_dx = [
        "I10",
        "E55.9",
        "E66.3",
        "E11.65",
        "Z79.899"
    ]

    print(
        generate_diagnosis_features(
            primary_dx,
            secondary_dx
        )
    )

    print(
        convert_cpt_category(
            "93000"
        )
    )

    print(
        convert_cpt_category(
            "99202"
        )
    )