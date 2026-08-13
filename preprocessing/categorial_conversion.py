"""
categorical_conversion.py

Purpose
-------
Converts raw ICD-10 diagnosis codes and CPT/activity codes into
higher-level categorical features required by the behavioral
XGBoost denial prediction model.

Model 2 expects:
    - diagnosis_code
    - activity_code
    - icd_category
    - cpt_category

Important
---------
The category mappings in this module MUST remain consistent with
the mappings used during model training.

Current training mappings are based on:
    1. ICD-10 first-character/domain mapping
    2. CPT numeric range mapping

This module does NOT:
    - calculate model probabilities
    - perform Pydantic validation
    - calculate log features
    - calculate high_cost_flag
    - create ICD-CPT interaction features
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# ============================================================================
# ICD CATEGORY MAPPING
# ============================================================================
#
# This follows the same first-character mapping used in the SQL training
# preprocessing logic.
#
# Example:
#     R50.9 -> R -> General_Symptoms
#     J18.9 -> J -> Pulmonology
#     I10   -> I -> Cardiology
#
# Note:
# These are project-level clinical domains, not official ICD hierarchy names.
# ============================================================================

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


# ============================================================================
# CPT CATEGORY RANGE DEFINITIONS
# ============================================================================
#
# These ranges reproduce the logic from the SQL training transformation.
#
# IMPORTANT:
# The ordering matters because some ranges overlap.
#
# 99202-99499 must be evaluated before >= 90000.
# ============================================================================

def _is_numeric_code(value: Any) -> bool:
    """
    Check whether an activity/CPT code contains only digits.

    Examples
    --------
    '87633'  -> True
    '99213'  -> True
    'J1234'  -> False
    None     -> False
    """
    if value is None:
        return False

    value_str = str(value).strip()

    return bool(re.fullmatch(r"\d+", value_str))


def _safe_numeric_code(value: Any) -> Optional[int]:
    """
    Convert a numeric activity code into an integer safely.

    Returns None for:
        - None
        - empty strings
        - alphanumeric codes
        - invalid values
    """
    if not _is_numeric_code(value):
        return None

    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


# ============================================================================
# PUBLIC CONVERSION FUNCTIONS
# ============================================================================

def convert_icd_category(diagnosis_code: Any) -> str:
    """
    Convert an ICD-10 diagnosis code into the project-level ICD category.

    Parameters
    ----------
    diagnosis_code:
        Raw ICD-10 diagnosis code.

    Returns
    -------
    str
        Derived ICD category.

    Examples
    --------
    >>> convert_icd_category("R50.9")
    'General_Symptoms'

    >>> convert_icd_category("J18.9")
    'Pulmonology'

    >>> convert_icd_category("I10")
    'Cardiology'

    >>> convert_icd_category("E11.9")
    'Endocrinology'
    """

    if diagnosis_code is None:
        return "Unknown_ICD"

    code = str(diagnosis_code).strip().upper()

    if not code:
        return "Unknown_ICD"

    first_character = code[0]

    return ICD_CATEGORY_MAP.get(first_character, "Unknown_ICD")


def convert_cpt_category(activity_code: Any) -> str:
    """
    Convert a CPT/activity code into the project-level CPT category.

    This reproduces the SQL training logic.

    Numeric codes:
        100-1999       -> Anesthesia
        10000-69999    -> Surgery
        70000-79999    -> Radiology
        80000-89999    -> Pathology_Laboratory
        99202-99499    -> Evaluation_Management
        >=90000        -> Medicine

    Alphanumeric codes:
        -> HCPCS_Supplies

    Unknown numeric ranges:
        -> Unknown_CPT
    """

    if activity_code is None:
        return "Unknown_CPT"

    code = str(activity_code).strip().upper()

    if not code:
        return "Unknown_CPT"

    numeric_code = _safe_numeric_code(code)

    # ------------------------------------------------------------------------
    # Alphanumeric activity codes
    # ------------------------------------------------------------------------
    #
    # Same behavior as:
    #
    # ELSE 'HCPCS_Supplies'
    #
    # in your SQL query.
    # ------------------------------------------------------------------------

    if numeric_code is None:
        return "HCPCS_Supplies"

    # ------------------------------------------------------------------------
    # Numeric CPT ranges
    # ------------------------------------------------------------------------

    if 100 <= numeric_code <= 1999:
        return "Anesthesia"

    if 10000 <= numeric_code <= 69999:
        return "Surgery"

    if 70000 <= numeric_code <= 79999:
        return "Radiology"

    if 80000 <= numeric_code <= 89999:
        return "Pathology_Laboratory"

    # Important:
    # This must be checked before >= 90000.
    if 99202 <= numeric_code <= 99499:
        return "Evaluation_Management"

    if numeric_code >= 90000:
        return "Medicine"

    return "Unknown_CPT"


# ============================================================================
# CLAIM-LEVEL CONVERSION
# ============================================================================

def add_categorical_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add the derived categorical features required by Model 2.

    Input
    -----
    A dictionary representing the validated claim payload.

    Output
    ------
    A NEW dictionary containing:
        - all original input fields
        - icd_category
        - cpt_category

    The original dictionary is not modified.

    Example
    -------
    Input:

        {
            "diagnosis_code": "R50.9",
            "activity_code": "87633"
        }

    Output:

        {
            "diagnosis_code": "R50.9",
            "activity_code": "87633",
            "icd_category": "General_Symptoms",
            "cpt_category": "Pathology_Laboratory"
        }
    """

    converted = dict(data)

    diagnosis_code = converted.get("diagnosis_code")
    activity_code = converted.get("activity_code")

    converted["icd_category"] = convert_icd_category(
        diagnosis_code
    )

    converted["cpt_category"] = convert_cpt_category(
        activity_code
    )

    return converted


# ============================================================================
# OPTIONAL: DOMAIN MATCH
# ============================================================================
#
# Your CURRENT Model 2 does NOT use this feature.
#
# It is provided here only as an optional utility because your old SQL
# pipeline generated icd_cpt_domain_match.
#
# DO NOT pass this feature to Model 2 unless you retrain the model with it.
# ============================================================================

def create_icd_cpt_domain_match(
    icd_category: str,
    cpt_category: str,
) -> str:
    """
    Create the old combined domain representation.

    Example:
        General_Symptoms + Pathology_Laboratory
        ->
        General_Symptoms_Pathology_Laboratory

    NOTE:
        This is NOT used by the current Model 2.
    """

    icd = str(icd_category).strip() if icd_category else "Unknown_ICD"
    cpt = str(cpt_category).strip() if cpt_category else "Unknown_CPT"

    return f"{icd}_{cpt}"


# ============================================================================
# DEBUG / TEST UTILITY
# ============================================================================

if __name__ == "__main__":

    test_claim = {
        "diagnosis_code": "R50.9",
        "diagnosis_type": "Principal",
        "activity_code": "87633",
        "activity_quantity": 1.0,
        "activity_gross": 1656.0,
        "claim_gross": 2470.13,
        "claim_net": 2470.13,
        "patient_age": 23,
        "gender": "MALE",
        "nationality": "EMIRATI",
        "payer_id": "E001",
        "insurance_plan_tier": "Insurance",
        "clinician_profession": "General Practitioner",
        "clinician_category": "General Practitioner",
        "facility_type": "Hospital",
        "billing_lag_days": 44,
        "length_of_stay": 0,
        "encounter_type": "OP",
    }

    print("=" * 70)
    print("CATEGORICAL CONVERSION TEST")
    print("=" * 70)

    print("\nInput:")
    print(f"  Diagnosis Code : {test_claim['diagnosis_code']}")
    print(f"  Activity Code  : {test_claim['activity_code']}")

    converted_claim = add_categorical_features(test_claim)

    print("\nDerived Features:")
    print(f"  ICD Category   : {converted_claim['icd_category']}")
    print(f"  CPT Category   : {converted_claim['cpt_category']}")

    print("\nComplete Converted Payload:")
    for key, value in converted_claim.items():
        print(f"  {key:<25} = {value}")

    print("\n" + "=" * 70)
    print("INDIVIDUAL TESTS")
    print("=" * 70)

    icd_tests = [
        "R50.9",
        "J18.9",
        "I10",
        "E11.9",
        "M54.5",
        "O80",
        "Z00.00",
        "INVALID",
        None,
    ]

    print("\nICD conversion:")
    for code in icd_tests:
        print(f"  {str(code):<10} -> {convert_icd_category(code)}")

    cpt_tests = [
        "87633",
        "70553",
        "99213",
        "100",
        "1500",
        "50000",
        "J1234",
        "EX1",
        None,
    ]

    print("\nCPT conversion:")
    for code in cpt_tests:
        print(f"  {str(code):<10} -> {convert_cpt_category(code)}")