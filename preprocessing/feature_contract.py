"""
feature_contract.py

Defines the exact feature contract for Behavioral XGBoost Model 2.

Model 2 expects exactly 19 features:

    11 categorical
    8 numeric

This module is responsible for:
    - Defining feature names
    - Defining feature order
    - Selecting only model features
    - Validating that all required features exist
    - Converting categorical columns to pandas category dtype
    - Normalizing numeric columns
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


# ============================================================================
# MODEL 2 FEATURE CONTRACT
# ============================================================================

CATEGORICAL_FEATURES = [
    "diagnosis_code",
    "diagnosis_type",
    "activity_code",
    "gender",
    "nationality",
    "encounter_type",
    "clinician_profession",
    "clinician_category",
    "facility_type",
    "icd_category",
    "cpt_category",
]


NUMERIC_FEATURES = [
    "activity_quantity",
    "patient_age",
    "activity_gross_log",
    "claim_gross_log",
    "claim_net_log",
    "high_cost_flag",
    "length_of_stay",
    "billing_lag_days",
]


# Exact order used by the trained XGBoost model.
FEATURE_ORDER = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# Expected feature count.
EXPECTED_FEATURE_COUNT = len(FEATURE_ORDER)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_feature_contract(data: Dict[str, Any]) -> None:
    """
    Validate that all required Model 2 features are present.

    Raises
    ------
    ValueError
        If one or more required features are missing.
    """

    missing_features = [
        feature
        for feature in FEATURE_ORDER
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )


# ============================================================================
# DATAFRAME CREATION
# ============================================================================

def build_model_dataframe(
    data: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build the exact DataFrame expected by Model 2.

    Parameters
    ----------
    data:
        Fully processed claim dictionary.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame containing exactly the 19 model features.
    """

    # ------------------------------------------------------------
    # 1. Validate
    # ------------------------------------------------------------

    validate_feature_contract(data)

    # ------------------------------------------------------------
    # 2. Create one-row DataFrame
    # ------------------------------------------------------------

    df = pd.DataFrame([data])

    # ------------------------------------------------------------
    # 3. Force exact feature order
    # ------------------------------------------------------------

    df = df[FEATURE_ORDER].copy()

    # ------------------------------------------------------------
    # 4. Convert categorical features
    # ------------------------------------------------------------

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype("category")

    # ------------------------------------------------------------
    # 5. Normalize numeric features
    # ------------------------------------------------------------

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    # ------------------------------------------------------------
    # 6. Final structural validation
    # ------------------------------------------------------------

    if list(df.columns) != FEATURE_ORDER:
        raise ValueError(
            "Model feature order mismatch.\n"
            f"Expected: {FEATURE_ORDER}\n"
            f"Received: {list(df.columns)}"
        )

    if len(df.columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but received {len(df.columns)}."
        )

    return df


# ============================================================================
# DEBUG INFORMATION
# ============================================================================

def get_feature_contract() -> Dict[str, Any]:
    """
    Return model feature-contract metadata.

    Useful for debugging and health checks.
    """

    return {
        "model": "behavioral_xgboost_model2_individual_cats",
        "feature_count": EXPECTED_FEATURE_COUNT,
        "categorical_features": CATEGORICAL_FEATURES.copy(),
        "numeric_features": NUMERIC_FEATURES.copy(),
        "feature_order": FEATURE_ORDER.copy(),
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":

    test_data = {
        # Categorical
        "diagnosis_code": "R50.9",
        "diagnosis_type": "Principal",
        "activity_code": "87633",
        "gender": "MALE",
        "nationality": "EMIRATI",
        "encounter_type": "OP",
        "clinician_profession": "General Practitioner",
        "clinician_category": "General Practitioner",
        "facility_type": "Hospital",
        "icd_category": "General_Symptoms",
        "cpt_category": "Pathology_Laboratory",

        # Numeric
        "activity_quantity": 1.0,
        "patient_age": 23,
        "activity_gross_log": 7.413,
        "claim_gross_log": 7.812,
        "claim_net_log": 7.812,
        "high_cost_flag": 1,
        "length_of_stay": 0,
        "billing_lag_days": 44,

        # Extra business fields are deliberately ignored.
        "payer_id": "E001",
        "insurance_plan_tier": "Insurance",
        "activity_gross": 1656.0,
        "claim_gross": 2470.13,
        "claim_net": 2470.13,
    }

    print("=" * 70)
    print("MODEL 2 FEATURE CONTRACT TEST")
    print("=" * 70)

    model_df = build_model_dataframe(test_data)

    print("\nFeature count:")
    print(len(model_df.columns))

    print("\nFeature order:")
    for index, feature in enumerate(model_df.columns, start=1):
        print(f"{index:02d}. {feature}")

    print("\nData types:")
    print(model_df.dtypes)

    print("\nModel input:")
    print(model_df.to_string(index=False))

    print("\nContract metadata:")
    print(get_feature_contract())

    print("\n Feature contract validation successful.")