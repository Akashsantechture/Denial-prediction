# """
# feature_engineering.py

# Feature engineering layer for the Healthcare Claim Denial
# Prediction Model - Model 2.

# Responsibilities
# ----------------
# 1. Generate log-transformed financial features.
# 2. Generate high_cost_flag.
# 3. Normalize numeric input values.
# 4. Handle missing/null numeric values safely.

# This module MUST reproduce the same feature transformations
# used during model training.

# Current Model 2 numeric features
# --------------------------------
#     activity_quantity
#     patient_age
#     activity_gross_log
#     claim_gross_log
#     claim_net_log
#     high_cost_flag
#     length_of_stay
#     billing_lag_days

# This module intentionally does NOT create:
#     - icd_category
#     - cpt_category
#     - icd_cpt_domain_match
#     - unit_cost
#     - discount_ratio
#     - ip_stay_cost_ratio
#     - gender_appropriate_flag
#     - age_appropriate_flag
#     - other interaction features
# """

# from __future__ import annotations

# import math
# from typing import Any, Dict


# # ============================================================================
# # MODEL CONFIGURATION
# # ============================================================================

# # IMPORTANT:
# # This threshold MUST be identical to the threshold used during training.
# #
# # Your previous FastAPI used:
# #
# #     activity_gross > 5000
# #
# # Therefore, until you establish a different canonical training threshold,
# # we preserve 5000 here.
# #
# # If your training dataset actually used a different percentile-derived
# # threshold, CHANGE THIS VALUE to the exact training value.
# HIGH_COST_THRESHOLD = 1000.00


# # ============================================================================
# # NUMERIC FEATURE CONTRACT
# # ============================================================================

# NUMERIC_FEATURES = [
#     "activity_quantity",
#     "patient_age",
#     "activity_gross_log",
#     "claim_gross_log",
#     "claim_net_log",
#     "high_cost_flag",
#     "length_of_stay",
#     "billing_lag_days",
# ]


# # Raw financial fields required to generate log features.
# RAW_FINANCIAL_FEATURES = [
#     "activity_gross",
#     "claim_gross",
#     "claim_net",
# ]


# # ============================================================================
# # SAFE NUMERIC CONVERSION
# # ============================================================================

# def safe_float(
#     value: Any,
#     default: float = 0.0,
# ) -> float:
#     """
#     Safely convert a value to float.

#     Handles:
#         - None
#         - NaN
#         - infinity
#         - empty strings
#         - invalid strings

#     Parameters
#     ----------
#     value:
#         Input value.

#     default:
#         Value returned when conversion fails.

#     Returns
#     -------
#     float
#     """

#     if value is None:
#         return default

#     try:
#         numeric_value = float(value)

#         if not math.isfinite(numeric_value):
#             return default

#         return numeric_value

#     except (TypeError, ValueError):
#         return default


# # ============================================================================
# # LOG TRANSFORMATION
# # ============================================================================

# def safe_log1p(value: Any) -> float:
#     """
#     Apply a safe log1p transformation.

#     Training transformation:
#         np.log1p(max(0, value))

#     The max(0, value) behavior matches the previous FastAPI
#     implementation.

#     Examples
#     --------
#     0       -> 0
#     100     -> log(101)
#     1656    -> log(1657)
#     None    -> 0
#     -100    -> 0
#     """

#     numeric_value = safe_float(value, default=0.0)

#     # Match the training/inference behavior:
#     # negative financial values are clamped to zero.
#     numeric_value = max(0.0, numeric_value)

#     return float(np_log1p(numeric_value))


# def np_log1p(value: float) -> float:
#     """
#     Small isolated wrapper around math.log1p.

#     Keeping this separate makes the transformation explicit and
#     easy to test.
#     """

#     return math.log1p(value)


# # ============================================================================
# # FINANCIAL FEATURE ENGINEERING
# # ============================================================================

# def add_financial_log_features(
#     data: Dict[str, Any],
# ) -> Dict[str, Any]:
#     """
#     Add the three log-transformed financial features required by Model 2.

#     Creates
#     -------
#     activity_gross_log
#     claim_gross_log
#     claim_net_log

#     Does not remove the original financial fields.

#     Example
#     -------
#     activity_gross = 1656

#     becomes approximately:

#     activity_gross_log = log1p(1656)
#     """

#     result = dict(data)

#     result["activity_gross_log"] = safe_log1p(
#         result.get("activity_gross", 0)
#     )

#     result["claim_gross_log"] = safe_log1p(
#         result.get("claim_gross", 0)
#     )

#     result["claim_net_log"] = safe_log1p(
#         result.get("claim_net", 0)
#     )

#     return result


# # ============================================================================
# # HIGH COST FLAG
# # ============================================================================

# def calculate_high_cost_flag(
#     activity_gross: Any,
#     threshold: float = HIGH_COST_THRESHOLD,
# ) -> int:
#     """
#     Calculate the high-cost indicator.

#     Logic
#     -----
#         activity_gross > threshold
#             -> 1

#         activity_gross <= threshold
#             -> 0

#     IMPORTANT
#     ---------
#     The threshold must match the value used during model training.

#     Parameters
#     ----------
#     activity_gross:
#         Raw activity gross amount.

#     threshold:
#         Canonical high-cost threshold.

#     Returns
#     -------
#     int
#         0 or 1
#     """

#     amount = safe_float(activity_gross, default=0.0)

#     threshold_value = safe_float(
#         threshold,
#         default=HIGH_COST_THRESHOLD,
#     )

#     return 1 if amount > threshold_value else 0


# def add_high_cost_flag(
#     data: Dict[str, Any],
#     threshold: float = HIGH_COST_THRESHOLD,
# ) -> Dict[str, Any]:
#     """
#     Add high_cost_flag to a claim dictionary.
#     """

#     result = dict(data)

#     result["high_cost_flag"] = calculate_high_cost_flag(
#         result.get("activity_gross", 0),
#         threshold=threshold,
#     )

#     return result


# # ============================================================================
# # BASIC NUMERIC NORMALIZATION
# # ============================================================================

# def normalize_numeric_features(
#     data: Dict[str, Any],
# ) -> Dict[str, Any]:
#     """
#     Normalize the raw numeric fields used directly by Model 2.

#     Direct numeric model features:
#         activity_quantity
#         patient_age
#         length_of_stay
#         billing_lag_days

#     The engineered fields are also normalized.

#     This function does not perform categorical conversion.
#     """

#     result = dict(data)

#     # ------------------------------------------------------------------------
#     # Direct numeric features
#     # ------------------------------------------------------------------------

#     result["activity_quantity"] = safe_float(
#         result.get("activity_quantity", 0),
#         default=0.0,
#     )

#     result["patient_age"] = safe_float(
#         result.get("patient_age", 0),
#         default=0.0,
#     )

#     result["length_of_stay"] = safe_float(
#         result.get("length_of_stay", 0),
#         default=0.0,
#     )

#     result["billing_lag_days"] = safe_float(
#         result.get("billing_lag_days", 0),
#         default=0.0,
#     )

#     # ------------------------------------------------------------------------
#     # Engineered numeric features
#     # ------------------------------------------------------------------------

#     result["activity_gross_log"] = safe_float(
#         result.get("activity_gross_log", 0),
#         default=0.0,
#     )

#     result["claim_gross_log"] = safe_float(
#         result.get("claim_gross_log", 0),
#         default=0.0,
#     )

#     result["claim_net_log"] = safe_float(
#         result.get("claim_net_log", 0),
#         default=0.0,
#     )

#     result["high_cost_flag"] = int(
#         safe_float(
#             result.get("high_cost_flag", 0),
#             default=0.0,
#         )
#     )

#     return result


# # ============================================================================
# # COMPLETE FEATURE ENGINEERING PIPELINE
# # ============================================================================

# def engineer_features(
#     data: Dict[str, Any],
#     high_cost_threshold: float = HIGH_COST_THRESHOLD,
# ) -> Dict[str, Any]:
#     """
#     Run the complete numeric feature-engineering pipeline.

#     Pipeline
#     --------
#     Raw claim
#         ↓
#     Financial log features
#         ↓
#     High-cost flag
#         ↓
#     Numeric normalization
#         ↓
#     Model-ready dictionary

#     Parameters
#     ----------
#     data:
#         Claim dictionary.

#     high_cost_threshold:
#         Threshold used for high_cost_flag.

#     Returns
#     -------
#     Dict[str, Any]
#         Claim with engineered numeric features.
#     """

#     result = dict(data)

#     # 1. Financial log transformations
#     result = add_financial_log_features(result)

#     # 2. High-cost indicator
#     result = add_high_cost_flag(
#         result,
#         threshold=high_cost_threshold,
#     )

#     # 3. Normalize numeric fields
#     result = normalize_numeric_features(result)

#     return result


# # ============================================================================
# # MODEL FEATURE EXTRACTION
# # ============================================================================

# def extract_numeric_model_features(
#     data: Dict[str, Any],
# ) -> Dict[str, Any]:
#     """
#     Extract ONLY the numeric features that Model 2 expects.

#     This prevents raw fields such as:
#         activity_gross
#         claim_gross
#         claim_net

#     from accidentally entering the model dataframe.
#     """

#     return {
#         feature: data.get(feature, 0)
#         for feature in NUMERIC_FEATURES
#     }


# # ============================================================================
# # TEST / DEBUG
# # ============================================================================

# if __name__ == "__main__":

#     test_claim = {
#         "activity_quantity": 1.0,
#         "activity_gross": 1656.0,
#         "claim_gross": 2470.13,
#         "claim_net": 2470.13,
#         "patient_age": 23,
#         "length_of_stay": 0,
#         "billing_lag_days": 44,
#     }

#     print("=" * 70)
#     print("FEATURE ENGINEERING TEST")
#     print("=" * 70)

#     print("\nRaw input:")
#     for key, value in test_claim.items():
#         print(f"  {key:<25} = {value}")

#     engineered = engineer_features(test_claim)

#     print("\nEngineered output:")

#     for key in [
#         "activity_gross_log",
#         "claim_gross_log",
#         "claim_net_log",
#         "high_cost_flag",
#     ]:
#         print(f"  {key:<25} = {engineered[key]}")

#     print("\nDirect numeric features:")

#     numeric = extract_numeric_model_features(engineered)

#     for key, value in numeric.items():
#         print(f"  {key:<25} = {value}")

#     print("\n" + "=" * 70)
#     print("HIGH COST TEST")
#     print("=" * 70)

#     test_amounts = [
#         100,
#         1656,
#         4999,
#         5000,
#         5000.01,
#         10000,
#     ]

#     for amount in test_amounts:
#         flag = calculate_high_cost_flag(amount)

#         print(
#             f"  activity_gross={amount:>10.2f}"
#             f"  -> high_cost_flag={flag}"
#         )





"""
feature_engineering.py
"""

import pandas as pd
from typing import Dict


def engineer_features(row: Dict):

    row["icd_cpt_domain_match"] = (
        f"{row['icd_category']}_{row['cpt_category']}"
    )

    return row


def build_activity_dataframe(request):

    rows = []

    for activity in request.activities:

        row = {
            "activity_code": activity.activity_code,
            "activity_quantity": activity.activity_quantity,
            "activity_gross": activity.activity_gross,

            "patient_age": request.patient_age,
            "gender": request.gender,
            "nationality": request.nationality,

            "claim_gross": request.claim_gross,
            "claim_net": request.claim_net,

            "encounter_type": request.encounter_type,
            "length_of_stay": request.length_of_stay,

            "clinician_profession": request.clinician_profession,
            "clinician_category": request.clinician_category,

            "facility_type": request.facility_type,
            "payer_classification": request.payer_classification,

            "diagnosis_code": request.diagnosis_code,
            "billing_lag_days": request.billing_lag_days,

            "icd_category": request.icd_category,
            "cpt_category": activity.cpt_category,
        }

        # ADD ENGINEERED FEATURES
        row = engineer_features(row)

        rows.append(row)

    return pd.DataFrame(rows)