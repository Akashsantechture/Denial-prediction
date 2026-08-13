# """
# model2main.py

# FastAPI inference service for Behavioral XGBoost Model 2.

# Pipeline
# --------
# Incoming JSON
#     ↓
# Pydantic schema validation
#     ↓
# categorical_conversion.py
#     ↓
# feature_engineering.py
#     ↓
# feature_contract.py
#     ↓
# XGBoost Model 2
#     ↓
# Denial probability
#     ↓
# Risk classification
#     ↓
# PredictionResponse
# """

# import os
# import joblib

# from contextlib import asynccontextmanager

# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse
# from fastapi.exceptions import RequestValidationError
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware

# from validators.schemas import UserClaimInput, PredictionResponse

# from preprocessing.categorial_conversion import (
#     add_categorical_features,
# )

# from preprocessing.feature_engineering import (
#     engineer_features,
# )

# from preprocessing.feature_contract import (
#     build_model_dataframe,
#     get_feature_contract,
# )


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# MODEL_PATH = "models/behavioral_xgboost_model2_individual_cats.joblib"

# MODEL_NAME = "Behavioral XGBoost Model 2 - Individual Categories"

# MODEL_VERSION = "2.0.0"


# # ============================================================================
# # GLOBAL MODEL
# # ============================================================================

# MODEL = None


# # ============================================================================
# # FASTAPI LIFESPAN
# # ============================================================================

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Load the XGBoost model once when the FastAPI application starts.
#     """

#     global MODEL

#     print("=" * 70)
#     print("Starting Behavioral XGBoost Model 2 API")
#     print("=" * 70)

#     if not os.path.exists(MODEL_PATH):
#         raise RuntimeError(
#             f"Model artifact not found: '{MODEL_PATH}'"
#         )

#     try:
#         print(f"Loading model from: {MODEL_PATH}")

#         MODEL = joblib.load(MODEL_PATH)

#         print("✅ Model loaded successfully!")
#         print(f"   Model: {MODEL_NAME}")
#         print(f"   Version: {MODEL_VERSION}")

#         contract = get_feature_contract()

#         print(
#             f"   Expected features: "
#             f"{contract['feature_count']}"
#         )

#         print("=" * 70)

#     except Exception as exc:
#         raise RuntimeError(
#             f"Failed to load XGBoost model: {exc}"
#         ) from exc

#     yield

#     # ------------------------------------------------------------
#     # Shutdown
#     # ------------------------------------------------------------

#     print("Shutting down Model 2 API...")

#     MODEL = None


# # ============================================================================
# # FASTAPI APPLICATION
# # ============================================================================

# app = FastAPI(
#     title="Healthcare Claim Denial Prediction API - Model 2",
#     description=(
#         "Production endpoint for real-time healthcare claim "
#         "denial risk prediction using Behavioral XGBoost Model 2. "
#         "The model uses raw ICD/CPT codes together with "
#         "ICD/CPT categorical domains and claim-level behavioral features."
#     ),
#     version=MODEL_VERSION,
#     lifespan=lifespan,
# )


# # ============================================================================
# # CORS
# # ============================================================================

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ============================================================================
# # REQUEST VALIDATION ERROR HANDLER
# # ============================================================================

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(
#     request: Request,
#     exc: RequestValidationError,
# ):
#     """
#     Return a cleaner validation response for invalid claim payloads.
#     """

#     errors = []

#     for error in exc.errors():

#         field_path = " → ".join(
#             str(loc)
#             for loc in error["loc"]
#             if loc != "body"
#         )

#         msg = error.get(
#             "msg",
#             "Invalid value",
#         )

#         if (
#             error["type"] == "value_error"
#             and "ctx" in error
#         ):
#             ctx_error = error["ctx"].get("error")

#             if ctx_error:
#                 msg = str(ctx_error)

#         errors.append(
#             {
#                 "field": field_path,
#                 "issue": msg,
#                 "provided_value": error.get("input"),
#             }
#         )

#     return JSONResponse(
#         status_code=422,
#         content={
#             "status": "validation_error",
#             "message": (
#                 "Request failed schema validation. "
#                 "Please correct the following fields "
#                 "before resubmitting."
#             ),
#             "errors": errors,
#         },
#     )


# # ============================================================================
# # MODEL INFERENCE PIPELINE
# # ============================================================================

# def run_model2_inference(
#     data_dict: dict,
# ) -> PredictionResponse:
#     """
#     Execute the complete Model 2 preprocessing and prediction pipeline.

#     Pipeline
#     --------
#     1. Convert raw ICD/CPT codes into categories.
#     2. Engineer numeric features.
#     3. Build exact 19-feature model DataFrame.
#     4. Run XGBoost prediction.
#     5. Convert probability into risk level.
#     """

#     if MODEL is None:
#         raise RuntimeError(
#             "Model 2 is not loaded."
#         )

#     # ------------------------------------------------------------------------
#     # Copy input so the original request dictionary is never modified.
#     # ------------------------------------------------------------------------

#     processed_data = dict(data_dict)

#     # ------------------------------------------------------------------------
#     # STEP 1
#     # CATEGORICAL CONVERSION
#     #
#     # diagnosis_code → icd_category
#     # activity_code  → cpt_category
#     # ------------------------------------------------------------------------

#     processed_data = add_categorical_features(
#         processed_data
#     )

#     # ------------------------------------------------------------------------
#     # STEP 2
#     # NUMERIC FEATURE ENGINEERING
#     #
#     # activity_gross → activity_gross_log
#     # claim_gross    → claim_gross_log
#     # claim_net      → claim_net_log
#     # activity_gross → high_cost_flag
#     # ------------------------------------------------------------------------

#     processed_data = engineer_features(
#         processed_data
#     )

#     # ------------------------------------------------------------------------
#     # STEP 3
#     # BUILD EXACT MODEL INPUT
#     #
#     # This selects exactly the 19 features expected by Model 2.
#     # ------------------------------------------------------------------------

#     X_inference = build_model_dataframe(
#         processed_data
#     )

#     # ------------------------------------------------------------------------
#     # STEP 4
#     # PREDICTION
#     # ------------------------------------------------------------------------

#     try:

#         probability = float(
#             MODEL.predict_proba(X_inference)[0, 1]
#         )

#     except Exception as exc:

#         raise RuntimeError(
#             f"XGBoost prediction failed: {exc}"
#         ) from exc

#     probability_pct = round(
#         probability * 100,
#         2,
#     )

#     # ------------------------------------------------------------------------
#     # STEP 5
#     # RISK CLASSIFICATION
#     #
#     # IMPORTANT:
#     # These are application thresholds, not model training thresholds.
#     # ------------------------------------------------------------------------

#     if probability_pct >= 50.0:

#         risk_level = "HIGH"

#         is_high_risk = True

#         action = (
#             "Flagged for manual review. "
#             "High predicted denial risk."
#         )

#     elif probability_pct >= 35.0:

#         risk_level = "MEDIUM"

#         is_high_risk = False

#         action = (
#             "Moderate predicted denial risk. "
#             "Review claim details before submission."
#         )

#     else:

#         risk_level = "LOW"

#         is_high_risk = False

#         action = (
#             "Low predicted denial risk. "
#             "Standard processing recommended."
#         )

#     # ------------------------------------------------------------------------
#     # STEP 6
#     # RESPONSE
#     # ------------------------------------------------------------------------

#     return PredictionResponse(
#         denial_probability_pct=probability_pct,
#         is_high_risk=is_high_risk,
#         risk_level=risk_level,
#         action_recommendation=action,
#     )


# # ============================================================================
# # PREDICTION ENDPOINT
# # ============================================================================

# @app.post(
#     "/predict_user_claim",
#     response_model=PredictionResponse,
# )
# def predict_user_claim_risk(
#     claim: UserClaimInput,
# ):
#     """
#     Predict denial probability for a submitted healthcare claim.
#     """

#     if MODEL is None:

#         raise HTTPException(
#             status_code=500,
#             detail="Model artifact is not loaded.",
#         )

#     try:

#         # ------------------------------------------------------------
#         # Pydantic → dictionary
#         #
#         # model_dump() is preferred for Pydantic v2.
#         # dict() is retained as fallback for Pydantic v1.
#         # ------------------------------------------------------------

#         if hasattr(claim, "model_dump"):

#             claim_data = claim.model_dump()

#         else:

#             claim_data = claim.dict()

#         # ------------------------------------------------------------
#         # Run complete inference pipeline
#         # ------------------------------------------------------------

#         return run_model2_inference(
#             claim_data
#         )

#     except ValueError as exc:

#         raise HTTPException(
#             status_code=422,
#             detail=str(exc),
#         ) from exc

#     except Exception as exc:

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 f"Error processing claim prediction: "
#                 f"{str(exc)}"
#             ),
#         ) from exc


# # ============================================================================
# # HEALTH CHECK
# # ============================================================================

# @app.get("/health")
# def health():
#     """
#     API health check.
#     """

#     contract = get_feature_contract()

#     return {
#         "status": "healthy",
#         "model_loaded": MODEL is not None,
#         "model": MODEL_NAME,
#         "version": MODEL_VERSION,
#         "feature_count": contract["feature_count"],
#         "categorical_feature_count": len(
#             contract["categorical_features"]
#         ),
#         "numeric_feature_count": len(
#             contract["numeric_features"]
#         ),
#     }


# # ============================================================================
# # MODEL INFORMATION
# # ============================================================================

# @app.get("/model-info")
# def model_info():
#     """
#     Return the Model 2 feature contract.

#     Useful for debugging and frontend integration.
#     """

#     if MODEL is None:

#         raise HTTPException(
#             status_code=500,
#             detail="Model artifact is not loaded.",
#         )

#     contract = get_feature_contract()

#     return {
#         "model": MODEL_NAME,
#         "version": MODEL_VERSION,
#         "feature_count": contract["feature_count"],
#         "categorical_features": (
#             contract["categorical_features"]
#         ),
#         "numeric_features": (
#             contract["numeric_features"]
#         ),
#         "feature_order": (
#             contract["feature_order"]
#         ),
#     }


# # ============================================================================
# # ROOT
# # ============================================================================

# @app.get("/api")
# def api_root():
#     """
#     API information endpoint.
#     """

#     return {
#         "service": "Healthcare Claim Denial Prediction API",
#         "model": MODEL_NAME,
#         "version": MODEL_VERSION,
#         "status": "running",
#         "prediction_endpoint": "/predict_user_claim",
#         "health_endpoint": "/health",
#         "model_info_endpoint": "/model-info",
#     }


# # ============================================================================
# # STATIC FRONTEND
# # ============================================================================
# #
# # Mount LAST so it does not interfere with API routes.
# # ============================================================================

# FRONTEND_DIRECTORY = "frontend3"

# if os.path.isdir(FRONTEND_DIRECTORY):

#     app.mount(
#         "/",
#         StaticFiles(
#             directory=FRONTEND_DIRECTORY,
#             html=True,
#         ),
#         name="frontend",
#     )

"""
model2main.py

FastAPI inference + explainability service for Behavioral XGBoost Model 2.

Pipeline
--------
Incoming JSON
    ↓
Pydantic schema validation
    ↓
categorical_conversion.py
    ↓
feature_engineering.py
    ↓
feature_contract.py
    ↓
XGBoost Model 2
    ↓
Actual prediction
    ↓
SHAP TreeExplainer
    ↓
Prediction + human-readable explanation
    ↓
Frontend charts/cards/tabs

Important
---------
The frontend must NOT estimate SHAP values.

FastAPI is the single source of truth for:
- prediction probability
- risk classification
- actual per-feature SHAP values
- SHAP base value
- top drivers
- supporting factors
- claim-level SHAP interactions
- model feature importance metadata
"""

import math
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from validators.schemas import UserClaimInput

from preprocessing.categorial_conversion import (
    add_categorical_features,
)

from preprocessing.feature_engineering import (
    engineer_features,
)

from preprocessing.feature_contract import (
    build_model_dataframe,
    get_feature_contract,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = (
    "models/"
    "behavioral_xgboost_model2_individual_cats.joblib"
)

MODEL_NAME = (
    "Behavioral XGBoost Model 2 - Individual Categories"
)

MODEL_VERSION = "2.0.0"

# Application-level risk thresholds.
# These are NOT model training thresholds.
HIGH_RISK_THRESHOLD = 50.0
MEDIUM_RISK_THRESHOLD = 35.0

# Number of SHAP features shown prominently by the frontend.
TOP_SHAP_FEATURES = 8

# Number of claim-level SHAP interactions returned.
TOP_INTERACTIONS = 8


# ============================================================================
# FEATURE DISPLAY METADATA
# ============================================================================
#
# These names are for UI readability only.
# They do not change model feature names or model calculations.
# ============================================================================

FEATURE_DISPLAY_NAMES = {
    "activity_code": "Activity / CPT Code",
    "activity_gross_log": "Activity Gross Amount",
    "activity_quantity": "Activity Quantity",
    "claim_gross_log": "Claim Gross Amount",
    "claim_net_log": "Net Claim Amount",
    "billing_lag_days": "Billing Lag",
    "cpt_category": "CPT Category",
    "icd_category": "ICD Category",
    "diagnosis_code": "Diagnosis Code",
    "diagnosis_type": "Diagnosis Type",
    "clinician_category": "Clinician Category",
    "clinician_profession": "Clinician Profession",
    "encounter_type": "Encounter Type",
    "nationality": "Nationality",
    "patient_age": "Patient Age",
    "gender": "Gender",
    "length_of_stay": "Length of Stay",
    "high_cost_flag": "High-Cost Flag",
    "facility_type": "Facility Type",
}


FEATURE_EXPLANATIONS = {
    "activity_code": (
        "The submitted activity/CPT code contributed "
        "{direction} to the model prediction."
    ),
    "activity_gross_log": (
        "The activity gross amount contributed "
        "{direction} to the model prediction."
    ),
    "activity_quantity": (
        "The activity quantity contributed "
        "{direction} to the model prediction."
    ),
    "claim_gross_log": (
        "The claim gross amount contributed "
        "{direction} to the model prediction."
    ),
    "claim_net_log": (
        "The net claim amount contributed "
        "{direction} to the model prediction."
    ),
    "billing_lag_days": (
        "The billing lag contributed "
        "{direction} to the model prediction."
    ),
    "cpt_category": (
        "The CPT category contributed "
        "{direction} to the model prediction."
    ),
    "icd_category": (
        "The ICD category contributed "
        "{direction} to the model prediction."
    ),
    "diagnosis_code": (
        "The diagnosis code contributed "
        "{direction} to the model prediction."
    ),
    "diagnosis_type": (
        "The diagnosis type contributed "
        "{direction} to the model prediction."
    ),
    "clinician_category": (
        "The clinician category contributed "
        "{direction} to the model prediction."
    ),
    "clinician_profession": (
        "The clinician profession contributed "
        "{direction} to the model prediction."
    ),
    "encounter_type": (
        "The encounter type contributed "
        "{direction} to the model prediction."
    ),
    "nationality": (
        "The nationality feature contributed "
        "{direction} to the model prediction."
    ),
    "patient_age": (
        "The patient's age contributed "
        "{direction} to the model prediction."
    ),
    "gender": (
        "The gender feature contributed "
        "{direction} to the model prediction."
    ),
    "length_of_stay": (
        "The length of stay contributed "
        "{direction} to the model prediction."
    ),
    "high_cost_flag": (
        "The high-cost flag contributed "
        "{direction} to the model prediction."
    ),
    "facility_type": (
        "The facility type contributed "
        "{direction} to the model prediction."
    ),
}


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class ShapFeatureExplanation(BaseModel):
    """
    One actual model feature explanation.

    shap_value is in the model's SHAP output space.
    For a binary XGBoost classifier using the default TreeExplainer
    raw output, this is normally the model margin/log-odds contribution.
    """

    feature: str
    display_name: str
    raw_value: Any = None
    shap_value: float
    absolute_shap: float
    direction: str
    impact_level: str
    contribution_share_pct: float
    explanation: str


class ShapInteractionExplanation(BaseModel):
    feature_1: str
    feature_1_display_name: str
    feature_2: str
    feature_2_display_name: str
    interaction_value: float
    absolute_interaction: float
    direction: str
    explanation: str


class PredictionExplanation(BaseModel):
    """
    Complete explanation returned by FastAPI for the frontend.
    """

    base_value: float
    final_model_value: float
    model_output_space: str = "raw_margin_log_odds"

    top_drivers: list[ShapFeatureExplanation] = Field(
        default_factory=list
    )

    supporting_factors: list[ShapFeatureExplanation] = Field(
        default_factory=list
    )

    all_features: list[ShapFeatureExplanation] = Field(
        default_factory=list
    )

    interactions: list[ShapInteractionExplanation] = Field(
        default_factory=list
    )


class PredictionResponse(BaseModel):
    """
    Backward-compatible prediction fields plus actual SHAP explanation.
    """

    denial_probability_pct: float
    is_high_risk: bool
    risk_level: str
    action_recommendation: str

    explanation: PredictionExplanation


# ============================================================================
# GLOBAL MODEL OBJECTS
# ============================================================================

MODEL = None
SHAP_EXPLAINER = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _safe_float(value: Any) -> Optional[float]:
    """
    Convert a value to a JSON-safe float.
    """
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return None


def _json_safe_value(value: Any) -> Any:
    """
    Convert numpy/pandas values into JSON-safe Python values.
    """
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (np.ndarray, list, tuple)):
        return [
            _json_safe_value(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def _display_name(feature: str) -> str:
    """
    Convert model feature name into user-facing label.
    """
    return FEATURE_DISPLAY_NAMES.get(
        feature,
        feature.replace("_", " ").title(),
    )


def _impact_level(abs_shap: float, abs_values: list[float]) -> str:
    """
    Relative impact label for UI cards.

    This is presentation metadata, not a model decision.
    """
    if not abs_values:
        return "LOW"

    sorted_values = sorted(
        abs_values,
        reverse=True,
    )

    if abs_shap >= np.percentile(
        sorted_values,
        75,
    ):
        return "HIGH"

    if abs_shap >= np.percentile(
        sorted_values,
        40,
    ):
        return "MODERATE"

    return "LOW"


def _direction_from_shap(
    shap_value: float,
) -> str:
    """
    Convert SHAP sign into user-friendly direction.
    """
    if shap_value > 0:
        return "increases_denial_risk"

    if shap_value < 0:
        return "reduces_denial_risk"

    return "neutral"


def _direction_text(
    shap_value: float,
) -> str:
    """
    Plain-English direction used in explanations.
    """
    if shap_value > 0:
        return "increasing the predicted denial risk"

    if shap_value < 0:
        return "reducing the predicted denial risk"

    return "having little or no effect on the prediction"


def _feature_explanation(
    feature: str,
    shap_value: float,
) -> str:
    """
    Generate a conservative, model-grounded explanation.

    Important:
    This does not claim causality or payer policy.
    It only describes the direction of the model contribution.
    """
    template = FEATURE_EXPLANATIONS.get(
        feature,
        "This feature contributed {direction} to the model prediction.",
    )

    return template.format(
        direction=_direction_text(shap_value)
    )


def _impact_badge(
    abs_shap: float,
    all_abs_values: list[float],
) -> str:
    return _impact_level(
        abs_shap,
        all_abs_values,
    )


def _extract_binary_shap_values(
    shap_result: Any,
) -> tuple[np.ndarray, float]:
    """
    Normalize SHAP output across common SHAP versions.

    Returns:
        values: shape (1, feature_count)
        base_value: scalar
    """

    values = getattr(
        shap_result,
        "values",
        shap_result,
    )

    base_values = getattr(
        shap_result,
        "base_values",
        None,
    )

    values = np.asarray(values)

    # Newer SHAP binary classifier output can be:
    # (samples, features)
    # or occasionally (samples, features, outputs).
    if values.ndim == 3:
        # Binary positive class.
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        else:
            values = values[:, :, -1]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP value shape: {values.shape}"
        )

    if base_values is None:
        raise RuntimeError(
            "SHAP explainer did not return a base value."
        )

    base_values = np.asarray(base_values)

    if base_values.ndim == 0:
        base_value = float(base_values)

    elif base_values.ndim == 1:
        if len(base_values) == 1:
            base_value = float(base_values[0])
        elif len(base_values) == 2:
            base_value = float(base_values[1])
        else:
            base_value = float(base_values[0])

    else:
        # Usually (samples,) or (samples, outputs)
        row = base_values[0]

        row = np.asarray(row)

        if row.ndim == 0:
            base_value = float(row)

        elif row.size == 1:
            base_value = float(row.flat[0])

        elif row.size == 2:
            base_value = float(row.flat[1])

        else:
            base_value = float(row.flat[0])

    return values, base_value


def _build_feature_explanations(
    X_inference: pd.DataFrame,
    shap_values: np.ndarray,
) -> list[ShapFeatureExplanation]:
    """
    Convert raw SHAP values into frontend-friendly feature objects.
    """

    feature_names = list(X_inference.columns)

    if shap_values.shape[1] != len(feature_names):
        raise RuntimeError(
            "SHAP feature count does not match model feature count. "
            f"SHAP={shap_values.shape[1]}, "
            f"features={len(feature_names)}."
        )

    row_values = shap_values[0]

    abs_values = [
        abs(float(value))
        for value in row_values
    ]

    total_abs = sum(abs_values)

    explanations = []

    for feature, shap_value in zip(
        feature_names,
        row_values,
    ):
        shap_value = float(shap_value)

        raw_value = X_inference.iloc[0][feature]

        if total_abs > 0:
            share = (
                abs(shap_value)
                / total_abs
                * 100.0
            )
        else:
            share = 0.0

        absolute_shap = abs(shap_value)

        explanations.append(
            ShapFeatureExplanation(
                feature=feature,
                display_name=_display_name(feature),
                raw_value=_json_safe_value(raw_value),
                shap_value=round(
                    shap_value,
                    6,
                ),
                absolute_shap=round(
                    absolute_shap,
                    6,
                ),
                direction=_direction_from_shap(
                    shap_value
                ),
                impact_level=_impact_badge(
                    absolute_shap,
                    abs_values,
                ),
                contribution_share_pct=round(
                    share,
                    2,
                ),
                explanation=_feature_explanation(
                    feature,
                    shap_value,
                ),
            )
        )

    explanations.sort(
        key=lambda item: item.absolute_shap,
        reverse=True,
    )

    return explanations


def _build_interaction_explanations(
    X_inference: pd.DataFrame,
) -> list[ShapInteractionExplanation]:
    """
    Calculate claim-level SHAP interaction values.

    Only the strongest interaction pairs are returned.
    """

    if SHAP_EXPLAINER is None:
        return []

    try:
        interaction_result = (
            SHAP_EXPLAINER.shap_interaction_values(
                X_inference
            )
        )

    except Exception as exc:
        print(
            "⚠️ SHAP interaction calculation failed: "
            f"{exc}"
        )
        return []

    interactions = np.asarray(
        interaction_result
    )

    # Common binary XGBoost format:
    # (samples, features, features)
    if interactions.ndim == 4:
        if interactions.shape[-1] == 2:
            interactions = interactions[:, :, :, 1]
        else:
            interactions = interactions[:, :, :, -1]

    if interactions.ndim != 3:
        return []

    matrix = interactions[0]

    feature_names = list(
        X_inference.columns
    )

    pairs = []

    for i in range(len(feature_names)):

        for j in range(i + 1, len(feature_names)):

            value = float(matrix[i, j])

            if not math.isfinite(value):
                continue

            pairs.append(
                (
                    abs(value),
                    value,
                    feature_names[i],
                    feature_names[j],
                )
            )

    pairs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    output = []

    for (
        absolute_value,
        value,
        feature_1,
        feature_2,
    ) in pairs[:TOP_INTERACTIONS]:

        if value > 0:
            direction = "increases_denial_risk"
            text = (
                "The combined interaction between "
                f"{_display_name(feature_1)} and "
                f"{_display_name(feature_2)} pushes the "
                "model prediction toward higher denial risk."
            )

        elif value < 0:
            direction = "reduces_denial_risk"
            text = (
                "The combined interaction between "
                f"{_display_name(feature_1)} and "
                f"{_display_name(feature_2)} pushes the "
                "model prediction toward lower denial risk."
            )

        else:
            direction = "neutral"
            text = (
                "The interaction between these two features "
                "has little effect on this prediction."
            )

        output.append(
            ShapInteractionExplanation(
                feature_1=feature_1,
                feature_1_display_name=_display_name(
                    feature_1
                ),
                feature_2=feature_2,
                feature_2_display_name=_display_name(
                    feature_2
                ),
                interaction_value=round(
                    value,
                    6,
                ),
                absolute_interaction=round(
                    absolute_value,
                    6,
                ),
                direction=direction,
                explanation=text,
            )
        )

    return output


def _build_explanation(
    X_inference: pd.DataFrame,
) -> PredictionExplanation:
    """
    Run actual TreeExplainer and build frontend explanation.
    """

    if SHAP_EXPLAINER is None:
        raise RuntimeError(
            "SHAP explainer is not loaded."
        )

    try:
        shap_result = SHAP_EXPLAINER(
            X_inference,
            check_additivity=False,
        )

    except TypeError:
        # Compatibility fallback for older SHAP versions.
        shap_result = SHAP_EXPLAINER(
            X_inference
        )

    shap_values, base_value = (
        _extract_binary_shap_values(
            shap_result
        )
    )

    feature_explanations = (
        _build_feature_explanations(
            X_inference,
            shap_values,
        )
    )

    top_drivers = [
        item
        for item in feature_explanations
        if item.shap_value > 0
    ][:TOP_SHAP_FEATURES]

    supporting_factors = [
        item
        for item in feature_explanations
        if item.shap_value < 0
    ][:TOP_SHAP_FEATURES]

    # The final raw model value should agree with:
    # base value + sum(SHAP values).
    final_model_value = (
        base_value
        + float(np.sum(shap_values[0]))
    )

    interactions = _build_interaction_explanations(
        X_inference
    )

    return PredictionExplanation(
        base_value=round(
            float(base_value),
            6,
        ),
        final_model_value=round(
            final_model_value,
            6,
        ),
        model_output_space=(
            "raw_margin_log_odds"
        ),
        top_drivers=top_drivers,
        supporting_factors=supporting_factors,
        all_features=feature_explanations,
        interactions=interactions,
    )


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model and SHAP explainer once at startup.
    """

    global MODEL
    global SHAP_EXPLAINER

    print("=" * 70)
    print(
        "Starting Behavioral XGBoost Model 2 API"
    )
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found: '{MODEL_PATH}'"
        )

    try:

        print(
            f"Loading model from: {MODEL_PATH}"
        )

        MODEL = joblib.load(
            MODEL_PATH
        )

        print(
            "✅ Model loaded successfully!"
        )

        print(
            f"   Model: {MODEL_NAME}"
        )

        print(
            f"   Version: {MODEL_VERSION}"
        )

        contract = get_feature_contract()

        print(
            "   Expected features: "
            f"{contract['feature_count']}"
        )

        # ------------------------------------------------------------
        # Load SHAP TreeExplainer once.
        # ------------------------------------------------------------

        print(
            "Loading SHAP TreeExplainer..."
        )

        SHAP_EXPLAINER = shap.TreeExplainer(
            MODEL
        )

        print(
            "✅ SHAP explainer loaded successfully!"
        )

        print("=" * 70)

    except Exception as exc:

        raise RuntimeError(
            f"Failed to initialize Model 2 API: {exc}"
        ) from exc

    yield

    print(
        "Shutting down Model 2 API..."
    )

    MODEL = None
    SHAP_EXPLAINER = None


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=(
        "Healthcare Claim Denial Prediction API - Model 2"
    ),
    description=(
        "Real-time healthcare claim denial prediction "
        "using Behavioral XGBoost Model 2 with actual "
        "SHAP-based explainability."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST VALIDATION ERROR HANDLER
# ============================================================================

@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    errors = []

    for error in exc.errors():

        field_path = " → ".join(
            str(loc)
            for loc in error["loc"]
            if loc != "body"
        )

        msg = error.get(
            "msg",
            "Invalid value",
        )

        if (
            error["type"] == "value_error"
            and "ctx" in error
        ):

            ctx_error = (
                error["ctx"].get("error")
            )

            if ctx_error:
                msg = str(ctx_error)

        errors.append(
            {
                "field": field_path,
                "issue": msg,
                "provided_value": (
                    error.get("input")
                ),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "status": "validation_error",
            "message": (
                "Request failed schema validation. "
                "Please correct the following fields "
                "before resubmitting."
            ),
            "errors": errors,
        },
    )


# ============================================================================
# MODEL INFERENCE PIPELINE
# ============================================================================

def run_model2_inference(
    data_dict: dict,
) -> PredictionResponse:

    if MODEL is None:
        raise RuntimeError(
            "Model 2 is not loaded."
        )

    if SHAP_EXPLAINER is None:
        raise RuntimeError(
            "SHAP explainer is not loaded."
        )

    # ------------------------------------------------------------
    # Never modify original request data.
    # ------------------------------------------------------------

    processed_data = dict(
        data_dict
    )

    # ------------------------------------------------------------
    # STEP 1
    # Convert raw ICD/CPT codes into categories.
    # ------------------------------------------------------------

    processed_data = add_categorical_features(
        processed_data
    )

    # ------------------------------------------------------------
    # STEP 2
    # Engineer numeric model features.
    # ------------------------------------------------------------

    processed_data = engineer_features(
        processed_data
    )

    # ------------------------------------------------------------
    # STEP 3
    # Build exact model DataFrame.
    # ------------------------------------------------------------

    X_inference = build_model_dataframe(
        processed_data
    )

    # ------------------------------------------------------------
    # STEP 4
    # Prediction.
    # ------------------------------------------------------------

    try:

        probability = float(
            MODEL.predict_proba(
                X_inference
            )[0, 1]
        )

    except Exception as exc:

        raise RuntimeError(
            f"XGBoost prediction failed: {exc}"
        ) from exc

    probability = min(
        max(probability, 0.0),
        1.0,
    )

    probability_pct = round(
        probability * 100,
        2,
    )

    # ------------------------------------------------------------
    # STEP 5
    # Risk classification.
    #
    # Application thresholds only.
    # ------------------------------------------------------------

    if probability_pct >= HIGH_RISK_THRESHOLD:

        risk_level = "HIGH"
        is_high_risk = True

        action = (
            "Flagged for manual review. "
            "High predicted denial risk."
        )

    elif probability_pct >= MEDIUM_RISK_THRESHOLD:

        risk_level = "MEDIUM"
        is_high_risk = False

        action = (
            "Moderate predicted denial risk. "
            "Review claim details before submission."
        )

    else:

        risk_level = "LOW"
        is_high_risk = False

        action = (
            "Low predicted denial risk. "
            "Standard processing recommended."
        )

    # ------------------------------------------------------------
    # STEP 6
    # Actual SHAP explanation.
    # ------------------------------------------------------------

    explanation = _build_explanation(
        X_inference
    )

    # ------------------------------------------------------------
    # STEP 7
    # Return prediction + explanation.
    # ------------------------------------------------------------

    return PredictionResponse(
        denial_probability_pct=probability_pct,
        is_high_risk=is_high_risk,
        risk_level=risk_level,
        action_recommendation=action,
        explanation=explanation,
    )


# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post(
    "/predict_user_claim",
    response_model=PredictionResponse,
)
def predict_user_claim_risk(
    claim: UserClaimInput,
):

    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifact is not loaded.",
        )

    try:

        if hasattr(
            claim,
            "model_dump",
        ):

            claim_data = (
                claim.model_dump()
            )

        else:

            claim_data = (
                claim.dict()
            )

        return run_model2_inference(
            claim_data
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Error processing claim prediction: "
                f"{str(exc)}"
            ),
        ) from exc


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
def health():

    contract = get_feature_contract()

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "shap_loaded": (
            SHAP_EXPLAINER is not None
        ),
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "feature_count": (
            contract["feature_count"]
        ),
        "categorical_feature_count": len(
            contract["categorical_features"]
        ),
        "numeric_feature_count": len(
            contract["numeric_features"]
        ),
    }


# ============================================================================
# MODEL INFORMATION
# ============================================================================

@app.get("/model-info")
def model_info():

    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifact is not loaded.",
        )

    contract = get_feature_contract()

    # ------------------------------------------------------------
    # Extract model feature importance when available.
    #
    # For XGBoost, feature_importances_ is normalized by the model
    # and is useful for global feature ranking. This is NOT the same
    # thing as an individual claim's SHAP value.
    # ------------------------------------------------------------

    feature_importance = []

    raw_importances = getattr(
        MODEL,
        "feature_importances_",
        None,
    )

    if raw_importances is not None:

        feature_order = list(
            contract["feature_order"]
        )

        for feature, importance in zip(
            feature_order,
            raw_importances,
        ):

            feature_importance.append(
                {
                    "feature": feature,
                    "display_name": _display_name(
                        feature
                    ),
                    "importance_pct": round(
                        float(importance) * 100,
                        4,
                    ),
                }
            )

        feature_importance.sort(
            key=lambda item: item[
                "importance_pct"
            ],
            reverse=True,
        )

    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "explainability": {
            "shap_available": (
                SHAP_EXPLAINER is not None
            ),
            "method": (
                "SHAP TreeExplainer"
            ),
            "individual_claim_values": (
                "actual_model_shap_values"
            ),
            "frontend_shap_estimation": False,
        },
        "risk_thresholds": {
            "medium": MEDIUM_RISK_THRESHOLD,
            "high": HIGH_RISK_THRESHOLD,
        },
        "feature_count": (
            contract["feature_count"]
        ),
        "categorical_features": (
            contract["categorical_features"]
        ),
        "numeric_features": (
            contract["numeric_features"]
        ),
        "feature_order": (
            contract["feature_order"]
        ),
        "feature_importance": (
            feature_importance
        ),
    }


# ============================================================================
# ROOT
# ============================================================================

@app.get("/api")
def api_root():

    return {
        "service": (
            "Healthcare Claim Denial "
            "Prediction API"
        ),
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": "running",
        "prediction_endpoint": (
            "/predict_user_claim"
        ),
        "health_endpoint": "/health",
        "model_info_endpoint": (
            "/model-info"
        ),
        "explainability": (
            "Actual SHAP TreeExplainer"
        ),
    }


# ============================================================================
# STATIC FRONTEND
# ============================================================================
#
# Mount LAST so it does not interfere with API routes.
# ============================================================================

FRONTEND_DIRECTORY = "frontend3"

if os.path.isdir(
    FRONTEND_DIRECTORY
):

    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIRECTORY,
            html=True,
        ),
        name="frontend",
    )
