"""
model2main.py

FastAPI inference service for Behavioral XGBoost Model 2.

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
Denial probability
    ↓
risk classification
    ↓
PredictionResponse
"""

import os
import joblib

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from validators.schemas import UserClaimInput, PredictionResponse

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

MODEL_PATH = "models/behavioral_xgboost_model2_individual_cats.joblib"

MODEL_NAME = "Behavioral XGBoost Model 2 - Individual Categories"

MODEL_VERSION = "2.0.0"


# ============================================================================
# GLOBAL MODEL
# ============================================================================

MODEL = None


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the XGBoost model once when the FastAPI application starts.
    """

    global MODEL

    print("=" * 70)
    print("Starting Behavioral XGBoost Model 2 API")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found: '{MODEL_PATH}'"
        )

    try:
        print(f"Loading model from: {MODEL_PATH}")

        MODEL = joblib.load(MODEL_PATH)

        print("✅ Model loaded successfully!")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Version: {MODEL_VERSION}")

        contract = get_feature_contract()

        print(
            f"   Expected features: "
            f"{contract['feature_count']}"
        )

        print("=" * 70)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load XGBoost model: {exc}"
        ) from exc

    yield

    # ------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------

    print("Shutting down Model 2 API...")

    MODEL = None


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Healthcare Claim Denial Prediction API - Model 2",
    description=(
        "Production endpoint for real-time healthcare claim "
        "denial risk prediction using Behavioral XGBoost Model 2. "
        "The model uses raw ICD/CPT codes together with "
        "ICD/CPT categorical domains and claim-level behavioral features."
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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Return a cleaner validation response for invalid claim payloads.
    """

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
            ctx_error = error["ctx"].get("error")

            if ctx_error:
                msg = str(ctx_error)

        errors.append(
            {
                "field": field_path,
                "issue": msg,
                "provided_value": error.get("input"),
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
    """
    Execute the complete Model 2 preprocessing and prediction pipeline.

    Pipeline
    --------
    1. Convert raw ICD/CPT codes into categories.
    2. Engineer numeric features.
    3. Build exact 19-feature model DataFrame.
    4. Run XGBoost prediction.
    5. Convert probability into risk level.
    """

    if MODEL is None:
        raise RuntimeError(
            "Model 2 is not loaded."
        )

    # ------------------------------------------------------------------------
    # Copy input so the original request dictionary is never modified.
    # ------------------------------------------------------------------------

    processed_data = dict(data_dict)

    # ------------------------------------------------------------------------
    # STEP 1
    # CATEGORICAL CONVERSION
    #
    # diagnosis_code → icd_category
    # activity_code  → cpt_category
    # ------------------------------------------------------------------------

    processed_data = add_categorical_features(
        processed_data
    )

    # ------------------------------------------------------------------------
    # STEP 2
    # NUMERIC FEATURE ENGINEERING
    #
    # activity_gross → activity_gross_log
    # claim_gross    → claim_gross_log
    # claim_net      → claim_net_log
    # activity_gross → high_cost_flag
    # ------------------------------------------------------------------------

    processed_data = engineer_features(
        processed_data
    )

    # ------------------------------------------------------------------------
    # STEP 3
    # BUILD EXACT MODEL INPUT
    #
    # This selects exactly the 19 features expected by Model 2.
    # ------------------------------------------------------------------------

    X_inference = build_model_dataframe(
        processed_data
    )

    # ------------------------------------------------------------------------
    # STEP 4
    # PREDICTION
    # ------------------------------------------------------------------------

    try:

        probability = float(
            MODEL.predict_proba(X_inference)[0, 1]
        )

    except Exception as exc:

        raise RuntimeError(
            f"XGBoost prediction failed: {exc}"
        ) from exc

    probability_pct = round(
        probability * 100,
        2,
    )

    # ------------------------------------------------------------------------
    # STEP 5
    # risk CLASSIFICATION
    #
    # IMPORTANT:
    # These are application thresholds, not model training thresholds.
    # ------------------------------------------------------------------------

    if probability_pct >= 50.0:

        risk_level = "HIGH"

        is_high_risk = True

        action = (
            "Flagged for manual review. "
            "High predicted denial risk."
        )

    elif probability_pct >= 35.0:

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

    # ------------------------------------------------------------------------
    # STEP 6
    # RESPONSE
    # ------------------------------------------------------------------------

    return PredictionResponse(
        denial_probability_pct=probability_pct,
        is_high_risk=is_high_risk,
        risk_level=risk_level,
        action_recommendation=action,
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
    """
    Predict denial probability for a submitted healthcare claim.
    """

    if MODEL is None:

        raise HTTPException(
            status_code=500,
            detail="Model artifact is not loaded.",
        )

    try:

        # ------------------------------------------------------------
        # Pydantic → dictionary
        #
        # model_dump() is preferred for Pydantic v2.
        # dict() is retained as fallback for Pydantic v1.
        # ------------------------------------------------------------

        if hasattr(claim, "model_dump"):

            claim_data = claim.model_dump()

        else:

            claim_data = claim.dict()

        # ------------------------------------------------------------
        # Run complete inference pipeline
        # ------------------------------------------------------------

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
                f"Error processing claim prediction: "
                f"{str(exc)}"
            ),
        ) from exc


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
def health():
    """
    API health check.
    """

    contract = get_feature_contract()

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "feature_count": contract["feature_count"],
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
    """
    Return the Model 2 feature contract.

    Useful for debugging and frontend integration.
    """

    if MODEL is None:

        raise HTTPException(
            status_code=500,
            detail="Model artifact is not loaded.",
        )

    contract = get_feature_contract()

    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "feature_count": contract["feature_count"],
        "categorical_features": (
            contract["categorical_features"]
        ),
        "numeric_features": (
            contract["numeric_features"]
        ),
        "feature_order": (
            contract["feature_order"]
        ),
    }


# ============================================================================
# ROOT
# ============================================================================

@app.get("/api")
def api_root():
    """
    API information endpoint.
    """

    return {
        "service": "Healthcare Claim Denial Prediction API",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": "running",
        "prediction_endpoint": "/predict_user_claim",
        "health_endpoint": "/health",
        "model_info_endpoint": "/model-info",
    }


# ============================================================================
# STATIC FRONTEND
# ============================================================================
#
# Mount LAST so it does not interfere with API routes.
# ============================================================================

FRONTEND_DIRECTORY = "frontend"

if os.path.isdir(FRONTEND_DIRECTORY):

    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIRECTORY,
            html=True,
        ),
        name="frontend",
    )