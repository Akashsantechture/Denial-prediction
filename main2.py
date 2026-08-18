import os
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from validators.schemas import ClaimInput, UserClaimInput, PredictionResponse

MODEL = None
PREPROCESSOR = None

# MODEL_PATH = "models/xgboost_denial_model.joblib"
MODEL_PATH = "models/xgboost_denial2.joblib"
PREPROCESSOR_PATH = "models/preprocessor_pipeline.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, PREPROCESSOR
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(f"Artifacts not found! Ensure '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' exist.")
    
    PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
    MODEL = joblib.load(MODEL_PATH)
    print("✅ Preprocessor and XGBoost Model loaded successfully!")
    yield

app = FastAPI(
    title="Healthcare Claim Denial Prediction API",
    description="Production endpoint for real-time RCM claim denial risk scoring.",
    version="1.0.0",
    lifespan=lifespan
)

# Custom exception handler for Pydantic validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Intercepts Pydantic validation errors and returns a clean, structured response.
    """
    errors = []
    for error in exc.errors():
        field_path = " → ".join(str(loc) for loc in error["loc"] if loc != "body")
        
        # Extract the human-readable message
        msg = error.get("msg", "Invalid value")
        
        # If it's a ValueError from our custom validators, extract the clean message
        if error["type"] == "value_error" and "ctx" in error:
            ctx_error = error["ctx"].get("error")
            if ctx_error:
                msg = str(ctx_error)
        
        errors.append({
            "field": field_path,
            "issue": msg,
            "provided_value": error.get("input")
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "validation_error",
            "message": "Request failed schema validation. Please correct the following fields before resubmitting.",
            "errors": errors
        }
    )

def run_pipeline_inference(data_dict: dict) -> PredictionResponse:
    df_input = pd.DataFrame([data_dict])

    df_input["cpt_icd_pair"] = df_input["activity_code"] + "_" + df_input["primary_diagnosis_code"]
    df_input["pa_exceeds_1000_flag"] = np.where(
        (df_input["activity_gross"] >= 1000) | (df_input["gross_amount"] >= 1000), 1, 0
    )

    monetary_cols = ["activity_gross", "gross_amount", "net_amount"]
    for col in monetary_cols:
        df_input[f"{col}_log"] = np.log1p(np.maximum(0, df_input[col]))

    X_raw = df_input.drop(columns=monetary_cols)
    X_transformed = PREPROCESSOR.transform(X_raw)

    prob = float(MODEL.predict_proba(X_transformed)[0, 1])
    prob_pct = round(prob * 100, 2)

    # --- RECALIBRATED risk THRESHOLDS ---
    if prob_pct >= 60.0:
        risk_level = "HIGH"
        is_high_risk = True
        action = "Flagged for manual review. Check medical necessity documentation and prior authorization requirements."
    elif prob_pct >= 35.0:  # <--- FIXED THRESHOLD FROM 45.0 TO 35.0
        risk_level = "MEDIUM"
        is_high_risk = False
        action = "Moderate risk. Verify CPT/ICD pairing before submission."
    else:
        risk_level = "LOW"
        is_high_risk = False
        action = "Ready for submission. Standard processing expected."

    return PredictionResponse(
        denial_probability_pct=prob_pct,
        is_high_risk=is_high_risk,
        risk_level=risk_level,
        action_recommendation=action
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_claim_risk(claim: ClaimInput):
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
    try:
        return run_pipeline_inference(claim.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")

@app.post("/predict_user_claim", response_model=PredictionResponse)
def predict_user_claim_risk(claim: UserClaimInput):
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
    try:
        data_dict = claim.dict()
        
        # Default baseline scores for user endpoint
        data_dict.setdefault("medical_necessity_score", 0.70)
        data_dict.setdefault("pa_risk_score", 0.10)
        data_dict.setdefault("coverage_score", 0.70)
        data_dict.setdefault("clinician_success_score", 0.75)
        data_dict.setdefault("facility_success_score", 0.75)

        return run_pipeline_inference(data_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")