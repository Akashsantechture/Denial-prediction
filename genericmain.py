import os
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from validators.schemas import UserClaimInput, PredictionResponse

# --- GLOBAL ARTIFACTS ---
MODEL = None
MODEL_PATH = "models/behavioral_xgboost_v1.5.joblib"

# # The exact features expected by the new v1.5 XGBoost model
# CATEGORICAL_FEATURES = [
#     'diagnosis_code', 'diagnosis_type', 'activity_code', 
#     'gender', 'nationality', 'encounter_type', 
#     'clinician_profession', 'clinician_category', 
#     'facility_type', 'insurance_plan_tier'
# ]

# FEATURE_ORDER = CATEGORICAL_FEATURES + [
#     'activity_quantity', 'patient_age', 'activity_gross', 
#     'claim_gross', 'claim_net', 'unit_cost', 'discount_ratio', 
#     'activity_gross_log', 'claim_gross_log', 'claim_net_log', 
#     'unit_cost_log', 'high_cost_flag', 'ip_stay_cost_ratio', 
#     'length_of_stay', 'billing_lag_days',
#     'gender_appropriate_flag', 'age_appropriate_flag', 'icd_cpt_chapter_match'
# ]
# The exact features expected by the new v1.5 XGBoost model
CATEGORICAL_FEATURES = [
    'diagnosis_code', 'diagnosis_type', 'activity_code', 
    'gender', 'nationality', 'encounter_type', 
    'clinician_profession', 'clinician_category', 
    'facility_type', 'insurance_plan_tier'
]

# FIX: The three extra clinical flags have been removed from this list
FEATURE_ORDER = CATEGORICAL_FEATURES + [
    'activity_quantity', 'patient_age', 'activity_gross', 
    'claim_gross', 'claim_net', 'unit_cost', 'discount_ratio', 
    'activity_gross_log', 'claim_gross_log', 'claim_net_log', 
    'unit_cost_log', 'high_cost_flag', 'ip_stay_cost_ratio', 
    'length_of_stay', 'billing_lag_days'
]
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Artifact not found! Ensure '{MODEL_PATH}' exists.")
    
    MODEL = joblib.load(MODEL_PATH)
    print("✅ Behavioral XGBoost Model loaded successfully!")
    yield

app = FastAPI(
    title="Healthcare Claim Denial Prediction API (Behavioral V1.5)",
    description="Production endpoint for real-time RCM claim denial risk scoring.",
    version="1.5.0",
    lifespan=lifespan
)

# Allow the HTML frontend (served from any origin during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field_path = " → ".join(str(loc) for loc in error["loc"] if loc != "body")
        msg = error.get("msg", "Invalid value")
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
    # --- FIX: Map Pydantic schema names to the new Model names ---
    if "gross_amount" in data_dict:
        data_dict["claim_gross"] = data_dict.pop("gross_amount")
    if "net_amount" in data_dict:
        data_dict["claim_net"] = data_dict.pop("net_amount")

    df = pd.DataFrame([data_dict])

    # 1. Engineer Financial Behaviors (Replicating SQL Logic)
    qty = max(df.get("activity_quantity", pd.Series([1])).iloc[0], 1)
    
    # Ensure activity_gross exists for math
    act_gross = df.get("activity_gross", pd.Series([0])).iloc[0]
    df["unit_cost"] = round(act_gross / qty, 2)
    
    # Now this will work perfectly because 'claim_gross' is mapped
    c_gross = max(df.get("claim_gross", pd.Series([1])).iloc[0], 1)
    
    # Safely calculate discount ratio (handling if claim_net is missing/0)
    c_net = df.get("claim_net", pd.Series([0])).iloc[0]
    df["discount_ratio"] = round((c_gross - c_net) / c_gross, 4)
    
    los = max(df.get("length_of_stay", pd.Series([1])).iloc[0], 1)
    enc_type = df.get("encounter_type", pd.Series(['OP'])).iloc[0]
    df["ip_stay_cost_ratio"] = np.where(enc_type == 'IP', round(c_gross / los, 2), 0)

    # Note: Replace 5000 with whatever your SQL 90th percentile threshold actually was
    df["high_cost_flag"] = np.where(act_gross > 5000, 1, 0) 

    # 2. Engineer Log Transforms safely
    # We must ensure the columns exist before applying log1p
    for col in ["activity_gross", "claim_gross", "claim_net", "unit_cost"]:
        val = df.get(col, pd.Series([0])).iloc[0]
        df[f"{col}_log"] = np.log1p(max(0, val))

    # 3. Engineer Clinical Consistency Flags (Simplified Regex/Logic for Python)
    diag = str(df.get("diagnosis_code", pd.Series([""])).iloc[0]).upper()
    cpt = str(df.get("activity_code", pd.Series([""])).iloc[0])
    gender = str(df.get("gender", pd.Series(["UNKNOWN"])).iloc[0]).upper()
    age = df.get("patient_age", pd.Series([30])).iloc[0]

    # Gender Flag
    if (diag.startswith('O') or (cpt.isdigit() and 59000 <= int(cpt) <= 59899)) and gender == 'MALE':
        df["gender_appropriate_flag"] = 0
    elif (cpt.isdigit() and 54000 <= int(cpt) <= 55899) and gender == 'FEMALE':
        df["gender_appropriate_flag"] = 0
    else:
        df["gender_appropriate_flag"] = 1

    # Age Flag
    df["age_appropriate_flag"] = 0 if diag.startswith('O') and (age < 12 or age > 60) else 1

    # ICD/CPT Match Flag
    if diag.startswith('J') and (cpt.isdigit() and 30000 <= int(cpt) <= 32999):
        df["icd_cpt_chapter_match"] = 1
    elif diag.startswith('I') and (cpt.isdigit() and 33000 <= int(cpt) <= 37799):
        df["icd_cpt_chapter_match"] = 1
    elif diag.startswith('Z'):
        df["icd_cpt_chapter_match"] = 1
    else:
        df["icd_cpt_chapter_match"] = 0

    # 4. Filter missing features, handle Nans, and cast Categoricals for XGBoost
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0 # Fallback for missing numeric features
            
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype('category')

    # Ensure column order matches training exactly
    X_inference = df[FEATURE_ORDER]

    # 5. Predict
    prob = float(MODEL.predict_proba(X_inference)[0, 1])
    prob_pct = round(prob * 100, 2)

    # --- RECALIBRATED RISK THRESHOLDS ---
    if prob_pct >= 50.0:
        risk_level = "HIGH"
        is_high_risk = True
        action = "Flagged for manual review. High probability of behavioral financial anomaly."
    elif prob_pct >= 35.0:
        risk_level = "MEDIUM"
        is_high_risk = False
        action = "Moderate risk. Verify clinical consistency and discount ratios."
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

@app.post("/predict_user_claim", response_model=PredictionResponse)
def predict_user_claim_risk(claim: UserClaimInput):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded.")
    try:
        # Pass the raw payload straight to the inference pipeline
        return run_pipeline_inference(claim.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model": "Behavioral XGBoost v1.5",
        "version": "1.5.0"
    }

# Serve the HTML/CSS/JS frontend — must be mounted LAST (after all API routes)
app.mount("/", StaticFiles(directory="frontend3", html=True), name="frontend")