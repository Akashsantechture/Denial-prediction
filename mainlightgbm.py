import os
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from validators.schemas import UserClaimInput, PredictionResponse

# --- GLOBAL ARTIFACTS & MEMORY LOOKUPS ---
MODEL = None
PREPROCESSOR = None

MODEL_PATH = "models/xgboost_denial2.joblib"
PREPROCESSOR_PATH = "models/lightgbm_denial_model2.joblib"

MED_NECESSITY_LOOKUP = {}
PA_RISK_LOOKUP = {}
COVERAGE_LOOKUP = {}
CLINICIAN_LOOKUP = {}
FACILITY_LOOKUP = {}

def load_lookups():
    """Loads CSVs from the validators folder into Python dictionaries using positional indexing."""
    global MED_NECESSITY_LOOKUP, PA_RISK_LOOKUP, COVERAGE_LOOKUP, CLINICIAN_LOOKUP, FACILITY_LOOKUP
    
    print("Loading CSV lookups into memory...")
    
    try:
        # 1. Medical Necessity (Matches: medical_necessacity_score.csv)
        df_med = pd.read_csv("validators/medical_necessacity_score.csv", on_bad_lines='skip').dropna()
        for _, row in df_med.iterrows():
            MED_NECESSITY_LOOKUP[(str(row.iloc[0]).strip(), str(row.iloc[1]).strip())] = float(row.iloc[2])

        # 2. PA Risk Score (Matches: pa_risk_score.csv)
        df_pa = pd.read_csv("validators/pa_risk_score.csv", on_bad_lines='skip').dropna()
        for _, row in df_pa.iterrows():
            PA_RISK_LOOKUP[(str(row.iloc[0]).strip(), str(row.iloc[1]).strip(), str(row.iloc[2]).strip())] = float(row.iloc[3])

        # 3. Coverage Score (Matches: coverage_sucess_score.csv)
        df_cov = pd.read_csv("validators/coverage_sucess_score.csv", on_bad_lines='skip').dropna()
        for _, row in df_cov.iterrows():
            COVERAGE_LOOKUP[(str(row.iloc[0]).strip(), str(row.iloc[1]).strip())] = float(row.iloc[2])

        # 4. Clinician Score (Matches: clinician_sucess_score.csv)
        df_clin = pd.read_csv("validators/clinician_sucess_score.csv", on_bad_lines='skip').dropna()
        for _, row in df_clin.iterrows():
            CLINICIAN_LOOKUP[str(row.iloc[0]).strip()] = float(row.iloc[1])

        # 5. Facility Score (Matches: facility_sucess_score.csv)
        df_fac = pd.read_csv("validators/facility_sucess_score.csv", on_bad_lines='skip').dropna()
        for _, row in df_fac.iterrows():
            FACILITY_LOOKUP[str(row.iloc[0]).strip()] = float(row.iloc[1])
            
        print("✅ All lookups loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load all lookups. Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, PREPROCESSOR
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(f"Artifacts not found! Ensure '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' exist.")
    
    PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
    MODEL = joblib.load(MODEL_PATH)
    print(" Preprocessor and Lightgbm Model loaded successfully!")
    
    # Load the CSV dictionaries into memory before accepting requests
    load_lookups()
    
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

    # --- RECALIBRATED RISK THRESHOLDS ---
    if prob_pct >= 45.0:
        risk_level = "HIGH"
        is_high_risk = True
        action = "Flagged for manual review. Check medical necessity documentation and prior authorization requirements."
    elif prob_pct >= 22.0:
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

@app.post("/predict_user_claim", response_model=PredictionResponse)
def predict_user_claim_risk(claim: UserClaimInput):
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
    try:
        data_dict = claim.dict()
        
        # Extract fields to use as lookup keys safely
        diag = str(data_dict.get("primary_diagnosis_code", ""))
        cpt = str(data_dict.get("activity_code", ""))
        payer = str(data_dict.get("payer_id", ""))
        clinician = str(data_dict.get("clinician_id", "")) 
        facility = str(data_dict.get("facility_type_id", "")) 
        
        # Fetch the historical scores from memory, falling back to safe defaults if the combo is brand new
        data_dict["medical_necessity_score"] = MED_NECESSITY_LOOKUP.get((diag, cpt), 0.70)
        data_dict["pa_risk_score"] = PA_RISK_LOOKUP.get((payer, diag, cpt), 0.10)
        data_dict["coverage_score"] = COVERAGE_LOOKUP.get((payer, cpt), 0.70)
        data_dict["clinician_success_score"] = CLINICIAN_LOOKUP.get(clinician, 0.75)
        data_dict["facility_success_score"] = FACILITY_LOOKUP.get(facility, 0.75)

        return run_pipeline_inference(data_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")