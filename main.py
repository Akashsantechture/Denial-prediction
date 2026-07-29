import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Healthcare Claim Denial Prediction API",
    description="Production endpoint for real-time RCM claim denial risk scoring.",
    version="1.0.0"
)

# Global variables for model and pipeline
MODEL = None
PREPROCESSOR = None

MODEL_PATH = "xgboost_denial_model.joblib"
PREPROCESSOR_PATH = "preprocessor_pipeline.joblib"

@app.on_event("startup")
def load_artifacts():
    """Load preprocessor and model into memory when server starts."""
    global MODEL, PREPROCESSOR
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(f"Artifacts not found! Ensure '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' exist.")
    
    PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
    MODEL = joblib.load(MODEL_PATH)
    print("✅ Preprocessor and XGBoost Model successfully loaded into API memory!")

# --- Request / Response Schemas ---
class ClaimInput(BaseModel):
    primary_diagnosis_code: str = Field(..., example="E11.22")
    activity_code: str = Field(..., example="84432")
    medical_necessity_score: float = Field(0.50, example=0.6813)
    pa_risk_score: float = Field(0.00, example=0.0124)
    coverage_score: float = Field(0.50, example=0.5164)
    clinician_success_score: float = Field(0.50, example=0.6259)
    facility_success_score: float = Field(0.50, example=0.6482)
    activity_gross: float = Field(..., example=1200.00)
    activity_quantity: int = Field(1, example=1)
    gross_amount: float = Field(..., example=1200.00)
    net_amount: float = Field(..., example=1000.00)
    patient_age: int = Field(..., example=45)
    gender: str = Field("UNKNOWN", example="MALE")
    nationality: str = Field("UNKNOWN", example="EMIRATI")
    payer_id: str = Field("UNKNOWN", example="E001")
    insurance_plan_tier: str = Field("Standard", example="Standard")
    profession: str = Field("UNKNOWN", example="Internal Medicine")
    category: str = Field("UNKNOWN", example="Pulmonology")
    facility_type_id: str = Field("UNKNOWN", example="Hospital")
    billing_lag_days: int = Field(0, example=2)
    length_of_stay: int = Field(0, example=0)
    encounter_type: str = Field("OP", example="OP")

class PredictionResponse(BaseModel):
    denial_probability_pct: float
    is_high_risk: bool
    risk_level: str
    action_recommendation: str

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_claim_risk(claim: ClaimInput):
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

    try:
        # Convert Pydantic object to Pandas DataFrame
        data_dict = claim.dict()
        df_input = pd.DataFrame([data_dict])

        # Feature Transformations
        df_input["cpt_icd_pair"] = df_input["activity_code"] + "_" + df_input["primary_diagnosis_code"]
        df_input["pa_exceeds_1000_flag"] = np.where(
            (df_input["activity_gross"] >= 1000) | (df_input["gross_amount"] >= 1000), 1, 0
        )

        monetary_cols = ["activity_gross", "gross_amount", "net_amount"]
        for col in monetary_cols:
            df_input[f"{col}_log"] = np.log1p(np.maximum(0, df_input[col]))

        # Drop raw unlogged monetary columns expected by pipeline
        X_raw = df_input.drop(columns=monetary_cols)

        # Transform using fitted preprocessor
        X_transformed = PREPROCESSOR.transform(X_raw)

        # Generate prediction
        prob = float(MODEL.predict_proba(X_transformed)[0, 1])
        prob_pct = round(prob * 100, 2)

        # Assign Risk Category & Action Guidance
        if prob_pct >= 60.0:
            risk_level = "HIGH"
            is_high_risk = True
            action = "Flagged for manual review. Check medical necessity documentation and prior authorization requirements."
        elif prob_pct >= 45.0:
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

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction: {str(e)}")