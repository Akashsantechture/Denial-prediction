import os
from contextlib import asynccontextmanager

import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from validators.schemas import ClaimRequest
from preprocessing.feature_engineering import build_activity_dataframe
from explainability.shap_engine import ShapEngine
from preprocessing.recommendation_engine import generate_recommendation


# ============================================================
# GLOBALS
# ============================================================

MODEL       = None
ENCODER     = None
SHAP_ENGINE = None


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_ORDER = [
    "activity_code",
    "activity_quantity",
    "activity_gross",
    "patient_age",
    "gender",
    "nationality",
    "claim_gross",
    "claim_net",
    "encounter_type",
    "length_of_stay",
    "clinician_profession",
    "clinician_category",
    "facility_type",
    "payer_classification",
    "diagnosis_code",
    "billing_lag_days",
    "icd_category",
    "cpt_category",
    "icd_cpt_domain_match",
]


# ============================================================
# LIFESPAN — load model once at startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, ENCODER, SHAP_ENGINE

    print("=" * 60)
    print("Loading Claim Denial Prediction Service")
    print("=" * 60)

    MODEL       = joblib.load("models/xgboost/model.joblib")
    ENCODER     = joblib.load("models/xgboost/target_encoder.joblib")
    SHAP_ENGINE = ShapEngine(MODEL)

    print("✅ Model Loaded")
    print("✅ Encoder Loaded")
    print("✅ SHAP Engine Loaded")
    print("=" * 60)

    yield

    MODEL = ENCODER = SHAP_ENGINE = None


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Claim Denial Prediction API",
    version="3.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# NO-CACHE MIDDLEWARE
# Ensures browser always gets fresh HTML/CSS/JS after changes.
# ============================================================

@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if (
        path == "/"
        or path.endswith(".html")
        or "/css/" in path
        or "/js/"  in path
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"]        = "no-cache"
        response.headers["Expires"]       = "0"
    return response


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "model_loaded":   MODEL   is not None,
        "encoder_loaded": ENCODER is not None,
        "shap_loaded":    SHAP_ENGINE is not None,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():
    return {
        "model_type":              "XGBoost",
        "feature_count":           len(FEATURE_ORDER),
        "features":                FEATURE_ORDER,
        "shap_enabled":            True,
        "recommendations_enabled": True,
        "claim_level_scoring":     True,
    }


# ============================================================
# PREDICT  —  POST /predict
# ============================================================

@app.post("/predict")
def predict(request: ClaimRequest):

    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Build feature DataFrame from the ClaimRequest
        df = build_activity_dataframe(request)

        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df         = df[FEATURE_ORDER]
        df_encoded = ENCODER.transform(df)
        df_encoded = df_encoded[FEATURE_ORDER]

        # Inference
        probabilities = MODEL.predict_proba(df_encoded)[:, 1]

        # Claim-level composite score  (0.7 × max  +  0.3 × avg)
        max_risk = float(np.max(probabilities))
        avg_risk = float(np.mean(probabilities))

        claim_prob     = round(0.7 * max_risk + 0.3 * avg_risk, 4)
        claim_prob_pct = round(claim_prob * 100, 2)

        if claim_prob_pct >= 70:
            risk_level = "HIGH"
        elif claim_prob_pct >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # SHAP explanations
        shap_results = SHAP_ENGINE.explain(df_encoded)

        # Per-activity predictions
        predictions = []
        for idx, activity in enumerate(request.activities):
            prob        = float(probabilities[idx])
            top_drivers = shap_results[idx] if idx < len(shap_results) else []

            predictions.append({
                "activity_code":      activity.activity_code,
                "cpt_category":       activity.cpt_category,
                "denial_probability": round(prob, 4),
                "predicted_denial":   prob >= 0.50,
                "top_drivers":        top_drivers,
                "recommendation":     generate_recommendation(
                    prob,
                    top_driver=top_drivers[0] if top_drivers else None,
                ),
            })

        return {
            "claim_id": request.claim_id,
            "claim_summary": {
                "claim_denial_probability":     claim_prob,
                "claim_denial_probability_pct": claim_prob_pct,
                "claim_risk_level":             risk_level,
                "highest_activity_risk":        round(max_risk, 4),
                "average_activity_risk":        round(avg_risk, 4),
                "activity_count":               len(predictions),
            },
            "predictions": predictions,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# STATIC FRONTEND
# NOTE: Must be mounted LAST — FastAPI routes above take
#       priority over the StaticFiles catch-all.
#       Serves frontend/index.html at http://127.0.0.1:8000/
# ============================================================

_FRONTEND = "frontend"
if os.path.isdir(_FRONTEND):
    app.mount(
        "/",
        StaticFiles(directory=_FRONTEND, html=True),
        name="frontend",
    )
