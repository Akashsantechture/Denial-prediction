
import os
import joblib
import numpy as np

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from validators.schemas import ClaimRequest
from preprocessing.feature_engineering import build_activity_dataframe
from explainability.shap_engine import ShapEngine


# ============================================================
# GLOBALS
# ============================================================

MODEL = None
ENCODER = None
SHAP_ENGINE = None

REASON_MODEL = None
REASON_ENCODER = None
REASON_LABEL_ENCODER = None


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
    "facility_type",
    "payer_classification",
    "primary_diagnosis_code",
    "primary_diagnosis_category",
    "secondary_dx_count",
    "secondary_infectious",
    "secondary_oncology",
    "secondary_oncology_hematology",
    "secondary_endocrinology",
    "secondary_psychiatry",
    "secondary_neurology",
    "secondary_eye_ear",
    "secondary_cardiology",
    "secondary_pulmonology",
    "secondary_gastroenterology_dental",
    "secondary_dermatology",
    "secondary_musculoskeletal",
    "secondary_genitourinary",
    "secondary_obgyn",
    "secondary_pediatrics",
    "secondary_congenital",
    "secondary_general_symptoms",
    "secondary_trauma_burns_poisoning",
    "secondary_external_causes",
    "secondary_factors_influencing_health_status",
    "secondary_unknown_icd_category",
    "cpt_category"
]


# ============================================================
# STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global MODEL, ENCODER, SHAP_ENGINE, REASON_MODEL, REASON_ENCODER, REASON_LABEL_ENCODER

    print("=" * 60)
    print("Loading Claim Denial Prediction Service")
    print("=" * 60)

    MODEL = joblib.load(
        "models/xgboost/model.joblib"
    )

    ENCODER = joblib.load(
        "models/xgboost/target_encoder.joblib"
    )

    SHAP_ENGINE = ShapEngine(MODEL)

    REASON_MODEL = joblib.load(
    "models/denial_reason/model.joblib"
    )

    REASON_ENCODER = joblib.load(
        "models/denial_reason/target_encoder.joblib"
    )

    REASON_LABEL_ENCODER = joblib.load(
        "models/denial_reason/label_encoder.joblib"
    )

    print("Model Loaded")
    print("Encoder Loaded")
    print("SHAP Loaded")
    print("Reason Model Loaded")
    print("Reason Encoder Loaded")
    print("Reason Label Encoder Loaded")

    print("=" * 60)

    yield

    MODEL = None
    ENCODER = None
    SHAP_ENGINE = None
    REASON_MODEL = None
    REASON_ENCODER = None
    REASON_LABEL_ENCODER = None


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Claim Denial Prediction API",
    version="4.0",
    lifespan=lifespan
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
# NO-CACHE MIDDLEWARE — always serve fresh static files
# ============================================================

@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith(".html") or "/css/" in path or "/js/" in path:
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
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "encoder_loaded": ENCODER is not None,
        "shap_loaded": SHAP_ENGINE is not None,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_type": "XGBoost",
        "feature_count": len(FEATURE_ORDER),
        "features": FEATURE_ORDER,
        "shap_enabled": True,
        "interaction_enabled": False
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: ClaimRequest):

    try:

        if MODEL is None:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded"
            )

        # =====================================================
        # BUILD FEATURES
        # =====================================================

        df = build_activity_dataframe(request)

        missing_cols = [
            col
            for col in FEATURE_ORDER
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing columns: {missing_cols}"
            )

        df = df[FEATURE_ORDER]

        # =====================================================
        # ENCODE
        # =====================================================

        df_encoded = ENCODER.transform(df)

        # =====================================================
        # SHAP
        # =====================================================

        shap_results = SHAP_ENGINE.explain(
            df_encoded
        )

        # =====================================================
        # INTERACTIONS
        # =====================================================

        # interaction_results = (
        #     SHAP_ENGINE.interaction_values(
        #         df_encoded
        #     )
        # )

        # =====================================================
        # SCORE
        # =====================================================

        probabilities = MODEL.predict_proba(
            df_encoded
        )[:, 1]

        # ==========================================
        # DENIAL REASON MODEL
        # ==========================================

        reason_df_encoded = REASON_ENCODER.transform(df)

        reason_probabilities = (
            REASON_MODEL.predict_proba(
                reason_df_encoded
            )
        )

        reason_predictions = (
            REASON_MODEL.predict(
                reason_df_encoded
            )
        )

        reason_labels = (
            REASON_LABEL_ENCODER.inverse_transform(
                reason_predictions
            )
        )

        # =====================================================
        # ACTIVITY RESULTS
        # =====================================================

        activity_predictions = []

        for idx, activity in enumerate(
            request.activities
        ):

            denial_probability = float(
                probabilities[idx]
            )

            predicted_denial = (
                denial_probability >= 0.50
            )

            reason_confidence = float(
                np.max(
                    reason_probabilities[idx]
                )
            )

            predicted_reason = None

            if predicted_denial:

                predicted_reason = (
                    str(
                        reason_labels[idx]
                    )
                )

            activity_predictions.append({

                "activity_code":
                    activity.activity_code,

                "denial_probability":
                    round(
                        denial_probability,
                        4
                    ),

                "predicted_denial":
                    predicted_denial,

                "predicted_denial_reason":
                    predicted_reason,

                "reason_confidence":
                    round(
                        reason_confidence,
                        4
                    ) if predicted_reason else None,

                "top_drivers":
                    shap_results[idx]
            })
        # =====================================================
        # CLAIM LEVEL SCORE
        # =====================================================

        max_risk = float(
            np.max(probabilities)
        )

        avg_risk = float(
            np.mean(probabilities)
        )

        claim_risk = (
            (0.7 * max_risk) +
            (0.3 * avg_risk)
        )

        claim_risk = round(
            claim_risk,
            4
        )

        # =====================================================
        # RISK BAND
        # =====================================================

        if claim_risk >= 0.70:
            risk_level = "HIGH"

        elif claim_risk >= 0.40:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # =====================================================
        # RESPONSE
        # =====================================================

        return {

            "claim_id":
                request.claim_id,

            "claim_summary": {

                "claim_denial_probability":
                    claim_risk,

                "claim_denial_probability_pct":
                    round(
                        claim_risk * 100,
                        2
                    ),

                "claim_risk_level":
                    risk_level,

                "highest_activity_risk":
                    round(
                        max_risk,
                        4
                    ),

                "average_activity_risk":
                    round(
                        avg_risk,
                        4
                    ),

                "activity_count":
                    len(
                        activity_predictions
                    )
            },

            "activity_predictions":
                activity_predictions
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# STATIC FRONTEND
# Mount LAST — API routes above take priority.
# Serves frontend/index.html at http://127.0.0.1:8000/
# ============================================================

_FRONTEND = "frontend"
if os.path.isdir(_FRONTEND):
    app.mount(
        "/",
        StaticFiles(directory=_FRONTEND, html=True),
        name="frontend",
    )
