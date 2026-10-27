
import os
import joblib
import numpy as np
import pandas as pd

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from validators.schemas import ClaimRequest, KSAClaimRequest
from preprocessing.feature_engineering import build_activity_dataframe, build_service_dataframe
from explainability.shap_engine import ShapEngine


# ============================================================
# UAE — GLOBALS
# ============================================================

MODEL = None
ENCODER = None
SHAP_ENGINE = None

REASON_MODEL = None
REASON_ENCODER = None
REASON_LABEL_ENCODER = None
REASON_SHAP_ENGINE = None

# ============================================================
# KSA — GLOBALS
# ============================================================
#
# Both KSA models are sklearn Pipelines — encoder baked in.
# No separate ENCODER object needed.
# SHAP requires the raw XGBoost step: pipeline.steps[-1][1]
# Pre-transform data via pipeline.steps[0][1].transform(df)
# before passing to ShapEngine.
# ============================================================

KSA_SCORE_PIPELINE  = None      # full sklearn Pipeline (preprocessor + XGBoost)
KSA_REASON_PIPELINE = None      # full sklearn Pipeline (preprocessor + XGBoost)
KSA_SCORE_SHAP      = None      # ShapEngine wrapping raw XGBoost step
KSA_REASON_SHAP     = None      # ShapEngine wrapping raw XGBoost step
KSA_REASON_LABEL_ENCODER = None


# ============================================================
# UAE — FEATURE ORDER
# ============================================================

# XGBoost denial score model (38 features)
FEATURE_ORDER = [
    "activity_code",
    "activity_quantity",
    "patient_age",
    "gender",
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
    "cpt_category",
    "payer_name",
    "network",
    "payer_id",
    "receiver_id",
]

# denial_reason SHAP model (36 features) — no financials, no payer_name/payer_id
REASON_FEATURE_ORDER = [
    "activity_code",
    "activity_quantity",
    "patient_age",
    "gender",
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
    "cpt_category",
    "network",
    "receiver_id",
]

# ============================================================
# KSA — FEATURE ORDER
# ============================================================
#
# Sourced from:
#   models/ksa/prediction_score/feature_columns.json  (score)
#   models/ksa/denial_reason/model.feature_names_in_  (reason)
#
# Score pipeline input  : 38 columns — includes "facility", no "policy_name"
# Reason pipeline input : 38 columns — includes "policy_name", no "facility"
# Both pipelines handle encoding internally — pass raw dataframe directly.
# ============================================================

KSA_FEATURE_ORDER = [
    "service_code",
    "gender",
    "encounter_type",
    "facility",
    "facility_type",
    "payer_id",
    "receiver_id",
    "payer_classification",
    "department",
    "claim_type",
    "claim_subtype",
    "primary_diagnosis",
    "primary_diagnosis_category",
    "patient_age",
    "quantity",
    "length_of_stay",
    "secondary_dx_infections",
    "secondary_dx_neoplasms",
    "secondary_dx_blood_immune",
    "secondary_dx_endocrine_metabolic",
    "secondary_dx_mental",
    "secondary_dx_nervous_system",
    "secondary_dx_eye",
    "secondary_dx_ear",
    "secondary_dx_circulatory",
    "secondary_dx_respiratory",
    "secondary_dx_digestive",
    "secondary_dx_skin",
    "secondary_dx_musculoskeletal",
    "secondary_dx_genitourinary",
    "secondary_dx_obstetrics",
    "secondary_dx_perinatal",
    "secondary_dx_congenital",
    "secondary_dx_symptoms",
    "secondary_dx_injury_poisoning",
    "secondary_dx_external_causes",
    "secondary_dx_health_status",
    "secondary_dx_special_purposes",
]

KSA_REASON_FEATURE_ORDER = [
    "service_code",
    "gender",
    "encounter_type",
    "facility_type",
    "payer_id",
    "receiver_id",
    "payer_classification",
    "department",
    "claim_type",
    "policy_name",
    "claim_subtype",
    "primary_diagnosis",
    "primary_diagnosis_category",
    "patient_age",
    "quantity",
    "length_of_stay",
    "secondary_dx_infections",
    "secondary_dx_neoplasms",
    "secondary_dx_blood_immune",
    "secondary_dx_endocrine_metabolic",
    "secondary_dx_mental",
    "secondary_dx_nervous_system",
    "secondary_dx_eye",
    "secondary_dx_ear",
    "secondary_dx_circulatory",
    "secondary_dx_respiratory",
    "secondary_dx_digestive",
    "secondary_dx_skin",
    "secondary_dx_musculoskeletal",
    "secondary_dx_genitourinary",
    "secondary_dx_obstetrics",
    "secondary_dx_perinatal",
    "secondary_dx_congenital",
    "secondary_dx_symptoms",
    "secondary_dx_injury_poisoning",
    "secondary_dx_external_causes",
    "secondary_dx_health_status",
    "secondary_dx_special_purposes",
]


# ============================================================
# STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global MODEL, ENCODER, SHAP_ENGINE, REASON_MODEL, REASON_ENCODER, REASON_LABEL_ENCODER, REASON_SHAP_ENGINE
    global KSA_SCORE_PIPELINE, KSA_REASON_PIPELINE, KSA_SCORE_SHAP, KSA_REASON_SHAP, KSA_REASON_LABEL_ENCODER

    print("=" * 60)
    print("Loading Claim Denial Prediction Service")
    print("=" * 60)

    # MODEL = joblib.load("models//xgboost/model.joblib")
    # ENCODER = joblib.load("models/xgboost/target_encoder.joblib")
    # SHAP_ENGINE = ShapEngine(MODEL)

    # REASON_MODEL = joblib.load("models/denial_reason/model.joblib")
    # REASON_SHAP_ENGINE = ShapEngine(REASON_MODEL)
    # REASON_ENCODER = joblib.load("models/denial_reason/target_encoder.joblib")
    # REASON_LABEL_ENCODER = joblib.load("models/denial_reason/label_encoder.joblib")

    MODEL = joblib.load("models/uae/prediction_score/model.joblib")
    ENCODER = joblib.load("models/uae/prediction_score/target_encoder.joblib")
    SHAP_ENGINE = ShapEngine(MODEL)

    REASON_MODEL = joblib.load("models/uae/denial_reason/model.joblib")
    REASON_SHAP_ENGINE = ShapEngine(REASON_MODEL)
    REASON_ENCODER = joblib.load("models/uae/denial_reason/target_encoder.joblib")
    REASON_LABEL_ENCODER = joblib.load("models/uae/denial_reason/label_encoder.joblib")

    print("Model Loaded")
    print("Encoder Loaded")
    print("SHAP Loaded")
    print("Reason Model Loaded")
    print("Reason Encoder Loaded")
    print("Reason Label Encoder Loaded")

    print("=" * 60)

    # ── KSA models (graceful skip if not present) ──────────
    print("Loading KSA Models…")
    try:
        KSA_SCORE_PIPELINE  = joblib.load("models/ksa/prediction_score/model_pipeline.joblib")
        KSA_REASON_PIPELINE = joblib.load("models/ksa/denial_reason/model.joblib")
        KSA_REASON_LABEL_ENCODER = joblib.load("models/ksa/denial_reason/label_encoder.joblib")

        # ShapEngine needs the raw XGBoost step, not the full pipeline
        KSA_SCORE_SHAP  = ShapEngine(KSA_SCORE_PIPELINE.steps[-1][1])
        KSA_REASON_SHAP = ShapEngine(KSA_REASON_PIPELINE.steps[-1][1])

        print("KSA Score Pipeline Loaded")
        print("KSA Reason Pipeline Loaded")
        print("KSA Reason Label Encoder Loaded")
    except FileNotFoundError as e:
        print(f"KSA models not found — /ksa/predict will return 503. ({e})")

    print("=" * 60)

    yield

    MODEL = None
    ENCODER = None
    SHAP_ENGINE = None
    REASON_MODEL = None
    REASON_ENCODER = None
    REASON_LABEL_ENCODER = None
    REASON_SHAP_ENGINE = None
    KSA_SCORE_PIPELINE  = None
    KSA_REASON_PIPELINE = None
    KSA_SCORE_SHAP      = None
    KSA_REASON_SHAP     = None
    KSA_REASON_LABEL_ENCODER = None


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Claim Denial & Reasoning Prediction API",
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
        # BUILD FEATURES — wide dataframe with all columns
        # =====================================================

        df = build_activity_dataframe(request)

        # ── Denial model ──────────────────────────────────
        missing_cols = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for denial model: {missing_cols}")
        df_denial = df[FEATURE_ORDER]

        # ── Reason model ──────────────────────────────────
        missing_reason = [c for c in REASON_FEATURE_ORDER if c not in df.columns]
        if missing_reason:
            raise ValueError(f"Missing columns for reason model: {missing_reason}")
        df_reason = df[REASON_FEATURE_ORDER]

        # =====================================================
        # ENCODE
        # =====================================================

        df_encoded        = ENCODER.transform(df_denial)
        reason_df_encoded = REASON_ENCODER.transform(df_reason)

        # =====================================================
        # SHAP — denial model
        # =====================================================

        shap_results = SHAP_ENGINE.explain(df_encoded)

        # =====================================================
        # INTERACTIONS
        # =====================================================

        # interaction_results = (
        #     SHAP_ENGINE.interaction_values(
        #         df_encoded
        #     )
        # )

        # =====================================================
        # SCORE — denial model
        # =====================================================

        probabilities = MODEL.predict_proba(df_encoded)[:, 1]

        # =====================================================
        # DENIAL REASON MODEL
        # =====================================================

        reason_probabilities = REASON_MODEL.predict_proba(reason_df_encoded)
        reason_predictions   = REASON_MODEL.predict(reason_df_encoded)
        reason_shap_results  = REASON_SHAP_ENGINE.explain_multiclass(
            reason_df_encoded,
            reason_predictions,
            original_df=df_reason
        )
        reason_labels = REASON_LABEL_ENCODER.inverse_transform(reason_predictions)

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
                denial_probability >= 0.40
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
                "activity_code":          activity.activity_code,
                "denial_probability":     round(denial_probability, 4),
                "predicted_denial":       predicted_denial,
                "predicted_denial_reason": predicted_reason,
                "reason_confidence":      round(reason_confidence, 4) if predicted_reason else None,
                "reason_drivers":         reason_shap_results[idx],
                "top_drivers":            shap_results[idx],
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


def _ksa_reason_shap(
    preprocessor,
    xgb_model,
    df_reason: pd.DataFrame,
    predicted_classes,
    top_n: int = 5,
) -> list:
    """
    Compute SHAP for KSA denial_reason model and roll up OHE columns
    back to the original 38 feature names.

    The reason model uses OneHotEncoder which expands 38 cols → 1808.
    TreeExplainer runs on the 1808-col transformed data, then the
    per-class SHAP values are summed back per original feature.

    Returns:
        List of per-row driver lists, same shape as ShapEngine.explain_multiclass().
    """
    import shap as _shap

    transformed = preprocessor.transform(df_reason)
    feat_names  = preprocessor.get_feature_names_out()

    # Build mapping: each expanded col → original col name
    original_cols = list(df_reason.columns)

    def _orig_col(ohe_name: str) -> str:
        if ohe_name.startswith("categorical__"):
            stripped = ohe_name[len("categorical__"):]
        elif ohe_name.startswith("numeric__"):
            return ohe_name[len("numeric__"):]
        else:
            stripped = ohe_name
        # Match longest original col that is a prefix of stripped
        for col in sorted(original_cols, key=len, reverse=True):
            if stripped.startswith(col):
                return col
        return stripped

    col_mapping = [_orig_col(f) for f in feat_names]

    # SHAP values: (n_rows, 1808, n_classes)
    explainer  = _shap.TreeExplainer(xgb_model)
    shap_array = np.array(explainer.shap_values(transformed))

    results = []
    for row_idx in range(len(df_reason)):
        cls      = int(predicted_classes[row_idx])
        row_shap = shap_array[row_idx, :, cls]  # (1808,)

        # Sum SHAP values per original feature
        orig_shap: dict = {}
        for shap_val, orig_col in zip(row_shap, col_mapping):
            orig_shap[orig_col] = orig_shap.get(orig_col, 0.0) + float(shap_val)

        # Format as driver dicts, sorted by absolute value
        drivers = sorted(
            [
                {
                    "feature":    feat,
                    "value":      str(df_reason.iloc[row_idx][feat]),
                    "shap_value": round(val, 4),
                    "impact":     "increase_reason_probability" if val > 0 else "decrease_reason_probability",
                }
                for feat, val in orig_shap.items()
            ],
            key=lambda d: abs(d["shap_value"]),
            reverse=True,
        )
        results.append(drivers[:top_n])

    return results
# ============================================================

@app.get("/ksa/health")
def ksa_health():
    return {
        "status":                   "healthy",
        "score_pipeline_loaded":    KSA_SCORE_PIPELINE  is not None,
        "reason_pipeline_loaded":   KSA_REASON_PIPELINE is not None,
        "shap_loaded":              KSA_SCORE_SHAP      is not None,
    }


@app.get("/ksa/model-info")
def ksa_model_info():
    return {
        "model_type":      "XGBoost Pipeline",
        "feature_count":   len(KSA_FEATURE_ORDER),
        "features":        KSA_FEATURE_ORDER,
        "shap_enabled":    True,
    }


# ============================================================
# KSA — PREDICT
# ============================================================
#
# Mirrors /predict exactly. Key implementation differences:
#
#   1. Uses build_service_dataframe() (services[] not activities[])
#   2. Both pipelines include encoder — call pipeline.predict_proba(df)
#      directly, NO separate encode step
#   3. SHAP requires pre-transformed data:
#      transformed = pipeline.steps[0][1].transform(df_slice)
#      then ShapEngine.explain(transformed_as_dataframe)
#   4. Response maps service_code → "activity_code" key so the
#      frontend receives the same shape as UAE responses
# ============================================================

@app.post("/ksa/predict")
def predict_ksa(request: KSAClaimRequest):

    try:
        if KSA_SCORE_PIPELINE is None:
            raise HTTPException(
                status_code=503,
                detail="KSA models not loaded. Drop model files into models/ksa/ and restart."
            )

        # ── Build wide dataframe ───────────────────────────
        df = build_service_dataframe(request)

        # ── Slice per model ────────────────────────────────
        missing_score  = [c for c in KSA_FEATURE_ORDER        if c not in df.columns]
        missing_reason = [c for c in KSA_REASON_FEATURE_ORDER if c not in df.columns]
        if missing_score:
            raise ValueError(f"Missing columns for KSA score model: {missing_score}")
        if missing_reason:
            raise ValueError(f"Missing columns for KSA reason model: {missing_reason}")

        df_score  = df[KSA_FEATURE_ORDER]
        df_reason = df[KSA_REASON_FEATURE_ORDER]

        # ── Score pipeline — encoder baked in ─────────────
        probabilities = KSA_SCORE_PIPELINE.predict_proba(df_score)[:, 1]

        # ── SHAP for score model ───────────────────────────
        # Pre-transform via the pipeline's preprocessor step.
        # Use get_feature_names_out() to preserve column names so
        # ShapEngine can label each driver correctly.
        score_preprocessor   = KSA_SCORE_PIPELINE.steps[0][1]
        score_transformed    = score_preprocessor.transform(df_score)
        score_feat_names     = score_preprocessor.get_feature_names_out()
        df_score_transformed = pd.DataFrame(score_transformed, columns=score_feat_names)
        shap_results = KSA_SCORE_SHAP.explain(df_score_transformed)

        # ── Reason pipeline — encoder baked in ────────────
        reason_probabilities = KSA_REASON_PIPELINE.predict_proba(df_reason)
        reason_predictions   = KSA_REASON_PIPELINE.predict(df_reason)
        reason_labels        = KSA_REASON_LABEL_ENCODER.inverse_transform(
            reason_predictions
        )

        # SHAP for reason model — OHE expansion rolled back to original features
        reason_shap_results = _ksa_reason_shap(
            preprocessor=KSA_REASON_PIPELINE.steps[0][1],
            xgb_model=KSA_REASON_PIPELINE.steps[-1][1],
            df_reason=df_reason,
            predicted_classes=reason_predictions,
        )

        # ── Per-service results ────────────────────────────
        activity_predictions = []

        for idx, service in enumerate(request.services):

            denial_probability = float(probabilities[idx])
            predicted_denial   = denial_probability >= 0.40
            reason_confidence  = float(np.max(reason_probabilities[idx]))
            predicted_reason   = str(reason_labels[idx]) if predicted_denial else None

            activity_predictions.append({
                "activity_code":           service.service_code,   # mapped for frontend compat
                "denial_probability":      round(denial_probability, 4),
                "predicted_denial":        predicted_denial,
                "predicted_denial_reason": predicted_reason,
                "reason_confidence":       round(reason_confidence, 4) if predicted_reason else None,
                "reason_drivers":          reason_shap_results[idx],
                "top_drivers":             shap_results[idx],
            })

        # ── Claim-level score ──────────────────────────────
        max_risk   = float(np.max(probabilities))
        avg_risk   = float(np.mean(probabilities))
        claim_risk = round((0.7 * max_risk) + (0.3 * avg_risk), 4)

        if claim_risk >= 0.70:
            risk_level = "HIGH"
        elif claim_risk >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ── Response — identical shape to /predict ─────────
        return {
            "claim_id": request.claim_id,
            "claim_summary": {
                "claim_denial_probability":     claim_risk,
                "claim_denial_probability_pct": round(claim_risk * 100, 2),
                "claim_risk_level":             risk_level,
                "highest_activity_risk":        round(max_risk, 4),
                "average_activity_risk":        round(avg_risk, 4),
                "activity_count":               len(activity_predictions),
            },
            "activity_predictions": activity_predictions,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
