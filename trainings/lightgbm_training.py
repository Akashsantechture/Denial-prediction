import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import train_test_split

from category_encoders import TargetEncoder

from lightgbm import LGBMClassifier


# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/denial_prediction_final.parquet"

MODEL_DIR = Path("models/lightgbm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "target_encoder.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "target_denied"


# =====================================================
# LOAD
# =====================================================

df = pd.read_parquet(DATA_PATH)

claims = (
    df["haad_claim_id"]
    .astype(str)
    .drop_duplicates()
    .to_numpy()
)

train_claims, test_claims = train_test_split(
    claims,
    test_size=0.20,
    random_state=42
)

train_df = df[df["haad_claim_id"].isin(train_claims)].copy()
test_df = df[df["haad_claim_id"].isin(test_claims)].copy()


# =====================================================
# FEATURES
# =====================================================

DROP_COLS = [
    "target_denied",
    "activity_denied",
    "haad_claim_id",
]

CAT_COLS = [
    "activity_code",
    "gender",
    "nationality",
    "encounter_type",
    "clinician_profession",
    "clinician_category",
    "facility_type",
    "payer_classification",
    "diagnosis_code",
    "icd_category",
    "cpt_category",
    "icd_cpt_domain_match",
]

X_train = train_df.drop(columns=DROP_COLS)
X_test = test_df.drop(columns=DROP_COLS)

y_train = train_df[TARGET]
y_test = test_df[TARGET]


# =====================================================
# TARGET ENCODING
# =====================================================

encoder = TargetEncoder(
    cols=CAT_COLS,
    handle_missing="value",
    handle_unknown="value",
)

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

joblib.dump(encoder, ENCODER_PATH)


# =====================================================
# LIGHTGBM
# =====================================================

model = LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


# =====================================================
# EVALUATION
# =====================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "pr_auc": float(average_precision_score(y_test, y_prob)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
}

print("\n========== RESULTS ==========")

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("=============================\n")


# =====================================================
# SAVE
# =====================================================

joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Saved successfully")