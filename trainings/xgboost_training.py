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

from xgboost import XGBClassifier


# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/claim_activity_denial_model_v3.parquet"

MODEL_DIR = Path("models/xgboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "target_encoder.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("Loading parquet...")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)

print(f"Dataset Shape: {df.shape}")


# ==========================================================
# CLAIM LEVEL SPLIT
# ==========================================================

claims = (
    df["haad_claim_id"]
    .astype(str)
    .drop_duplicates()
    .to_numpy()
)

train_claims, test_claims = train_test_split(
    claims,
    test_size=0.20,
    random_state=42,
)

train_df = df[
    df["haad_claim_id"].isin(train_claims)
].copy()

test_df = df[
    df["haad_claim_id"].isin(test_claims)
].copy()

print(f"Train Rows : {len(train_df):,}")
print(f"Test Rows  : {len(test_df):,}")


# ==========================================================
# TARGET
# ==========================================================

TARGET = "activity_denied"

DROP_COLS = [
    TARGET,
    "haad_claim_id",
    "billing_lag_days",
    "billing_lag_missing_flag",
]

CAT_COLS = [
    "activity_code",
    "gender",
    "nationality",
    "encounter_type",
    "clinician_profession",
    "facility_type",
    "payer_classification",
    "primary_diagnosis_code",
    "primary_diagnosis_category",
    "cpt_category",
]

X_train = train_df.drop(columns=DROP_COLS)
X_test = test_df.drop(columns=DROP_COLS)

y_train = train_df[TARGET]
y_test = test_df[TARGET]


# ==========================================================
# TARGET ENCODING
# ==========================================================

print("\nTarget Encoding...")

encoder = TargetEncoder(
    cols=CAT_COLS,
    handle_missing="value",
    handle_unknown="value",
)

X_train = encoder.fit_transform(
    X_train,
    y_train
)

X_test = encoder.transform(
    X_test
)

joblib.dump(
    encoder,
    ENCODER_PATH
)

print("Encoder saved.")


# ==========================================================
# SAFETY CHECK
# ==========================================================

remaining_strings = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nRemaining String Columns:")
print(remaining_strings)

if len(remaining_strings) > 0:
    raise ValueError(
        f"Still contains string columns: {remaining_strings}"
    )


# ==========================================================
# CLASS BALANCING
# ==========================================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print(
    f"\nscale_pos_weight = {scale_pos_weight:.3f}"
)


# ==========================================================
# MODEL
# ==========================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",

    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=5,
    gamma=0.2,

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(
    X_test
)[:, 1]

metrics = {
    "roc_auc": float(
        roc_auc_score(
            y_test,
            y_prob
        )
    ),

    "pr_auc": float(
        average_precision_score(
            y_test,
            y_prob
        )
    ),

    "accuracy": float(
        accuracy_score(
            y_test,
            y_pred
        )
    ),

    "precision": float(
        precision_score(
            y_test,
            y_pred
        )
    ),

    "recall": float(
        recall_score(
            y_test,
            y_pred
        )
    ),

    "f1": float(
        f1_score(
            y_test,
            y_pred
        )
    ),
}

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

for k, v in metrics.items():
    print(f"{k:<12}: {v:.4f}")

print("=" * 60)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_importance = (
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })
    .sort_values(
        "importance",
        ascending=False
    )
)

print("\nTop 25 Features\n")

print(
    feature_importance
    .head(25)
    .to_string(index=False)
)


# ==========================================================
# SAVE
# ==========================================================

joblib.dump(
    model,
    MODEL_PATH
)

with open(
    METRICS_PATH,
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("\nSaved Artifacts")
print(f"Model   : {MODEL_PATH}")
print(f"Encoder : {ENCODER_PATH}")
print(f"Metrics : {METRICS_PATH}")

print("\nTraining Complete.")