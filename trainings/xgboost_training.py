# import json
# import joblib
# import pandas as pd

# from pathlib import Path

# from sklearn.metrics import (
#     roc_auc_score,
#     average_precision_score,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
# )

# from sklearn.model_selection import train_test_split

# from category_encoders import TargetEncoder

# from xgboost import XGBClassifier


# # ==========================================================
# # CONFIG
# # ==========================================================

# DATA_PATH = "data/claim_activity_denial_model_v3.parquet"

# MODEL_DIR = Path("models/xgboost")
# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# MODEL_PATH = MODEL_DIR / "model.joblib"
# ENCODER_PATH = MODEL_DIR / "target_encoder.joblib"
# METRICS_PATH = MODEL_DIR / "metrics.json"


# # ==========================================================
# # LOAD DATA
# # ==========================================================

# print("=" * 60)
# print("Loading parquet...")
# print("=" * 60)

# df = pd.read_parquet(DATA_PATH)

# print(f"Dataset Shape: {df.shape}")


# # ==========================================================
# # CLAIM LEVEL SPLIT
# # ==========================================================

# claims = (
#     df["haad_claim_id"]
#     .astype(str)
#     .drop_duplicates()
#     .to_numpy()
# )

# train_claims, test_claims = train_test_split(
#     claims,
#     test_size=0.20,
#     random_state=42,
# )

# train_df = df[
#     df["haad_claim_id"].isin(train_claims)
# ].copy()

# test_df = df[
#     df["haad_claim_id"].isin(test_claims)
# ].copy()

# print(f"Train Rows : {len(train_df):,}")
# print(f"Test Rows  : {len(test_df):,}")


# # ==========================================================
# # TARGET
# # ==========================================================

# TARGET = "activity_denied"

# DROP_COLS = [
#     TARGET,
#     "haad_claim_id",
#     "billing_lag_days",
#     "billing_lag_missing_flag",
# ]

# CAT_COLS = [
#     "activity_code",
#     "gender",
#     "nationality",
#     "encounter_type",
#     "clinician_profession",
#     "facility_type",
#     "payer_classification",
#     "primary_diagnosis_code",
#     "primary_diagnosis_category",
#     "cpt_category",
#     "receiver_id"
# ]

# X_train = train_df.drop(columns=DROP_COLS)
# X_test = test_df.drop(columns=DROP_COLS)

# y_train = train_df[TARGET]
# y_test = test_df[TARGET]


# # ==========================================================
# # TARGET ENCODING
# # ==========================================================

# print("\nTarget Encoding...")

# encoder = TargetEncoder(
#     cols=CAT_COLS,
#     handle_missing="value",
#     handle_unknown="value",
# )

# X_train = encoder.fit_transform(
#     X_train,
#     y_train
# )

# X_test = encoder.transform(
#     X_test
# )

# joblib.dump(
#     encoder,
#     ENCODER_PATH
# )

# print("Encoder saved.")


# # ==========================================================
# # SAFETY CHECK
# # ==========================================================

# remaining_strings = X_train.select_dtypes(
#     include=["object", "string"]
# ).columns.tolist()

# print("\nRemaining String Columns:")
# print(remaining_strings)

# if len(remaining_strings) > 0:
#     raise ValueError(
#         f"Still contains string columns: {remaining_strings}"
#     )


# # ==========================================================
# # CLASS BALANCING
# # ==========================================================

# neg = (y_train == 0).sum()
# pos = (y_train == 1).sum()

# scale_pos_weight = neg / pos

# print(
#     f"\nscale_pos_weight = {scale_pos_weight:.3f}"
# )


# # ==========================================================
# # MODEL
# # ==========================================================

# print("\nTraining XGBoost...")

# model = XGBClassifier(
#     objective="binary:logistic",
#     eval_metric="aucpr",

#     n_estimators=1000,
#     max_depth=8,
#     learning_rate=0.03,

#     subsample=0.8,
#     colsample_bytree=0.8,

#     min_child_weight=5,
#     gamma=0.2,

#     scale_pos_weight=scale_pos_weight,

#     tree_method="hist",

#     random_state=42,
#     n_jobs=-1,
# )

# model.fit(
#     X_train,
#     y_train
# )

# print("Training completed.")


# # ==========================================================
# # EVALUATION
# # ==========================================================

# print("\nEvaluating...")

# y_pred = model.predict(X_test)

# y_prob = model.predict_proba(
#     X_test
# )[:, 1]

# metrics = {
#     "roc_auc": float(
#         roc_auc_score(
#             y_test,
#             y_prob
#         )
#     ),

#     "pr_auc": float(
#         average_precision_score(
#             y_test,
#             y_prob
#         )
#     ),

#     "accuracy": float(
#         accuracy_score(
#             y_test,
#             y_pred
#         )
#     ),

#     "precision": float(
#         precision_score(
#             y_test,
#             y_pred
#         )
#     ),

#     "recall": float(
#         recall_score(
#             y_test,
#             y_pred
#         )
#     ),

#     "f1": float(
#         f1_score(
#             y_test,
#             y_pred
#         )
#     ),
# }

# print("\n" + "=" * 60)
# print("RESULTS")
# print("=" * 60)

# for k, v in metrics.items():
#     print(f"{k:<12}: {v:.4f}")

# print("=" * 60)


# # ==========================================================
# # FEATURE IMPORTANCE
# # ==========================================================

# feature_importance = (
#     pd.DataFrame({
#         "feature": X_train.columns,
#         "importance": model.feature_importances_
#     })
#     .sort_values(
#         "importance",
#         ascending=False
#     )
# )

# print("\nTop 25 Features\n")

# print(
#     feature_importance
#     .head(25)
#     .to_string(index=False)
# )


# # ==========================================================
# # SAVE
# # ==========================================================

# joblib.dump(
#     model,
#     MODEL_PATH
# )

# with open(
#     METRICS_PATH,
#     "w"
# ) as f:
#     json.dump(
#         metrics,
#         f,
#         indent=4
#     )

# print("\nSaved Artifacts")
# print(f"Model   : {MODEL_PATH}")
# print(f"Encoder : {ENCODER_PATH}")
# print(f"Metrics : {METRICS_PATH}")

# print("\nTraining Complete.")


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

DATA_PATH = Path("data/claim_activity_denial_model_v3.parquet")

MODEL_DIR = Path("models/xgboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "target_encoder.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

TARGET = "activity_denied"
CLAIM_ID = "haad_claim_id"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==========================================================
# FEATURE EXCLUSIONS
# ==========================================================

# Identifiers and target are not model inputs.
# Financial amounts and nationality are intentionally removed.
EXCLUDED_FEATURES = [
    TARGET,
    CLAIM_ID,
    "nationality",
    "claim_gross",
    "claim_net",
    "activity_gross",
    "billing_lag_days",
    "billing_lag_missing_flag",
]

# These are the categorical columns that remain eligible.
CAT_COLS = [
    "activity_code",
    "gender",
    "encounter_type",
    "clinician_profession",
    "facility_type",
    "payer_classification",
    "primary_diagnosis_code",
    "primary_diagnosis_category",
    "cpt_category",
    "payer_name",
    "network",
    "payer_id",
    "receiver_id",
]


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("LOADING CLAIM DENIAL DATA")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)

print(f"Dataset shape: {df.shape}")

required_columns = [
    TARGET,
    CLAIM_ID,
    "receiver_id",
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

if df[TARGET].isna().any():
    raise ValueError(
        f"{TARGET} contains missing target values. "
        "Clean these rows before training."
    )

if df[CLAIM_ID].isna().any():
    raise ValueError(
        f"{CLAIM_ID} contains null values. "
        "Resolve them before the claim-level split."
    )

target_values = set(df[TARGET].unique())

if not target_values.issubset({0, 1}):
    raise ValueError(
        f"{TARGET} must contain only 0 and 1. "
        f"Found: {target_values}"
    )

if df[TARGET].nunique() != 2:
    raise ValueError(
        "Both approved (0) and denied (1) rows "
        "must be present."
    )


# ==========================================================
# CLAIM-LEVEL TRAIN / TEST SPLIT
# ==========================================================

# Normalize IDs consistently for splitting.
# This avoids string-vs-numeric ID mismatches.
df["_claim_split_key"] = df[CLAIM_ID].astype(str)

unique_claims = df["_claim_split_key"].unique()

train_claims, test_claims = train_test_split(
    unique_claims,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_df = df[
    df["_claim_split_key"].isin(train_claims)
].copy()

test_df = df[
    df["_claim_split_key"].isin(test_claims)
].copy()

if set(train_df["_claim_split_key"]) & set(
    test_df["_claim_split_key"]
):
    raise RuntimeError(
        "Claim leakage detected between train and test."
    )

print(f"\nUnique claims: {len(unique_claims):,}")
print(f"Train rows:    {len(train_df):,}")
print(f"Test rows:     {len(test_df):,}")
print(
    f"Train denial rate: "
    f"{train_df[TARGET].mean():.2%}"
)
print(
    f"Test denial rate:  "
    f"{test_df[TARGET].mean():.2%}"
)


# ==========================================================
# FEATURE SELECTION
# ==========================================================

# Use the available dataset columns except explicitly excluded
# features and the internal split key.
FEATURE_COLS = [
    col for col in df.columns
    if col not in EXCLUDED_FEATURES
    and col != "_claim_split_key"
]

# Do not allow accidental inclusion of identifiers.
assert CLAIM_ID not in FEATURE_COLS
assert TARGET not in FEATURE_COLS

# Confirm the requested exclusions are absent.
for col in [
    "nationality",
    "claim_gross",
    "claim_net",
    "activity_gross",
    "billing_lag_days",
    "billing_lag_missing_flag",
]:
    assert col not in FEATURE_COLS, (
        f"Excluded feature unexpectedly included: {col}"
    )

# Verify receiver_id is present as an input.
assert "receiver_id" in FEATURE_COLS

# Retain only categorical columns actually present.
CAT_COLS = [
    col for col in CAT_COLS
    if col in FEATURE_COLS
]

X_train = train_df[FEATURE_COLS].copy()
X_test = test_df[FEATURE_COLS].copy()

y_train = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

print(f"\nNumber of input features: {len(FEATURE_COLS)}")
print(f"Categorical features: {CAT_COLS}")
print(f"Receiver ID included: {'receiver_id' in FEATURE_COLS}")

print("\nExcluded features:")
for col in EXCLUDED_FEATURES:
    print(f"  - {col}")


# ==========================================================
# DATA CLEANING
# ==========================================================

# Convert categorical columns to strings so mixed values
# are handled consistently by TargetEncoder.
for col in CAT_COLS:
    X_train[col] = (
        X_train[col]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

    X_test[col] = (
        X_test[col]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

# Reject unexpected text columns rather than silently
# encoding or dropping them.
remaining_strings = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

unexpected_strings = [
    col for col in remaining_strings
    if col not in CAT_COLS
]

if unexpected_strings:
    raise ValueError(
        "Unconfigured categorical/text columns found: "
        f"{unexpected_strings}. Add them to CAT_COLS "
        "or clean them explicitly."
    )

# Ensure test data has the same column order.
X_test = X_test[FEATURE_COLS]


# ==========================================================
# TARGET ENCODING
# ==========================================================

print("\nEncoding categorical features...")

encoder = TargetEncoder(
    cols=CAT_COLS,
    handle_missing="value",
    handle_unknown="value",
    smoothing=10.0,
)

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# Ensure all model inputs are numeric.
X_train = X_train.apply(pd.to_numeric, errors="raise")
X_test = X_test.apply(pd.to_numeric, errors="raise")

if X_train.isna().all(axis=0).any():
    bad_columns = X_train.columns[
        X_train.isna().all(axis=0)
    ].tolist()

    raise ValueError(
        f"Features are entirely missing after encoding: {bad_columns}"
    )

# Save the encoder and feature contract.
joblib.dump(encoder, ENCODER_PATH)

with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print("Encoder and feature schema saved.")


# ==========================================================
# CLASS BALANCING
# ==========================================================

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())

if neg == 0 or pos == 0:
    raise ValueError(
        "Training split must contain both target classes."
    )

scale_pos_weight = neg / pos

print(f"\nApproved training rows: {neg:,}")
print(f"Denied training rows:   {pos:,}")
print(f"scale_pos_weight:       {scale_pos_weight:.4f}")


# ==========================================================
# TRAIN XGBOOST
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
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating on held-out claims...")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "pr_auc": float(average_precision_score(y_test, y_prob)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_unique_claims": int(len(train_claims)),
    "test_unique_claims": int(len(test_claims)),
    "feature_count": int(len(FEATURE_COLS)),
    "excluded_features": EXCLUDED_FEATURES,
    "categorical_features": CAT_COLS,
    "random_state": RANDOM_STATE,
}

print("\n" + "=" * 70)
print("HELD-OUT TEST RESULTS")
print("=" * 70)

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key:<24}: {value:.4f}")
    else:
        print(f"{key:<24}: {value}")

print("=" * 70)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_importance = (
    pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    })
    .sort_values("importance", ascending=False)
)

print("\nTop 25 Features:\n")
print(feature_importance.head(25).to_string(index=False))

feature_importance.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False,
)


# ==========================================================
# SAVE MODEL AND METRICS
# ==========================================================

joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nSaved artifacts:")
print(f"Model:              {MODEL_PATH}")
print(f"Encoder:            {ENCODER_PATH}")
print(f"Metrics:            {METRICS_PATH}")
print(f"Feature schema:     {FEATURES_PATH}")
print(f"Feature importance: {MODEL_DIR / 'feature_importance.csv'}")

print("\nTraining complete.")


