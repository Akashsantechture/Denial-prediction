
import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ==========================================================
# CONFIG
# ==========================================================

PROJECT_DIR = Path(
    r"C:\Users\akash.subramani.SANTECHTURE\Desktop\projects\Denial-prediction"
)

# Export public.ksa_denial_reason_features to this Parquet file.
DATA_PATH = PROJECT_DIR / "data" / "ksa_denial_reason_features.parquet"

# Fixed output directory requested
MODEL_DIR = Path(
    r"C:\Users\akash.subramani.SANTECHTURE\Desktop\projects\Denial-prediction\models\ksa\denial_reason"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

TARGET = "denial_category"
CLAIM_ID = "claim_id"
DENIED_FLAG = "activity_denied"

RANDOM_STATE = 42
TEST_SIZE_FOLDS = 5  # approximately 20% held out


# ==========================================================
# KSA FEATURES
# ==========================================================

CAT_COLS = [
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
]

NUMERIC_COLS = [
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

FEATURE_ORDER = CAT_COLS + NUMERIC_COLS


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 75)
print("LOADING KSA DENIAL CATEGORY DATASET")
print("=" * 75)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Export public.ksa_denial_reason_features to this path "
        "or update DATA_PATH."
    )

df = pd.read_parquet(DATA_PATH)

print(f"Original dataset shape: {df.shape}")

required_cols = [
    CLAIM_ID,
    DENIED_FLAG,
    TARGET,
    *FEATURE_ORDER,
]

missing_cols = sorted(set(required_cols) - set(df.columns))

if missing_cols:
    raise ValueError(
        f"Missing required dataset columns: {missing_cols}"
    )


# ==========================================================
# KEEP ONLY DENIED ACTIVITIES
# ==========================================================

df[DENIED_FLAG] = pd.to_numeric(
    df[DENIED_FLAG],
    errors="coerce",
)

df = df.loc[
    (df[DENIED_FLAG] == 1)
    & df[TARGET].notna()
    & df[CLAIM_ID].notna()
].copy()

df[TARGET] = df[TARGET].astype(str).str.strip()
df = df.loc[df[TARGET] != ""].copy()

# Normalize claim identifiers for grouping.
df[CLAIM_ID] = df[CLAIM_ID].astype(str)

# Reset indexes to keep all training arrays aligned.
df.reset_index(drop=True, inplace=True)

if df.empty:
    raise ValueError(
        "No denied activities with a valid denial_category and claim_id."
    )

print(f"Denied activities used: {len(df):,}")
print(f"Unique claims: {df[CLAIM_ID].nunique():,}")

print("\nHigh-level category distribution:")
print(df[TARGET].value_counts().to_string())


# ==========================================================
# VALIDATE FEATURES
# ==========================================================

# Do not use claim identifiers, denial details, or financial fields
# as model inputs. Only FEATURE_ORDER is used below.
X = df[FEATURE_ORDER].copy()
y_text = df[TARGET].copy()
groups = df[CLAIM_ID].to_numpy()

# Convert categorical inputs to strings while preserving missing values.
for col in CAT_COLS:
    X[col] = X[col].map(
        lambda value: str(value) if pd.notna(value) else np.nan
    )

# Convert numeric inputs safely.
for col in NUMERIC_COLS:
    X[col] = pd.to_numeric(X[col], errors="coerce")

if y_text.nunique() < 2:
    raise ValueError(
        "At least two denial categories are required for classification."
    )


# ==========================================================
# ENCODE TARGET LABELS
# ==========================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

print("\nTarget class mapping:")
for idx, category in enumerate(label_encoder.classes_):
    print(f"{idx} -> {category}")

# Save the mapping for inference and decoding predictions.
joblib.dump(label_encoder, LABEL_ENCODER_PATH)


# ==========================================================
# CLAIM-LEVEL TRAIN / TEST SPLIT
# ==========================================================

# StratifiedGroupKFold approximately preserves class distribution
# while keeping each claim entirely within one split.
splitter = StratifiedGroupKFold(
    n_splits=TEST_SIZE_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y[train_idx]
y_test = y[test_idx]

train_claims = set(groups[train_idx])
test_claims = set(groups[test_idx])

assert train_claims.isdisjoint(test_claims), (
    "Claim leakage detected between training and test sets."
)

print("\nSplit summary:")
print(f"Train rows: {len(X_train):,}")
print(f"Test rows : {len(X_test):,}")
print(f"Train claims: {len(train_claims):,}")
print(f"Test claims : {len(test_claims):,}")

print("\nTraining category counts:")
print(
    pd.Series(
        label_encoder.inverse_transform(y_train)
    ).value_counts().to_string()
)

print("\nTest category counts:")
print(
    pd.Series(
        label_encoder.inverse_transform(y_test)
    ).value_counts().to_string()
)

# Fail rather than silently train/evaluate without a category.
if len(np.unique(y_train)) != len(label_encoder.classes_):
    raise ValueError(
        "At least one category is absent from training. "
        "Review category frequency and the claim-level split."
    )


# ==========================================================
# PREPROCESSING
# ==========================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, CAT_COLS),
        ("numeric", numeric_pipeline, NUMERIC_COLS),
    ],
    remainder="drop",
)


# ==========================================================
# CLASS BALANCING
# ==========================================================

# Balance the training data only.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

print("\nClass balancing enabled.")


# ==========================================================
# XGBOOST MODEL
# ==========================================================

num_classes = len(label_encoder.classes_)

classifier = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",

    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=5,
    gamma=0.2,

    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Save the preprocessor and classifier together so inference uses
# exactly the same feature transformation as training.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)

print("\n" + "=" * 75)
print(f"TRAINING KSA MODEL ({num_classes} CATEGORIES)")
print("=" * 75)

model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights,
)

print("Training completed.")


# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating on held-out claims...")

y_pred = model.predict(X_test).astype(int)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

balanced_acc = balanced_accuracy_score(
    y_test,
    y_pred,
)

all_labels = np.arange(num_classes)

report = classification_report(
    y_test,
    y_pred,
    labels=all_labels,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)

print("\n" + "=" * 75)
print("KSA DENIAL CATEGORY MODEL RESULTS")
print("=" * 75)

print(f"Weighted F1       : {weighted_f1:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Balanced Accuracy : {balanced_acc:.4f}")

print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

print("\nConfusion matrix (rows=true, columns=predicted):\n")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=all_labels,
    )
)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

encoded_feature_names = fitted_preprocessor.get_feature_names_out()

feature_importance = (
    pd.DataFrame(
        {
            "feature": encoded_feature_names,
            "importance": fitted_classifier.feature_importances_,
        }
    )
    .sort_values("importance", ascending=False)
)

feature_importance.to_csv(
    IMPORTANCE_PATH,
    index=False,
)

print("\nTop 25 encoded features:\n")
print(feature_importance.head(25).to_string(index=False))


# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = {
    "model_type": "ksa_denial_category_classifier",
    "target": TARGET,
    "dataset": str(DATA_PATH),
    "rows_used": int(len(df)),
    "unique_claims": int(df[CLAIM_ID].nunique()),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_claims": int(len(train_claims)),
    "test_claims": int(len(test_claims)),
    "categories": label_encoder.classes_.tolist(),
    "category_counts": {
        str(k): int(v)
        for k, v in df[TARGET].value_counts().items()
    },
    "weighted_f1": float(weighted_f1),
    "macro_f1": float(macro_f1),
    "balanced_accuracy": float(balanced_acc),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(
        y_test,
        y_pred,
        labels=all_labels,
    ).tolist(),
    "random_state": RANDOM_STATE,
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)


# ==========================================================
# SAVE MODEL PIPELINE
# ==========================================================

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 75)
print("KSA MODEL ARTIFACTS SAVED")
print("=" * 75)

print(f"Model pipeline : {MODEL_PATH}")
print(f"Label encoder  : {LABEL_ENCODER_PATH}")
print(f"Metrics        : {METRICS_PATH}")
print(f"Feature import.: {IMPORTANCE_PATH}")

print("\nKSA denial category training complete.")
