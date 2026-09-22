import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from category_encoders import TargetEncoder
from xgboost import XGBClassifier


# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/claim_denial_reason_model_v2.parquet"

MODEL_DIR = Path("models/denial_reason")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "target_encoder.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("Loading Dataset")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)

print(f"Dataset Shape: {df.shape}")


# ==========================================================
# KEEP ONLY DENIED ACTIVITIES
# ==========================================================

df = df[
    df["denial_category"].notna()
].copy()

print(
    f"Denied Activities: {len(df):,}"
)


# ==========================================================
# TARGET
# ==========================================================

TARGET = "denial_category"

DROP_COLS = [

    TARGET,

    # IDs
    "haad_claim_id",
    "mrn",
    "provider_id",
    "haad_claim_line_id",
    "claim_ref_no",
    "mrn",
    "provider_id",
    "payer_id",
    "accumed_patient_id",
    "datestamp",
    "date_submitted",

    # Payment
    "payment_reference",
    "activity_net",
    "payment_amount",

    # Denial leakage
    "activity_denial_code",
    "denial_description",
    "denial_type",
    "activity_denied",

    # Provider identifiers
    "activity_clinician",
    "activity_ordering_clinician",
    "clinician_license",
    "clinician_category",
    "facility_license",
    "facility_name",

    # Raw diagnosis fields
    "diagnosis_code",
    "diagnosis_description",
    "diagnosis_type",
    "icd_type",

    # Dates
    "date_of_birth",

    # Misc
    "claim_route"
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



X = df.drop(
    columns=DROP_COLS,
    errors="ignore"
)
X = X[FEATURE_ORDER]

print("\nTraining Feature Order:")
for col in X.columns:
    print(col)

y = df[TARGET]


# ==========================================================
# LABEL ENCODING TARGET
# ==========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

joblib.dump(
    label_encoder,
    LABEL_ENCODER_PATH
)

print("\nClasses Learned:\n")

for idx, cls in enumerate(
    label_encoder.classes_
):
    print(f"{idx} -> {cls}")


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

train_mask = (
    df["haad_claim_id"]
    .astype(str)
    .isin(train_claims)
)

test_mask = (
    df["haad_claim_id"]
    .astype(str)
    .isin(test_claims)
)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y[train_mask]
y_test = y[test_mask]

print(
    f"\nTrain Rows : {len(X_train):,}"
)

print(
    f"Test Rows  : {len(X_test):,}"
)


# ==========================================================
# TARGET ENCODING
# ==========================================================

print("\nApplying Target Encoding...")

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

print("Encoder Saved")


# ==========================================================
# SAFETY CHECK
# ==========================================================

remaining_strings = (
    X_train
    .select_dtypes(
        include=["object", "string"]
    )
    .columns
    .tolist()
)

if len(remaining_strings) > 0:

    raise ValueError(
        f"String Columns Found: {remaining_strings}"
    )

print(
    "\nAll Features Numeric ✓"
)


# ==========================================================
# CLASS BALANCING
# ==========================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print(
    "\nClass Balancing Enabled"
)


# ==========================================================
# MODEL
# ==========================================================

num_classes = len(
    label_encoder.classes_
)

print("\n" + "=" * 70)
print(
    f"Training XGBoost ({num_classes} Classes)"
)
print("=" * 70)

model = XGBClassifier(

    objective="multi:softprob",
    num_class=num_classes,

    eval_metric="mlogloss",

    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=5,
    gamma=0.2,

    tree_method="hist",

    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
)

print("Training Completed")


# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating...")

y_pred = model.predict(
    X_test
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

balanced_acc = balanced_accuracy_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Weighted F1      : {weighted_f1:.4f}"
)

print(
    f"Macro F1         : {macro_f1:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_acc:.4f}"
)

print("=" * 70)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


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
# SAVE METRICS
# ==========================================================

metrics = {

    "weighted_f1":
        float(weighted_f1),

    "macro_f1":
        float(macro_f1),

    "balanced_accuracy":
        float(balanced_acc),
}

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("ARTIFACTS SAVED")
print("=" * 70)

print(
    f"Model         : {MODEL_PATH}"
)

print(
    f"Encoder       : {ENCODER_PATH}"
)

print(
    f"Label Encoder : {LABEL_ENCODER_PATH}"
)

print(
    f"Metrics       : {METRICS_PATH}"
)

print("\nTraining Complete.")