import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

# Update DATA_PATH if your exported KSA Parquet has a different filename.
DATA_PATH = Path("data/ksa_prediction_model_features.parquet")
MODEL_DIR = Path("models/ksa/prediction_score")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "activity_denied"
CLAIM_ID = "claim_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50

# Explicit pre-submission feature allowlist. Uses service_code, NOT CPT.
CAT_COLS = [
    "service_code", "gender", "encounter_type", "facility", "facility_type",
    "payer_id", "receiver_id", "payer_classification", "department",
    "claim_type", "claim_subtype", "primary_diagnosis","plan_name",
    "primary_diagnosis_category",
]
NUMERIC_COLS = [
    "patient_age", "quantity", "length_of_stay",
    "secondary_dx_infections", "secondary_dx_neoplasms",
    "secondary_dx_blood_immune", "secondary_dx_endocrine_metabolic",
    "secondary_dx_mental", "secondary_dx_nervous_system",
    "secondary_dx_eye", "secondary_dx_ear", "secondary_dx_circulatory",
    "secondary_dx_respiratory", "secondary_dx_digestive", "secondary_dx_skin",
    "secondary_dx_musculoskeletal", "secondary_dx_genitourinary",
    "secondary_dx_obstetrics", "secondary_dx_perinatal",
    "secondary_dx_congenital", "secondary_dx_symptoms",
    "secondary_dx_injury_poisoning", "secondary_dx_external_causes",
    "secondary_dx_health_status", "secondary_dx_special_purposes",
]
# These must never be model inputs for pre-submission prediction.
FORBIDDEN = {
    TARGET, CLAIM_ID, "denial_code", "denial_reason", "denial_category",
    "claim_status", "adjudication_status", "cpt_category", "cpt_code",
    "nationality", "unit_price", "net_amount", "claim_gross", "claim_net",
    "activity_gross", "billing_lag_days", "billing_lag_missing_flag",
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}. Update DATA_PATH to your KSA Parquet."
        )

    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    for col in (TARGET, CLAIM_ID):
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
    if df[TARGET].isna().any() or df[CLAIM_ID].isna().any():
        raise ValueError(f"{TARGET} and {CLAIM_ID} must not contain nulls.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{TARGET} must contain both 0 and 1.")

    requested = CAT_COLS + NUMERIC_COLS
    missing = [c for c in requested if c not in df.columns]
    if missing:
        warnings.warn("Skipping missing configured features: " + ", ".join(missing))

    features = [c for c in requested if c in df.columns and c not in FORBIDDEN]
    cats = [c for c in CAT_COLS if c in features]
    nums = [c for c in NUMERIC_COLS if c in features]
    if "service_code" not in features:
        raise ValueError("service_code is missing. This script intentionally does not use CPT.")
    if not cats or not nums:
        warnings.warn("No categorical or no numeric features are configured/present.")

    # Group by claim so activities from the same claim cannot cross the split.
    df["_claim_key"] = df[CLAIM_ID].astype(str)
    claims = df["_claim_key"].drop_duplicates().to_numpy()
    train_claims, test_claims = train_test_split(
        claims, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_set, test_set = set(train_claims), set(test_claims)
    if train_set & test_set:
        raise RuntimeError("Claim leakage between train and test.")
    train_df = df[df["_claim_key"].isin(train_set)].copy()
    test_df = df[df["_claim_key"].isin(test_set)].copy()

    X_train, X_test = train_df[features].copy(), test_df[features].copy()
    y_train, y_test = train_df[TARGET], test_df[TARGET]
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both train and test must contain approved and denied rows.")

    for col in cats:
        X_train[col] = X_train[col].astype("string").fillna("__MISSING__").astype(str)
        X_test[col] = X_test[col].astype("string").fillna("__MISSING__").astype(str)
    for col in nums:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    transformers = []
    if cats:
        # sklearn TargetEncoder.fit_transform cross-fits training encodings.
        transformers.append((
            "categorical",
            TargetEncoder(
                target_type="binary", smooth="auto", cv=5,
                shuffle=True, random_state=RANDOM_STATE,
            ),
            cats,
        ))
    if nums:
        transformers.append(("numeric", "passthrough", nums))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    if not neg or not pos:
        raise ValueError("Training data must contain both target classes.")
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
        scale_pos_weight=neg / pos,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    print(f"Unique claims: {len(claims):,}")
    print(f"Train claims/rows: {len(train_claims):,} / {len(train_df):,}")
    print(f"Test claims/rows:  {len(test_claims):,} / {len(test_df):,}")
    print(f"Train/test denial rates: {y_train.mean():.2%} / {y_test.mean():.2%}")
    print(f"Features ({len(features)}): {features}")
    print(f"scale_pos_weight: {neg / pos:.4f}")

    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "model_type": "KSA activity-level binary denial prediction",
        "target": TARGET,
        "prediction_unit": "activity row",
        "service_code_used": True,
        "cpt_features_used": False,
        "threshold": THRESHOLD,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_unique_claims": len(train_claims),
        "test_unique_claims": len(test_claims),
        "train_denial_rate": float(y_train.mean()),
        "test_denial_rate": float(y_test.mean()),
        "feature_count": len(features),
        "feature_columns": features,
        "categorical_features": cats,
        "numeric_features": nums,
        "excluded_features": sorted(FORBIDDEN),
    }

    print("\\nHeld-out test metrics:")
    for key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        print(f"{key:<20}: {metrics[key]:.4f}")
    report = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["approved", "denied"], output_dict=True, zero_division=0,
    )
    print("\\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["approved", "denied"], zero_division=0,
    ))

    model_dir = MODEL_DIR
    joblib.dump(pipeline, model_dir / "model_pipeline.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (model_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        index=["actual_approved", "actual_denied"],
        columns=["predicted_approved", "predicted_denied"],
    ).to_csv(model_dir / "confusion_matrix.csv")

    try:
        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.DataFrame({
            "feature": transformed_names,
            "importance": pipeline.named_steps["model"].feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(model_dir / "feature_importance.csv", index=False)
        print("\\nTop 25 transformed features:")
        print(importance.head(25).to_string(index=False))
    except Exception as exc:
        warnings.warn(f"Could not export feature importance: {exc}")

    print("\\nSaved artifacts to:", model_dir.resolve())
    print("  model_pipeline.joblib")
    print("  metrics.json")
    print("  feature_columns.json")
    print("  classification_report.json")
    print("  confusion_matrix.csv")
    print("  feature_importance.csv (if supported)")


if __name__ == "__main__":
    main()
