import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

DATA_PATH = "data/denial_prediction_final.parquet"

MODEL_PATH = "models/xgboost/model.joblib"
ENCODER_PATH = "models/xgboost/target_encoder.joblib"

OUTPUT_DIR = Path("models/xgboost")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading data...")

df = pd.read_parquet(DATA_PATH)

X = df.drop(
    columns=[
        "target_denied",
        "activity_denied",
        "haad_claim_id"
    ],
    errors="ignore"
)

# ==========================================================
# LOAD TARGET ENCODER
# ==========================================================

print("Loading encoder...")

encoder = joblib.load(ENCODER_PATH)

X_encoded = encoder.transform(X)

# ==========================================================
# SAMPLE DATA
# ==========================================================

print("Sampling rows for SHAP...")

sample_size = min(10000, len(X_encoded))

X_sample = X_encoded.sample(
    n=sample_size,
    random_state=42
)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

# ==========================================================
# SHAP CALCULATION
# ==========================================================

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

# ==========================================================
# FEATURE CONTRIBUTION TABLE
# ==========================================================

print("Generating feature contribution table...")

importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "feature_contribution.csv",
    index=False
)

print("\nTop 20 Features")
print(importance.head(20))

# ==========================================================
# FEATURE IMPORTANCE BAR PLOT
# ==========================================================

print("Generating feature importance plot...")

top20 = importance.head(20)

plt.figure(figsize=(12, 8))

plt.barh(
    top20["feature"][::-1],
    top20["mean_abs_shap"][::-1]
)

plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Feature")
plt.title("Top 20 Feature Contributions")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SHAP SUMMARY PLOT
# ==========================================================

print("Generating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# DEPENDENCE PLOTS
# ==========================================================

print("Generating dependence plots...")

candidate_features = [
    "billing_lag_days",
    "activity_gross",
    "activity_code",
    "diagnosis_code",
    "icd_cpt_domain_match"
]

for feature in candidate_features:

    if feature not in X_sample.columns:
        print(f"Skipping {feature}")
        continue

    try:

        shap.dependence_plot(
            feature,
            shap_values,
            X_sample,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{feature}_dependence.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {feature}_dependence.png")

    except Exception as e:
        print(f"Failed {feature}: {e}")

# ==========================================================
# DONE
# ==========================================================

print("\n===================================")
print("SHAP ANALYSIS COMPLETED")
print("===================================")

print(f"Results saved in: {OUTPUT_DIR}")