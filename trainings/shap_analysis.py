import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/claim_activity_denial_model_v3.parquet"

MODEL_PATH = "models/xgboost/model.joblib"
ENCODER_PATH = "models/xgboost/target_encoder.joblib"

OUTPUT_DIR = Path("models/xgboost")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)

print(f"Dataset Shape: {df.shape}")

# ==========================================================
# FEATURE PREPARATION
# ==========================================================

DROP_COLS = [
    "activity_denied",
    "haad_claim_id",
    "billing_lag_days",
    "billing_lag_missing_flag"
]

X = df.drop(
    columns=DROP_COLS,
    errors="ignore"
)

print(f"Feature Shape: {X.shape}")

# ==========================================================
# LOAD TARGET ENCODER
# ==========================================================

print("\nLoading Target Encoder...")

encoder = joblib.load(ENCODER_PATH)

X_encoded = encoder.transform(X)

print("Encoding Completed.")

# ==========================================================
# VALIDATE DATA TYPES
# ==========================================================

string_cols = X_encoded.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nRemaining String Columns:")
print(string_cols)

if len(string_cols) > 0:
    raise ValueError(
        f"String columns still exist: {string_cols}"
    )

# ==========================================================
# SAMPLE DATA
# ==========================================================

print("\nSampling data for SHAP...")

sample_size = min(
    10000,
    len(X_encoded)
)

X_sample = X_encoded.sample(
    n=sample_size,
    random_state=42
)

print(f"SHAP Sample Size: {len(X_sample):,}")

# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Model...")

model = joblib.load(MODEL_PATH)

print("Model Loaded.")

# ==========================================================
# XGBOOST FEATURE IMPORTANCE
# ==========================================================

print("\nGenerating XGBoost Feature Importance...")

xgb_importance = pd.DataFrame({
    "feature": X_encoded.columns,
    "gain": model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    "gain",
    ascending=False
)

xgb_importance.to_csv(
    OUTPUT_DIR / "xgb_feature_importance.csv",
    index=False
)

print("\nTop 25 XGBoost Features")

print(
    xgb_importance.head(25)
    .to_string(index=False)
)

# ==========================================================
# SHAP VALUES
# ==========================================================

print("\nCalculating SHAP Values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_sample
)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("SHAP Values Calculated.")

# ==========================================================
# SHAP INTERACTION VALUES
# ==========================================================

print("\nCalculating SHAP Interactions...")

interaction_sample = X_sample.sample(
    min(1000, len(X_sample)),
    random_state=42
)

interaction_values = (
    explainer.shap_interaction_values(
        interaction_sample
    )
)

feature_names = interaction_sample.columns

interaction_scores = []

for i in range(len(feature_names)):

    for j in range(i + 1, len(feature_names)):

        score = abs(
            interaction_values[:, i, j]
        ).mean()

        interaction_scores.append([
            feature_names[i],
            feature_names[j],
            score
        ])

interaction_df = pd.DataFrame(
    interaction_scores,
    columns=[
        "feature_1",
        "feature_2",
        "interaction_strength"
    ]
)

interaction_df = interaction_df.sort_values(
    "interaction_strength",
    ascending=False
)

interaction_df.to_csv(
    OUTPUT_DIR / "feature_interactions.csv",
    index=False
)

print("\nTop 20 Feature Interactions")

print(
    interaction_df.head(20)
    .to_string(index=False)
)

# ==========================================================
# CPT + ICD INTERACTIONS
# ==========================================================

print("\nSearching for CPT/ICD Interactions...")

keywords = [
    "activity_code",
    "primary_diagnosis_code",
    "primary_diagnosis_category",
    "cpt_category",
    "secondary"
]

medical_interactions = interaction_df[
    interaction_df["feature_1"].str.contains(
        "|".join(keywords),
        case=False
    )
    |
    interaction_df["feature_2"].str.contains(
        "|".join(keywords),
        case=False
    )
]

medical_interactions.to_csv(
    OUTPUT_DIR / "medical_interactions.csv",
    index=False
)

print("\nTop Medical Interactions")

print(
    medical_interactions.head(30)
    .to_string(index=False)
)

# ==========================================================
# SHAP FEATURE CONTRIBUTION
# ==========================================================

print("\nGenerating SHAP Contribution Table...")

importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": abs(
        shap_values
    ).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "feature_contribution.csv",
    index=False
)

print("\nTop 30 SHAP Features")

print(
    importance.head(30)
    .to_string(index=False)
)

# ==========================================================
# TOP 50 FEATURES EXPORT
# ==========================================================

importance.head(50).to_csv(
    OUTPUT_DIR / "top_50_shap_features.csv",
    index=False
)

# ==========================================================
# SHAP BAR PLOT
# ==========================================================

print("\nGenerating SHAP Importance Plot...")

top25 = importance.head(25)

plt.figure(figsize=(12, 10))

plt.barh(
    top25["feature"][::-1],
    top25["mean_abs_shap"][::-1]
)

plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Feature")
plt.title(
    "Top 25 SHAP Feature Contributions"
)

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

print("Generating SHAP Summary Plot...")

plt.figure(figsize=(14, 10))

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

print("\nGenerating Dependence Plots...")

candidate_features = [

    "activity_code",
    "activity_gross",

    "primary_diagnosis_code",
    "primary_diagnosis_category",

    "secondary_dx_count",

    "claim_gross",
    "claim_net",

    "patient_age",

    "cpt_category",

    "secondary_cardiology",
    "secondary_pulmonology",
    "secondary_musculoskeletal",
    "secondary_endocrinology",
    "secondary_infectious"
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

        print(
            f"Saved: {feature}_dependence.png"
        )

    except Exception as e:

        print(
            f"Failed {feature}: {e}"
        )

# ==========================================================
# SHAP BEESWARM (BEST VISUAL)
# ==========================================================

print("\nGenerating SHAP Beeswarm Plot...")

plt.figure(figsize=(14, 10))

shap.plots.beeswarm(
    shap.Explanation(
        values=shap_values,
        data=X_sample.values,
        feature_names=X_sample.columns
    ),
    max_display=25,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nResults Saved To: {OUTPUT_DIR}")

print("\nGenerated Files")

print("- xgb_feature_importance.csv")
print("- feature_contribution.csv")
print("- top_50_shap_features.csv")
print("- feature_interactions.csv")
print("- medical_interactions.csv")
print("- feature_importance.png")
print("- shap_summary.png")
print("- shap_beeswarm.png")

print("\nDependence Plots Generated For:")

for feature in candidate_features:
    print(f"- {feature}")

print("\nReady for Santosh Review.")