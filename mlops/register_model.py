from pathlib import Path
import json
import platform
import warnings

import joblib
import mlflow
import mlflow.xgboost
import xgboost


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "behavioral_xgboost_model2_individual_cats.joblib"
)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "Denial-Prediction"

REGISTERED_MODEL_NAME = "DenialPredictionXGBoost"


FEATURE_NAMES = [
    "diagnosis_code",
    "diagnosis_type",
    "activity_code",
    "gender",
    "nationality",
    "encounter_type",
    "clinician_profession",
    "clinician_category",
    "facility_type",
    "icd_category",
    "cpt_category",
    "activity_quantity",
    "patient_age",
    "activity_gross_log",
    "claim_gross_log",
    "claim_net_log",
    "high_cost_flag",
    "length_of_stay",
    "billing_lag_days",
]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DENIAL PREDICTION - MLOPS MODEL REGISTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate model path
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    print(f"\nModel path:")
    print(f"  {MODEL_PATH}")

    # --------------------------------------------------------
    # Connect to MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("\nMLflow tracking URI:")
    print(f"  {MLFLOW_TRACKING_URI}")

    # --------------------------------------------------------
    # Create / get experiment
    # --------------------------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nMLflow experiment:")
    print(f"  {EXPERIMENT_NAME}")

    # --------------------------------------------------------
    # Load existing model
    # --------------------------------------------------------

    print("\nLoading existing model...")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        model = joblib.load(MODEL_PATH)

    print("  Model loaded successfully.")

    # --------------------------------------------------------
    # Validate model type
    # --------------------------------------------------------

    if not isinstance(model, xgboost.XGBClassifier):
        raise TypeError(
            f"Expected XGBClassifier, got {type(model)}"
        )

    print("\nModel validation:")
    print(f"  Type: {type(model).__name__}")

    # --------------------------------------------------------
    # Extract model information
    # --------------------------------------------------------

    booster = model.get_booster()

    actual_feature_names = booster.feature_names

    if actual_feature_names:
        print(f"  Feature count: {len(actual_feature_names)}")
        print(f"  Features match expected: "
              f"{actual_feature_names == FEATURE_NAMES}")
    else:
        print("  Feature names: Not available")

    print(f"  Number of input features: {model.n_features_in_}")
    print(f"  Classes: {model.classes_}")
    print(f"  Device: {model.get_params().get('device')}")
    print(
        f"  Enable categorical: "
        f"{model.get_params().get('enable_categorical')}"
    )
    print(f"  XGBoost version: {xgboost.__version__}")

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="existing_behavioral_xgboost_model"
    ) as run:

        run_id = run.info.run_id

        print("\nMLflow run started:")
        print(f"  Run ID: {run_id}")

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        params = model.get_params()

        # MLflow parameters must be scalar/string-compatible.
        safe_params = {}

        for key, value in params.items():

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                safe_params[key] = value
            else:
                safe_params[key] = str(value)

        mlflow.log_params(safe_params)

        # ----------------------------------------------------
        # Log model metadata
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "model_type": "XGBClassifier",
                "model_source": "existing_joblib",
                "source_model_path": str(MODEL_PATH),
                "framework": "xgboost",
                "xgboost_version": xgboost.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "device": str(model.get_params().get("device")),
                "enable_categorical": str(
                    model.get_params().get("enable_categorical")
                ),
                "feature_count": str(model.n_features_in_),
                "model_status": "candidate",
                "artifact_validation": "load_success",
            }
        )

        # ----------------------------------------------------
        # Log feature metadata
        # ----------------------------------------------------

        feature_metadata = {
            "feature_count": len(FEATURE_NAMES),
            "features": FEATURE_NAMES,
            "classes": model.classes_.tolist(),
            "n_features_in": int(model.n_features_in_),
            "device": model.get_params().get("device"),
            "enable_categorical": model.get_params().get(
                "enable_categorical"
            ),
            "xgboost_version": xgboost.__version__,
        }

        feature_metadata_path = (
            PROJECT_ROOT
            / "mlops"
            / "feature_metadata.json"
        )

        with open(feature_metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                feature_metadata,
                f,
                indent=2
            )

        mlflow.log_artifact(
            str(feature_metadata_path),
            artifact_path="metadata"
        )

        # ----------------------------------------------------
        # Log original joblib artifact
        # ----------------------------------------------------

        print("\nLogging original .joblib artifact...")

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="source_model"
        )

        print("  Source artifact logged.")

        # ----------------------------------------------------
        # Log XGBoost model using MLflow XGBoost flavor
        # ----------------------------------------------------

        print("\nLogging XGBoost model to MLflow...")

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="denial_prediction_model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("  XGBoost model logged.")

        # ----------------------------------------------------
        # Log compatibility warning information
        # ----------------------------------------------------

        if caught_warnings:

            compatibility_warnings = []

            for warning in caught_warnings:
                compatibility_warnings.append(
                    str(warning.message)
                )

            warning_path = (
                PROJECT_ROOT
                / "mlops"
                / "model_load_warnings.txt"
            )

            with open(
                warning_path,
                "w",
                encoding="utf-8"
            ) as f:

                for warning_text in compatibility_warnings:
                    f.write(warning_text)
                    f.write("\n\n")

            mlflow.log_artifact(
                str(warning_path),
                artifact_path="validation"
            )

            mlflow.set_tag(
                "serialization_warning",
                "true"
            )

            print(
                "\nWARNING:"
                "\n  Existing model produced a serialization "
                "compatibility warning."
            )

        else:

            mlflow.set_tag(
                "serialization_warning",
                "false"
            )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("MODEL REGISTRATION COMPLETE")
        print("=" * 70)

        print(f"\nExperiment:")
        print(f"  {EXPERIMENT_NAME}")

        print(f"\nRun ID:")
        print(f"  {run_id}")

        print(f"\nModel:")
        print(f"  {REGISTERED_MODEL_NAME}")

        print(f"\nModel URI:")
        print(f"  {model_info.model_uri}")

        print("\nMLflow UI:")
        print(f"  {MLFLOW_TRACKING_URI}")

        print("\nOriginal model:")
        print(f"  {MODEL_PATH}")

        print("\nIMPORTANT:")
        print(
            "  The model has been registered as a candidate."
        )
        print(
            "  Do NOT consider it production-ready yet."
        )

        print("=" * 70)


if __name__ == "__main__":
    main()