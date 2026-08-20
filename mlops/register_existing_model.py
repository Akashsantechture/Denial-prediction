import joblib
import mlflow
import mlflow.sklearn

MODEL_PATH = "models/behavioral_xgboost_model2_individual_cats.joblib"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Denial-Prediction")

with mlflow.start_run(run_name="existing-denial-model") as run:

    model = joblib.load(MODEL_PATH)

    mlflow.log_artifact(
        MODEL_PATH,
        artifact_path="model"
    )

    mlflow.set_tag("model_source", "existing_joblib")
    mlflow.set_tag("model_framework", "xgboost")

    print("MLflow Run ID:", run.info.run_id)
    print("Model logged successfully.")