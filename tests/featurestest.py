import joblib

model = joblib.load(
    "models/denial_reason/model.joblib"
)

print("Feature Count:", model.n_features_in_)

print("\nFeatures:")
for f in model.feature_names_in_:
    print(f)

# reason_encoder = joblib.load(
#     "models/denial_reason/target_encoder.joblib"
# )

# prediction_encoder = joblib.load(
#     "models/xgboost/target_encoder.joblib"
# )

# print("Reason Encoder Columns:")
# print(reason_encoder.cols)

# print("\nPrediction Encoder Columns:")
# print(prediction_encoder.cols)


# import pandas as pd

# df = pd.read_parquet(
#     "data/claim_denial_reason_model_v2.parquet"
# )

# print(df.columns.tolist())
# print("\nTotal:", len(df.columns))