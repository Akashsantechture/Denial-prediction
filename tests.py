import pandas as pd

# df = pd.read_parquet("data/denial_prediction_final.parquet")

# features = [
#     c for c in df.columns
#     if c not in [
#         "target_denied",
#         "activity_denied",
#         "haad_claim_id"
#     ]
# ]

# print(len(features))
# print(features)




# import joblib

# encoder = joblib.load(
#     "models/xgboost/target_encoder.joblib"
# )

# print(encoder.feature_names_in_)
# print(len(encoder.feature_names_in_))

from json import encoder
import pandas as pd
df = pd.read_parquet("data/denial_prediction_final.parquet")

EXPECTED_COLUMNS = [
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
    "clinician_category",
    "facility_type",
    "payer_classification",
    "diagnosis_code",
    "billing_lag_days",
    "icd_category",
    "cpt_category",
    "icd_cpt_domain_match"
]

df = df[EXPECTED_COLUMNS]
df_encoded = encoder.transform(df)