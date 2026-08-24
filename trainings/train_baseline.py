import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/denial_prediction_final.parquet")

claims = df["haad_claim_id"].astype(str).unique().tolist()

train_claims, test_claims = train_test_split(
    claims,
    test_size=0.2,
    random_state=42
)

train_df = df[df["haad_claim_id"].isin(train_claims)]
test_df = df[df["haad_claim_id"].isin(test_claims)]

print("Train rows:", len(train_df))
print("Test rows :", len(test_df))
print("Train claims:", train_df["haad_claim_id"].nunique())
print("Test claims :", test_df["haad_claim_id"].nunique())