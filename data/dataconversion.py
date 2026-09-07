import os
import pandas as pd
from sqlalchemy import create_engine

# 1. Connect locally
engine = create_engine('postgresql://postgres:2004@localhost:5432/Denial-prediction')

print("Fetching data and converting to Parquet...")
# df = pd.read_sql("SELECT * FROM public.denialclaims_features;", engine)
df = pd.read_sql("SELECT * FROM public.claim_activity_denial_model_v3;", engine)
print("current working directory",os.getcwd())

# 2. Save locally as compressed Parquet
df.to_parquet("claim_activity_denial_model_v3.parquet", index=False)
print("Done! Check your working folder for 'claim_activity_denial_model_v3.parquet'.")