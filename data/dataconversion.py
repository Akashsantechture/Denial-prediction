import os
import pandas as pd
from sqlalchemy import create_engine

# 1. Connect locally
engine = create_engine('postgresql://postgres:2004@localhost:5432/Denial-prediction')

print("Fetching data and converting to Parquet...")
# df = pd.read_sql("SELECT * FROM public.denialclaims_features;", engine)
df = pd.read_sql("SELECT * FROM public.denial_prediction_final;", engine)
print("current working directory",os.getcwd())

# 2. Save locally as compressed Parquet
df.to_parquet("denial_prediction_final.parquet", index=False)
print("Done! Check your working folder for 'denial_prdiction_final.parquet'.")