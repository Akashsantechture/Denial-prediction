import pandas as pd
from sqlalchemy import create_engine

# 1. Connect locally
engine = create_engine('postgresql://postgres:2004@localhost:5432/Denial-prediction')

print("Fetching data and converting to Parquet...")
df = pd.read_sql("SELECT * FROM public.denial_prediction_features;", engine)

# 2. Save locally as compressed Parquet
df.to_parquet("denial_prediction_features.parquet", index=False)
print("Done! Check your working folder for 'denial_prediction_features.parquet'.")