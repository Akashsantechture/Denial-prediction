import pandas as pd
from sqlalchemy import create_engine

# 1. Read CSV
csv_path = r"C:\Users\akash.subramani.SANTECHTURE\Downloads\FINNEVA CLAIMS.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

# 2. Connect to your database (insert your PostgreSQL password)
engine = create_engine(
    "postgresql+psycopg2://postgres:2004@localhost:5432/Denial-prediction"
)

# 3. 'replace' automatically creates the table structure from the CSV headers
df.to_sql("FINNEVA CLAIMS", engine, if_exists="replace", index=False)

print(f"Table created with {len(df.columns)} columns and {len(df):,} rows inserted!")