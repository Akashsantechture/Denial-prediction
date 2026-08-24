"""data configs of row and health checks on parquetfiles"""
# import pandas as pd

# df = pd.read_parquet("data/denial_prediction_final.parquet")

# print(df.shape)
# print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

"""DATAtesting of data types on parquetfiles"""

import pandas as pd

df = pd.read_parquet("data/denial_prediction_final.parquet")

print(df.dtypes)