"""
build_claim_dataframe.py
"""

import pandas as pd

from preprocessing.feature_contract import (
    FEATURE_ORDER
)


def build_claim_dataframe(rows):

    df = pd.DataFrame(rows)

    df = df[FEATURE_ORDER]

    return df