
from __future__ import annotations

import pandas as pd


def apply_basic_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values before model-level preprocessing runs."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    result = df.copy()
    numeric_columns = result.select_dtypes(include=["number"]).columns
    categorical_columns = result.select_dtypes(exclude=["number"]).columns

    for column in numeric_columns:
        result[column] = result[column].fillna(result[column].median())

    for column in categorical_columns:
        mode = result[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        result[column] = result[column].fillna(fill_value)

    return result
