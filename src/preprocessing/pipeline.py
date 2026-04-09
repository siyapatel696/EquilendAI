
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Any

from .imputation import apply_basic_imputation


TARGET_COLUMN = "default_status"
PROTECTED_COLUMN = "gender"


def summarize_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    """Return a compact data-understanding summary for reporting/debugging."""
    if df is None or df.empty:
        raise ValueError("Dataset cannot be empty.")

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "numeric_summary": df.describe(include=["number"]).to_dict(),
        "target_distribution": df[target_column].value_counts(dropna=False).to_dict()
        if target_column in df.columns
        else {},
    }


def clean_and_engineer_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Clean data, handle missing values, engineer features, and cap outliers."""
    if df is None or df.empty:
        raise ValueError("Dataset cannot be empty.")

    cleaned = df.drop_duplicates().copy()
    cleaned = apply_basic_imputation(cleaned)

    if {"monthly_income", "utility_bill_average"}.issubset(cleaned.columns):
        income_denominator = cleaned["monthly_income"].clip(lower=1.0)
        cleaned["utility_to_income_ratio"] = (
            cleaned["utility_bill_average"] / income_denominator
        )
        cleaned["disposable_income"] = (
            cleaned["monthly_income"] - cleaned["utility_bill_average"]
        )

    if {"repayment_history_pct", "monthly_income"}.issubset(cleaned.columns):
        cleaned["repayment_income_interaction"] = (
            cleaned["repayment_history_pct"] * cleaned["monthly_income"]
        )

    if {"repayment_history_pct", "utility_bill_average"}.issubset(cleaned.columns):
        cleaned["bill_repayment_pressure"] = cleaned["repayment_history_pct"] / (
            cleaned["utility_bill_average"] + 1.0
        )

    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        if column == target_column:
            continue
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned[column] = cleaned[column].clip(lower=lower_bound, upper=upper_bound)

    return cleaned


def select_model_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    protected_column: str = PROTECTED_COLUMN,
    correlation_threshold: float = 0.98,
) -> pd.DataFrame:
    """Drop highly correlated numeric features while preserving target/protected fields."""
    if df is None or df.empty:
        raise ValueError("Dataset cannot be empty.")

    numeric_features = [
        column
        for column in df.select_dtypes(include=["number"]).columns
        if column != target_column
    ]
    if len(numeric_features) < 2:
        return df.copy()

    corr_matrix = df[numeric_features].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > correlation_threshold)
    ]

    protected_to_keep = {target_column, protected_column}
    final_drop = [column for column in to_drop if column not in protected_to_keep]
    return df.drop(columns=final_drop, errors="ignore")


def prepare_training_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    protected_column: str = PROTECTED_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full cleaning and feature-prep workflow for model training."""
    summary_before = summarize_dataset(df, target_column=target_column)
    processed = clean_and_engineer_features(df, target_column=target_column)
    processed = select_model_features(
        processed,
        target_column=target_column,
        protected_column=protected_column,
    )
    summary_after = summarize_dataset(processed, target_column=target_column)

    return processed, {
        "before": summary_before,
        "after": summary_after,
    }


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    TASK 04: Scaling & Encoding Pipeline

    Build a reusable preprocessing pipeline for EquiLend training/inference.
    """
    if X is None or X.empty:
        raise ValueError("Input feature DataFrame cannot be empty.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No numeric or categorical features found for preprocessing.")

    return ColumnTransformer(transformers=transformers)
