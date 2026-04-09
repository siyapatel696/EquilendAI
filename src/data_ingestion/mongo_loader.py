
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pymongo import MongoClient


DEFAULT_DATA_PATH = Path("data/equilend_mock_data.csv")
DEFAULT_DB_NAME = "equilend_ai"
DEFAULT_COLLECTION_NAME = "loan_applications"
DEFAULT_DECISIONS_COLLECTION = "loan_decisions"
REQUIRED_TRAINING_COLUMNS = {
    "gender",
    "monthly_income",
    "utility_bill_average",
    "repayment_history_pct",
    "employment_length",
    "default_status",
}


def get_mongo_client(uri: str | None = None) -> MongoClient:
    """Create a MongoDB client using an explicit URI or environment variable."""
    mongo_uri = uri or os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("Missing MONGODB_URI environment variable.")
    return MongoClient(mongo_uri)


def load_data_from_mongo(
    uri: str | None = None,
    db_name: str = DEFAULT_DB_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> pd.DataFrame:
    """Load application records from MongoDB Atlas into a DataFrame."""
    client = get_mongo_client(uri)
    try:
        records = list(client[db_name][collection_name].find({}, {"_id": 0}))
    finally:
        client.close()

    return pd.DataFrame(records)


def load_data_from_csv(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the synthetic training dataset from local storage."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run `python scripts/generate_data.py` first."
        )
    return pd.read_csv(path)


def validate_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the ingested training data matches the expected EquiLend schema."""
    if df is None or df.empty:
        raise ValueError("Training data is empty.")

    missing_columns = REQUIRED_TRAINING_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: {sorted(missing_columns)}"
        )

    cleaned = df.drop_duplicates().copy()
    cleaned["monthly_income"] = pd.to_numeric(
        cleaned["monthly_income"], errors="coerce"
    )
    cleaned["utility_bill_average"] = pd.to_numeric(
        cleaned["utility_bill_average"], errors="coerce"
    )
    cleaned["repayment_history_pct"] = pd.to_numeric(
        cleaned["repayment_history_pct"], errors="coerce"
    )
    cleaned["default_status"] = pd.to_numeric(
        cleaned["default_status"], errors="coerce"
    )

    return cleaned


def load_training_dataframe(
    prefer_mongo: bool = True,
    csv_path: str | Path = DEFAULT_DATA_PATH,
    uri: str | None = None,
    db_name: str = DEFAULT_DB_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> pd.DataFrame:
    """
    Load training data from the ingestion layer.

    Prefers MongoDB when configured and populated, then falls back to the generated CSV.
    """
    if prefer_mongo:
        try:
            df = validate_training_dataframe(
                load_data_from_mongo(
                uri=uri,
                db_name=db_name,
                collection_name=collection_name,
                )
            )
            if not df.empty:
                return df
        except Exception:
            pass

    return validate_training_dataframe(load_data_from_csv(csv_path))


def save_decision_record(
    record: dict,
    uri: str | None = None,
    db_name: str = DEFAULT_DB_NAME,
    collection_name: str = DEFAULT_DECISIONS_COLLECTION,
) -> str:
    """Persist a single credit decision to MongoDB and return the inserted id."""
    client = get_mongo_client(uri)
    try:
        result = client[db_name][collection_name].insert_one(record)
        return str(result.inserted_id)
    finally:
        client.close()


def load_decision_records(
    uri: str | None = None,
    db_name: str = DEFAULT_DB_NAME,
    collection_name: str = DEFAULT_DECISIONS_COLLECTION,
    limit: int = 100,
) -> pd.DataFrame:
    """Load recent saved credit decisions from MongoDB."""
    client = get_mongo_client(uri)
    try:
        cursor = (
            client[db_name][collection_name]
            .find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        return pd.DataFrame(list(cursor))
    finally:
        client.close()
