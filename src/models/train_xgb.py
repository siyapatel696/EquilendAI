from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.data_ingestion.mongo_loader import load_training_dataframe
    from src.preprocessing.oversampling import compute_class_weight_ratio
    from src.preprocessing.pipeline import (
        build_preprocessing_pipeline,
        prepare_training_dataset,
    )
except ModuleNotFoundError:
    from data_ingestion.mongo_loader import load_training_dataframe
    from preprocessing.oversampling import compute_class_weight_ratio
    from preprocessing.pipeline import (
        build_preprocessing_pipeline,
        prepare_training_dataset,
    )


TARGET_COLUMN = "default_status"
PROTECTED_COLUMN = "gender"
RANDOM_STATE = 42


@dataclass
class FairnessSummary:
    demographic_parity_difference: float
    equal_opportunity_difference: float
    positive_prediction_rate_by_group: dict[str, float]
    true_positive_rate_by_group: dict[str, float]


def build_preprocessor(X: pd.DataFrame):
    """Build the preprocessing graph for numeric and categorical features."""
    return build_preprocessing_pipeline(X)


def evaluate_fairness(
    protected_values: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> FairnessSummary:
    """Compute simple group fairness diagnostics for the protected attribute."""
    group_rates: dict[str, float] = {}
    group_tprs: dict[str, float] = {}

    protected = protected_values.fillna("Unknown").astype(str).reset_index(drop=True)
    truth = y_true.reset_index(drop=True)
    pred = pd.Series(y_pred).reset_index(drop=True)

    for group in sorted(protected.unique()):
        mask = protected == group
        group_truth = truth[mask]
        group_pred = pred[mask]

        group_rates[group] = float(group_pred.mean()) if len(group_pred) else 0.0

        positives = group_truth == 1
        if positives.any():
            group_tprs[group] = float(group_pred[positives].mean())
        else:
            group_tprs[group] = 0.0

    return FairnessSummary(
        demographic_parity_difference=max(group_rates.values()) - min(group_rates.values()),
        equal_opportunity_difference=max(group_tprs.values()) - min(group_tprs.values()),
        positive_prediction_rate_by_group=group_rates,
        true_positive_rate_by_group=group_tprs,
    )


def build_xgb_model(scale_pos_weight: float) -> XGBClassifier:
    """Return the tuned XGBoost model used for credit-risk training."""
    return XGBClassifier(
        n_estimators=1200,
        max_depth=3,
        learning_rate=0.02,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=2.5,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
    )


def train_optimized_model(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    protected_column: str = PROTECTED_COLUMN,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Train an optimized XGBoost pipeline for credit-risk prediction.

    The dataset is expected to come from the data-ingestion layer and include the
    protected attribute for fairness analysis.
    """
    if target_column not in df.columns:
        raise ValueError(f"Expected target column '{target_column}' in training data.")
    if protected_column not in df.columns:
        raise ValueError(
            f"Expected protected attribute column '{protected_column}' in training data."
        )

    dataset, preprocessing_report = prepare_training_dataset(
        df,
        target_column=target_column,
        protected_column=protected_column,
    )
    dataset = dataset.dropna(subset=[target_column]).copy()
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scale_pos_weight = compute_class_weight_ratio(y_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", build_xgb_model(scale_pos_weight=scale_pos_weight)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    auc_score = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)

    fairness = evaluate_fairness(
        protected_values=X_test[protected_column],
        y_true=y_test,
        y_pred=pd.Series(y_pred),
    )

    risk_labels = pd.Series(
        pd.cut(
            y_proba,
            bins=[-0.01, 0.30, 0.70, 1.0],
            labels=["Low", "Medium", "High"],
        )
    )
    loan_decisions = risk_labels.map(
        {"Low": "Approve", "Medium": "Review", "High": "Reject"}
    )
    credit_scores = 300 + (1 - y_proba) * 600

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
        .to_dict(orient="records")
    )

    return {
        "model_name": "xgboost",
        "pipeline": pipeline,
        "model": pipeline.named_steps["model"],
        "preprocessor": pipeline.named_steps["preprocessor"],
        "y_test": y_test,
        "X_test": X_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": {
            "auc": auc_score,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "log_loss": float(log_loss(y_test, y_proba)),
            "r2_score": float(r2_score(y_test, y_proba)),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
            "confusion_matrix": cm.tolist(),
            "class_distribution": y.value_counts().sort_index().to_dict(),
            "risk_category_counts": risk_labels.value_counts().to_dict(),
            "loan_decision_counts": loan_decisions.value_counts().to_dict(),
            "credit_score_range": {
                "min": float(credit_scores.min()),
                "max": float(credit_scores.max()),
            },
            "top_feature_importance": feature_importance,
            "fairness": fairness,
        },
        "feature_names": feature_names,
        "preprocessing_report": preprocessing_report,
    }


def train_from_generated_data() -> dict[str, Any]:
    """Load the synthetic dataset generated by scripts/generate_data.py."""
    df = load_training_dataframe(prefer_mongo=False)
    return train_optimized_model(df)


def train_xgb_model() -> dict[str, Any]:
    """
    Backward-compatible entrypoint for the project.

    Loads the generated CSV dataset and trains the optimized XGBoost model.
    """
    return train_from_generated_data()


if __name__ == "__main__":
    results = train_xgb_model()
    fairness = results["metrics"]["fairness"]

    print(f"Selected model: {results['model_name']}")
    print(f"AUC: {results['metrics']['auc']:.4f}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Log Loss: {results['metrics']['log_loss']:.4f}")
    print(f"R2 Score: {results['metrics']['r2_score']:.4f}")
    print("Confusion Matrix:", results["metrics"]["confusion_matrix"])
    print("Class Distribution:", results["metrics"]["class_distribution"])
    print("Preprocessing Summary:", results["preprocessing_report"]["after"]["shape"])
    print("Risk Category Counts:", results["metrics"]["risk_category_counts"])
    print("Loan Decision Counts:", results["metrics"]["loan_decision_counts"])
    print("Credit Score Range:", results["metrics"]["credit_score_range"])
    print("Top Feature Importance:", results["metrics"]["top_feature_importance"][:5])
    print(
        "Demographic parity difference:",
        f"{fairness.demographic_parity_difference:.4f}",
    )
    print(
        "Equal opportunity difference:",
        f"{fairness.equal_opportunity_difference:.4f}",
    )
