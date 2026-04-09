
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroupFairnessMetrics:
    group: str
    approval_rate: float
    default_rate: float
    sample_size: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    disparate_impact_ratio: float
    compliance_flag: str


def compute_group_fairness(
    protected_values: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> list[GroupFairnessMetrics]:
    """Compute approval/default rates per protected group."""
    protected = pd.Series(protected_values).fillna("Unknown").astype(str).reset_index(drop=True)
    truth = pd.Series(y_true).astype(int).reset_index(drop=True)
    pred = pd.Series(y_pred).astype(int).reset_index(drop=True)

    if not (len(protected) == len(truth) == len(pred)):
        raise ValueError("protected_values, y_true, and y_pred must have equal length.")

    rows: list[GroupFairnessMetrics] = []
    approval_lookup: dict[str, float] = {}
    for group in sorted(protected.unique()):
        mask = protected == group
        if mask.sum() == 0:
            continue
        approval_lookup[group] = float((pred[mask] == 0).mean())

    max_approval = max(approval_lookup.values()) if approval_lookup else 0.0

    for group in sorted(protected.unique()):
        mask = protected == group
        sample_size = int(mask.sum())
        if sample_size == 0:
            continue

        # Approval means predicted non-default in this lending setup.
        approval_rate = float((pred[mask] == 0).mean())
        default_rate = float((truth[mask] == 1).mean())
        tp = int(((truth[mask] == 1) & (pred[mask] == 1)).sum())
        tn = int(((truth[mask] == 0) & (pred[mask] == 0)).sum())
        fp = int(((truth[mask] == 0) & (pred[mask] == 1)).sum())
        fn = int(((truth[mask] == 1) & (pred[mask] == 0)).sum())

        accuracy = (tp + tn) / sample_size if sample_size else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        dir_value = approval_rate / max_approval if max_approval > 0 else 0.0
        compliance_flag = "PASS" if dir_value >= 0.8 else "WARN" if dir_value >= 0.7 else "FAIL"

        rows.append(
            GroupFairnessMetrics(
                group=group,
                approval_rate=approval_rate,
                default_rate=default_rate,
                sample_size=sample_size,
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                disparate_impact_ratio=float(dir_value),
                compliance_flag=compliance_flag,
            )
        )
    return rows


def compute_disparate_impact_ratio(
    protected_values: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Compute disparate impact ratio using approval rates.

    DIR = min group approval rate / max group approval rate
    """
    protected = pd.Series(protected_values).fillna("Unknown").astype(str).reset_index(drop=True)
    pred = pd.Series(y_pred).astype(int).reset_index(drop=True)

    if len(protected) != len(pred):
        raise ValueError("protected_values and y_pred must have equal length.")

    approval_rates = []
    for group in sorted(protected.unique()):
        mask = protected == group
        if mask.sum() == 0:
            continue
        approval_rates.append(float((pred[mask] == 0).mean()))

    if not approval_rates or max(approval_rates) == 0:
        return 0.0

    return float(min(approval_rates) / max(approval_rates))


def fairness_summary_payload(
    protected_values: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Build an auditor-friendly fairness summary payload."""
    groups = compute_group_fairness(protected_values, y_true, y_pred)
    dir_value = compute_disparate_impact_ratio(protected_values, y_pred)

    return {
        "disparate_impact_ratio": dir_value,
        "passes_four_fifths_rule": dir_value >= 0.8,
        "compliance_flag": "PASS" if dir_value >= 0.8 else "WARN" if dir_value >= 0.7 else "FAIL",
        "group_metrics": [asdict(row) for row in groups],
    }
