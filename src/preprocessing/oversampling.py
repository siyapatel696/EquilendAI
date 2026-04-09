
from __future__ import annotations

import pandas as pd


def compute_class_weight_ratio(y: pd.Series) -> float:
    """Compute a positive-class weight ratio for imbalanced binary targets."""
    if y is None or y.empty:
        raise ValueError("Target series cannot be empty.")

    counts = y.value_counts()
    negative_count = int(counts.get(0, 0))
    positive_count = int(counts.get(1, 0))

    if positive_count == 0:
        raise ValueError("Positive class is missing from the target.")

    return negative_count / positive_count
