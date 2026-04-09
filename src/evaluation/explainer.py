from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import re
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import xgboost as xgb


def compute_shap_feature_summary(
    model: Any,
    transformed_frame: pd.DataFrame,
    max_display: int = 15,
) -> pd.DataFrame:
    """Compute mean absolute contribution values for the transformed feature matrix."""
    original_feature_names = transformed_frame.columns.tolist()
    safe_feature_names = [
        re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or f"feature_{idx}"
        for idx, name in enumerate(original_feature_names)
    ]
    dmatrix = xgb.DMatrix(
        transformed_frame.values,
        feature_names=safe_feature_names,
    )
    shap_array = model.get_booster().predict(dmatrix, pred_contribs=True)
    # The final column is the bias term; drop it for feature-level reporting.
    shap_array = np.asarray(shap_array)[:, :-1]

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    summary = pd.DataFrame(
        {
            "feature": original_feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return summary.head(max_display).reset_index(drop=True)


def save_shap_summary_plot(
    model: Any,
    transformed_frame: pd.DataFrame,
    output_path: str | Path,
    max_display: int = 15,
) -> Path:
    """
    Save a lightweight SVG SHAP summary bar chart without external plotting deps.
    """
    summary = compute_shap_feature_summary(
        model=model,
        transformed_frame=transformed_frame,
        max_display=max_display,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    width = 1100
    row_height = 32
    top_margin = 60
    left_margin = 360
    right_margin = 60
    bottom_margin = 40
    plot_width = width - left_margin - right_margin
    height = top_margin + bottom_margin + row_height * len(summary)
    max_value = float(summary["mean_abs_shap"].max()) if not summary.empty else 1.0
    max_value = max(max_value, 1e-9)

    bars = []
    for idx, row in enumerate(summary.itertuples(index=False)):
        y = top_margin + idx * row_height
        bar_width = (float(row.mean_abs_shap) / max_value) * plot_width
        label = escape(str(row.feature))
        value_label = f"{float(row.mean_abs_shap):.4f}"

        bars.append(
            f"""
            <text x="{left_margin - 10}" y="{y + 20}" text-anchor="end" font-size="14" fill="#1f2937">{label}</text>
            <rect x="{left_margin}" y="{y + 6}" width="{bar_width:.2f}" height="18" rx="4" fill="#2E7D32" />
            <text x="{left_margin + bar_width + 8:.2f}" y="{y + 20}" font-size="13" fill="#1f2937">{value_label}</text>
            """
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <rect width="100%" height="100%" fill="white"/>
    <text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-weight="bold" fill="#111827">
      SHAP Feature Importance Summary
    </text>
    <text x="{left_margin}" y="{height - 12}" font-size="12" fill="#4b5563">Mean |SHAP value|</text>
    {''.join(bars)}
    </svg>
    """

    output.write_text(svg, encoding="utf-8")
    return output


def build_shap_summary_base64(
    model: Any,
    transformed_frame: pd.DataFrame,
    max_display: int = 15,
) -> tuple[str, pd.DataFrame]:
    """Return a base64-encoded SVG SHAP summary plus the summary table."""
    summary = compute_shap_feature_summary(
        model=model,
        transformed_frame=transformed_frame,
        max_display=max_display,
    )

    width = 1100
    row_height = 32
    top_margin = 60
    left_margin = 360
    right_margin = 60
    bottom_margin = 40
    plot_width = width - left_margin - right_margin
    height = top_margin + bottom_margin + row_height * len(summary)
    max_value = float(summary["mean_abs_shap"].max()) if not summary.empty else 1.0
    max_value = max(max_value, 1e-9)

    bars = []
    for idx, row in enumerate(summary.itertuples(index=False)):
        y = top_margin + idx * row_height
        bar_width = (float(row.mean_abs_shap) / max_value) * plot_width
        label = escape(str(row.feature))
        value_label = f"{float(row.mean_abs_shap):.4f}"
        bars.append(
            f"""
            <text x="{left_margin - 10}" y="{y + 20}" text-anchor="end" font-size="14" fill="#1f2937">{label}</text>
            <rect x="{left_margin}" y="{y + 6}" width="{bar_width:.2f}" height="18" rx="4" fill="#2E7D32" />
            <text x="{left_margin + bar_width + 8:.2f}" y="{y + 20}" font-size="13" fill="#1f2937">{value_label}</text>
            """
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <rect width="100%" height="100%" fill="white"/>
    <text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-weight="bold" fill="#111827">
      SHAP Feature Importance Summary
    </text>
    <text x="{left_margin}" y="{height - 12}" font-size="12" fill="#4b5563">Mean |SHAP value|</text>
    {''.join(bars)}
    </svg>
    """

    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return encoded, summary
