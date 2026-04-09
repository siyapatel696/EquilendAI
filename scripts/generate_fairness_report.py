from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.explainer import build_shap_summary_base64
from evaluation.fairness import fairness_summary_payload
from models.train_xgb import DEFAULT_DATA_PATH, DEFAULT_MODELS_DIR, load_artifacts
from preprocessing.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES


TARGET_COLUMN = "default_status"
PROTECTED_COLUMN = "gender"
RANDOM_STATE = 42
REPORT_PATH = ROOT_DIR / "Fairness_Report.html"


def _load_test_split(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruct the same test split used in training."""
    df = pd.read_csv(data_path)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def _compliance_badge(flag: str) -> str:
    colors = {"PASS": "#166534", "WARN": "#b45309", "FAIL": "#b91c1c"}
    background = {"PASS": "#dcfce7", "WARN": "#fef3c7", "FAIL": "#fee2e2"}
    return (
        f"<span style='display:inline-block;padding:6px 12px;border-radius:999px;"
        f"font-weight:700;color:{colors.get(flag, '#111827')};"
        f"background:{background.get(flag, '#e5e7eb')};'>{html.escape(flag)}</span>"
    )


def generate_fairness_report_from_predictions(
    *,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    protected_values: np.ndarray | pd.Series,
    shap_image_base64: str,
    shap_summary: pd.DataFrame,
    threshold: float,
    model_name: str,
    dataset_size: int,
    report_path: Path = REPORT_PATH,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Generate a self-contained HTML fairness report from predictions and SHAP outputs."""
    fairness = fairness_summary_payload(
        protected_values=protected_values,
        y_true=y_true,
        y_pred=y_pred,
    )

    overall_metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    fairness_flag = fairness["compliance_flag"]
    metadata = metadata or {}
    metadata.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    metadata.setdefault("model_name", model_name)
    metadata.setdefault("dataset_size", dataset_size)
    metadata.setdefault("threshold", threshold)

    group_rows = "".join(
        [
            "<tr>"
            f"<td>{html.escape(str(row['group']))}</td>"
            f"<td>{row['sample_size']}</td>"
            f"<td>{row['approval_rate']:.3f}</td>"
            f"<td>{row['default_rate']:.3f}</td>"
            f"<td>{row['accuracy']:.3f}</td>"
            f"<td>{row['precision']:.3f}</td>"
            f"<td>{row['recall']:.3f}</td>"
            f"<td>{row['f1']:.3f}</td>"
            f"<td>{row['disparate_impact_ratio']:.3f}</td>"
            f"<td>{_compliance_badge(row['compliance_flag'])}</td>"
            "</tr>"
            for row in fairness["group_metrics"]
        ]
    )

    shap_rows = "".join(
        [
            "<tr>"
            f"<td>{html.escape(str(row.feature))}</td>"
            f"<td>{row.mean_abs_shap:.6f}</td>"
            "</tr>"
            for row in shap_summary.itertuples(index=False)
        ]
    )

    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>EquiLend AI Fairness Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      color: #111827;
      background: #f9fafb;
      line-height: 1.5;
    }}
    .card {{
      background: white;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 10px;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f3f4f6;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 16px;
    }}
    .metric {{
      background: #f3f4f6;
      border-radius: 12px;
      padding: 16px;
    }}
    .metric-label {{
      font-size: 12px;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .small {{
      color: #4b5563;
      font-size: 14px;
    }}
    img {{
      width: 100%;
      max-width: 1100px;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      background: white;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>EquiLend AI Fairness Report</h1>
    <p class="small">Auditor-ready compliance summary for the deployed credit-risk model.</p>
    <p><strong>Overall Compliance:</strong> {_compliance_badge(fairness_flag)}</p>
    <p><strong>Disparate Impact Ratio:</strong> {fairness['disparate_impact_ratio']:.3f}</p>
    <p><strong>Decision Threshold:</strong> {threshold:.3f}</p>
    <p><strong>Model:</strong> {html.escape(model_name)}</p>
    <p><strong>Dataset Size:</strong> {dataset_size}</p>
    <p><strong>Generated:</strong> {html.escape(str(metadata['generated_at_utc']))}</p>
  </div>

  <div class="card">
    <h2>Overall Model Metrics</h2>
    <div class="metrics">
      <div class="metric"><div class="metric-label">ROC-AUC</div><div class="metric-value">{overall_metrics['auc']:.4f}</div></div>
      <div class="metric"><div class="metric-label">Accuracy</div><div class="metric-value">{overall_metrics['accuracy']:.4f}</div></div>
      <div class="metric"><div class="metric-label">Precision</div><div class="metric-value">{overall_metrics['precision']:.4f}</div></div>
      <div class="metric"><div class="metric-label">Recall</div><div class="metric-value">{overall_metrics['recall']:.4f}</div></div>
      <div class="metric"><div class="metric-label">F1 Score</div><div class="metric-value">{overall_metrics['f1']:.4f}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Fairness by Protected Group</h2>
    <p class="small">
      Approval means the model predicted <strong>non-default</strong>. The 80% rule is used as the primary fairness audit threshold.
    </p>
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th>Sample Size</th>
          <th>Approval Rate</th>
          <th>Default Rate</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>DIR</th>
          <th>Compliance</th>
        </tr>
      </thead>
      <tbody>{group_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Disparate Impact Interpretation</h2>
    <p>
      The Disparate Impact Ratio (DIR) is calculated as
      <code>min(group approval rate) / max(group approval rate)</code>.
    </p>
    <ul>
      <li><strong>PASS</strong>: DIR ≥ 0.80</li>
      <li><strong>WARN</strong>: 0.70 ≤ DIR &lt; 0.80</li>
      <li><strong>FAIL</strong>: DIR &lt; 0.70</li>
    </ul>
  </div>

  <div class="card">
    <h2>SHAP Explainability Summary</h2>
    <p class="small">Embedded as a base64 SVG so the report is fully self-contained.</p>
    <img alt="SHAP summary plot" src="data:image/svg+xml;base64,{shap_image_base64}" />
    <table>
      <thead>
        <tr><th>Feature</th><th>Mean |SHAP|</th></tr>
      </thead>
      <tbody>{shap_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Audit Metadata</h2>
    <pre>{html.escape(json.dumps(metadata, indent=2))}</pre>
  </div>
</body>
</html>"""

    report_path.write_text(html_report, encoding="utf-8")
    return report_path


def generate_fairness_report(
    report_path: Path = REPORT_PATH,
    models_dir: Path = Path(DEFAULT_MODELS_DIR),
    data_path: Path = Path(DEFAULT_DATA_PATH),
) -> Path:
    """Generate a self-contained HTML fairness report from current saved artifacts."""
    artifacts = load_artifacts(str(models_dir))
    if artifacts is None:
        raise FileNotFoundError(
            "Model artifacts not found. Train the model before generating the fairness report."
        )

    model, preprocessor, y_test_saved, y_prob_saved, threshold_info = artifacts
    X_test, y_test = _load_test_split(data_path)

    if len(y_test) != len(y_test_saved):
        raise ValueError("Saved test predictions do not align with reconstructed test split.")

    y_proba = np.asarray(y_prob_saved, dtype=float)
    threshold = float(threshold_info.get("threshold", 0.5))
    y_pred = (y_proba >= threshold).astype(int)

    transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    transformed_frame = pd.DataFrame(transformed, columns=feature_names)
    shap_image_base64, shap_summary = build_shap_summary_base64(
        model=model,
        transformed_frame=transformed_frame,
        max_display=10,
    )

    return generate_fairness_report_from_predictions(
        y_true=y_test,
        y_prob=y_proba,
        y_pred=y_pred,
        protected_values=X_test[PROTECTED_COLUMN],
        shap_image_base64=shap_image_base64,
        shap_summary=shap_summary,
        threshold=threshold,
        model_name=type(model).__name__,
        dataset_size=len(X_test),
        report_path=report_path,
        metadata={
            "protected_attribute": PROTECTED_COLUMN,
            "threshold_objective": threshold_info.get("objective", "business"),
            "source_artifacts_dir": str(models_dir),
            "source_dataset": str(data_path),
        },
    )


if __name__ == "__main__":
    output = generate_fairness_report()
    print(f"Fairness report written to {output}")
