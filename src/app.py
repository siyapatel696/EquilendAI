from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    from src.data_ingestion.mongo_loader import (
        load_decision_records,
        save_decision_record,
    )
    from src.models.train_xgb import train_xgb_model
except ModuleNotFoundError:
    from data_ingestion.mongo_loader import load_decision_records, save_decision_record
    from models.train_xgb import train_xgb_model


PRIMARY_COLOR = "#2E7D32"
ACCENT_COLOR = "#5D4037"
DECISION_THRESHOLD = 0.5


@st.cache_resource(show_spinner=False)
def get_trained_artifacts() -> dict:
    """Train and cache the scoring model once per app session."""
    return train_xgb_model()


def build_application_frame(
    name: str,
    age: int,
    income: float,
    utility_bill: float,
    repayment_history: int,
) -> pd.DataFrame:
    """Convert UI inputs into the feature schema expected by the trained model."""
    employment_length = "1-3 years" if age < 25 else "4-7 years"
    return pd.DataFrame(
        [
            {
                "gender": "Unknown",
                "monthly_income": float(income),
                "utility_bill_average": float(utility_bill),
                "repayment_history_pct": int(repayment_history),
                "employment_length": employment_length,
            }
        ]
    )


def score_application(
    name: str,
    age: int,
    income: float,
    utility_bill: float,
    repayment_history: int,
) -> dict:
    """Run model inference and format a user-facing decision payload."""
    artifacts = get_trained_artifacts()
    features = build_application_frame(
        name=name,
        age=age,
        income=income,
        utility_bill=utility_bill,
        repayment_history=repayment_history,
    )
    probability = float(artifacts["pipeline"].predict_proba(features)[0][1])
    prediction = int(probability >= DECISION_THRESHOLD)
    credit_score = int(round(300 + (1 - probability) * 600))

    if probability < 0.30:
        risk_level = "Low"
        decision = "Approve"
    elif probability < 0.70:
        risk_level = "Medium"
        decision = "Review"
    else:
        risk_level = "High"
        decision = "Reject"

    return {
        "applicant_name": name,
        "age": int(age),
        "monthly_income": float(income),
        "utility_bill_average": float(utility_bill),
        "repayment_history_pct": int(repayment_history),
        "default_probability": round(probability, 4),
        "predicted_default": prediction,
        "risk_level": risk_level,
        "decision": decision,
        "credit_score": credit_score,
        "model_name": artifacts["model_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def render_decision(result: dict) -> None:
    """Display the score and recommendation in the app."""
    st.success(f"Analysis Complete for {result['applicant_name']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Credit Score", result["credit_score"])
    col2.metric("Default Probability", f"{result['default_probability']:.2%}")
    col3.metric("Recommended Decision", result["decision"])
    st.write(f"Risk Level: **{result['risk_level']}**")


def render_saved_decisions(limit: int = 50) -> None:
    """Render saved decisions from MongoDB or explain why they are unavailable."""
    try:
        records = load_decision_records(limit=limit)
    except Exception as exc:
        st.info(
            "MongoDB decisions are unavailable right now. "
            "Set `MONGODB_URI` to enable persistent dashboard history."
        )
        st.caption(str(exc))
        return

    if records.empty:
        st.info("No saved decisions found yet.")
        return

    st.dataframe(records, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")

    st.title("EquiLend AI: Transparent Credit Scoring")
    st.markdown("### Assessing creditworthiness through alternative data.")

    menu = ["New Application", "Dashboard", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "New Application":
        st.subheader("Manual Loan Application")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=18)
            income = st.number_input("Monthly Income (INR)", min_value=0.0, value=25000.0)

        with col2:
            utility_bill = st.number_input(
                "Average Utility Bill (INR)", min_value=0.0, value=2500.0
            )
            repayment_history = st.slider(
                "Past Repayment Consistency (%)", 0, 100, 50
            )

        if st.button("Analyze Risk"):
            if age < 18:
                st.error("Applicants must be at least 18 years old to be scored.")
                return

            if not name.strip():
                st.error("Enter the applicant's full name before scoring.")
                return

            with st.spinner("AI model calculating..."):
                result = score_application(
                    name=name.strip(),
                    age=int(age),
                    income=float(income),
                    utility_bill=float(utility_bill),
                    repayment_history=int(repayment_history),
                )

            render_decision(result)

            try:
                save_decision_record(result)
                st.caption("Decision saved to MongoDB.")
            except Exception as exc:
                st.warning(
                    "Decision was scored locally, but MongoDB persistence is not configured."
                )
                st.caption(str(exc))

    elif choice == "Dashboard":
        st.subheader("Lender Rules Engine Overview")
        render_saved_decisions(limit=25)

    elif choice == "Audit Logs":
        st.subheader("Decision Audit Log")
        render_saved_decisions(limit=100)


if __name__ == "__main__":
    main()
