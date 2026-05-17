"""Streamlit interface for sending customer records to the FastAPI scorer."""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Bank Deposit Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

JOBS: list[str] = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MARITAL: list[str] = ["married", "single", "divorced"]
EDUCATION: list[str] = ["primary", "secondary", "tertiary", "unknown"]
YES_NO: list[str] = ["no", "yes"]
CONTACT: list[str] = ["cellular", "telephone", "unknown"]
MONTHS: list[str] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME: list[str] = ["unknown", "failure", "success", "other"]


def get_api_endpoint() -> str:
    """Resolve the FastAPI base URL for local or Docker Compose execution."""
    return os.getenv("API_URL", "http://localhost:8000").rstrip("/")


@st.cache_data(ttl=15, show_spinner=False)
def check_api_health(api_endpoint: str) -> dict[str, Any]:
    """Fetch API readiness so the UI can show whether scoring is available."""
    health_url = f"{api_endpoint}/health"
    response = requests.get(health_url, timeout=3)
    response.raise_for_status()
    return response.json()


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Submit form values to the FastAPI prediction endpoint.

    The API URL is environment-driven so Docker Compose can route to the
    ``fastapi`` service while local runs can default to ``localhost``.
    """
    predict_url = f"{get_api_endpoint()}/predict"
    response = requests.post(predict_url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def describe_api_error(exc: requests.RequestException) -> str:
    """Convert API/network exceptions into concise user-facing messages."""
    response = getattr(exc, "response", None)
    if response is None:
        return f"Could not reach the FastAPI service: {exc}"

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    if isinstance(detail, list) and detail:
        first_error = detail[0]
        location = ".".join(str(item) for item in first_error.get("loc", []))
        message = first_error.get("msg", "Invalid request")
        return f"API rejected `{location}`: {message}"
    return f"API request failed with status {response.status_code}: {detail}"


def clear_stale_prediction(payload: dict[str, Any], predict_button: bool) -> None:
    """Remove previous results when form values no longer match the last score."""
    last_payload = st.session_state.get("last_payload")
    if not predict_button and last_payload is not None and last_payload != payload:
        st.session_state.pop("prediction", None)
        st.session_state.pop("latency", None)
        st.session_state.pop("last_payload", None)


def factor_importance_series(factors: dict[str, float]) -> pd.Series:
    """Convert API feature importances into a chart-ready Series."""
    return pd.Series(factors, name="Importance")


api_endpoint = get_api_endpoint()

st.title("Bank Marketing Term Deposit Prediction")
st.caption("Production-style bank marketing classifier for term-deposit subscription prediction.")

try:
    api_health = check_api_health(api_endpoint)
    health_text = "Ready" if api_health.get("model_loaded") else "Unavailable"
except requests.RequestException:
    health_text = "Unavailable"

with st.sidebar:
    st.subheader("Runtime")
    st.metric("API status", health_text)
    st.caption(api_endpoint)

if health_text != "Ready":
    st.warning("API service is unavailable.")

left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.subheader("Customer and Campaign Details")

    # The UI exposes only production-safe fields. Post-call duration is not
    # collected because the production model is intended for pre-call scoring.
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        age = st.slider("Age", min_value=18, max_value=100, value=41)
    with row1_col2:
        balance = st.number_input("Average yearly balance", value=1200, step=100)
    with row1_col3:
        day = st.slider("Contact day", min_value=1, max_value=31, value=15)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        job = st.selectbox("Job", JOBS, index=8)
    with row2_col2:
        marital = st.selectbox("Marital status", MARITAL, index=1)
    with row2_col3:
        education = st.selectbox("Education", EDUCATION, index=1)

    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        default = st.selectbox("Credit in default", YES_NO)
    with row3_col2:
        housing = st.selectbox("Housing loan", YES_NO, index=1)
    with row3_col3:
        loan = st.selectbox("Personal loan", YES_NO)

    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        contact = st.selectbox("Contact type", CONTACT)
    with row4_col2:
        month = st.selectbox("Contact month", MONTHS, index=4)
    with row4_col3:
        poutcome = st.selectbox("Previous outcome", POUTCOME)

    row5_col1, row5_col2 = st.columns(2)
    with row5_col1:
        campaign = st.number_input("Contacts in current campaign", min_value=1, value=2, step=1)
    with row5_col2:
        previous = st.number_input("Previous contacts", min_value=0, value=0, step=1)

    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "previous": previous,
        "poutcome": poutcome,
    }

    predict_button = st.button("Predict Subscription", use_container_width=True)
    clear_stale_prediction(payload, predict_button)

with right:
    st.subheader("Prediction")

    if predict_button:
        with st.spinner("Scoring customer..."):
            try:
                started_at = time.time()
                prediction = predict(payload)
                st.session_state.prediction = prediction
                st.session_state.latency = time.time() - started_at
                st.session_state.last_payload = payload.copy()
            except requests.RequestException as exc:
                st.error(describe_api_error(exc))

    if "prediction" in st.session_state:
        prediction = st.session_state.prediction
        probability = prediction["subscription_probability"]
        label = prediction["predicted_deposit"].upper()

        st.metric("Predicted deposit", label)
        st.metric("Subscription probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

        lower, upper = prediction["probability_range"]
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Probability range", f"{lower:.1%} - {upper:.1%}")
        metric_col2.metric("API latency", f"{st.session_state.get('latency', 0):.2f}s")

        factors = prediction.get("top_model_factors", {})
        if factors:
            st.subheader("Global Model Factors")
            st.bar_chart(factor_importance_series(factors))
    else:
        st.info("No prediction yet.")

with st.expander("API payload"):
    st.json(payload)
