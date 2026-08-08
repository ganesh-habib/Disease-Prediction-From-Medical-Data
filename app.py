"""Streamlit dashboard for comparing models and estimating a risk probability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import load_breast_cancer, load_csv
from src.modeling import evaluate_models, train_best_model

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")
st.title("Disease Prediction from Medical Data")
st.caption("Educational decision-support only — not a diagnosis or a substitute for a clinician.")

st.sidebar.header("Dataset")
dataset_choice = st.sidebar.radio(
    "Choose a disease dataset",
    ("Breast Cancer", "Heart Disease", "Diabetes"),
)

DATASET_CONFIG = {
    "Breast Cancer": {"path": None, "name": "Breast Cancer", "uploader_key": "breast_upload"},
    "Heart Disease": {"path": Path("data/heart.csv"), "name": "Heart Disease", "uploader_key": "heart_upload"},
    "Diabetes": {"path": Path("data/diabetes.csv"), "name": "Diabetes", "uploader_key": "diabetes_upload"},
}
config = DATASET_CONFIG[dataset_choice]
uploaded = st.sidebar.file_uploader(
    f"Upload a {dataset_choice} CSV (optional)",
    type="csv",
    key=config["uploader_key"],
    help="A replacement CSV must include a binary target column: target, Outcome, diagnosis, class, or label.",
)

if uploaded is not None:
    try:
        dataset = load_csv(uploaded, f"Uploaded {dataset_choice}")
    except ValueError as error:
        st.error(str(error))
        st.stop()
elif config["path"] is None:
    dataset = load_breast_cancer()
else:
    dataset = load_csv(config["path"], config["name"])

st.subheader(dataset.name)
st.write(f"{len(dataset.features):,} records · {dataset.features.shape[1]} clinical features · binary target")

ENTRY_GUIDANCE = {
    "Breast Cancer": "Enter the 30 tumour-measurement values below. The result reports the probability of the dataset's benign class.",
    "Heart Disease": "Enter the 13 clinical measurements below. A target of 1 represents heart disease present.",
    "Diabetes": "Enter the 8 patient measurements below. An Outcome of 1 represents diabetes present.",
}
DISEASE_NAMES = {
    "Breast Cancer": "breast cancer",
    "Heart Disease": "heart disease",
    "Diabetes": "diabetes",
}

@st.cache_data(show_spinner=False)
def compare(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    return evaluate_models(features, target)

with st.spinner("Evaluating SVM, Logistic Regression, Random Forest, and XGBoost..."):
    scores = compare(dataset.features, dataset.target)
st.subheader("Cross-validated model comparison")
st.dataframe(scores.style.format({column: "{:.3f}" for column in scores.columns if column != "Model"}), use_container_width=True)

selected = st.selectbox("Model for prediction", scores["Model"].tolist())
@st.cache_resource(show_spinner=False)
def fit(features: pd.DataFrame, target: pd.Series, name: str):
    return train_best_model(features, target, name)

result = fit(dataset.features, dataset.target, selected)
st.caption("Held-out test metrics: " + " · ".join(f"{key} {value:.3f}" for key, value in result.metrics.items()))
st.divider()
st.subheader(f"Manual {dataset_choice} patient entry")
st.caption(ENTRY_GUIDANCE[dataset_choice])
with st.form("prediction"):
    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(dataset.features.columns):
        series = dataset.features[feature]
        with columns[index % 3]:
            if pd.api.types.is_numeric_dtype(series):
                default = float(series.median())
                values[feature] = st.number_input(feature, value=default, format="%.4f")
            else:
                options = sorted(series.dropna().astype(str).unique())
                values[feature] = st.selectbox(feature, options)
    submitted = st.form_submit_button(f"Estimate {dataset_choice} probability")
if submitted:
    positive_probability = float(result.pipeline.predict_proba(pd.DataFrame([values]))[0, 1])
    # sklearn's breast-cancer target uses 1 for benign; the other datasets use 1 for disease.
    disease_probability = 1 - positive_probability if dataset_choice == "Breast Cancer" else positive_probability
    if disease_probability < 0.30:
        possibility = "Lower estimated possibility"
    elif disease_probability < 0.70:
        possibility = "Moderate estimated possibility"
    else:
        possibility = "Higher estimated possibility"

    st.metric(
        f"Estimated possibility of {DISEASE_NAMES[dataset_choice]}",
        f"{disease_probability:.1%}",
        possibility,
    )
    if disease_probability >= 0.70:
        st.error(
            f"High estimated possibility of {DISEASE_NAMES[dataset_choice]}. "
            "Please discuss this result with a qualified healthcare professional.",
            icon="🚨",
        )
    elif disease_probability >= 0.30:
        st.warning(
            f"Moderate estimated possibility of {DISEASE_NAMES[dataset_choice]}. "
            "Consider discussing relevant symptoms and risk factors with a clinician.",
            icon="⚠️",
        )
    else:
        st.success(f"Lower estimated possibility of {DISEASE_NAMES[dataset_choice]}.", icon="✅")
    if dataset_choice == "Breast Cancer":
        st.caption(f"Estimated benign-class probability: {positive_probability:.1%}")
    st.info("This is a model estimate, not a diagnosis. Interpret it with clinical context and qualified medical advice.")
