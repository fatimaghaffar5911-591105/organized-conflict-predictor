
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
st.set_page_config(
    page_title="Conflict Risk Prediction",
    page_icon="🌍",
    layout="centered"
)

@st.cache_resource
def load_models():
    base_dir = Path(__file__).resolve().parent

    with open(base_dir / "lagged_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(base_dir / "lagged_random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    return preprocessor, model

preprocessor, model = load_models()

st.title("🌍 Conflict Risk Prediction")

st.markdown(
    "### A Machine Learning Approach to Organized State-Based Conflict"
)

st.write(
    "Exploring whether previous-year socioeconomic conditions "
    "can help predict organized state-based conflict."
)

st.info(
    "This is an academic machine-learning project. "
    "Predictions should not be interpreted as causal explanations "
    "or definitive forecasts."
)

st.header("🔮 Generate Prediction")

st.write("Enter the country's previous-year socioeconomic conditions.")

gdp_per_capita = st.number_input(
    "GDP per capita (US$)",
    min_value=0.0,
    value=3000.0
)

gdp_growth = st.number_input(
    "GDP growth (%)",
    value=3.0
)

inflation = st.number_input(
    "Inflation (%)",
    value=5.0
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=10000000.0
)

unemployment = st.number_input(
    "Unemployment (%)",
    min_value=0.0,
    value=6.0
)

military_expenditure = st.number_input(
    "Military expenditure (% of GDP)",
    min_value=0.0,
    value=2.0
)

if st.button("🔮 Generate Prediction", use_container_width=True):

    input_data = pd.DataFrame({
        "GDP_per_capita_lag1": [gdp_per_capita],
        "GDP_growth_lag1": [gdp_growth],
        "Inflation_lag1": [inflation],
        "Population_lag1": [population],
        "Unemployment_lag1": [unemployment],
        "Military_expenditure_lag1": [military_expenditure]
    })

    processed = preprocessor.transform(input_data)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    st.divider()
    st.subheader("Prediction")

    if prediction == 1:
        st.error("🔴 Conflict Predicted")
    else:
        st.success("🟢 No Conflict Predicted")

    st.metric(
        "Estimated probability of conflict",
        f"{probability * 100:.1f}%"
    )

st.divider()

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Lagged Model Accuracy", "92.49%")

with col2:
    st.metric("Country-Held-Out Accuracy", "83.12%")

st.write(
    "The country-held-out evaluation tests the model on countries "
    "that were completely excluded from training."
)

st.subheader("🔍 Most Influential Features")

importance = pd.DataFrame({
    "Feature": [
        "Population",
        "GDP per capita",
        "Military expenditure",
        "Unemployment",
        "Inflation",
        "GDP growth"
    ],
    "Importance": [
        35.59,
        20.69,
        17.64,
        12.04,
        7.79,
        6.25
    ]
})

st.bar_chart(
    importance.set_index("Feature")
)

st.caption(
    "Feature importance represents predictive contribution and "
    "does not establish causality."
)

st.divider()

st.caption(
    "Academic project | International Relations × Machine Learning"
)
