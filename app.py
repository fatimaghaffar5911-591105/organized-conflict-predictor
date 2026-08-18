import streamlit as st
import pandas as pd
import pickle
import hashlib
from pathlib import Path

st.set_page_config(
    page_title="Conflict Risk Prediction",
    page_icon="🌍",
    layout="centered"
)

base_dir = Path(__file__).resolve().parent


# Temporary file verification
for filename in ["lagged_preprocessor.pkl", "lagged_random_forest.pkl"]:
    path = base_dir / filename

    with open(path, "rb") as f:
        data = f.read()

    st.write(filename)
    st.write("Size:", len(data), "bytes")
    st.write("SHA256:", hashlib.sha256(data).hexdigest())


@st.cache_resource
def load_models():

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
