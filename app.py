import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Conflict Risk Prediction",
    page_icon=None,
    layout="wide"
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Main title */

h1 {
    color: #17191c !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em;
    margin-bottom: 0.3rem !important;
}

h2 {
    color: #20252a !important;
    font-weight: 650 !important;
}

h3 {
    color: #30363b !important;
}

/* Text */

p {
    color: #4d555d;
    line-height: 1.65;
}

/* Small labels */

.label {
    color: #31516b;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 2.4rem;
    margin-bottom: 0.5rem;
}

/* Intro */

.intro {
    max-width: 820px;
    color: #555d65;
    font-size: 1.05rem;
    line-height: 1.7;
    margin-bottom: 1rem;
}

/* Notice */

.notice {
    background: #f5f7f8;
    border-left: 3px solid #31516b;
    padding: 0.9rem 1.1rem;
    color: #555e66;
    font-size: 0.84rem;
    line-height: 1.55;
    margin: 1.2rem 0 2rem 0;
}

/* Simple information cards */

.info-card {
    background: #fafbfc;
    border: 1px solid #dfe3e6;
    border-radius: 5px;
    padding: 1rem 1.1rem;
    min-height: 115px;
}

.info-title {
    color: #252a2f;
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 0.35rem;
}

.info-text {
    color: #626a72;
    font-size: 0.79rem;
    line-height: 1.5;
}

/* Result */

.result {
    background: #f6f8f9;
    border: 1px solid #ccd5dc;
    border-left: 4px solid #31516b;
    border-radius: 5px;
    padding: 1.4rem;
}

.result-label {
    color: #69737c;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.result-number {
    color: #18232d;
    font-size: 3rem;
    font-weight: 700;
    margin: 0.25rem 0;
}

.result-text {
    color: #626c75;
    font-size: 0.8rem;
    line-height: 1.5;
}

/* Performance */

.metric {
    border-top: 2px solid #31516b;
    padding-top: 0.7rem;
}

.metric-label {
    color: #68727b;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.metric-value {
    color: #18232d;
    font-size: 2.4rem;
    font-weight: 700;
}

.metric-text {
    color: #68727b;
    font-size: 0.78rem;
}

/* Footer */

.footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #dfe3e6;
    text-align: center;
    color: #7b848c;
    font-size: 0.72rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():

    with open(BASE_DIR / "lagged_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(BASE_DIR / "lagged_random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    return preprocessor, model


preprocessor, model = load_models()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="label">International Relations × Machine Learning</div>',
    unsafe_allow_html=True
)

st.title("Conflict Risk Prediction")

st.markdown(
    """
    <div class="intro">
    An interactive machine-learning application exploring whether
    previous-year socioeconomic conditions can provide useful
    information about organized state-based conflict.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="notice">
    <strong>How to use this tool:</strong>
    Enter a country's previous-year socioeconomic conditions below.
    The model will return an estimated probability based on patterns
    learned from historical data.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# QUICK EXPLANATION
# ============================================================

st.markdown(
    '<div class="label">Before you begin</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">What goes in?</div>
            <div class="info-text">
                Six socioeconomic and security indicators from the
                previous year.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">What does it do?</div>
            <div class="info-text">
                A Random Forest model compares the supplied profile
                with patterns learned from historical data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">What comes out?</div>
            <div class="info-text">
                An estimated probability for the model's positive
                conflict class.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INPUTS
# ============================================================

st.markdown(
    '<div class="label">Country Profile</div>',
    unsafe_allow_html=True
)

st.header("Enter previous-year conditions")

st.caption(
    "Use values from the same country and year where possible."
)

st.markdown("#### Economy")

c1, c2, c3 = st.columns(3)

with c1:
    gdp_per_capita = st.number_input(
        "GDP per capita (US$)",
        min_value=0.0,
        max_value=150000.0,
        value=3000.0,
        step=100.0
    )

with c2:
    gdp_growth = st.number_input(
        "GDP growth (%)",
        min_value=-50.0,
        max_value=50.0,
        value=3.0,
        step=0.1
    )

with c3:
    inflation = st.number_input(
        "Inflation (%)",
        min_value=0.0,
        max_value=500.0,
        value=5.0,
        step=0.1
    )

st.markdown("#### Population and employment")

c1, c2 = st.columns(2)

with c1:
    population = st.number_input(
        "Population",
        min_value=0.0,
        max_value=2000000000.0,
        value=10000000.0,
        step=100000.0
    )

with c2:
    unemployment = st.number_input(
        "Unemployment (%)",
        min_value=0.0,
        max_value=100.0,
        value=6.0,
        step=0.1
    )

st.markdown("#### Security")

military_expenditure = st.number_input(
    "Military expenditure (% of GDP)",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.1
)

st.write("")

predict = st.button(
    "Generate Prediction",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    input_data = pd.DataFrame({
        "GDP_per_capita_lag1": [gdp_per_capita],
        "GDP_growth_lag1": [gdp_growth],
        "Inflation_lag1": [inflation],
        "Population_lag1": [population],
        "Unemployment_lag1": [unemployment],
        "Military_expenditure_lag1": [military_expenditure]
    })

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    st.markdown(
        '<div class="label">Model Result</div>',
        unsafe_allow_html=True
    )

    st.header("Estimated conflict probability")

    r1, r2 = st.columns([1, 1.5])

    with r1:

        st.markdown(
            f"""
            <div class="result">

                <div class="result-label">
                    Estimated probability
                </div>

                <div class="result-number">
                    {probability * 100:.1f}%
                </div>

                <div class="result-text">
                    Probability assigned by the trained Random Forest
                    classifier to the positive conflict class.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:

        if prediction == 1:

            st.subheader("Model classification")

            st.warning(
                "The supplied profile is classified in the model's "
                "positive conflict class."
            )

        else:

            st.subheader("Model classification")

            st.info(
                "The supplied profile is classified in the model's "
                "negative conflict class."
            )

        st.caption(
            "This is a model-based estimate, not a prediction of what "
            "will necessarily happen in the real world."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="label">Model Performance</div>',
    unsafe_allow_html=True
)

st.header("How did the model perform?")

p1, p2 = st.columns(2)

with p1:

    st.markdown(
        """
        <div class="metric">

            <div class="metric-label">
                Lagged model accuracy
            </div>

            <div class="metric-value">
                92.49%
            </div>

            <div class="metric-text">
                Accuracy using the previous-year evaluation framework.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with p2:

    st.markdown(
        """
        <div class="metric">

            <div class="metric-label">
                Country-held-out accuracy
            </div>

            <div class="metric-value">
                83.12%
            </div>

            <div class="metric-text">
                Accuracy when evaluating on countries excluded from
                model training.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    "The country-held-out test provides a stronger check of whether "
    "the model can generalize beyond the countries used during training."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="label">Understanding the Model</div>',
    unsafe_allow_html=True
)

st.header("Which factors mattered most?")

st.write(
    "The chart below shows the relative feature importance calculated "
    "by the Random Forest model."
)

feature_names = [
    "GDP per capita",
    "GDP growth",
    "Inflation",
    "Population",
    "Unemployment",
    "Military expenditure"
]

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    st.bar_chart(
        importance_df.set_index("Feature"),
        horizontal=True
    )

    st.caption(
        "Feature importance indicates predictive contribution within "
        "the model. It does not mean that a variable causes conflict."
    )


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.markdown(
    '<div class="label">About the Project</div>',
    unsafe_allow_html=True
)

st.header("Why this project?")

st.write(
    """
    This project explores the intersection of International Relations
    and machine learning. Conflict observations were combined with
    country-level socioeconomic indicators, including data from the
    World Bank, to examine whether previous-year conditions contain
    useful predictive information about subsequent conflict outcomes.
    """
)

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                Data sources
            </div>

            <div class="info-text">
                Organized state-based conflict data and World Bank
                socioeconomic indicators.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                Machine-learning model
            </div>

            <div class="info-text">
                Random Forest classification with a preprocessing
                pipeline and previous-year predictors.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LIMITATIONS
# ============================================================

st.markdown(
    '<div class="label">Important Context</div>',
    unsafe_allow_html=True
)

st.header("What should you take from the result?")

st.write(
    """
    The model identifies statistical patterns in historical data.
    It does not establish causal relationships and cannot capture
    every political, historical, geographic or strategic factor
    involved in conflict.
    """
)

st.caption(
    "For academic and exploratory use only. The output should not be "
    "treated as a definitive forecast or real-world risk assessment."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Academic project · International Relations × Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
