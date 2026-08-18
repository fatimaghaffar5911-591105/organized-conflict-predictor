import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Conflict Risk Prediction",
    page_icon=None,
    layout="wide"
)

# ============================================================
# VISUAL STYLE
# ============================================================

st.markdown("""
<style>

    /* Overall page */

    .stApp {
        background-color: #f8f7f4;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 3.2rem;
        padding-bottom: 4rem;
    }

    /* Main typography */

    h1 {
        color: #111418 !important;
        font-size: 3.4rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.055em !important;
        line-height: 1.05 !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        color: #171b20 !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    h3 {
        color: #20252a !important;
        font-weight: 680 !important;
    }

    p {
        color: #343a40;
        line-height: 1.7;
    }

    /* Small section labels */

    .section-label {
        color: #29445b;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-top: 2.8rem;
        margin-bottom: 0.45rem;
    }

    /* Intro text */

    .intro {
        max-width: 820px;
        color: #3d444b;
        font-size: 1.12rem;
        line-height: 1.7;
        margin-bottom: 1.3rem;
    }

    /* Quiet explanation box */

    .stAlert {
        border-radius: 5px;
    }

    /* Buttons */

    .stButton > button {
        background-color: #172b3b;
        color: white;
        border: 1px solid #172b3b;
        border-radius: 4px;
        font-weight: 650;
        padding: 0.65rem 1.2rem;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        background-color: #253f54;
        border-color: #253f54;
        color: white;
    }

    /* Input labels */

    label {
        color: #252a2f !important;
        font-weight: 600 !important;
    }

    /* Metric styling */

    [data-testid="stMetricValue"] {
        color: #111418 !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #4d555d !important;
        font-weight: 600 !important;
    }

    /* Dividers */

    hr {
        border: none;
        border-top: 1px solid #d8d7d3;
        margin: 2.5rem 0;
    }

    /* Footer */

    .footer-text {
        color: #777c81;
        font-size: 0.75rem;
        text-align: center;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED MODEL
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
# HERO
# ============================================================

st.markdown(
    '<p class="section-label">International Relations × Machine Learning</p>',
    unsafe_allow_html=True
)

st.title("Conflict Risk Prediction")

st.markdown(
    """
    <p class="intro">
    Can a country's socioeconomic conditions help us understand
    its likelihood of experiencing organized state-based conflict?
    This interactive project uses historical data and machine learning
    to explore that question.
    </p>
    """,
    unsafe_allow_html=True
)

st.info(
    "This is an academic research project. The model identifies "
    "patterns in historical data; its results should not be treated "
    "as definitive predictions or explanations of why conflicts occur."
)


# ============================================================
# THE IDEA
# ============================================================

st.markdown(
    '<p class="section-label">The idea</p>',
    unsafe_allow_html=True
)

st.header("What does this model actually do?")

st.write(
    """
    The model looks at six conditions from the previous year —
    economic performance, prices, population, employment and
    military spending — and compares them with patterns found
    in historical data.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("1. Enter conditions")
    st.write(
        "Provide the previous year's socioeconomic information "
        "for a hypothetical or real country."
    )

with c2:
    st.subheader("2. The model compares")
    st.write(
        "A Random Forest model looks for patterns it learned "
        "from the historical dataset."
    )

with c3:
    st.subheader("3. See the estimate")
    st.write(
        "The application returns an estimated probability "
        "for the conflict outcome identified by the model."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<p class="section-label">Try the model</p>',
    unsafe_allow_html=True
)

st.header("Enter previous-year conditions")

st.write(
    "Enter the country's conditions from the previous year. "
    "You can experiment with different values to see how the "
    "model responds."
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
        '<p class="section-label">Your result</p>',
        unsafe_allow_html=True
    )

    st.header("Estimated conflict probability")

    r1, r2 = st.columns([1, 1.7])

    with r1:

        st.metric(
            "Estimated probability",
            f"{probability * 100:.1f}%"
        )

    with r2:

        if prediction == 1:

            st.subheader("The model identifies a higher-risk pattern")

            st.write(
                "Based on the information entered, the model assigns "
                "this profile to its positive conflict class."
            )

        else:

            st.subheader("The model identifies a lower-risk pattern")

            st.write(
                "Based on the information entered, the model assigns "
                "this profile to its negative conflict class."
            )

    st.caption(
        "The percentage is a model-generated estimate based on "
        "historical patterns. It is not a forecast of a country's future."
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.markdown(
    '<p class="section-label">Model performance</p>',
    unsafe_allow_html=True
)

st.header("How well does it perform?")

st.write(
    """
    The model was evaluated in two ways. The second test is especially
    useful because it asks how well the model performs on countries
    that were not included in its training data.
    """
)

p1, p2 = st.columns(2)

with p1:

    st.metric(
        "Main evaluation accuracy",
        "92.49%"
    )

    st.caption(
        "Accuracy using the previous-year evaluation framework."
    )

with p2:

    st.metric(
        "Unseen-country accuracy",
        "83.12%"
    )

    st.caption(
        "Accuracy when tested on countries excluded from training."
    )


# ============================================================
# WHAT INFLUENCED THE MODEL
# ============================================================

st.markdown(
    '<p class="section-label">Inside the model</p>',
    unsafe_allow_html=True
)

st.header("Which factors mattered most?")

st.write(
    """
    Machine-learning models can consider several variables at once.
    The chart below shows which of the six inputs contributed most
    strongly to the Random Forest's decisions.
    """
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
        "Higher importance means greater contribution to the model's "
        "predictions. It does not mean the factor causes conflict."
    )


# ============================================================
# DATA SOURCES
# ============================================================

st.markdown(
    '<p class="section-label">Data</p>',
    unsafe_allow_html=True
)

st.header("What information was used?")

st.write(
    """
    The project combines organized state-based conflict observations
    with country-level socioeconomic indicators. The socioeconomic
    indicators include data obtained from the World Bank.
    """
)

source_df = pd.DataFrame({
    "Information": [
        "Organized state-based conflict",
        "GDP per capita",
        "GDP growth",
        "Inflation",
        "Population",
        "Unemployment",
        "Military expenditure"
    ],
    "Role in the project": [
        "Outcome",
        "Economic condition",
        "Economic condition",
        "Economic condition",
        "Population",
        "Employment condition",
        "Security condition"
    ]
})

st.dataframe(
    source_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ABOUT
# ============================================================

st.markdown(
    '<p class="section-label">About the project</p>',
    unsafe_allow_html=True
)

st.header("International Relations meets Machine Learning")

st.write(
    """
    This project was developed to explore how computational methods
    can be applied to an International Relations question. Historical
    conflict observations were combined with socioeconomic indicators,
    previous-year conditions were used as inputs, and a Random Forest
    classifier was trained to identify patterns in the data.
    """
)

st.write(
    """
    The purpose is not to reduce conflict to economics. Rather, the
    project asks whether measurable socioeconomic conditions contain
    information that can contribute to understanding patterns of conflict.
    """
)


# ============================================================
# LIMITATIONS
# ============================================================

st.markdown(
    '<p class="section-label">Important context</p>',
    unsafe_allow_html=True
)

st.header("What the result does — and does not — mean")

st.write(
    """
    Conflict is influenced by political institutions, history,
    geography, leadership, identity, external intervention and many
    other factors that are not represented by these six variables.
    """
)

st.write(
    """
    Therefore, a high or low model estimate should be understood as
    a statistical result from historical data, not as a definitive
    statement about a country's future.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<p class="footer-text">'
    'Academic project · International Relations × Machine Learning'
    '</p>',
    unsafe_allow_html=True
)
