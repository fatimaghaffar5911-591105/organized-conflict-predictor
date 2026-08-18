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
# VISUAL DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #fbfaf8;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}


/* ============================================================
   MAIN TYPOGRAPHY
   ============================================================ */

h1, h2, h3, h4 {
    color: #303236 !important;
}

h1 {
    font-size: 2.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    margin-bottom: 0.4rem !important;
}

h2 {
    font-size: 1.55rem !important;
    font-weight: 680 !important;
    letter-spacing: -0.02em !important;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 650 !important;
}

p, li {
    color: #303236 !important;
    line-height: 1.65 !important;
}


/* ============================================================
   SECTION LABELS
   ============================================================ */

.section-label {
    color: #66696c;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 2.6rem;
    margin-bottom: 0.45rem;
}


/* ============================================================
   RESEARCH QUESTION
   ============================================================ */

.question-box {
    border: 1px solid #d7d5d1;
    border-left: 4px solid #55585b;
    background: #f3f2ef;
    border-radius: 4px;
    padding: 1.15rem 1.3rem;
    margin: 1rem 0 1.2rem 0;
}

.question-label {
    color: #66696c;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.question-text {
    color: #292b2e;
    font-size: 1.12rem;
    font-weight: 650;
    line-height: 1.5;
}


/* ============================================================
   PREDICTION TOOL
   ============================================================ */

.tool-heading {
    color: #292b2e;
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.tool-description {
    color: #303236;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}


/* ============================================================
   GENERATE PREDICTION BUTTON
   ============================================================ */

/*
   Force the button itself and every common nested element
   to use a black background and pure white typography.
*/

div.stButton > button {
    width: 100% !important;
    height: 3.4rem !important;
    min-height: 3.4rem !important;

    background: #111111 !important;
    background-color: #111111 !important;

    color: #ffffff !important;

    border: 1px solid #111111 !important;
    border-radius: 5px !important;

    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;

    opacity: 1 !important;
    box-shadow: none !important;
}


/* Force all text inside the button to white */

div.stButton > button *,
div.stButton > button p,
div.stButton > button span,
div.stButton > button div {
    color: #ffffff !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}


/* Hover */

div.stButton > button:hover {
    background: #000000 !important;
    background-color: #000000 !important;
    color: #ffffff !important;

    border-color: #000000 !important;
}


/* Hover text */

div.stButton > button:hover *,
div.stButton > button:hover p,
div.stButton > button:hover span,
div.stButton > button:hover div {
    color: #ffffff !important;
}


/* Focus */

div.stButton > button:focus,
div.stButton > button:focus-visible,
div.stButton > button:active {
    background: #111111 !important;
    background-color: #111111 !important;
    color: #ffffff !important;

    border-color: #111111 !important;
    box-shadow: none !important;
}


/* Focus text */

div.stButton > button:focus *,
div.stButton > button:focus-visible *,
div.stButton > button:active * {
    color: #ffffff !important;
}


/* ============================================================
   RESULT
   ============================================================ */

.result-box {
    border: 1px solid #d2d0cc;
    border-left: 4px solid #55585b;
    border-radius: 5px;
    background: #f3f2ef;
    padding: 1.3rem;
}

.result-number {
    color: #292b2e;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0.2rem 0;
}

.result-label {
    color: #66696c;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}


/* ============================================================
   NATIVE STREAMLIT CONTAINERS
   ============================================================ */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #d7d5d1 !important;
    border-radius: 5px !important;
}


/* ============================================================
   METRICS
   ============================================================ */

[data-testid="stMetricLabel"] {
    color: #55585b !important;
    font-weight: 650 !important;
}

[data-testid="stMetricValue"] {
    color: #292b2e !important;
}


/* ============================================================
   INPUTS
   ============================================================ */

label {
    color: #303236 !important;
    font-weight: 600 !important;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    border: 1px solid #d7d5d1;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #777a7d;
    font-size: 0.72rem;
    margin-top: 2.5rem;
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
    '<div class="section-label">'
    'International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.title("Conflict Risk Prediction")

st.write(
    "A machine-learning study examining whether previous-year "
    "socioeconomic conditions contain predictive information about "
    "organized state-based conflict."
)


# ============================================================
# RESEARCH QUESTION
# ============================================================

st.markdown(
    '<div class="section-label">Research Question</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="question-box">
        <div class="question-label">Central question</div>
        <div class="question-text">
            To what extent can previous-year socioeconomic conditions
            help predict organized state-based conflict?
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "The project examines whether measurable socioeconomic conditions "
    "in the preceding year contain useful predictive information about "
    "subsequent conflict outcomes."
)


# ============================================================
# HOW THE MODEL WORKS
# ============================================================

st.markdown(
    '<div class="section-label">How the Model Works</div>',
    unsafe_allow_html=True
)

st.header("From socioeconomic data to prediction")

st.write(
    "The model uses six previous-year socioeconomic indicators and a "
    "Random Forest classifier to identify patterns associated with "
    "historical organized state-based conflict."
)

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("**Previous-year indicators**")
        st.write(
            "GDP per capita, GDP growth, inflation, population, "
            "unemployment and military expenditure."
        )

with c2:
    with st.container(border=True):
        st.markdown("**Random Forest classifier**")
        st.write(
            "A trained machine-learning model identifies patterns "
            "across historical country-year observations."
        )

with c3:
    with st.container(border=True):
        st.markdown("**Model output**")
        st.write(
            "The model produces a probability and a corresponding "
            "conflict classification."
        )


# ============================================================
# INTERACTIVE MODEL
# ============================================================

st.markdown(
    '<div class="section-label">Interactive Model</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown(
        '<div class="tool-heading">Try the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tool-description">'
        'Enter a hypothetical country profile using previous-year '
        'socioeconomic conditions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### Economic conditions")

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

    st.markdown("#### Demographic and labour conditions")

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

    st.markdown("#### Security conditions")

    military_expenditure = st.number_input(
        "Military expenditure (% of GDP)",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        step=0.1
    )

    st.caption(
        "Use hypothetical values to explore how the trained model "
        "responds to different socioeconomic profiles."
    )

    st.write("")

    assess = st.button(
        "Generate Prediction",
        use_container_width=True
    )


# ============================================================
# MODEL OUTPUT
# ============================================================

if assess:

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
        '<div class="section-label">Prediction Result</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1, 1.6])

    with c1:

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">Estimated probability</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-number">{probability * 100:.1f}%</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Probability assigned by the trained Random Forest "
            "to the positive conflict class."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:

        if prediction == 1:
            st.subheader("Positive conflict classification")
            st.write(
                "Under the supplied hypothetical conditions, the model "
                "assigns the scenario to the positive conflict class."
            )
        else:
            st.subheader("Negative conflict classification")
            st.write(
                "Under the supplied hypothetical conditions, the model "
                "assigns the scenario to the negative conflict class."
            )

        st.caption(
            "This is a model-based estimate, not a definitive forecast."
        )


# ============================================================
# DATA AND SOURCES
# ============================================================

st.markdown(
    '<div class="section-label">Data & Sources</div>',
    unsafe_allow_html=True
)

st.header("Data foundation")

st.write(
    "The project combines organized state-based conflict observations "
    "with country-level socioeconomic indicators. World Bank data "
    "provide the socioeconomic predictors, which were integrated with "
    "conflict observations at the country-year level."
)

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("**Organized state-based conflict data**")
        st.write(
            "Historical conflict observations provide the outcome "
            "used for classification."
        )

with c2:
    with st.container(border=True):
        st.markdown("**World Bank socioeconomic data**")
        st.write(
            "Country-level economic, demographic, labour and security "
            "indicators provide the model's predictors."
        )

st.markdown("#### Predictors")

indicator_df = pd.DataFrame({
    "Indicator": [
        "GDP per capita",
        "GDP growth",
        "Inflation",
        "Population",
        "Unemployment",
        "Military expenditure"
    ],
    "Category": [
        "Economic",
        "Economic",
        "Economic",
        "Demographic",
        "Labour",
        "Security"
    ],
    "Time reference": [
        "Previous year",
        "Previous year",
        "Previous year",
        "Previous year",
        "Previous year",
        "Previous year"
    ]
})

st.dataframe(
    indicator_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-label">Model Evaluation</div>',
    unsafe_allow_html=True
)

st.header("Model performance")

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.metric(
            "Lagged Model Accuracy",
            "92.49%"
        )
        st.write(
            "Accuracy under the lagged evaluation framework."
        )

with c2:
    with st.container(border=True):
        st.metric(
            "Country-Held-Out Accuracy",
            "83.12%"
        )
        st.write(
            "Accuracy when evaluated on countries completely "
            "excluded from training."
        )

st.write(
    "The country-held-out evaluation provides an additional test of "
    "generalization because the model is evaluated on countries that "
    "were not included during training."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-label">Model Interpretation</div>',
    unsafe_allow_html=True
)

st.header("Which variables influenced the model most?")

st.write(
    "Feature importance shows the relative contribution of each "
    "predictor to the trained Random Forest's decisions. It does not "
    "establish that any variable causes conflict."
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


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-label">Interpretation</div>',
    unsafe_allow_html=True
)

st.header("Understanding the model's output")

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("**What the model can indicate**")
        st.write(
            "Whether a hypothetical socioeconomic profile resembles "
            "patterns associated with historical conflict outcomes "
            "in the training data."
        )

with c2:
    with st.container(border=True):
        st.markdown("**What the model cannot establish**")
        st.write(
            "It cannot establish causality or provide a complete "
            "explanation of why conflict occurs in a particular "
            "country or period."
        )


# ============================================================
# LIMITATIONS
# ============================================================

st.markdown(
    '<div class="section-label">Limitations</div>',
    unsafe_allow_html=True
)

st.header("Important limitations")

st.write(
    "Conflict is shaped by political, historical, geographic, "
    "institutional and strategic factors that are not fully represented "
    "by the six variables used in this model."
)

st.write(
    "The predictions should therefore be understood as statistical "
    "estimates based on historical patterns, rather than definitive "
    "forecasts of future conflict."
)


# ============================================================
# PROJECT
# ============================================================

st.markdown(
    '<div class="section-label">Project</div>',
    unsafe_allow_html=True
)

st.header("International Relations × Machine Learning")

st.write(
    "This project applies machine-learning methods to an International "
    "Relations research question through data integration, temporal "
    "feature engineering, classification, evaluation and interactive "
    "deployment."
)

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("**Research focus**")
        st.write(
            "Socioeconomic conditions and organized state-based conflict."
        )

with c2:
    with st.container(border=True):
        st.markdown("**Technical approach**")
        st.write(
            "Python, preprocessing, Random Forest classification "
            "and model evaluation."
        )

with c3:
    with st.container(border=True):
        st.markdown("**Deployment**")
        st.write(
            "An interactive web application for exploring "
            "hypothetical scenarios."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Academic project · International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
