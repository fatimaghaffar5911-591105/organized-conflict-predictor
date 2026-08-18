import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Conflict Risk Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .main {
        color: #17191c;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.035em;
        color: #15171a;
    }

    h2 {
        font-size: 1.65rem !important;
        font-weight: 650 !important;
        color: #191c20;
        margin-top: 1rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 650 !important;
        color: #24272b;
    }

    p {
        color: #353a40;
        line-height: 1.65;
    }

    .eyebrow {
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #243b53;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1.18rem;
        line-height: 1.65;
        color: #4b5560;
        max-width: 850px;
        margin-bottom: 1.5rem;
    }

    .hero-note {
        border-left: 3px solid #243b53;
        padding: 0.7rem 1rem;
        background: #f6f8fa;
        color: #4a5057;
        font-size: 0.9rem;
        line-height: 1.55;
        margin: 1.5rem 0;
    }

    .section-divider {
        height: 1px;
        background: #dfe3e7;
        margin: 3rem 0 2rem 0;
    }

    .card {
        border: 1px solid #dfe3e7;
        border-radius: 10px;
        padding: 1.35rem 1.4rem;
        background: #fbfcfd;
        min-height: 150px;
    }

    .card-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #20252a;
        margin-bottom: 0.55rem;
    }

    .card-text {
        font-size: 0.87rem;
        line-height: 1.55;
        color: #59616a;
    }

    .result-card {
        border: 1px solid #cfd6dd;
        border-radius: 12px;
        background: #f8fafb;
        padding: 1.6rem;
        margin-top: 1rem;
    }

    .result-label {
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #59636e;
    }

    .result-number {
        font-size: 3.4rem;
        line-height: 1.05;
        font-weight: 750;
        color: #17202a;
        margin: 0.4rem 0 0.6rem 0;
    }

    .result-text {
        font-size: 0.88rem;
        color: #5c6670;
        line-height: 1.55;
    }

    .method-card {
        border-top: 3px solid #243b53;
        border-left: 1px solid #dfe3e7;
        border-right: 1px solid #dfe3e7;
        border-bottom: 1px solid #dfe3e7;
        border-radius: 0 0 9px 9px;
        padding: 1rem;
        min-height: 135px;
        background: #fbfcfd;
    }

    .method-number {
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        color: #68737e;
    }

    .method-title {
        font-weight: 700;
        margin-top: 0.45rem;
        color: #20252a;
    }

    .method-text {
        font-size: 0.78rem;
        color: #68717b;
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    .source-box {
        border: 1px solid #dfe3e7;
        border-radius: 10px;
        background: #fbfcfd;
        padding: 1.15rem 1.3rem;
        margin-bottom: 0.8rem;
    }

    .source-title {
        font-weight: 700;
        color: #20252a;
        margin-bottom: 0.25rem;
    }

    .source-text {
        font-size: 0.86rem;
        color: #5d6670;
        line-height: 1.5;
    }

    .footer {
        border-top: 1px solid #dfe3e7;
        margin-top: 3rem;
        padding-top: 1.2rem;
        text-align: center;
        color: #747c84;
        font-size: 0.78rem;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():

    with open(BASE_DIR / "lagged_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(BASE_DIR / "lagged_random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    return preprocessor, model


preprocessor, model = load_models()

# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="eyebrow">International Relations × Machine Learning</div>',
    unsafe_allow_html=True
)

st.title("Conflict Risk Prediction")

st.markdown(
    '<div class="hero-subtitle">'
    'A machine-learning study examining whether previous-year '
    'socioeconomic indicators can provide predictive information '
    'about organized state-based conflict.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-note">'
    '<strong>Academic research application.</strong> '
    'The model identifies statistical patterns in historical data. '
    'Its outputs are predictive estimates and should not be interpreted '
    'as causal explanations or definitive forecasts of future conflict.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# 01 — RESEARCH QUESTION
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">01 — Research Question</div>',
    unsafe_allow_html=True
)

st.header("What is this project examining?")

st.write(
    """
    This project asks whether socioeconomic conditions observed in the
    previous year contain useful predictive information about organized
    state-based conflict in the following period.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Temporal Approach</div>
            <div class="card-text">
                Socioeconomic predictors are lagged by one year so that
                preceding conditions are used to assess the subsequent
                conflict outcome.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Analytical Approach</div>
            <div class="card-text">
                A Random Forest classifier is used to identify nonlinear
                patterns and interactions among the socioeconomic indicators.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Research Boundary</div>
            <div class="card-text">
                The model examines predictive association. It does not
                attempt to establish that socioeconomic indicators cause conflict.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# 02 — DATA & SOURCES
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">02 — Data & Sources</div>',
    unsafe_allow_html=True
)

st.header("Data Foundation")

st.write(
    """
    The project combines organized state-based conflict observations with
    country-level socioeconomic indicators. The datasets were integrated
    at the country-year level before constructing previous-year predictors
    for model development.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="source-box">
            <div class="source-title">Conflict Data</div>
            <div class="source-text">
                Organized state-based conflict observations provide the
                outcome variable used for classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="source-box">
            <div class="source-title">World Bank Indicators</div>
            <div class="source-text">
                Country-level socioeconomic indicators provide the
                explanatory variables used by the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("#### Indicators used")

indicator_data = pd.DataFrame({
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
    "Temporal role": [
        "Previous year",
        "Previous year",
        "Previous year",
        "Previous year",
        "Previous year",
        "Previous year"
    ]
})

st.dataframe(
    indicator_data,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 03 — PREDICTION
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">03 — Model Assessment</div>',
    unsafe_allow_html=True
)

st.header("Generate an Assessment")

st.write(
    "Enter the country's previous-year socioeconomic conditions."
)

st.markdown("#### Economic conditions")

col1, col2, col3 = st.columns(3)

with col1:
    gdp_per_capita = st.number_input(
        "GDP per capita (US$)",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

with col2:
    gdp_growth = st.number_input(
        "GDP growth (%)",
        value=3.0,
        step=0.1
    )

with col3:
    inflation = st.number_input(
        "Inflation (%)",
        value=5.0,
        step=0.1
    )

st.markdown("#### Demographic and labour conditions")

col1, col2 = st.columns(2)

with col1:
    population = st.number_input(
        "Population",
        min_value=0.0,
        value=10000000.0,
        step=100000.0
    )

with col2:
    unemployment = st.number_input(
        "Unemployment (%)",
        min_value=0.0,
        value=6.0,
        step=0.1
    )

st.markdown("#### Security conditions")

military_expenditure = st.number_input(
    "Military expenditure (% of GDP)",
    min_value=0.0,
    value=2.0,
    step=0.1
)

st.write("")

predict_button = st.button(
    "Generate Model Assessment",
    use_container_width=True
)

# =========================================================
# 04 — RESULT
# =========================================================

if predict_button:

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
        '<div class="section-divider"></div>'
        '<div class="eyebrow">Model Output</div>',
        unsafe_allow_html=True
    )

    st.header("Assessment Result")

    col1, col2 = st.columns([1, 1.4])

    with col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated probability</div>
                <div class="result-number">{probability * 100:.1f}%</div>
                <div class="result-text">
                    Probability assigned by the trained Random Forest
                    classifier to the positive conflict class.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        if prediction == 1:

            st.subheader("Higher-risk classification")

            st.warning(
                "For the supplied conditions, the model assigns the "
                "observation to the positive conflict class."
            )

        else:

            st.subheader("Lower-risk classification")

            st.info(
                "For the supplied conditions, the model assigns the "
                "observation to the negative conflict class."
            )

        st.caption(
            "This is a model-based classification and should not be "
            "interpreted as a definitive forecast."
        )

# =========================================================
# 05 — PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">04 — Model Evaluation</div>',
    unsafe_allow_html=True
)

st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Lagged Model Accuracy",
        value="92.49%"
    )

with col2:
    st.metric(
        label="Country-Held-Out Accuracy",
        value="83.12%"
    )

st.write(
    """
    The country-held-out evaluation tests the model on countries that
    were completely excluded from training. This provides a more demanding
    assessment of generalization to countries not encountered during training.
    """
)

st.caption(
    "Accuracy is reported for the evaluation procedures used in this project."
)

# =========================================================
# 06 — FEATURE IMPORTANCE
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">05 — Model Interpretation</div>',
    unsafe_allow_html=True
)

st.header("Predictive Feature Importance")

feature_names = [
    "GDP per capita",
    "GDP growth",
    "Inflation",
    "Population",
    "Unemployment",
    "Military expenditure"
]

if hasattr(model, "feature_importances_"):

    importance_values = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    }).sort_values(
        "Importance",
        ascending=True
    )

    st.bar_chart(
        importance_df.set_index("Feature"),
        horizontal=True
    )

    st.caption(
        "Feature importance represents relative predictive contribution "
        "within the trained model. It does not establish causality."
    )

else:

    st.info(
        "Feature-importance information is not available for this model."
    )

# =========================================================
# 07 — METHODOLOGY
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">06 — Methodology</div>',
    unsafe_allow_html=True
)

st.header("Research Pipeline")

methodology = [
    ("01", "Data integration", "Combine conflict and socioeconomic observations."),
    ("02", "Temporal engineering", "Construct previous-year predictors."),
    ("03", "Preprocessing", "Apply the trained transformation pipeline."),
    ("04", "Classification", "Generate predictions using Random Forest."),
    ("05", "Evaluation", "Assess accuracy and country-level generalization.")
]

method_cols = st.columns(5)

for col, (number, title, description) in zip(method_cols, methodology):

    with col:

        st.markdown(
            f"""
            <div class="method-card">
                <div class="method-number">{number}</div>
                <div class="method-title">{title}</div>
                <div class="method-text">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# 08 — INTERPRETATION
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">07 — Interpretation & Limitations</div>',
    unsafe_allow_html=True
)

st.header("What the Model Can — and Cannot — Tell Us")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">What it can indicate</div>
            <div class="card-text">
                Whether the socioeconomic indicators supplied to the model
                resemble patterns associated with the observed conflict
                outcomes in the training data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">What it cannot establish</div>
            <div class="card-text">
                The model does not establish causality and does not account
                for every political, historical, geographic or strategic
                factor that may influence conflict.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

st.write(
    """
    The resulting probability should therefore be interpreted as a
    statistical output of the trained model rather than a definitive
    geopolitical forecast. The project is intended as an academic
    exploration of predictive machine learning in International Relations.
    """
)

# =========================================================
# 09 — PROJECT SUMMARY
# =========================================================

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="eyebrow">08 — Project Summary</div>',
    unsafe_allow_html=True
)

st.header("International Relations × Machine Learning")

st.write(
    """
    This project demonstrates how quantitative and computational methods
    can be applied to an International Relations research question.
    It combines data integration, temporal feature engineering,
    machine-learning classification, model evaluation and public deployment
    in a single analytical workflow.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Research</div>
            <div class="card-text">
                Conflict studies and socioeconomic conditions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Technical</div>
            <div class="card-text">
                Python, preprocessing, Random Forest and evaluation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Deployment</div>
            <div class="card-text">
                Interactive public application built with Streamlit.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'Academic project · International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
