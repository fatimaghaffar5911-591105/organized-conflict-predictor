import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Conflict Risk Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Main application */
    .main {
        padding-top: 2rem;
    }

    /* Limit overall content width */
    .block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-bottom: 4rem;
    padding-left: 3rem;
    padding-right: 3rem;
}
    /* Typography */
    h1 {
        font-size: 2.7rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.03em;
    }

    h2 {
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }

    /* Hero subtitle */
    .hero-subtitle {
        font-size: 1.15rem;
        color: #5f6368;
        line-height: 1.6;
        max-width: 850px;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Small label */
    .eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5f6368;
        margin-bottom: 0.4rem;
    }

    /* Information cards */
    .info-card {
        background: #f7f8fa;
        border: 1px solid #e4e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        height: 100%;
    }

    .info-card-title {
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .info-card-text {
        font-size: 0.9rem;
        color: #606770;
        line-height: 1.55;
    }

    /* Result area */
    .result-card {
        background: #f7f8fa;
        border: 1px solid #dfe3e8;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0 1.5rem 0;
    }

    .result-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6b7280;
    }

    .result-number {
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.3rem 0;
    }

    .result-description {
        color: #606770;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Section divider */
    .section-line {
        height: 1px;
        background: #e5e7eb;
        margin: 2.5rem 0;
    }

    /* Methodology steps */
    .method-step {
        background: #f7f8fa;
        border: 1px solid #e4e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        min-height: 100px;
    }

    .method-number {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.08em;
    }

    .method-title {
        font-weight: 650;
        margin-top: 0.35rem;
    }

    .method-description {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #73777d;
        font-size: 0.82rem;
        padding-top: 1rem;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

base_dir = Path(__file__).resolve().parent


@st.cache_resource
def load_models():

    with open(base_dir / "lagged_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(base_dir / "lagged_random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    return preprocessor, model


preprocessor, model = load_models()

# ---------------------------------------------------------
# HERO / INTRODUCTION
# ---------------------------------------------------------

st.markdown(
    '<div class="eyebrow">International Relations × Machine Learning</div>',
    unsafe_allow_html=True
)

st.title("Conflict Risk Prediction")

st.markdown(
    '<div class="hero-subtitle">'
    'A machine-learning study exploring whether previous-year '
    'socioeconomic conditions can provide predictive information '
    'about organized state-based conflict.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    This application applies a Random Forest classifier to
    country-year socioeconomic indicators. The model uses
    previous-year conditions as predictors and produces a
    model-based probability for the positive conflict class.
    """
)

# ---------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">01 — Project Overview</div>', unsafe_allow_html=True)

st.header("Research Context")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Research Question</div>
            <div class="info-card-text">
                Can previous-year socioeconomic conditions provide
                useful predictive information about organized
                state-based conflict?
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Analytical Approach</div>
            <div class="info-card-text">
                A Random Forest classifier is trained using
                lagged socioeconomic indicators to identify
                predictive patterns in the data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Interpretation</div>
            <div class="info-card-text">
                Results represent statistical predictions from
                the trained model and should not be interpreted
                as causal explanations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">02 — Model Assessment</div>', unsafe_allow_html=True)

st.header("Generate an Assessment")

st.write(
    "Enter the country's previous-year socioeconomic conditions."
)

st.markdown("### Economic Conditions")

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

st.markdown("### Demographic and Labour Conditions")

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

st.markdown("### Security Conditions")

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

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

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

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Model Output</div>', unsafe_allow_html=True)

    st.header("Assessment Result")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Estimated probability
                </div>

                <div class="result-number">
                    {probability * 100:.1f}%
                </div>

                <div class="result-description">
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
                "The model classified the supplied conditions "
                "as belonging to the positive conflict class."
            )

        else:

            st.subheader("Lower-risk classification")

            st.success(
                "The model classified the supplied conditions "
                "as belonging to the negative conflict class."
            )

        st.caption(
            "This classification is a model output, not a definitive "
            "forecast of future conflict."
        )

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">03 — Model Evaluation</div>', unsafe_allow_html=True)

st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Lagged Model Accuracy",
        "92.49%"
    )

with col2:
    st.metric(
        "Country-Held-Out Accuracy",
        "83.12%"
    )

st.write(
    "The country-held-out evaluation tests the model on countries "
    "that were completely excluded from training. This provides "
    "a more demanding assessment of how the model performs on "
    "countries it did not encounter during training."
)

# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">04 — Model Interpretation</div>', unsafe_allow_html=True)

st.header("Predictive Feature Importance")

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
    "Feature importance indicates the relative predictive contribution "
    "of each variable within the trained model. It does not establish "
    "that a variable causes conflict."
)

# ---------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">05 — Methodology</div>', unsafe_allow_html=True)

st.header("Research Pipeline")

method_cols = st.columns(5)

steps = [
    ("01", "Data Integration", "Conflict and socioeconomic data"),
    ("02", "Lagged Features", "Previous-year predictors"),
    ("03", "Preprocessing", "Transformation of model inputs"),
    ("04", "Random Forest", "Classification model"),
    ("05", "Evaluation", "Accuracy and country-held-out testing")
]

for col, (number, title, description) in zip(method_cols, steps):

    with col:

        st.markdown(
            f"""
            <div class="method-step">
                <div class="method-number">{number}</div>
                <div class="method-title">{title}</div>
                <div class="method-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">06 — Interpretation</div>', unsafe_allow_html=True)

st.header("Interpretation and Limitations")

st.write(
    """
    This application is designed as an academic machine-learning
    experiment. Its purpose is to examine whether socioeconomic
    conditions contain predictive information associated with
    organized state-based conflict.
    """
)

st.write(
    """
    The model identifies statistical patterns in the training data.
    A predictive relationship should not be interpreted as evidence
    of a causal relationship. The probability generated by the model
    should therefore be understood as a model-based estimate rather
    than a definitive geopolitical forecast.
    """
)

# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">07 — Project Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        **Domain**

        International Relations  
        Political Analysis  
        Machine Learning  
        Data Science
        """
    )

with col2:

    st.markdown(
        """
        **Model**

        Random Forest Classifier  
        Lagged socioeconomic predictors  
        Streamlit deployment
        """
    )

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">'
    'Academic Project · International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
