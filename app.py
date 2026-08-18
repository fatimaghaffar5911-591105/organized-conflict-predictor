import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Conflict Risk Prediction | IR × Machine Learning",
    page_icon=None,
    layout="wide"
)

# ============================================================
# VISUAL DESIGN
# ============================================================

st.markdown("""
<style>

    /* ---------- Page ---------- */

    .block-container {
        max-width: 1120px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* ---------- Typography ---------- */

    h1 {
        color: #17191c !important;
        font-size: 3.15rem !important;
        font-weight: 720 !important;
        letter-spacing: -0.045em;
        line-height: 1.05 !important;
        margin-bottom: 0.6rem !important;
    }

    h2 {
        color: #1d2125 !important;
        font-size: 1.7rem !important;
        font-weight: 680 !important;
        letter-spacing: -0.02em;
    }

    h3 {
        color: #252a2f !important;
        font-size: 1.1rem !important;
        font-weight: 650 !important;
    }

    p {
        color: #3f464d;
        line-height: 1.7;
    }

    /* ---------- Section labels ---------- */

    .section-label {
        color: #29465f;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-top: 3.1rem;
        margin-bottom: 0.55rem;
    }

    /* ---------- Hero ---------- */

    .hero-kicker {
        color: #29465f;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-description {
        max-width: 860px;
        color: #525a62;
        font-size: 1.15rem;
        line-height: 1.7;
        margin-bottom: 1.3rem;
    }

    .research-note {
        max-width: 900px;
        border-left: 3px solid #29465f;
        background: #f5f7f9;
        padding: 0.9rem 1.1rem;
        color: #505860;
        font-size: 0.87rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* ---------- Quiet highlight ---------- */

    .quiet-box {
        background: #fafbfc;
        border: 1px solid #dce1e5;
        border-radius: 5px;
        padding: 1.2rem 1.3rem;
        height: 100%;
    }

    .quiet-title {
        color: #20252a;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .quiet-text {
        color: #626a72;
        font-size: 0.83rem;
        line-height: 1.55;
    }

    /* ---------- Scenario area ---------- */

    .scenario-intro {
        color: #555e66;
        max-width: 820px;
        line-height: 1.65;
        margin-bottom: 1.2rem;
    }

    /* ---------- Result ---------- */

    .result-box {
        background: #f7f9fa;
        border: 1px solid #ccd5dc;
        border-left: 4px solid #29465f;
        border-radius: 5px;
        padding: 1.4rem 1.5rem;
    }

    .result-label {
        color: #66717b;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .result-number {
        color: #18232d;
        font-size: 3.15rem;
        font-weight: 720;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin: 0.35rem 0 0.5rem 0;
    }

    .result-description {
        color: #626c75;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* ---------- Performance ---------- */

    .performance {
        border-top: 2px solid #29465f;
        padding-top: 0.8rem;
    }

    .performance-label {
        color: #69737d;
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .performance-number {
        color: #18232d;
        font-size: 2.55rem;
        font-weight: 720;
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }

    .performance-description {
        color: #6a737b;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* ---------- Methodology ---------- */

    .method-step {
        border-top: 2px solid #29465f;
        padding-top: 0.75rem;
        min-height: 130px;
    }

    .method-number {
        color: #7a848d;
        font-size: 0.66rem;
        font-weight: 750;
        letter-spacing: 0.1em;
    }

    .method-title {
        color: #24292e;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    .method-description {
        color: #69727b;
        font-size: 0.77rem;
        line-height: 1.5;
        margin-top: 0.25rem;
    }

    /* ---------- Footer ---------- */

    .footer {
        margin-top: 3.5rem;
        padding-top: 1rem;
        border-top: 1px solid #dce1e5;
        text-align: center;
        color: #7c858d;
        font-size: 0.73rem;
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
    '<div class="hero-kicker">'
    'International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.title("Conflict Risk Prediction")

st.markdown(
    '<div class="hero-description">'
    'A machine-learning study examining whether previous-year '
    'socioeconomic indicators contain predictive information about '
    'organized state-based conflict.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="research-note">'
    '<strong>Academic research application.</strong> '
    'The model identifies statistical patterns in historical data. '
    'Its outputs represent model-based estimates and should not be '
    'interpreted as causal explanations or definitive forecasts.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# RESEARCH QUESTION
# ============================================================

st.markdown(
    '<div class="section-label">Research Question</div>',
    unsafe_allow_html=True
)

st.header(
    "To what extent can previous-year socioeconomic indicators "
    "help predict organized state-based conflict?"
)

st.write(
    """
    The project examines whether socioeconomic conditions observed in
    the previous year contain useful predictive information about
    conflict outcomes in the following period.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="quiet-box">
            <div class="quiet-title">Temporal design</div>
            <div class="quiet-text">
                Socioeconomic variables are lagged by one year so that
                preceding conditions are used to assess the subsequent
                outcome.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="quiet-box">
            <div class="quiet-title">Analytical approach</div>
            <div class="quiet-text">
                A Random Forest classifier is used to identify patterns
                across multiple socioeconomic indicators.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="quiet-box">
            <div class="quiet-title">Research boundary</div>
            <div class="quiet-text">
                The model evaluates predictive association and does not
                attempt to establish causal relationships.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA & SOURCES
# ============================================================

st.markdown(
    '<div class="section-label">Data & Sources</div>',
    unsafe_allow_html=True
)

st.header("Data Foundation")

st.write(
    """
    The project combines organized state-based conflict observations
    with country-level socioeconomic indicators. The datasets were
    integrated at the country-year level before constructing
    previous-year predictors.
    """
)

s1, s2 = st.columns(2)

with s1:
    st.markdown(
        """
        <div class="quiet-box">
            <div class="quiet-title">Conflict data</div>
            <div class="quiet-text">
                Organized state-based conflict observations provide
                the outcome used for classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s2:
    st.markdown(
        """
        <div class="quiet-box">
            <div class="quiet-title">World Bank</div>
            <div class="quiet-text">
                Country-level socioeconomic indicators provide the
                predictor variables used by the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("#### Indicators")

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
    indicator_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SCENARIO EXPLORER
# ============================================================

st.markdown(
    '<div class="section-label">Scenario Explorer</div>',
    unsafe_allow_html=True
)

st.header("Explore a hypothetical country profile")

st.markdown(
    '<div class="scenario-intro">'
    'Enter previous-year socioeconomic conditions and examine how '
    'the trained model responds to different hypothetical profiles.'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "This tool is intended for scenario exploration. "
    "It does not provide real-time country risk assessments."
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

st.write("")

assess = st.button(
    "Assess Scenario",
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
        '<div class="section-label">Model Output</div>',
        unsafe_allow_html=True
    )

    st.header("Scenario Assessment")

    r1, r2 = st.columns([1, 1.5])

    with r1:

        st.markdown(
            f"""
            <div class="result-box">

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

    with r2:

        if prediction == 1:

            st.subheader("Positive conflict classification")

            st.warning(
                "Under the supplied hypothetical conditions, the model "
                "assigns the scenario to the positive conflict class."
            )

        else:

            st.subheader("Negative conflict classification")

            st.info(
                "Under the supplied hypothetical conditions, the model "
                "assigns the scenario to the negative conflict class."
            )

        st.caption(
            "This is a model-based classification, not a definitive forecast."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-label">Model Evaluation</div>',
    unsafe_allow_html=True
)

st.header("Performance")

p1, p2 = st.columns(2)

with p1:

    st.markdown(
        """
        <div class="performance">

            <div class="performance-label">
                Lagged evaluation accuracy
            </div>

            <div class="performance-number">
                92.49%
            </div>

            <div class="performance-description">
                Accuracy under the lagged evaluation framework.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with p2:

    st.markdown(
        """
        <div class="performance">

            <div class="performance-label">
                Country-held-out accuracy
            </div>

            <div class="performance-number">
                83.12%
            </div>

            <div class="performance-description">
                Evaluation on countries completely excluded from training.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    """
    The country-held-out evaluation provides an additional test of
    generalization by evaluating the model on countries that were not
    included in the training data.
    """
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-label">Model Interpretation</div>',
    unsafe_allow_html=True
)

st.header("What influenced the model's predictions?")

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
        "Feature importance represents relative predictive contribution "
        "within the trained model. It does not establish causality."
    )

else:

    st.info(
        "Feature-importance information is not available for this model."
    )


# ============================================================
# METHODOLOGY
# ============================================================

st.markdown(
    '<div class="section-label">Methodology</div>',
    unsafe_allow_html=True
)

st.header("From data to model")

steps = [
    (
        "01",
        "Data integration",
        "Combine conflict and socioeconomic observations."
    ),
    (
        "02",
        "Temporal engineering",
        "Construct previous-year predictor variables."
    ),
    (
        "03",
        "Preprocessing",
        "Apply the trained preprocessing pipeline."
    ),
    (
        "04",
        "Classification",
        "Generate predictions using Random Forest."
    ),
    (
        "05",
        "Evaluation",
        "Assess performance and country-level generalization."
    )
]

method_cols = st.columns(5)

for col, (number, title, description) in zip(method_cols, steps):

    with col:

        st.markdown(
            f"""
            <div class="method-step">

                <div class="method-number">
                    {number}
                </div>

                <div class="method-title">
                    {title}
                </div>

                <div class="method-description">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# INTERPRETATION & LIMITATIONS
# ============================================================

st.markdown(
    '<div class="section-label">Interpretation & Limitations</div>',
    unsafe_allow_html=True
)

st.header("What the model can — and cannot — tell us")

l1, l2 = st.columns(2)

with l1:

    st.markdown(
        """
        <div class="quiet-box">

            <div class="quiet-title">
                What it can indicate
            </div>

            <div class="quiet-text">
                Whether a supplied socioeconomic profile resembles
                patterns associated with observed conflict outcomes
                in the historical data used to train the model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with l2:

    st.markdown(
        """
        <div class="quiet-box">

            <div class="quiet-title">
                What it cannot establish
            </div>

            <div class="quiet-text">
                The model does not establish causality and does not
                capture the full political, historical, geographic
                and strategic complexity of conflict.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

st.write(
    """
    The project should therefore be understood as an exploration of
    predictive machine learning applied to an International Relations
    question, rather than as a comprehensive conflict forecasting system.
    """
)


# ============================================================
# PROJECT CONTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-label">Project Contribution</div>',
    unsafe_allow_html=True
)

st.header("International Relations × Machine Learning")

st.write(
    """
    This project demonstrates how computational methods can be applied
    to an International Relations research question through data
    integration, temporal feature engineering, machine-learning
    classification, evaluation and public deployment.
    """
)

a1, a2, a3 = st.columns(3)

with a1:

    st.markdown(
        """
        <div class="quiet-box">

            <div class="quiet-title">
                Research
            </div>

            <div class="quiet-text">
                Conflict studies and socioeconomic conditions.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with a2:

    st.markdown(
        """
        <div class="quiet-box">

            <div class="quiet-title">
                Technical work
            </div>

            <div class="quiet-text">
                Python, preprocessing, Random Forest and model evaluation.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with a3:

    st.markdown(
        """
        <div class="quiet-box">

            <div class="quiet-title">
                Deployment
            </div>

            <div class="quiet-text">
                Interactive public application developed with Streamlit.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Academic project · International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
