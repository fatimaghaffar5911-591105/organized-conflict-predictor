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

    /* ---------- Page ---------- */

    .stApp {
        background-color: #faf9f7;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 3.2rem;
        padding-bottom: 4rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
    }

    /* ---------- Typography ---------- */

    h1 {
        color: #25282b !important;
        font-size: 2.7rem !important;
        font-weight: 720 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.08 !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        color: #292d31 !important;
        font-size: 1.75rem !important;
        font-weight: 680 !important;
        letter-spacing: -0.025em !important;
    }

    h3 {
        color: #303438 !important;
        font-size: 1.05rem !important;
        font-weight: 650 !important;
    }

    p {
        color: #25282b !important;
        line-height: 1.7 !important;
    }

    /* ---------- Section labels ---------- */

    .section-label {
        color: #4b5055;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-top: 3rem;
        margin-bottom: 0.55rem;
    }

    /* ---------- Hero ---------- */

    .hero-description {
        max-width: 850px;
        color: #303438;
        font-size: 1.08rem;
        line-height: 1.7;
        margin-bottom: 1.4rem;
    }

    /* ---------- Research Question ---------- */

    .research-question {
        background: #f1f0ed;
        border-left: 4px solid #3d4246;
        border-radius: 3px;
        padding: 1.35rem 1.5rem;
        margin: 1rem 0 1.4rem 0;
    }

    .research-question-title {
        color: #555b60;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .research-question-text {
        color: #202326;
        font-size: 1.32rem;
        font-weight: 650;
        line-height: 1.45;
    }

    /* ---------- Information Cards ---------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #dededb;
        border-radius: 5px;
        padding: 1.1rem 1.15rem;
        min-height: 125px;
    }

    .info-title {
        color: #25282b;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .info-text {
        color: #25282b;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* ---------- Inputs ---------- */

    label {
        color: #25282b !important;
        font-weight: 600 !important;
    }

    [data-baseweb="input"] {
        border-radius: 4px;
    }

    /* ---------- Button ---------- */

    .stButton > button {
        background-color: #303438;
        color: #ffffff;
        border: 1px solid #303438;
        border-radius: 4px;
        font-weight: 650;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        background-color: #454a4f;
        border-color: #454a4f;
        color: #ffffff;
    }

    /* ---------- Result ---------- */

    .result-box {
        background: #f1f0ed;
        border-left: 4px solid #3d4246;
        border-radius: 4px;
        padding: 1.35rem 1.45rem;
    }

    .result-label {
        color: #5d6368;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .result-number {
        color: #202326;
        font-size: 3rem;
        font-weight: 720;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin: 0.3rem 0 0.5rem 0;
    }

    .result-description {
        color: #25282b;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* ---------- Metrics ---------- */

    [data-testid="stMetricValue"] {
        color: #202326 !important;
        font-weight: 720 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #40464b !important;
        font-weight: 650 !important;
    }

    /* ---------- Dividers ---------- */

    hr {
        border: none;
        border-top: 1px solid #d8d7d3;
        margin: 2.3rem 0;
    }

    /* ---------- Footer ---------- */

    .footer {
        color: #777b7f;
        font-size: 0.73rem;
        text-align: center;
        margin-top: 2rem;
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

st.markdown(
    """
    <div class="hero-description">
        An interactive research project examining whether
        socioeconomic conditions from the previous year can provide
        useful predictive information about organized state-based
        conflict.
    </div>
    """,
    unsafe_allow_html=True
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
    <div class="research-question">

        <div class="research-question-title">
            The central question of this project
        </div>

        <div class="research-question-text">
            To what extent can previous-year socioeconomic conditions
            help predict organized state-based conflict?
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    The project examines whether socioeconomic conditions observed
    in the previous year contain useful predictive information
    about conflict outcomes in the following period.
    """
)


# ============================================================
# APPROACH
# ============================================================

st.markdown(
    '<div class="section-label">Research Approach</div>',
    unsafe_allow_html=True
)

st.header("How the model approaches the question")

st.write(
    """
    Six country-level indicators from the previous year are used
    as predictors. A Random Forest classifier then identifies
    patterns between these conditions and historical conflict
    outcomes.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Temporal design</div>
            <div class="info-text">
                Previous-year socioeconomic conditions are used
                to examine the subsequent conflict outcome.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Machine-learning method</div>
            <div class="info-text">
                A Random Forest classifier learns patterns from
                the historical training data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Research boundary</div>
            <div class="info-text">
                The model examines predictive patterns and does
                not establish causal relationships.
            </div>
        </div>
        """,
        unsafe_allow_html=True
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
    """
    The project combines organized state-based conflict observations
    with country-level socioeconomic indicators. The socioeconomic
    variables were obtained from World Bank data and integrated at
    the country-year level before creating previous-year predictors.
    """
)

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

st.write(
    """
    Enter previous-year socioeconomic conditions and examine how
    the trained model responds to different hypothetical profiles.
    """
)

st.caption(
    "This tool is intended for scenario exploration and does not "
    "provide real-time country risk assessments."
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
        '<div class="section-label">Model Output</div>',
        unsafe_allow_html=True
    )

    st.header("Scenario assessment")

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

            st.write(
                """
                Under the supplied hypothetical conditions, the
                model assigns the scenario to the positive conflict
                class.
                """
            )

        else:

            st.subheader("Negative conflict classification")

            st.write(
                """
                Under the supplied hypothetical conditions, the
                model assigns the scenario to the negative conflict
                class.
                """
            )

    st.caption(
        "This is a model-based classification, not a definitive forecast."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

st.markdown(
    '<div class="section-label">Model Evaluation</div>',
    unsafe_allow_html=True
)

st.header("Performance")

p1, p2 = st.columns(2)

with p1:
    st.metric(
        "Lagged evaluation accuracy",
        "92.49%"
    )

    st.caption(
        "Accuracy under the lagged evaluation framework."
    )

with p2:
    st.metric(
        "Country-held-out accuracy",
        "83.12%"
    )

    st.caption(
        "Accuracy when tested on countries completely excluded "
        "from training."
    )

st.write(
    """
    The country-held-out evaluation provides an additional test
    of generalization by evaluating the model on countries that
    were not included in training.
    """
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-label">Model Interpretation</div>',
    unsafe_allow_html=True
)

st.header("Which factors influenced the model?")

st.write(
    """
    The chart shows the relative predictive importance of the
    six socioeconomic and security indicators within the trained
    Random Forest model.
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
        "Feature importance represents relative predictive contribution "
        "within the model. It does not establish causality."
    )


# ============================================================
# METHODOLOGY
# ============================================================

st.markdown(
    '<div class="section-label">Methodology</div>',
    unsafe_allow_html=True
)

st.header("From data to model output")

method_df = pd.DataFrame({
    "Stage": [
        "Data integration",
        "Temporal engineering",
        "Preprocessing",
        "Classification",
        "Evaluation"
    ],
    "Description": [
        "Conflict and socioeconomic observations are combined.",
        "Previous-year predictor variables are constructed.",
        "The trained preprocessing pipeline transforms the inputs.",
        "Random Forest generates the classification and probability.",
        "Performance is assessed using multiple evaluation approaches."
    ]
})

st.dataframe(
    method_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INTERPRETATION AND LIMITATIONS
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
        <div class="info-card">

            <div class="info-title">
                What it can indicate
            </div>

            <div class="info-text">
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
        <div class="info-card">

            <div class="info-title">
                What it cannot establish
            </div>

            <div class="info-text">
                The model does not establish causality or capture
                the full political, historical, geographic and
                strategic complexity of conflict.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    """
    Conflict is shaped by many factors beyond the variables used
    here, including political institutions, history, geography,
    leadership, identity, external intervention and strategic
    decisions.
    """
)

st.write(
    """
    The project should therefore be understood as an exploration
    of predictive machine learning applied to an International
    Relations question, rather than as a comprehensive conflict
    forecasting system.
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
    This project demonstrates how computational methods can be
    applied to an International Relations research question through
    data integration, temporal feature engineering, machine-learning
    classification, model evaluation and public deployment.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                Research
            </div>

            <div class="info-text">
                Examining the relationship between socioeconomic
                conditions and organized state-based conflict.
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
                Technical work
            </div>

            <div class="info-text">
                Python, preprocessing, Random Forest classification,
                feature analysis and model evaluation.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                Deployment
            </div>

            <div class="info-text">
                An interactive public application developed with
                Streamlit.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FINAL NOTE
# ============================================================

st.markdown("---")

st.write(
    """
    This application is an academic demonstration of applying
    machine-learning methods to an International Relations question.
    Its outputs are statistical model estimates and should not be
    interpreted as definitive statements about future conflict.
    """
)

st.markdown(
    '<div class="footer">'
    'Academic project · International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
