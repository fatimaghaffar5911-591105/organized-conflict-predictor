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
    background-color: #faf9f7;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.8rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}


/* ============================================================
   TYPOGRAPHY
   ============================================================ */

h1 {
    color: #34383b !important;
    font-size: 2.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.1 !important;
}

h2 {
    color: #383c40 !important;
    font-size: 1.65rem !important;
    font-weight: 680 !important;
}

h3 {
    color: #3c4044 !important;
    font-size: 1rem !important;
    font-weight: 680 !important;
}

p {
    color: #303437 !important;
    line-height: 1.7 !important;
}


/* ============================================================
   SECTION LABELS
   ============================================================ */

.section-label {
    color: #656a6e;
    font-size: 0.66rem;
    font-weight: 750;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2.8rem;
    margin-bottom: 0.55rem;
}


/* ============================================================
   RESEARCH QUESTION
   ============================================================ */

.question-box {
    background: #f1f0ed;
    border: 1px solid #d8d6d1;
    border-left: 4px solid #555b60;
    border-radius: 5px;
    padding: 1.45rem 1.55rem;
    margin: 1rem 0 1.6rem 0;
}

.question-title {
    color: #656a6e;
    font-size: 0.65rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.question-text {
    color: #292d30;
    font-size: 1.28rem;
    font-weight: 650;
    line-height: 1.48;
}


/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.info-card {
    background: #ffffff;
    border: 1px solid #dddcd8;
    border-radius: 5px;
    padding: 1.25rem 1.3rem;
    min-height: 145px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.025);
}

.info-title {
    color: #303438;
    font-size: 0.91rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}

.info-text {
    color: #303438;
    font-size: 0.82rem;
    line-height: 1.62;
}


/* ============================================================
   SOURCE CARDS
   ============================================================ */

.source-card {
    background: #ffffff;
    border: 1px solid #d9d8d4;
    border-radius: 5px;
    padding: 1.3rem 1.35rem;
    min-height: 145px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.025);
}

.source-title {
    color: #292d30;
    font-size: 0.96rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}

.source-text {
    color: #303437;
    font-size: 0.83rem;
    line-height: 1.65;
}


/* ============================================================
   SCENARIO NOTICE
   ============================================================ */

.scenario-box {
    background: #f3f2ef;
    border: 1px solid #d8d6d1;
    border-radius: 5px;
    padding: 1.05rem 1.2rem;
    margin: 0.8rem 0 1.4rem 0;
}

.scenario-title {
    color: #34383b;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.scenario-text {
    color: #303437;
    font-size: 0.82rem;
    line-height: 1.6;
}


/* ============================================================
   RESULT CARD
   ============================================================ */

.result-box {
    background: #f1f0ed;
    border: 1px solid #d5d3ce;
    border-left: 4px solid #555b60;
    border-radius: 5px;
    padding: 1.45rem 1.5rem;
    min-height: 190px;
}

.result-label {
    color: #62676b;
    font-size: 0.65rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.result-number {
    color: #292d30;
    font-size: 3rem;
    font-weight: 720;
    line-height: 1.1;
    margin: 0.3rem 0 0.55rem 0;
}

.result-description {
    color: #303437;
    font-size: 0.82rem;
    line-height: 1.58;
}


/* ============================================================
   INTERPRETATION CARDS
   ============================================================ */

.interpretation-card {
    background: #ffffff;
    border: 1px solid #dddcd8;
    border-radius: 5px;
    padding: 1.3rem 1.35rem;
    min-height: 170px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.025);
}

.interpretation-title {
    color: #303438;
    font-size: 0.91rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.interpretation-text {
    color: #303437;
    font-size: 0.82rem;
    line-height: 1.65;
}


/* ============================================================
   PROJECT CARDS
   ============================================================ */

.project-card {
    background: #ffffff;
    border: 1px solid #dddcd8;
    border-radius: 5px;
    padding: 1.25rem 1.3rem;
    min-height: 145px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.025);
}

.project-title {
    color: #303438;
    font-size: 0.91rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}

.project-text {
    color: #303437;
    font-size: 0.82rem;
    line-height: 1.62;
}


/* ============================================================
   INPUTS
   ============================================================ */

label {
    color: #303437 !important;
    font-weight: 600 !important;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    background-color: #41464a;
    color: #ffffff;
    border: 1px solid #41464a;
    border-radius: 4px;
    font-weight: 650;
    min-height: 2.8rem;
}

.stButton > button:hover {
    background-color: #555b60;
    border-color: #555b60;
}


/* ============================================================
   METRICS
   ============================================================ */

[data-testid="stMetricValue"] {
    color: #292d30 !important;
    font-weight: 720 !important;
}

[data-testid="stMetricLabel"] {
    color: #454a4e !important;
    font-weight: 650 !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    color: #777b7f;
    font-size: 0.73rem;
    text-align: center;
    margin-top: 1.5rem;
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
    """
    An interactive machine-learning research project examining whether
    socioeconomic conditions observed in the previous year contain
    useful predictive information about organized state-based conflict.
    """
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
        <div class="question-title">
            Central research question
        </div>

        <div class="question-text">
            To what extent can previous-year socioeconomic conditions
            help predict the occurrence of organized state-based conflict?
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    The project examines whether conditions in the year preceding
    a conflict outcome contain measurable predictive information.
    Several socioeconomic and security indicators are considered
    together rather than treating any single variable as an explanation
    for conflict.
    """
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-label">Model</div>',
    unsafe_allow_html=True
)

st.header("How the model approaches the question")

st.write(
    """
    The application uses a Random Forest classifier trained on
    historical country-year observations. Previous-year socioeconomic
    conditions are provided to the model, which identifies patterns
    associated with observed conflict outcomes.
    """
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">
                Previous-year conditions
            </div>

            <div class="info-text">
                Six socioeconomic and security indicators from the
                preceding year are used as model inputs.
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
                Random Forest
            </div>

            <div class="info-text">
                Multiple decision trees are combined to identify
                patterns across the historical observations.
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
                Model estimate
            </div>

            <div class="info-text">
                The trained model produces a probability together
                with a corresponding conflict classification.
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

st.header("The evidence behind the model")

st.write(
    """
    The project combines historical organized state-based conflict
    observations with country-level socioeconomic indicators.
    The socioeconomic indicators were obtained from World Bank data
    and integrated with conflict observations at the country-year level.
    """
)

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="source-card">
            <div class="source-title">
                Organized conflict data
            </div>

            <div class="source-text">
                Historical observations of organized state-based conflict
                provide the outcome used for classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="source-card">
            <div class="source-title">
                World Bank
            </div>

            <div class="source-text">
                Country-level socioeconomic indicators provide the
                predictor variables used by the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("#### Indicators included in the model")

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
# INTERACTIVE PREDICTION
# ============================================================

st.markdown(
    '<div class="section-label">Interactive Prediction</div>',
    unsafe_allow_html=True
)

st.header("Explore a hypothetical country profile")

st.markdown(
    """
    <div class="scenario-box">
        <div class="scenario-title">
            About this tool
        </div>

        <div class="scenario-text">
            Enter previous-year socioeconomic conditions to see how
            the trained model responds to a hypothetical country
            profile. This is a model-based estimate, not a real-time
            assessment of any country's current conflict risk.
        </div>
    </div>
    """,
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
        '<div class="section-label">Result</div>',
        unsafe_allow_html=True
    )

    st.header("Model assessment")

    c1, c2 = st.columns([1, 1.5])

    with c1:

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

    with c2:

        if prediction == 1:

            st.subheader("Positive conflict classification")

            st.write(
                """
                Under the supplied hypothetical conditions, the model
                assigns the scenario to the positive conflict class.
                """
            )

        else:

            st.subheader("Negative conflict classification")

            st.write(
                """
                Under the supplied hypothetical conditions, the model
                assigns the scenario to the negative conflict class.
                """
            )

        st.caption(
            "This is a model-based estimate, not a definitive forecast."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-label">Model Performance</div>',
    unsafe_allow_html=True
)

st.header("How well does the model generalize?")

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">
                Lagged Model Accuracy
            </div>

            <div style="
                color:#292d30;
                font-size:2.35rem;
                font-weight:720;
                margin:0.35rem 0;
            ">
                92.49%
            </div>

            <div class="info-text">
                Accuracy under the lagged evaluation framework.
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
                Country-Held-Out Accuracy
            </div>

            <div style="
                color:#292d30;
                font-size:2.35rem;
                font-weight:720;
                margin:0.35rem 0;
            ">
                83.12%
            </div>

            <div class="info-text">
                Performance when evaluated on countries completely
                excluded from training.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    """
    The country-held-out evaluation provides a stricter test of
    generalization because the model is evaluated on countries that
    were not included in its training data.
    """
)


# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-label">Model Interpretation</div>',
    unsafe_allow_html=True
)

st.header("Which factors influenced the model most?")

st.write(
    """
    Feature importance shows the relative contribution of each
    predictor to the Random Forest's decisions. These values describe
    how the model uses the variables and should not be interpreted as
    evidence that a particular factor causes conflict.
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
        "within the trained model and does not establish causality."
    )


# ============================================================
# INTERPRETATION & LIMITATIONS
# ============================================================

st.markdown(
    '<div class="section-label">Interpretation & Limitations</div>',
    unsafe_allow_html=True
)

st.header("What do these predictions actually mean?")

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="interpretation-card">

            <div class="interpretation-title">
                What the model can indicate
            </div>

            <div class="interpretation-text">
                Whether a supplied socioeconomic profile resembles
                patterns associated with historical conflict outcomes
                in the data used to train the model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="interpretation-card">

            <div class="interpretation-title">
                What the model cannot establish
            </div>

            <div class="interpretation-text">
                It cannot establish causality or capture the complete
                political, historical, geographic and strategic
                complexity behind organized conflict.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    """
    Conflict is influenced by many factors beyond the variables
    included in this model, including political institutions,
    historical grievances, identity, geography, leadership,
    external intervention and strategic decisions.
    """
)

st.write(
    """
    The application should therefore be understood as an exploration
    of predictive machine learning applied to an International
    Relations question, rather than as a comprehensive conflict
    forecasting system.
    """
)


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.markdown(
    '<div class="section-label">About the Project</div>',
    unsafe_allow_html=True
)

st.header("International Relations × Machine Learning")

st.write(
    """
    This project explores how computational methods can be applied
    to an International Relations question by combining country-level
    socioeconomic data, historical conflict observations and
    machine-learning classification in an interactive application.
    """
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
        <div class="project-card">

            <div class="project-title">
                Research
            </div>

            <div class="project-text">
                An empirical exploration of socioeconomic conditions
                and organized state-based conflict.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="project-card">

            <div class="project-title">
                Machine Learning
            </div>

            <div class="project-text">
                Random Forest classification, preprocessing,
                probability estimation and model evaluation.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
        <div class="project-card">

            <div class="project-title">
                Application
            </div>

            <div class="project-text">
                An interactive interface allowing users to explore
                hypothetical socioeconomic scenarios.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Academic project · International Relations × Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
