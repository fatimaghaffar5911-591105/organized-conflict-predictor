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

/* Typography */

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
}

p {
    color: #303437 !important;
    line-height: 1.7 !important;
}

/* Section labels */

.section-label {
    color: #60656a;
    font-size: 0.66rem;
    font-weight: 750;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2.8rem;
    margin-bottom: 0.55rem;
}

/* Research question */

.question-box {
    background: #f1f0ed;
    border-left: 4px solid #555b60;
    padding: 1.4rem 1.5rem;
    margin: 1rem 0 1.5rem 0;
    border-radius: 3px;
}

.question-title {
    color: #60656a;
    font-size: 0.66rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.question-text {
    color: #272a2d;
    font-size: 1.28rem;
    font-weight: 650;
    line-height: 1.48;
}

/* Cards */

.info-card {
    background: #ffffff;
    border: 1px solid #deddd9;
    border-radius: 5px;
    padding: 1.15rem 1.2rem;
    min-height: 125px;
}

.info-title {
    color: #303438;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.info-text {
    color: #303438;
    font-size: 0.82rem;
    line-height: 1.62;
}

/* Scenario note */

.scenario-note {
    background: #f3f2ef;
    border: 1px solid #deddd9;
    border-radius: 4px;
    padding: 0.95rem 1rem;
    color: #303438;
    font-size: 0.82rem;
    line-height: 1.58;
    margin-bottom: 1.15rem;
}

/* Inputs */

label {
    color: #303438 !important;
    font-weight: 600 !important;
}

/* Button */

.stButton > button {
    background-color: #41464a;
    color: white;
    border: 1px solid #41464a;
    border-radius: 4px;
    font-weight: 650;
    min-height: 2.8rem;
}

.stButton > button:hover {
    background-color: #555b60;
    border-color: #555b60;
}

/* Result */

.result-box {
    background: #f1f0ed;
    border-left: 4px solid #555b60;
    border-radius: 4px;
    padding: 1.4rem 1.5rem;
}

.result-label {
    color: #60656a;
    font-size: 0.66rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.result-number {
    color: #282b2e;
    font-size: 3rem;
    font-weight: 720;
    line-height: 1.1;
    margin: 0.3rem 0 0.5rem 0;
}

.result-description {
    color: #303438;
    font-size: 0.82rem;
    line-height: 1.55;
}

/* Metrics */

[data-testid="stMetricValue"] {
    color: #292d30 !important;
    font-weight: 720 !important;
}

[data-testid="stMetricLabel"] {
    color: #454a4e !important;
    font-weight: 650 !important;
}

/* Footer */

.footer {
    color: #777b7f;
    font-size: 0.73rem;
    text-align: center;
    margin-top: 2rem;
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
    '<div class="question-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="question-title">Central question</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="question-text">'
    'To what extent can previous-year socioeconomic conditions '
    'help predict the occurrence of organized state-based conflict?'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
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
    '<div class="section-label">How It Works</div>',
    unsafe_allow_html=True
)

st.header("From previous-year conditions to a model estimate")

st.write(
    """
    The application uses a Random Forest classifier trained on
    historical country-year observations. The model receives
    socioeconomic conditions from the previous year and identifies
    patterns associated with observed conflict outcomes.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Previous-year conditions")
    st.write(
        """
        Six socioeconomic and security indicators from the preceding
        year are used as inputs.
        """
    )

with c2:
    st.subheader("Pattern recognition")
    st.write(
        """
        A Random Forest combines multiple decision trees to identify
        patterns within the historical data.
        """
    )

with c3:
    st.subheader("Model estimate")
    st.write(
        """
        The trained model produces a probability and a corresponding
        conflict classification.
        """
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
    st.subheader("Organized conflict data")
    st.write(
        """
        Historical observations of organized state-based conflict
        provide the outcome used for classification.
        """
    )

with c2:
    st.subheader("World Bank data")
    st.write(
        """
        Country-level socioeconomic indicators provide the predictor
        variables used by the model.
        """
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
    <div class="scenario-note">
        Enter previous-year socioeconomic conditions to see how the
        trained model responds to a hypothetical country profile.
        The result is a model-based estimate and is not a real-time
        assessment of any country's current conflict risk.
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
    st.metric(
        "Lagged Model Accuracy",
        "92.49%"
    )

with c2:
    st.metric(
        "Country-Held-Out Accuracy",
        "83.12%"
    )

st.write(
    """
    The country-held-out evaluation provides a stricter test of
    generalization by evaluating the model on countries that were
    completely excluded from training.
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
# INTERPRETATION AND LIMITATIONS
# ============================================================

st.markdown(
    '<div class="section-label">Interpretation & Limitations</div>',
    unsafe_allow_html=True
)

st.header("What do these predictions actually mean?")

c1, c2 = st.columns(2)

with c1:

    st.subheader("What the model can indicate")

    st.write(
        """
        Whether a supplied socioeconomic profile resembles patterns
        associated with historical conflict outcomes in the data used
        to train the model.
        """
    )

with c2:

    st.subheader("What the model cannot establish")

    st.write(
        """
        It cannot establish causality or capture the complete
        political, historical, geographic and strategic complexity
        behind organized conflict.
        """
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
    st.subheader("Research")
    st.write(
        """
        An empirical exploration of socioeconomic conditions and
        organized state-based conflict.
        """
    )

with c2:
    st.subheader("Machine Learning")
    st.write(
        """
        Random Forest classification, preprocessing, probability
        estimation and model evaluation.
        """
    )

with c3:
    st.subheader("Application")
    st.write(
        """
        An interactive interface allowing users to explore
        hypothetical socioeconomic scenarios.
        """
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
