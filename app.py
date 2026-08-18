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
    background-color: #fbfaf8;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}


/* ------------------------------------------------------------
   Typography
   ------------------------------------------------------------ */

h1, h2, h3, h4 {
    color: #292c2f !important;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.1 !important;
    margin-bottom: 0.45rem !important;
}

h2 {
    font-size: 1.6rem !important;
    font-weight: 680 !important;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 650 !important;
}

p {
    color: #303336 !important;
    line-height: 1.68 !important;
}


/* ------------------------------------------------------------
   Section labels
   ------------------------------------------------------------ */

.section-label {
    color: #55595d;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 2.55rem;
    margin-bottom: 0.5rem;
}


/* ------------------------------------------------------------
   Research question
   ------------------------------------------------------------ */

.question-box {
    border: 1px solid #d8d6d2;
    border-left: 3px solid #55595d;
    border-radius: 4px;
    padding: 1.15rem 1.35rem;
    margin: 1rem 0 1.4rem 0;
    background: #f5f4f1;
}

.question-label {
    color: #55595d;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.question-text {
    color: #292c2f;
    font-size: 1.18rem;
    font-weight: 650;
    line-height: 1.5;
}


/* ------------------------------------------------------------
   Model working cards
   ------------------------------------------------------------ */

.model-card {
    border: 1px solid #dedcd8;
    border-radius: 4px;
    background: #ffffff;
    padding: 1.1rem 1.2rem;
    min-height: 125px;
}

.model-number {
    color: #6a6e72;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}

.model-title {
    color: #292c2f;
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.model-text {
    color: #303336;
    font-size: 0.81rem;
    line-height: 1.58;
}


/* ------------------------------------------------------------
   Main prediction tool
   ------------------------------------------------------------ */

.tool-container {
    border: 1px solid #cfc from;
    border: 1px solid #ceccc7;
    border-radius: 5px;
    background: #ffffff;
    padding: 1.45rem 1.5rem 1.2rem 1.5rem;
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}

.tool-title {
    color: #292c2f;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.tool-description {
    color: #303336;
    font-size: 0.84rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}


/* ------------------------------------------------------------
   Result card
   ------------------------------------------------------------ */

.result-card {
    border: 1px solid #d5d3cf;
    border-left: 3px solid #55595d;
    border-radius: 4px;
    background: #f5f4f1;
    padding: 1.3rem 1.4rem;
}

.result-label {
    color: #5f6367;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.result-number {
    color: #292c2f;
    font-size: 2.85rem;
    font-weight: 700;
    margin: 0.25rem 0;
}

.result-text {
    color: #303336;
    font-size: 0.81rem;
    line-height: 1.58;
}


/* ------------------------------------------------------------
   General cards
   ------------------------------------------------------------ */

.info-card {
    border: 1px solid #dedcd8;
    border-radius: 4px;
    background: #ffffff;
    padding: 1.1rem 1.2rem;
    min-height: 125px;
}

.info-title {
    color: #292c2f;
    font-size: 0.91rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.info-text {
    color: #303336;
    font-size: 0.81rem;
    line-height: 1.6;
}


/* ------------------------------------------------------------
   Performance cards
   ------------------------------------------------------------ */

.performance-card {
    border: 1px solid #dedcd8;
    border-radius: 4px;
    background: #ffffff;
    padding: 1.15rem 1.25rem;
}

.performance-title {
    color: #303336;
    font-size: 0.82rem;
    font-weight: 700;
}

.performance-number {
    color: #292c2f;
    font-size: 2.25rem;
    font-weight: 700;
    margin: 0.25rem 0;
}

.performance-text {
    color: #303336;
    font-size: 0.78rem;
    line-height: 1.55;
}


/* ------------------------------------------------------------
   Interpretation cards
   ------------------------------------------------------------ */

.interpretation-card {
    border: 1px solid #dedcd8;
    border-radius: 4px;
    background: #ffffff;
    padding: 1.2rem 1.25rem;
    min-height: 145px;
}

.interpretation-title {
    color: #292c2f;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.interpretation-text {
    color: #303336;
    font-size: 0.81rem;
    line-height: 1.62;
}


/* ------------------------------------------------------------
   Inputs
   ------------------------------------------------------------ */

label {
    color: #292c2f !important;
    font-weight: 600 !important;
}


/* ------------------------------------------------------------
   Button
   ------------------------------------------------------------ */

.stButton > button {
    background-color: #44484c;
    color: #ffffff;
    border: 1px solid #44484c;
    border-radius: 4px;
    font-weight: 650;
    min-height: 2.8rem;
}

.stButton > button:hover {
    background-color: #363a3e;
    border-color: #363a3e;
}


/* ------------------------------------------------------------
   Footer
   ------------------------------------------------------------ */

.footer {
    text-align: center;
    color: #777a7d;
    font-size: 0.72rem;
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
# TITLE
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
    "The project investigates whether measurable socioeconomic "
    "conditions in the preceding year can provide useful predictive "
    "information about subsequent conflict outcomes."
)


# ============================================================
# HOW THE MODEL WORKS
# ============================================================

st.markdown(
    '<div class="section-label">How the Model Works</div>',
    unsafe_allow_html=True
)

st.header("How the model turns data into a prediction")

st.write(
    "The model uses six previous-year socioeconomic indicators and "
    "a Random Forest classifier to identify patterns associated with "
    "historical organized state-based conflict."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-number">01</div>
            <div class="model-title">Previous-year indicators</div>
            <div class="model-text">
                GDP per capita, GDP growth, inflation, population,
                unemployment and military expenditure.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-number">02</div>
            <div class="model-title">Random Forest</div>
            <div class="model-text">
                The trained classifier identifies patterns across
                historical country-year observations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-number">03</div>
            <div class="model-title">Prediction</div>
            <div class="model-text">
                The model produces a probability and a corresponding
                conflict classification for the supplied profile.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN MODEL TOOL
# ============================================================

st.markdown(
    '<div class="section-label">Interactive Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="tool-container">
        <div class="tool-title">
            Try the Model
        </div>
        <div class="tool-description">
            Enter a hypothetical country's previous-year socioeconomic
            conditions to generate a model-based prediction.
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

st.caption(
    "This tool explores hypothetical scenarios. It is not a real-time "
    "assessment of any country's current conflict risk."
)

assess = st.button(
    "Generate Prediction",
    use_container_width=True
)


# ============================================================
# MODEL RESULT
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

    c1, c2 = st.columns([1, 1.5])

    with c1:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Estimated probability
                </div>

                <div class="result-number">
                    {probability * 100:.1f}%
                </div>

                <div class="result-text">
                    Probability assigned by the trained Random Forest
                    to the positive conflict class.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

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

st.header("Data used in the analysis")

st.write(
    "The project combines organized state-based conflict observations "
    "with country-level socioeconomic indicators. The socioeconomic "
    "variables were obtained from World Bank data and integrated with "
    "conflict observations at the country-year level."
)

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">
                Organized state-based conflict data
            </div>
            <div class="info-text">
                Historical conflict observations provide the outcome
                used by the classification model.
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
                World Bank socioeconomic data
            </div>
            <div class="info-text">
                Country-level economic, demographic, labour and
                security indicators provide the model's predictors.
            </div>
        </div>
        """,
        unsafe_allow_html=True
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

    st.markdown(
        """
        <div class="performance-card">

            <div class="performance-title">
                Lagged Model Accuracy
            </div>

            <div class="performance-number">
                92.49%
            </div>

            <div class="performance-text">
                Accuracy under the lagged evaluation framework.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="performance-card">

            <div class="performance-title">
                Country-Held-Out Accuracy
            </div>

            <div class="performance-number">
                83.12%
            </div>

            <div class="performance-text">
                Accuracy when evaluated on countries completely
                excluded from training.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    "The country-held-out evaluation provides an additional test of "
    "generalization because the model is evaluated on countries that "
    "were not used during training."
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
    "Feature importance indicates how strongly the trained Random Forest "
    "uses each predictor when making its decisions. It describes model "
    "behaviour rather than proving that a variable causes conflict."
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
        "Higher relative importance means the model relied more heavily "
        "on that feature when constructing its predictions."
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-label">Interpretation</div>',
    unsafe_allow_html=True
)

st.header("How should the results be understood?")

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="interpretation-card">

            <div class="interpretation-title">
                What the model can tell us
            </div>

            <div class="interpretation-text">
                It can indicate whether a hypothetical socioeconomic
                profile resembles patterns associated with historical
                conflict outcomes in the training data.
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
                What the model cannot tell us
            </div>

            <div class="interpretation-text">
                It cannot establish causality or provide a complete
                explanation of why conflict occurs in a particular
                country or period.
            </div>

        </div>
        """,
        unsafe_allow_html=True
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
    "Organized conflict is shaped by political, historical, geographic, "
    "institutional and strategic factors that are not fully represented "
    "by the six variables used in this model."
)

st.write(
    "The predictions should therefore be viewed as statistical estimates "
    "generated from historical patterns, rather than definitive forecasts "
    "of future conflict."
)


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.markdown(
    '<div class="section-label">Project</div>',
    unsafe_allow_html=True
)

st.header("International Relations × Machine Learning")

st.write(
    "This project demonstrates the application of machine-learning "
    "methods to an International Relations research question through "
    "data integration, temporal feature engineering, classification, "
    "evaluation and interactive deployment."
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">
                Research focus
            </div>
            <div class="info-text">
                Socioeconomic conditions and organized state-based conflict.
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
                Technical approach
            </div>
            <div class="info-text">
                Python, preprocessing, Random Forest classification
                and model evaluation.
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
                An interactive web application for exploring
                hypothetical scenarios.
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
    '<div class="footer">'
    'Academic project · International Relations × Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
