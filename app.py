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

    /* --------------------------------------------------------
       PAGE
    -------------------------------------------------------- */

    .stApp {
        background-color: #faf9f7;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 3.2rem;
        padding-bottom: 4rem;
    }

    /* --------------------------------------------------------
       TYPOGRAPHY
    -------------------------------------------------------- */

    h1 {
        color: #25282b !important;
        font-size: 3.35rem !important;
        font-weight: 720 !important;
        letter-spacing: -0.05em !important;
        line-height: 1.05 !important;
        margin-bottom: 0.45rem !important;
    }

    h2 {
        color: #292d31 !important;
        font-size: 1.8rem !important;
        font-weight: 680 !important;
        letter-spacing: -0.025em !important;
    }

    h3 {
        color: #303438 !important;
        font-weight: 650 !important;
    }

    p {
        color: #25282b !important;
        line-height: 1.7 !important;
    }

    /* --------------------------------------------------------
       SECTION LABELS
    -------------------------------------------------------- */

    .section-label {
        color: #4a5055;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-top: 2.8rem;
        margin-bottom: 0.5rem;
    }

    /* --------------------------------------------------------
       INTRODUCTION
    -------------------------------------------------------- */

    .intro {
        max-width: 820px;
        color: #303438 !important;
        font-size: 1.1rem;
        line-height: 1.7;
        margin-bottom: 1.3rem;
    }

    /* --------------------------------------------------------
       RESEARCH QUESTION
    -------------------------------------------------------- */

    .research-question {
        background: #f1f0ed;
        border-left: 4px solid #3b4044;
        padding: 1.25rem 1.45rem;
        margin: 1rem 0 1.5rem 0;
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
        font-size: 1.35rem;
        font-weight: 650;
        line-height: 1.45;
    }

    /* --------------------------------------------------------
       SIMPLE CARDS
    -------------------------------------------------------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #dededb;
        border-radius: 5px;
        padding: 1rem 1.1rem;
        min-height: 125px;
    }

    .info-title {
        color: #25282b;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .info-text {
        color: #25282b;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* --------------------------------------------------------
       INPUTS
    -------------------------------------------------------- */

    label {
        color: #25282b !important;
        font-weight: 600 !important;
    }

    [data-baseweb="input"] {
        border-radius: 4px;
    }

    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    .stButton > button {
        background-color: #303438;
        color: white;
        border: 1px solid #303438;
        border-radius: 4px;
        font-weight: 650;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        background-color: #454a4f;
        border-color: #454a4f;
        color: white;
    }

    /* --------------------------------------------------------
       RESULT
    -------------------------------------------------------- */

    .result-box {
        background: #f1f0ed;
        border-left: 4px solid #3b4044;
        padding: 1.3rem 1.4rem;
        border-radius: 4px;
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
        font-size: 3.1rem;
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

    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    [data-testid="stMetricValue"] {
        color: #202326 !important;
        font-weight: 720 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #40464b !important;
        font-weight: 650 !important;
    }

    /* --------------------------------------------------------
       DIVIDERS
    -------------------------------------------------------- */

    hr {
        border: none;
        border-top: 1px solid #d8d7d3;
        margin: 2.4rem 0;
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer-text {
        color: #777b7f !important;
        font-size: 0.73rem;
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
    <div class="intro">
    An interactive research project examining whether socioeconomic
    conditions from the previous year can provide useful information
    about organized state-based conflict.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RESEARCH QUESTION
# ============================================================

st.markdown(
    '<p class="section-label">Research Question</p>',
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
    The project examines whether measurable socioeconomic conditions
    observed before a conflict contain predictive information about
    conflict outcomes in the following period.
    """
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<p class="section-label">The Approach</p>',
    unsafe_allow_html=True
)

st.header("How does it work?")

st.write(
    """
    The model uses six conditions from the previous year and looks
    for patterns between those conditions and historical conflict
    observations.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">The inputs</div>
            <div class="info-text">
                Six socioeconomic and security indicators from
                the previous year.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">The analysis</div>
            <div class="info-text">
                A Random Forest model examines patterns learned
                from historical observations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">The result</div>
            <div class="info-text">
                The model produces an estimated probability for
                the conflict outcome it was trained to identify.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL INPUT
# ============================================================

st.markdown(
    '<p class="section-label">Explore the Model</p>',
    unsafe_allow_html=True
)

st.header("Enter previous-year conditions")

st.write(
    """
    Enter the socioeconomic conditions for a hypothetical or real
    country. You can experiment with different profiles and see
    how the model responds.
    """
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


st.markdown("#### Security conditions")

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

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.markdown(
        '<p class="section-label">Your Result</p>',
        unsafe_allow_html=True
    )

    st.header("Estimated conflict probability")

    r1, r2 = st.columns([1, 1.7])

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
                    Model-generated estimate based on the conditions
                    entered above.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:

        if prediction == 1:

            st.subheader("The model identifies a higher-risk pattern")

            st.write(
                """
                Based on the information provided, the model assigns
                this profile to the conflict class it was trained to
                identify.
                """
            )

        else:

            st.subheader("The model identifies a lower-risk pattern")

            st.write(
                """
                Based on the information provided, the model assigns
                this profile to the non-conflict class it was trained
                to identify.
                """
            )

    st.caption(
        "This is a statistical estimate based on historical data. "
        "It is not a forecast of what will necessarily happen in reality."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<p class="section-label">Model Performance</p>',
    unsafe_allow_html=True
)

st.header("How well does the model perform?")

st.write(
    """
    The model was evaluated using two approaches. The second test
    is particularly useful because it evaluates the model on countries
    that were completely excluded from training.
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
# MODEL INTERPRETATION
# ============================================================

st.markdown(
    '<p class="section-label">Understanding the Model</p>',
    unsafe_allow_html=True
)

st.header("Which factors mattered most?")

st.write(
    """
    The chart shows the relative importance of the six indicators
    within the Random Forest model.
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
        "Feature importance shows predictive contribution within "
        "the model. It does not establish that a factor causes conflict."
    )


# ============================================================
# DATA AND SOURCES
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
    indicators include data from the World Bank.
    """
)

source_df = pd.DataFrame({
    "Indicator": [
        "Organized state-based conflict",
        "GDP per capita",
        "GDP growth",
        "Inflation",
        "Population",
        "Unemployment",
        "Military expenditure"
    ],
    "Role": [
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
# ABOUT THE PROJECT
# ============================================================

st.markdown(
    '<p class="section-label">About the Project</p>',
    unsafe_allow_html=True
)

st.header("International Relations meets Machine Learning")

st.write(
    """
    This project explores how computational methods can be applied
    to an International Relations question. Historical conflict
    observations were combined with socioeconomic indicators, with
    previous-year conditions used as inputs to a Random Forest
    classification model.
    """
)

st.write(
    """
    The purpose is not to reduce conflict to economic conditions.
    Instead, the project asks whether measurable socioeconomic
    circumstances contain information that can contribute to
    identifying patterns associated with conflict.
    """
)


# ============================================================
# LIMITATIONS
# ============================================================

st.markdown(
    '<p class="section-label">Important Context</p>',
    unsafe_allow_html=True
)

st.header("What does the result mean?")

st.write(
    """
    Conflict is influenced by many factors, including political
    institutions, history, geography, leadership, identity,
    external intervention and strategic decisions. These factors
    are not fully represented by the six variables used here.
    """
)

st.write(
    """
    The model should therefore be understood as an exploration
    of predictive patterns in historical data — not as a definitive
    conflict forecasting system.
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
