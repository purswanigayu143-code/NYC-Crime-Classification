# ============================================================
# NYC SHOOTING INCIDENT CLASSIFICATION
# Machine Learning + Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NYC Shooting Incident Classification",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LIGHT PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ============================================================
   MAIN BACKGROUND
   ============================================================ */

.stApp {

    background:
        linear-gradient(
            rgba(248, 250, 252, 0.94),
            rgba(241, 245, 249, 0.96)
        ),
        url("https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?auto=format&fit=crop&w=2400&q=85");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}


/* ============================================================
   MAIN CONTENT
   ============================================================ */

.block-container {

    padding-top: 2rem;
    padding-bottom: 3rem;

    max-width: 1400px;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background: #ffffff;

    border-right: 1px solid #e2e8f0;

    box-shadow:
        3px 0 15px rgba(15, 23, 42, 0.06);
}

section[data-testid="stSidebar"] * {

    color: #1e293b !important;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {

    padding: 35px 40px;

    border-radius: 22px;

    margin-bottom: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.97),
            rgba(254,242,242,0.97)
        );

    border: 1px solid #fecaca;

    box-shadow:
        0 12px 35px rgba(15, 23, 42, 0.08);
}


.hero h1 {

    color: #991b1b;

    font-size: 40px;

    font-weight: 800;

    margin-bottom: 8px;
}


.hero p {

    color: #475569;

    font-size: 16px;

    line-height: 1.6;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {

    color: #0f172a;

    font-size: 26px;

    font-weight: 750;

    margin-top: 30px;

    margin-bottom: 18px;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {

    background: rgba(255,255,255,0.97);

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 24px;

    margin-bottom: 20px;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);
}


.card h3 {

    color: #991b1b;

    margin-top: 0;
}


.card p {

    color: #475569;

    line-height: 1.7;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {

    background: #ffffff;

    border-radius: 16px;

    padding: 22px;

    text-align: center;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 7px 20px rgba(15,23,42,0.06);
}


.metric-title {

    color: #64748b;

    font-size: 13px;

    font-weight: 600;

    letter-spacing: 0.5px;
}


.metric-value {

    color: #991b1b;

    font-size: 25px;

    font-weight: 800;

    margin-top: 7px;
}


/* ============================================================
   PREDICTION BOX
   ============================================================ */

.prediction-box {

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            #fff1f2,
            #ffffff
        );

    border: 1px solid #fecaca;

    box-shadow:
        0 10px 30px rgba(127,29,29,0.08);
}


.prediction-box h1 {

    color: #991b1b;

    font-size: 36px;

    margin-bottom: 8px;
}


.prediction-box p {

    color: #475569;

    font-size: 16px;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {

    width: 100%;

    border-radius: 12px;

    border: none;

    background:
        linear-gradient(
            135deg,
            #dc2626,
            #991b1b
        );

    color: white;

    font-weight: 700;

    font-size: 16px;

    padding: 12px 20px;

    transition: 0.25s;
}


.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 8px 22px rgba(220,38,38,0.25);
}


/* ============================================================
   INPUTS
   ============================================================ */

div[data-baseweb="input"],
div[data-baseweb="select"] {

    border-radius: 10px;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

.stDataFrame {

    border-radius: 12px;

    overflow: hidden;

    border: 1px solid #e2e8f0;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    text-align: center;

    color: #64748b;

    padding: 35px 0 10px 0;

    font-size: 13px;
}


/* ============================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
   ============================================================ */

#MainMenu {

    visibility: hidden;
}

footer {

    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "best_model.pkl"


if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ best_model.pkl not found. "
        "Please keep best_model.pkl in the same folder as app.py."
    )

    st.stop()


try:

    best_model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()


# ============================================================
# MODEL FEATURES
# ============================================================

model_features = [

    "YEAR",
    "PERP_SEX",
    "HOUR",
    "PERP_AGE_GROUP",
    "Longitude",
    "DAY_OF_WEEK",
    "Latitude",
    "VIC_SEX",
    "DAY",
    "PERP_RACE",
    "VIC_AGE_GROUP",
    "MONTH",
    "IS_WEEKEND",
    "VIC_RACE",
    "Lon_Lat",
    "BORO",
    "LOC_OF_OCCUR_DESC",
    "PRECINCT",
    "JURISDICTION_CODE",
    "LOC_CLASSFCTN_DESC",
    "LOCATION_DESC",
    "X_COORD_CD",
    "Y_COORD_CD"

]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div style="
text-align:center;
padding:15px 5px 25px 5px;
">

<h1 style="
color:#dc2626;
font-size:44px;
margin-bottom:0;
">
🚨
</h1>

<h2 style="
color:#0f172a;
font-size:20px;
margin-top:5px;
">
NYC Crime Analytics
</h2>

<p style="
color:#64748b;
font-size:13px;
">
Machine Learning Classification
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(

    "NAVIGATION",

    [

        "🏠 Crime Prediction",

        "📊 Model Information",

        "🔍 Feature Importance",

        "ℹ️ About Project"

    ]

)


st.sidebar.markdown("---")


st.sidebar.markdown("""
### 🎯 Prediction Target

`STATISTICAL_MURDER_FLAG`

**TRUE** → Fatal

**FALSE** → Non-Fatal
""")


st.sidebar.markdown("---")


st.sidebar.markdown("""
### 🧠 Project Domain

**Data Science & Artificial Intelligence**

Historical NYC Shooting Incident Classification
""")


st.sidebar.markdown("---")


st.sidebar.caption(
    "NYC Shooting Incident Classification\n\n"
    "Machine Learning • Streamlit"
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>
🚨 NYC Shooting Incident Classification
</h1>

<p>
<b>Machine Learning Based Fatal & Non-Fatal Incident Classification</b>
</p>

<p>
Analyze historical shooting incident characteristics and generate
a model-based classification using an interactive machine learning
dashboard.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1
# CRIME PREDICTION
# ============================================================

if page == "🏠 Crime Prediction":


    # ========================================================
    # PROJECT METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-title">
        PROJECT TYPE
        </div>

        <div class="metric-value">
        Classification
        </div>

        </div>
        """, unsafe_allow_html=True)


    with c2:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-title">
        TARGET
        </div>

        <div class="metric-value">
        Fatality
        </div>

        </div>
        """, unsafe_allow_html=True)


    with c3:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-title">
        FEATURES
        </div>

        <div class="metric-value">
        23+
        </div>

        </div>
        """, unsafe_allow_html=True)


    with c4:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-title">
        DEPLOYMENT
        </div>

        <div class="metric-value">
        Streamlit
        </div>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # INCIDENT DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">📅 Incident Details</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        year = st.number_input(

            "Year",

            min_value=2000,

            max_value=2030,

            value=2023,

            step=1

        )


    with col2:

        month = st.selectbox(

            "Month",

            list(range(1, 13)),

            index=0

        )


    with col3:

        day = st.number_input(

            "Day",

            min_value=1,

            max_value=31,

            value=15

        )


    with col4:

        hour = st.number_input(

            "Hour",

            min_value=0,

            max_value=23,

            value=20

        )


    # ========================================================
    # DAY OF WEEK
    # ========================================================

    days = [

        "Monday",

        "Tuesday",

        "Wednesday",

        "Thursday",

        "Friday",

        "Saturday",

        "Sunday"

    ]


    day_name = st.selectbox(

        "Day of Week",

        days

    )


    day_mapping = {

        "Monday": 0,

        "Tuesday": 1,

        "Wednesday": 2,

        "Thursday": 3,

        "Friday": 4,

        "Saturday": 5,

        "Sunday": 6

    }


    day_of_week = day_mapping[day_name]


    is_weekend = (

        1

        if day_of_week >= 5

        else 0

    )


    # ========================================================
    # VICTIM INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">👤 Victim Information</div>',
        unsafe_allow_html=True
    )


    v1, v2, v3 = st.columns(3)


    with v1:

        vic_age = st.selectbox(

            "Victim Age Group",

            [

                "<18",

                "18-24",

                "25-44",

                "45-64",

                "65+",

                "UNKNOWN"

            ]

        )


    with v2:

        vic_sex = st.selectbox(

            "Victim Sex",

            [

                "M",

                "F",

                "U"

            ]

        )


    with v3:

        vic_race = st.selectbox(

            "Victim Race",

            [

                "BLACK",

                "WHITE",

                "ASIAN / PACIFIC ISLANDER",

                "WHITE HISPANIC",

                "BLACK HISPANIC",

                "AMERICAN INDIAN/ALASKAN NATIVE",

                "UNKNOWN"

            ]

        )


    # ========================================================
    # PERPETRATOR INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🕵️ Perpetrator Information</div>',
        unsafe_allow_html=True
    )


    p1, p2, p3 = st.columns(3)


    with p1:

        perp_age = st.selectbox(

            "Perpetrator Age Group",

            [

                "<18",

                "18-24",

                "25-44",

                "45-64",

                "65+",

                "UNKNOWN"

            ]

        )


    with p2:

        perp_sex = st.selectbox(

            "Perpetrator Sex",

            [

                "M",

                "F",

                "U"

            ]

        )


    with p3:

        perp_race = st.selectbox(

            "Perpetrator Race",

            [

                "BLACK",

                "WHITE",

                "ASIAN / PACIFIC ISLANDER",

                "WHITE HISPANIC",

                "BLACK HISPANIC",

                "AMERICAN INDIAN/ALASKAN NATIVE",

                "UNKNOWN"

            ]

        )


    # ========================================================
    # NYC LOCATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📍 NYC Location Details</div>',
        unsafe_allow_html=True
    )


    l1, l2 = st.columns(2)


    with l1:

        boro = st.selectbox(

            "Borough",

            [

                "BRONX",

                "BROOKLYN",

                "MANHATTAN",

                "QUEENS",

                "STATEN ISLAND"

            ]

        )


        loc_occ = st.selectbox(

            "Location of Occurrence",

            [

                "INSIDE",

                "OUTSIDE"

            ]

        )


        precinct = st.number_input(

            "Precinct",

            min_value=1,

            max_value=200,

            value=14,

            step=1

        )


        jurisdiction = st.number_input(

            "Jurisdiction Code",

            min_value=0,

            max_value=100,

            value=0,

            step=1

        )


    with l2:

        location_desc = st.selectbox(

            "Location Description",

            [

                "STREET",

                "RESIDENCE - APT. HOUSE",

                "RESIDENCE-HOUSE",

                "COMMERCIAL BUILDING",

                "PARKING LOT/GARAGE",

                "OTHER"

            ]

        )


        loc_class = st.selectbox(

            "Location Classification",

            [

                "STREET",

                "DWELLING",

                "COMMERCIAL",

                "OTHER"

            ]

        )


        latitude = st.number_input(

            "Latitude",

            value=40.7128,

            format="%.6f"

        )


        longitude = st.number_input(

            "Longitude",

            value=-74.0060,

            format="%.6f"

        )


    # ========================================================
    # COORDINATES
    # ========================================================

    st.markdown(
        '<div class="section-title">🗺️ Geographic Coordinates</div>',
        unsafe_allow_html=True
    )


    x1, x2 = st.columns(2)


    with x1:

        x_coord = st.number_input(

            "X Coordinate",

            value=985000.0

        )


    with x2:

        y_coord = st.number_input(

            "Y Coordinate",

            value=200000.0

        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)


    predict_button = st.button(

        "🚨 GENERATE INCIDENT CLASSIFICATION"

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # Lon_Lat
            # ------------------------------------------------

            lon_lat = (

                f"{longitude}_{latitude}"

            )


            # ------------------------------------------------
            # INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame([{

                "YEAR": year,

                "PERP_SEX": perp_sex,

                "HOUR": hour,

                "PERP_AGE_GROUP": perp_age,

                "Longitude": longitude,

                "DAY_OF_WEEK": day_of_week,

                "Latitude": latitude,

                "VIC_SEX": vic_sex,

                "DAY": day,

                "PERP_RACE": perp_race,

                "VIC_AGE_GROUP": vic_age,

                "MONTH": month,

                "IS_WEEKEND": is_weekend,

                "VIC_RACE": vic_race,

                "Lon_Lat": lon_lat,

                "BORO": boro,

                "LOC_OF_OCCUR_DESC": loc_occ,

                "PRECINCT": precinct,

                "JURISDICTION_CODE": jurisdiction,

                "LOC_CLASSFCTN_DESC": loc_class,

                "LOCATION_DESC": location_desc,

                "X_COORD_CD": x_coord,

                "Y_COORD_CD": y_coord

            }])


            # ------------------------------------------------
            # EXACT MODEL FEATURE ORDER
            # ------------------------------------------------

            input_data = input_data[

                model_features

            ]


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = best_model.predict(

                input_data

            )[0]


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probability = None


            if hasattr(

                best_model,

                "predict_proba"

            ):

                probabilities = best_model.predict_proba(

                    input_data

                )[0]


                probability = float(

                    np.max(

                        probabilities

                    )

                )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction in [

                True,

                1,

                "TRUE",

                "True"

            ]:

                result_text = (

                    "FATAL INCIDENT"

                )

            else:

                result_text = (

                    "NON-FATAL INCIDENT"

                )


            # ------------------------------------------------
            # RESULT DISPLAY
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)


            st.markdown(

                f"""
                <div class="prediction-box">

                <h1>
                🚨 {result_text}
                </h1>

                <p>
                Model classification based on the
                provided shooting incident characteristics.
                </p>

                </div>
                """,

                unsafe_allow_html=True

            )


            # ------------------------------------------------
            # RESULT METRICS
            # ------------------------------------------------

            if probability is not None:

                st.markdown("<br>", unsafe_allow_html=True)


                pc1, pc2, pc3 = st.columns(3)


                with pc1:

                    st.markdown(

                        f"""
                        <div class="metric-card">

                        <div class="metric-title">
                        PREDICTION CONFIDENCE
                        </div>

                        <div class="metric-value">
                        {probability * 100:.2f}%
                        </div>

                        </div>
                        """,

                        unsafe_allow_html=True

                    )


                with pc2:

                    st.markdown(

                        f"""
                        <div class="metric-card">

                        <div class="metric-title">
                        BOROUGH
                        </div>

                        <div class="metric-value">
                        {boro}
                        </div>

                        </div>
                        """,

                        unsafe_allow_html=True

                    )


                with pc3:

                    st.markdown(

                        f"""
                        <div class="metric-card">

                        <div class="metric-title">
                        OCCURRENCE
                        </div>

                        <div class="metric-value">
                        {loc_occ}
                        </div>

                        </div>
                        """,

                        unsafe_allow_html=True

                    )


                # ------------------------------------------------
                # PROBABILITY CHART
                # ------------------------------------------------

                st.markdown(

                    '<div class="section-title">'
                    '📊 Prediction Probability'
                    '</div>',

                    unsafe_allow_html=True

                )


                class_names = [

                    str(x)

                    for x in best_model.classes_

                ]


                probability_df = pd.DataFrame({

                    "Class": class_names,

                    "Probability (%)":

                        probabilities * 100

                })


                st.bar_chart(

                    probability_df.set_index(

                        "Class"

                    )

                )


                # ------------------------------------------------
                # INPUT SUMMARY
                # ------------------------------------------------

                st.markdown(

                    '<div class="section-title">'
                    '📋 Prediction Input Summary'
                    '</div>',

                    unsafe_allow_html=True

                )


                st.dataframe(

                    input_data.T.rename(

                        columns={0: "Input Value"}

                    ),

                    use_container_width=True

                )


        except Exception as e:

            st.error(

                f"❌ Prediction Error: {e}"

            )


# ============================================================
# PAGE 2
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":


    st.markdown(

        '<div class="section-title">'
        '📊 Machine Learning Model'
        '</div>',

        unsafe_allow_html=True

    )


    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    try:

        classifier = best_model.named_steps[

            "classifier"

        ]

        model_name = type(

            classifier

        ).__name__


    except Exception:

        classifier = best_model

        model_name = type(

            best_model

        ).__name__


    m1, m2, m3 = st.columns(3)


    with m1:

        st.markdown(

            f"""
            <div class="metric-card">

            <div class="metric-title">
            MODEL
            </div>

            <div class="metric-value">
            {model_name}
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    with m2:

        st.markdown(

            """
            <div class="metric-card">

            <div class="metric-title">
            TARGET
            </div>

            <div class="metric-value">
            STATISTICAL_MURDER_FLAG
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    with m3:

        st.markdown(

            """
            <div class="metric-card">

            <div class="metric-title">
            FEATURES
            </div>

            <div class="metric-value">
            23+
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.markdown("""

    <div class="card">

    <h3>
    🎯 Classification Objective
    </h3>

    <p>

    The objective of this project is to develop a machine learning
    model capable of classifying historical NYC shooting incidents
    as <b>fatal</b> or <b>non-fatal</b> using the target variable
    <b>STATISTICAL_MURDER_FLAG</b>.

    </p>

    </div>

    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.markdown(

        '<div class="section-title">'
        '🔄 Machine Learning Workflow'
        '</div>',

        unsafe_allow_html=True

    )


    st.markdown("""

    <div class="card">

    <p style="
    font-size:18px;
    text-align:center;
    color:#334155;
    ">

    📂 Historical Data

    &nbsp; → &nbsp;

    🧹 Data Cleaning

    &nbsp; → &nbsp;

    ⚙️ Feature Engineering

    &nbsp; → &nbsp;

    🔢 Preprocessing

    &nbsp; → &nbsp;

    🤖 Model Training

    &nbsp; → &nbsp;

    📊 Evaluation

    &nbsp; → &nbsp;

    🚨 Prediction

    </p>

    </div>

    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.markdown(

        '<div class="section-title">'
        '🧩 Model Features'
        '</div>',

        unsafe_allow_html=True

    )


    feature_df = pd.DataFrame({

        "Feature": model_features

    })


    st.dataframe(

        feature_df,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# PAGE 3
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":


    st.markdown(

        '<div class="section-title">'
        '🔍 Feature Importance Analysis'
        '</div>',

        unsafe_allow_html=True

    )


    st.markdown("""

    <div class="card">

    <h3>
    🧠 Understanding Model Decisions
    </h3>

    <p>

    Feature importance helps identify the variables that contribute
    most to the machine learning model's classification decisions.
    It improves model interpretability and supports analytical insights.

    </p>

    </div>

    """, unsafe_allow_html=True)


    try:

        classifier = best_model.named_steps[

            "classifier"

        ]

        preprocessor = best_model.named_steps[

            "preprocessor"

        ]


        feature_names = (

            preprocessor

            .get_feature_names_out()

        )


        # ----------------------------------------------------
        # TREE MODELS
        # ----------------------------------------------------

        if hasattr(

            classifier,

            "feature_importances_"

        ):

            importance_values = (

                classifier.feature_importances_

            )


        # ----------------------------------------------------
        # LINEAR MODELS
        # ----------------------------------------------------

        elif hasattr(

            classifier,

            "coef_"

        ):

            importance_values = np.abs(

                classifier.coef_[0]

            )


        else:

            st.warning(

                "Feature importance is not available "
                "for the selected model."

            )

            importance_values = None


        if importance_values is not None:


            feature_importance = pd.DataFrame({

                "Feature":

                    feature_names,

                "Importance":

                    importance_values

            })


            feature_importance = (

                feature_importance

                .sort_values(

                    "Importance",

                    ascending=False

                )

                .head(15)

            )


            # ------------------------------------------------
            # TOP FEATURES
            # ------------------------------------------------

            st.markdown(

                '<div class="section-title">'
                '🏆 Top 15 Important Features'
                '</div>',

                unsafe_allow_html=True

            )


            chart_data = (

                feature_importance

                .set_index(

                    "Feature"

                )

            )


            st.bar_chart(

                chart_data

            )


            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            st.markdown(

                '<div class="section-title">'
                '📋 Feature Importance Table'
                '</div>',

                unsafe_allow_html=True

            )


            st.dataframe(

                feature_importance,

                use_container_width=True,

                hide_index=True

            )


    except Exception as e:

        st.error(

            f"Unable to calculate feature importance: {e}"

        )


# ============================================================
# PAGE 4
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":


    st.markdown(

        '<div class="section-title">'
        'ℹ️ About the Project'
        '</div>',

        unsafe_allow_html=True

    )


    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.markdown("""

    <div class="card">

    <h3>
    🎯 Project Overview
    </h3>

    <p>

    This project develops a machine learning classification system
    for historical NYC shooting incidents.

    The model predicts whether an incident is classified as
    <b>fatal</b> or <b>non-fatal</b> using
    <b>STATISTICAL_MURDER_FLAG</b> as the target variable.

    </p>

    </div>

    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # PROJECT OBJECTIVES
    # --------------------------------------------------------

    st.markdown("""

    <div class="card">

    <h3>
    🎯 Project Objectives
    </h3>

    <ul style="
    color:#475569;
    line-height:2;
    ">

    <li>
    Clean and preprocess historical shooting incident data
    </li>

    <li>
    Perform exploratory data analysis
    </li>

    <li>
    Engineer meaningful temporal features
    </li>

    <li>
    Train and compare machine learning classification models
    </li>

    <li>
    Identify important factors using feature importance
    </li>

    <li>
    Deploy the trained model using Streamlit
    </li>

    </ul>

    </div>

    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # PROJECT BENEFITS
    # --------------------------------------------------------

    st.markdown("""

    <div class="card">

    <h3>
    📌 Project Benefits
    </h3>

    <ul style="
    color:#475569;
    line-height:2;
    ">

    <li>
    Improved response planning
    </li>

    <li>
    Data-driven resource allocation
    </li>

    <li>
    Identification of factors associated with incident fatality
    </li>

    <li>
    Interactive analytical visualization
    </li>

    <li>
    Machine learning based incident classification
    </li>

    </ul>

    </div>

    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.markdown("""

    <div class="card">

    <h3>
    🛠️ Technologies Used
    </h3>

    <p>

    <b>Python</b> •
    <b>Pandas</b> •
    <b>NumPy</b> •
    <b>Scikit-learn</b> •
    <b>Joblib</b> •
    <b>Matplotlib</b> •
    <b>Streamlit</b>

    </p>

    </div>

    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # DELIVERABLES
    # --------------------------------------------------------

    st.markdown("""

    <div class="card">

    <h3>
    📋 Project Deliverables
    </h3>

    <ul style="
    color:#475569;
    line-height:2;
    ">

    <li>
    Trained and validated machine learning model
    </li>

    <li>
    Exploratory data analysis
    </li>

    <li>
    Feature engineering
    </li>

    <li>
    Model evaluation
    </li>

    <li>
    Feature importance analysis
    </li>

    <li>
    Interactive Streamlit application
    </li>

    <li>
    Final report and presentation
    </li>

    </ul>

    </div>

    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""

<div class="footer">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<br><br>

🚨

<b>
NYC Shooting Incident Classification
</b>

<br>

Machine Learning • Data Science • Streamlit

<br><br>

Developed as a Data Science & Artificial Intelligence Project

<br><br>

<b>
From Historical Data to Machine Learning Insights
</b>

</div>

""", unsafe_allow_html=True)