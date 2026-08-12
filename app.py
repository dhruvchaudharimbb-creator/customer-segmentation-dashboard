import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "SEGMENTATION.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

else:
    rng = np.random.default_rng(42)

    n = 500

    genders = [
        "Male",
        "Female"
    ]

    factors = [
        "Price",
        "Service",
        "Quality",
        "Delivery",
        "Support"
    ]

    loyalty_levels = [
        "Bronze",
        "Silver",
        "Gold",
        "Platinum"
    ]

    age = rng.integers(
        18,
        70,
        n
    )

    gender = rng.choice(
        genders,
        n
    )

    satisfaction_factor = rng.choice(
        factors,
        n
    )

    loyalty_level = rng.choice(
        loyalty_levels,
        n
    )

    spend_base = {
        "Bronze": 400,
        "Silver": 1000,
        "Gold": 2500,
        "Platinum": 5000
    }

    annual_spend = np.array([
        max(
            50,
            rng.normal(
                spend_base[level],
                spend_base[level] * 0.25
            )
        )
        for level in loyalty_level
    ])

    annual_spend = np.round(
        annual_spend,
        2
    )

    visit_frequency = np.clip(
        annual_spend / 350
        + rng.normal(
            0,
            2,
            n
        ),
        1,
        None
    )

    visit_frequency = np.round(
        visit_frequency,
        1
    )

    satisfaction_score = np.clip(
        rng.normal(
            6,
            1.8,
            n
        )
        + annual_spend / 5000,
        1,
        10
    )

    satisfaction_score = np.round(
        satisfaction_score,
        1
    )

    groups = []

    for spend, score in zip(
        annual_spend,
        satisfaction_score
    ):

        if score < 4:
            groups.append(
                "At Risk"
            )

        elif (
            spend >= 1500
            and score >= 6
        ):
            groups.append(
                "Settled"
            )

        else:
            groups.append(
                "Attention Required"
            )

    df = pd.DataFrame({

        "Customer_ID": [
            f"CUST{i + 1:04d}"
            for i in range(n)
        ],

        "Age": age,

        "Gender": gender,

        "Satisfaction_Factor":
            satisfaction_factor,

        "Satisfaction_Score":
            satisfaction_score,

        "Loyalty_Level":
            loyalty_level,

        "Annual_Spend":
            annual_spend,

        "Visit_Frequency":
            visit_frequency,

        "Group": groups
    })

    df.to_csv(
        CSV_PATH,
        index=False
    )


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "Customer_ID",
    "Age",
    "Gender",
    "Satisfaction_Factor",
    "Satisfaction_Score",
    "Loyalty_Level",
    "Annual_Spend",
    "Visit_Frequency",
    "Group"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# MACHINE LEARNING
# ============================================================

model_df = df.copy()


categorical_columns = [
    "Gender",
    "Satisfaction_Factor",
    "Loyalty_Level"
]


X = pd.get_dummies(
    model_df.drop(
        columns=[
            "Customer_ID",
            "Group"
        ]
    ),
    columns=categorical_columns,
    drop_first=True
)


label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    model_df["Group"]
)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10
)


model.fit(
    X_train,
    y_train
)


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)


report_df = (
    pd.DataFrame(report)
    .transpose()
    .round(3)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# GLASS CSS
# ============================================================

st.markdown(
    """
<style>

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #030406 !important;
}

.stApp {

    background:

        radial-gradient(
            circle at 5% 5%,
            rgba(255,215,0,0.12),
            transparent 28%
        ),

        radial-gradient(
            circle at 95% 10%,
            rgba(70,120,255,0.14),
            transparent 30%
        ),

        radial-gradient(
            circle at 50% 90%,
            rgba(0,180,255,0.08),
            transparent 35%
        ),

        linear-gradient(
            135deg,
            #020304 0%,
            #080c16 50%,
            #020304 100%
        );

    background-attachment: fixed;
}


/* STREAMLIT HEADER */

[data-testid="stHeader"] {
    background: transparent !important;
}


/* HIDE DEFAULT FOOTER */

footer {
    visibility: hidden !important;
}

#MainMenu {
    visibility: hidden !important;
}


/* MAIN AREA */

.main .block-container {

    max-width: 1500px;

    padding-top: 1.5rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}


/* ============================================================
   GLASS CARDS
   ============================================================ */

.glass-card,
.prediction-card,
.footer-card,
[data-testid="stMetric"],
div[data-testid="stPlotlyChart"] {

    position: relative;

    overflow: hidden;

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.14),
            rgba(255,255,255,0.055) 45%,
            rgba(255,255,255,0.018)
        );

    border:
        1px solid
        rgba(255,255,255,0.20);

    box-shadow:

        0 25px 70px
        rgba(0,0,0,0.45),

        inset 0 1px 0
        rgba(255,255,255,0.30),

        inset 0 -1px 0
        rgba(255,255,255,0.05);

    backdrop-filter:
        blur(32px)
        saturate(160%);

    -webkit-backdrop-filter:
        blur(32px)
        saturate(160%);
}


/* TOP GLASS REFLECTION */

.glass-card::before,
.prediction-card::before,
.footer-card::before,
[data-testid="stMetric"]::before,
div[data-testid="stPlotlyChart"]::before {

    content: "";

    position: absolute;

    top: 0;
    left: 4%;

    width: 92%;
    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.75),
            rgba(255,255,255,0.20),
            transparent
        );

    pointer-events: none;
}


/* MAC / IOS DIAGONAL SHINE */

.glass-card::after,
.prediction-card::after,
.footer-card::after,
[data-testid="stMetric"]::after,
div[data-testid="stPlotlyChart"]::after {

    content: "";

    position: absolute;

    top: -100%;
    left: -55%;

    width: 70%;
    height: 250%;

    transform: rotate(8deg);

    background:

        linear-gradient(
            105deg,
            transparent 35%,
            rgba(255,255,255,0.12) 48%,
            rgba(255,255,255,0.025) 53%,
            transparent 65%
        );

    pointer-events: none;

    opacity: 0.7;
}


/* ============================================================
   MAIN TITLE
   ============================================================ */

.glass-card {

    text-align: center;

    padding: 35px 20px;

    margin-bottom: 30px;

    border-radius: 30px;
}


.main-title {

    color: white;

    font-size:
        clamp(
            2rem,
            5vw,
            4rem
        );

    font-weight: 800;

    letter-spacing: 3px;

    text-shadow:
        0 0 30px
        rgba(255,255,255,0.10);
}


.title-line {

    width: 120px;

    height: 2px;

    margin: 16px auto 0;

    background:
        linear-gradient(
            90deg,
            transparent,
            #FFD700,
            transparent
        );
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {

    color: white;

    font-size: 1.35rem;

    font-weight: 700;

    padding: 15px 18px;

    margin-top: 30px;
    margin-bottom: 16px;

    border-radius: 17px;

    background:
        linear-gradient(
            90deg,
            rgba(255,255,255,0.08),
            transparent
        );

    border-left:
        3px solid
        #FFD700;
}


.section-caption {

    color:
        rgba(255,255,255,0.50);

    font-size:
        0.75rem;

    font-weight:
        700;

    letter-spacing:
        2.5px;

    margin:
        20px 0 14px 4px;
}


/* ============================================================
   METRICS
   ============================================================ */

[data-testid="stMetric"] {

    min-height: 150px;

    padding: 25px !important;

    border-radius: 27px;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}


[data-testid="stMetric"]:hover {

    transform:
        translateY(-5px);

    box-shadow:

        0 30px 70px
        rgba(0,0,0,0.55),

        0 0 30px
        rgba(255,215,0,0.08),

        inset 0 1px 0
        rgba(255,255,255,0.40);
}


[data-testid="stMetricLabel"] {

    color:
        rgba(255,255,255,0.55)
        !important;

    font-size:
        0.75rem
        !important;

    font-weight:
        700
        !important;

    letter-spacing:
        1.5px
        !important;
}


[data-testid="stMetricValue"] {

    color:
        white
        !important;

    font-size:
        2.2rem
        !important;

    font-weight:
        800
        !important;
}


/* ============================================================
   PLOTLY
   ============================================================ */

div[data-testid="stPlotlyChart"] {

    border-radius: 27px;

    padding: 8px;
}


/* ============================================================
   SELECT BOX
   ============================================================ */

div[data-baseweb="select"] > div {

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.13),
            rgba(255,255,255,0.045)
        ) !important;

    border:
        1px solid
        rgba(255,255,255,0.20)
        !important;

    border-radius:
        17px
        !important;

    box-shadow:

        inset 0 1px 0
        rgba(255,255,255,0.25),

        0 10px 30px
        rgba(0,0,0,0.30);

    backdrop-filter:
        blur(25px)
        saturate(160%);
}


div[data-baseweb="select"] span {
    color: white !important;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {

    width: 100%;

    min-height: 52px;

    border-radius: 18px;

    border:
        1px solid
        rgba(255,255,255,0.22);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.16),
            rgba(255,255,255,0.045)
        );

    color: white;

    font-weight: 700;

    box-shadow:

        0 15px 40px
        rgba(0,0,0,0.35),

        inset 0 1px 0
        rgba(255,255,255,0.28);

    backdrop-filter:
        blur(25px)
        saturate(160%);

    transition:
        all 0.25s ease;
}


.stButton > button:hover {

    color: #FFD700;

    border-color:
        rgba(255,215,0,0.65);

    transform:
        translateY(-3px);

    box-shadow:

        0 20px 45px
        rgba(0,0,0,0.45),

        0 0 30px
        rgba(255,215,0,0.10),

        inset 0 1px 0
        rgba(255,255,255,0.35);
}


/* ============================================================
   PREDICTION CARD
   ============================================================ */

.prediction-card {

    text-align: center;

    padding: 34px;

    margin-top: 20px;

    margin-bottom: 25px;

    border-radius: 28px;

    background:

        linear-gradient(
            145deg,
            rgba(255,215,0,0.15),
            rgba(255,255,255,0.065) 45%,
            rgba(255,255,255,0.02)
        );

    border:
        1px solid
        rgba(255,215,0,0.38);

    box-shadow:

        0 25px 70px
        rgba(0,0,0,0.48),

        0 0 45px
        rgba(255,215,0,0.08),

        inset 0 1px 0
        rgba(255,255,255,0.30);
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer-card {

    text-align: center;

    margin-top: 45px;

    padding: 22px;

    border-radius: 22px;
}


.footer-text {

    color:
        rgba(255,255,255,0.55);

    font-size:
        0.75rem;

    letter-spacing:
        2px;

    font-weight:
        600;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .main .block-container {

        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-top: 0.8rem;
    }

    .main-title {

        font-size: 1.55rem;

        letter-spacing: 1.5px;
    }

    .glass-card {

        padding: 28px 15px;

        border-radius: 24px;
    }

    [data-testid="stMetric"] {

        min-height: 135px;

        padding: 18px !important;

        border-radius: 23px;
    }

    [data-testid="stMetricValue"] {

        font-size:
            1.75rem !important;
    }

    div[data-testid="stPlotlyChart"] {

        border-radius: 23px;

        padding: 3px;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="glass-card">

        <div class="main-title">
            CUSTOMER SEGMENTATION ANALYTICS
        </div>

        <div class="title-line"></div>

    </div>
    """
)


# ============================================================
# CUSTOMER INSIGHTS
# ============================================================

st.markdown(
    """
    <div class="section-caption">
        CUSTOMER INSIGHTS
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# METRICS
# ============================================================

metric1, metric2, metric3 = st.columns(
    3,
    gap="medium"
)


with metric1:

    st.metric(
        label="👥  TOTAL CUSTOMERS",
        value=f"{len(df):,}"
    )


with metric2:

    st.metric(
        label="◈  CUSTOMER SEGMENTS",
        value=df["Group"].nunique()
    )


with metric3:

    st.metric(
        label="⚡  MODEL ACCURACY",
        value=f"{accuracy:.2%}"
    )


# ============================================================
# CUSTOMER PORTFOLIO
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Customer Portfolio
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FACTOR IMPACT
# ============================================================

factor_data = (
    df.groupby(
        "Satisfaction_Factor"
    )["Satisfaction_Score"]
    .sum()
    .reset_index()
    .sort_values(
        "Satisfaction_Score",
        ascending=False
    )
)


fig1 = px.bar(
    factor_data,
    x="Satisfaction_Factor",
    y="Satisfaction_Score",
    color="Satisfaction_Score",
    color_continuous_scale="Cividis",
    title="Factor Impact"
)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

fig2 = px.pie(
    factor_data,
    names="Satisfaction_Factor",
    values="Satisfaction_Score",
    hole=0.55,
    title="Score Distribution"
)


fig2.update_traces(
    textfont=dict(
        color="white"
    )
)


# ============================================================
# AGE TREND
# ============================================================

age_data = (
    df.groupby(
        "Age"
    )["Satisfaction_Score"]
    .sum()
    .reset_index()
)


fig3 = px.line(
    age_data,
    x="Age",
    y="Satisfaction_Score",
    markers=True,
    title="Age Trend"
)


# ============================================================
# SCORE FREQUENCY
# ============================================================

fig4 = px.histogram(
    df,
    x="Satisfaction_Score",
    title="Score Frequency"
)


fig4.update_traces(
    marker=dict(
        color="#FFD700",
        opacity=0.78
    )
)


# ============================================================
# STATISTICAL RANGE
# ============================================================

fig5 = px.box(
    df,
    y="Satisfaction_Score",
    title="Statistical Range"
)


fig5.update_traces(
    marker=dict(
        color="#FFD700"
    )
)


# ============================================================
# DEMOGRAPHIC ANALYSIS
# ============================================================

fig6 = px.scatter(
    df,
    x="Age",
    y="Satisfaction_Score",
    color="Loyalty_Level",
    symbol="Gender",
    title="Demographic Analysis"
)


# ============================================================
# GLASS CHART STYLE
# ============================================================

def make_glass_chart(fig):

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="rgba(255,255,255,0.88)"
        ),

        margin=dict(
            l=45,
            r=30,
            t=65,
            b=45
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="rgba(255,255,255,0.80)"
            )
        )
    )

    fig.update_xaxes(

        gridcolor=
            "rgba(255,255,255,0.07)",

        zerolinecolor=
            "rgba(255,255,255,0.10)",

        color=
            "rgba(255,255,255,0.72)"
    )

    fig.update_yaxes(

        gridcolor=
            "rgba(255,255,255,0.07)",

        zerolinecolor=
            "rgba(255,255,255,0.10)",

        color=
            "rgba(255,255,255,0.72)"
    )

    return fig


# ============================================================
# DISPLAY CHARTS
# ============================================================

chart1, chart2 = st.columns(
    2,
    gap="medium"
)

with chart1:

    fig1 = make_glass_chart(
        fig1
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


with chart2:

    fig2 = make_glass_chart(
        fig2
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


chart3, chart4 = st.columns(
    2,
    gap="medium"
)

with chart3:

    fig3 = make_glass_chart(
        fig3
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


with chart4:

    fig4 = make_glass_chart(
        fig4
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


chart5, chart6 = st.columns(
    2,
    gap="medium"
)

with chart5:

    fig5 = make_glass_chart(
        fig5
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


with chart6:

    fig6 = make_glass_chart(
        fig6
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )


# ============================================================
# SEGMENT BREAKDOWN
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Segment Breakdown
    </div>
    """,
    unsafe_allow_html=True
)


segment_counts = (
    df["Group"]
    .value_counts()
    .reset_index()
)


segment_counts.columns = [
    "Group",
    "Count"
]


fig_segments = px.bar(
    segment_counts,
    x="Group",
    y="Count",
    color="Group",
    title="Customers per Segment"
)


fig_segments = make_glass_chart(
    fig_segments
)


st.plotly_chart(
    fig_segments,
    use_container_width=True
)


# ============================================================
# PREDICTION INTELLIGENCE
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Prediction Intelligence
    </div>
    """,
    unsafe_allow_html=True
)


feature_importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        model.feature_importances_
})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(10)
)


fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Cividis",
    title="Top Features Driving Segment Prediction"
)


fig_importance.update_layout(
    yaxis=dict(
        autorange="reversed"
    )
)


fig_importance = make_glass_chart(
    fig_importance
)


st.plotly_chart(
    fig_importance,
    use_container_width=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

fig_cm = px.imshow(
    cm,
    x=list(
        label_encoder.classes_
    ),
    y=list(
        label_encoder.classes_
    ),
    text_auto=True,
    color_continuous_scale="Cividis",
    labels={
        "x": "Predicted",
        "y": "Actual",
        "color": "Count"
    },
    title="Prediction Accuracy Matrix"
)


fig_cm = make_glass_chart(
    fig_cm
)


st.plotly_chart(
    fig_cm,
    use_container_width=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Model Performance
    </div>
    """,
    unsafe_allow_html=True
)


st.dataframe(
    report_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# LIVE CUSTOMER PREDICTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Customer Prediction
    </div>
    """,
    unsafe_allow_html=True
)


selected_customer_id = st.selectbox(
    "Select Customer",
    sorted(
        df["Customer_ID"].unique()
    )
)


if st.button(
    "⚡  Predict Customer Segment",
    type="primary"
):

    # --------------------------------------------------------
    # GET SELECTED CUSTOMER
    # --------------------------------------------------------

    input_df_original = (
        df[
            df["Customer_ID"]
            == selected_customer_id
        ]
        .drop(
            columns=[
                "Customer_ID",
                "Group"
            ]
        )
    )


    # --------------------------------------------------------
    # ENCODE CUSTOMER
    # --------------------------------------------------------

    input_encoded = pd.get_dummies(
        input_df_original,
        columns=[
            col
            for col in categorical_columns
            if col in input_df_original.columns
        ],
        drop_first=True
    )


    # --------------------------------------------------------
    # MATCH TRAINING COLUMNS
    # --------------------------------------------------------

    input_encoded = (
        input_encoded
        .reindex(
            columns=X.columns,
            fill_value=0
        )
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_encoded
    )


    predicted_label = (
        label_encoder
        .inverse_transform(
            prediction
        )[0]
    )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        model
        .predict_proba(
            input_encoded
        )[0]
    )


    confidence = probabilities.max()


    # --------------------------------------------------------
    # PREDICTION CARD
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="prediction-card">

            <div style="
                color: rgba(255,255,255,0.52);
                font-size: 0.75rem;
                letter-spacing: 2px;
                font-weight: 700;
                margin-bottom: 12px;
            ">
                PREDICTED CUSTOMER SEGMENT
            </div>

            <div style="
                color: #FFD700;
                font-size: 2.4rem;
                font-weight: 800;
                margin-bottom: 10px;
                text-shadow:
                    0 0 25px
                    rgba(255,215,0,0.25);
            ">
                {predicted_label}
            </div>

            <div style="
                color: rgba(255,255,255,0.72);
                font-size: 1rem;
            ">
                Confidence: {confidence:.1%}
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # PROBABILITY DATAFRAME
    # --------------------------------------------------------

    prob_df = pd.DataFrame({

        "Segment":
            label_encoder.classes_,

        "Probability":
            probabilities
    })


    prob_df = (
        prob_df
        .sort_values(
            "Probability",
            ascending=False
        )
    )


    # --------------------------------------------------------
    # CONFIDENCE CHART
    # --------------------------------------------------------

    fig_prob = px.bar(

        prob_df,

        x="Segment",

        y="Probability",

        color="Segment",

        title=
            "Segment Prediction Confidence"
    )


    fig_prob = make_glass_chart(
        fig_prob
    )


    fig_prob.update_yaxes(
        tickformat=".0%"
    )


    st.plotly_chart(
        fig_prob,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer-card">

        <div class="footer-text">
            CUSTOMER INSIGHTS
            &nbsp; • &nbsp;
            ANALYTICS DASHBOARD
        </div>

    </div>
    """
)
