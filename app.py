
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# COLORS
# ============================================================

GOLD = "#FFD700"


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
# GENERATE DATA
# ============================================================

def generate_sample_data(
    path=CSV_PATH,
    n=500,
    seed=42
):

    rng = np.random.default_rng(seed)

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
        n,
        p=[
            0.25,
            0.25,
            0.20,
            0.15,
            0.15
        ]
    )

    loyalty_level = rng.choice(
        loyalty_levels,
        n,
        p=[
            0.35,
            0.30,
            0.25,
            0.10
        ]
    )

    loyalty_spend_base = {
        "Bronze": 300,
        "Silver": 900,
        "Gold": 2200,
        "Platinum": 5000
    }

    annual_spend = np.array([

        max(
            20,
            rng.normal(
                loyalty_spend_base[level],
                loyalty_spend_base[level] * 0.30
            )
        )

        for level in loyalty_level

    ]).round(2)

    visit_frequency = np.clip(

        (annual_spend / 300)
        + rng.normal(0, 2, n),

        0.5,

        None

    ).round(1)

    satisfaction_score = np.clip(

        rng.normal(
            5.5,
            2.0,
            n
        )
        + (annual_spend / 2000),

        1,

        10

    ).round(1)

    def assign_group(
        spend,
        score
    ):

        if score < 4:
            return "At Risk"

        elif spend > 1500 and score >= 6:
            return "Settled"

        return "Attention Required"

    group = [

        assign_group(
            spend,
            score
        )

        for spend, score
        in zip(
            annual_spend,
            satisfaction_score
        )
    ]

    data = pd.DataFrame({

        "Customer_ID": [
            f"CUST{i+1:04d}"
            for i in range(n)
        ],

        "Age":
            age,

        "Gender":
            gender,

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

        "Group":
            group
    })

    data.to_csv(
        path,
        index=False
    )

    return data


# ============================================================
# LOAD DATA
# ============================================================

df = generate_sample_data()


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_COLS = {

    "Customer_ID",
    "Group",
    "Satisfaction_Factor",
    "Satisfaction_Score",
    "Age",
    "Loyalty_Level",
    "Gender"
}

missing = (
    REQUIRED_COLS
    - set(df.columns)
)

if missing:

    st.error(
        f"Missing required columns: {missing}"
    )

    st.stop()


# ============================================================
# MACHINE LEARNING
# ============================================================

df_encoded = df.copy()


categorical_cols = (
    df_encoded
    .select_dtypes(
        include=["object"]
    )
    .columns
    .tolist()
)


for col in [
    "Customer_ID",
    "Group"
]:

    if col in categorical_cols:

        categorical_cols.remove(
            col
        )


df_encoded = pd.get_dummies(

    df_encoded,

    columns=categorical_cols,

    drop_first=True
)


label_encoder = LabelEncoder()


df_encoded["Group_encoded"] = (

    label_encoder
    .fit_transform(
        df_encoded["Group"]
    )
)


X = df_encoded.drop(

    columns=[
        "Customer_ID",
        "Group",
        "Group_encoded"
    ]
)


y = df_encoded[
    "Group_encoded"
]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


model = RandomForestClassifier(

    n_estimators=200,

    random_state=42
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


report_df = pd.DataFrame(

    classification_report(

        y_test,

        y_pred,

        target_names=
        label_encoder.classes_,

        output_dict=True,

        zero_division=0

    )

).transpose().round(3)


cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# LIQUID GLASS CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL BACKGROUND
       ====================================================== */

    .stApp {

        background:
            radial-gradient(
                circle at 8% 8%,
                rgba(255,215,0,0.12),
                transparent 28%
            ),

            radial-gradient(
                circle at 92% 15%,
                rgba(80,120,255,0.14),
                transparent 30%
            ),

            radial-gradient(
                circle at 50% 95%,
                rgba(0,180,255,0.08),
                transparent 35%
            ),

            linear-gradient(
                135deg,
                #030303 0%,
                #090d19 48%,
                #030303 100%
            );

        background-attachment:
            fixed;
    }


    /* ======================================================
       STREAMLIT BACKGROUND
       ====================================================== */

    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    .main {
        background: transparent !important;
    }


    .main .block-container {

        max-width:
            1500px;

        padding-top:
            1.5rem;

        padding-bottom:
            3rem;

        padding-left:
            3rem;

        padding-right:
            3rem;
    }


    /* ======================================================
       LIQUID GLASS HEADER
       ====================================================== */

    .header-glass {

        padding:
            32px 20px;

        border-radius:
            28px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.13),
                rgba(255,255,255,0.035)
            );

        border:
            1px solid
            rgba(255,255,255,0.20);

        box-shadow:
            0 15px 45px
            rgba(0,0,0,0.40),

            inset 0 1px 1px
            rgba(255,255,255,0.25),

            0 0 35px
            rgba(255,215,0,0.08);

        backdrop-filter:
            blur(25px);

        -webkit-backdrop-filter:
            blur(25px);

        text-align:
            center;

        margin-bottom:
            28px;
    }


    /* ======================================================
       HEADER TEXT
       ====================================================== */

    .header-text {

        color:
            #ffffff;

        font-size:
            clamp(2rem, 5vw, 4rem);

        font-weight:
            800;

        letter-spacing:
            3px;

        line-height:
            1.05;

        margin:
            0;
    }


    .header-glow {

        width:
            120px;

        height:
            3px;

        margin:
            18px auto 0;

        border-radius:
            50%;

        background:
            linear-gradient(
                90deg,
                transparent,
                #FFD700,
                transparent
            );

        box-shadow:
            0 0 15px
            rgba(255,215,0,0.8);
    }


    /* ======================================================
       SECTION TITLE
       ====================================================== */

    .section-title {

        font-size:
            1.35rem;

        font-weight:
            700;

        color:
            #ffffff;

        margin-top:
            28px;

        margin-bottom:
            14px;

        padding-left:
            14px;

        border-left:
            3px solid #FFD700;
    }


    .section-caption {

        color:
            rgba(255,255,255,0.45);

        font-size:
            0.75rem;

        font-weight:
            700;

        letter-spacing:
            2px;

        margin-bottom:
            14px;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    [data-testid="stMetric"] {

        min-height:
            155px;

        padding:
            22px !important;

        border-radius:
            24px;

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.12),
                rgba(255,255,255,0.035)
            ) !important;

        border:
            1px solid
            rgba(255,255,255,0.16);

        box-shadow:
            0 15px 40px
            rgba(0,0,0,0.35),

            inset 0 1px 1px
            rgba(255,255,255,0.18);

        backdrop-filter:
            blur(25px);

        -webkit-backdrop-filter:
            blur(25px);

        transition:
            all 0.25s ease;
    }


    [data-testid="stMetric"]:hover {

        transform:
            translateY(-5px);

        border-color:
            rgba(255,215,0,0.40);

        box-shadow:
            0 20px 45px
            rgba(0,0,0,0.45),

            0 0 25px
            rgba(255,215,0,0.08);
    }


    [data-testid="stMetricLabel"] {

        color:
            rgba(255,255,255,0.52)
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
            #ffffff
            !important;

        font-size:
            2.2rem
            !important;

        font-weight:
            750
            !important;
    }


    /* ======================================================
       PLOTLY GLASS
       ====================================================== */

    div[data-testid="stPlotlyChart"] {

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.075),
                rgba(255,255,255,0.025)
            ) !important;

        border:
            1px solid
            rgba(255,255,255,0.14);

        border-radius:
            24px;

        padding:
            8px;

        box-shadow:
            0 12px 38px
            rgba(0,0,0,0.35),

            inset 0 1px 1px
            rgba(255,255,255,0.12);

        backdrop-filter:
            blur(20px);

        -webkit-backdrop-filter:
            blur(20px);

        overflow:
            hidden;

        margin-bottom:
            18px;
    }


    div[data-testid="stPlotlyChart"] iframe {

        background:
            transparent !important;
    }


    /* ======================================================
       SELECTBOX
       ====================================================== */

    div[data-baseweb="select"] > div {

        background:
            rgba(255,255,255,0.07)
            !important;

        border:
            1px solid
            rgba(255,255,255,0.18)
            !important;

        border-radius:
            16px
            !important;

        color:
            white
            !important;

        backdrop-filter:
            blur(20px);

        -webkit-backdrop-filter:
            blur(20px);

        box-shadow:
            inset 0 1px 1px
            rgba(255,255,255,0.15),

            0 5px 20px
            rgba(0,0,0,0.25);
    }


    div[data-baseweb="select"] span {

        color:
            #ffffff
            !important;
    }


    div[data-baseweb="popover"] {

        background:
            rgba(10,15,28,0.95)
            !important;

        border:
            1px solid
            rgba(255,255,255,0.18)
            !important;

        border-radius:
            16px
            !important;

        backdrop-filter:
            blur(25px);
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {

        width:
            100%;

        min-height:
            48px;

        border-radius:
            16px;

        border:
            1px solid
            rgba(255,255,255,0.20);

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.14),
                rgba(255,255,255,0.045)
            );

        color:
            #ffffff;

        font-weight:
            700;

        letter-spacing:
            0.5px;

        backdrop-filter:
            blur(20px);

        -webkit-backdrop-filter:
            blur(20px);

        box-shadow:
            0 8px 25px
            rgba(0,0,0,0.30),

            inset 0 1px 1px
            rgba(255,255,255,0.20);
    }


    .stButton > button:hover {

        color:
            #FFD700;

        border-color:
            rgba(255,215,0,0.65);

        transform:
            translateY(-2px);

        box-shadow:
            0 10px 30px
            rgba(0,0,0,0.40),

            0 0 25px
            rgba(255,215,0,0.12);
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {

        border-radius:
            20px;

        overflow:
            hidden;

        border:
            1px solid
            rgba(255,255,255,0.15);

        box-shadow:
            0 10px 35px
            rgba(0,0,0,0.35);
    }


    /* ======================================================
       PREDICTION CARD
       ====================================================== */

    .prediction-card {

        text-align:
            center;

        padding:
            25px;

        border-radius:
            24px;

        background:
            linear-gradient(
                145deg,
                rgba(255,215,0,0.13),
                rgba(255,255,255,0.035)
            );

        border:
            1px solid
            rgba(255,215,0,0.35);

        box-shadow:
            0 15px 40px
            rgba(0,0,0,0.40),

            0 0 30px
            rgba(255,215,0,0.08),

            inset 0 1px 1px
            rgba(255,255,255,0.18);

        backdrop-filter:
            blur(25px);

        -webkit-backdrop-filter:
            blur(25px);

        margin:
            18px 0;
    }


    /* ======================================================
       FOOTER GLASS
       ====================================================== */

    .footer-glass {

        text-align:
            center;

        padding:
            22px 18px;

        margin-top:
            35px;

        border-radius:
            22px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.09),
                rgba(255,255,255,0.025)
            );

        border:
            1px solid
            rgba(255,255,255,0.15);

        box-shadow:
            0 12px 35px
            rgba(0,0,0,0.35),

            inset 0 1px 1px
            rgba(255,255,255,0.15);

        backdrop-filter:
            blur(22px);

        -webkit-backdrop-filter:
            blur(22px);
    }


    .footer-text {

        color:
            rgba(255,215,0,0.85);

        font-size:
            0.78rem;

        font-weight:
            700;

        letter-spacing:
            1.8px;

        margin:
            0;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 768px) {

        .main .block-container {

            padding-left:
                0.8rem;

            padding-right:
                0.8rem;

            padding-top:
                0.8rem;
        }


        .header-glass {

            padding:
                28px 14px;

            border-radius:
                22px;
        }


        .header-text {

            font-size:
                2rem;

            letter-spacing:
                1.5px;
        }


        .section-title {

            font-size:
                1.15rem;
        }


        [data-testid="stMetric"] {

            min-height:
                130px;

            padding:
                17px !important;

            border-radius:
                20px;
        }


        [data-testid="stMetricValue"] {

            font-size:
                1.7rem !important;
        }


        div[data-testid="stPlotlyChart"] {

            border-radius:
                20px;

            padding:
                3px;
        }


        .footer-glass {

            border-radius:
                18px;

            padding:
                18px 10px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# IMPORTANT:
# NO HTML IS USED FOR THE ACTUAL HEADER CONTENT.
# ============================================================

with st.container():

    st.markdown(
        '<div class="header-glass">',
        unsafe_allow_html=True
    )

    st.markdown(
        "CUSTOMER SEGMENTATION ANALYTICS"
    )

    st.markdown(
        '<div class="header-glow"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
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
# PORTFOLIO TITLE
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
# CHART 1 — FACTOR IMPACT
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

    color_continuous_scale="Cividis"
)


# ============================================================
# CHART 2 — SCORE DISTRIBUTION
# ============================================================

fig2 = px.pie(

    factor_data,

    names="Satisfaction_Factor",

    values="Satisfaction_Score",

    hole=0.55
)


fig2.update_traces(

    textfont=dict(
        color="white"
    )
)


# ============================================================
# CHART 3 — AGE TREND
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

    markers=True
)


# ============================================================
# CHART 4 — SCORE FREQUENCY
# ============================================================

fig4 = px.histogram(

    df,

    x="Satisfaction_Score"
)


fig4.update_traces(

    marker=dict(
        color=GOLD,
        opacity=0.78
    )
)


# ============================================================
# CHART 5 — STATISTICAL RANGE
# ============================================================

fig5 = px.box(

    df,

    y="Satisfaction_Score"
)


fig5.update_traces(

    marker=dict(
        color=GOLD
    )
)


# ============================================================
# CHART 6 — DEMOGRAPHIC MAP
# ============================================================

fig6 = px.scatter(

    df,

    x="Age",

    y="Satisfaction_Score",

    color="Loyalty_Level",

    symbol="Gender"
)


# ============================================================
# GLASS CHART FUNCTION
# ============================================================

def make_glass_chart(fig):

    fig.update_layout(

        paper_bgcolor=
            "rgba(0,0,0,0)",

        plot_bgcolor=
            "rgba(0,0,0,0)",

        font=dict(
            color=
                "rgba(255,255,255,0.88)"
        ),

        legend=dict(

            bgcolor=
                "rgba(0,0,0,0)",

            font=dict(
                color=
                    "rgba(255,255,255,0.82)"
            )
        ),

        margin=dict(

            l=45,

            r=30,

            t=55,

            b=45
        )
    )


    fig.update_xaxes(

        color=
            "rgba(255,255,255,0.72)",

        gridcolor=
            "rgba(255,255,255,0.07)",

        zerolinecolor=
            "rgba(255,255,255,0.10)"
    )


    fig.update_yaxes(

        color=
            "rgba(255,255,255,0.72)",

        gridcolor=
            "rgba(255,255,255,0.07)",

        zerolinecolor=
            "rgba(255,255,255,0.10)"
    )


    return fig


fig1 = make_glass_chart(fig1)

fig2 = make_glass_chart(fig2)

fig3 = make_glass_chart(fig3)

fig4 = make_glass_chart(fig4)

fig5 = make_glass_chart(fig5)

fig6 = make_glass_chart(fig6)


# ============================================================
# COMBINE CHARTS
# ============================================================

dashboard = make_subplots(

    rows=3,

    cols=2,

    subplot_titles=[

        "Factor Impact",

        "Score Distribution",

        "Age Trend",

        "Score Frequency",

        "Statistical Range",

        "Demographic Map"

    ],

    specs=[

        [
            {"type": "xy"},
            {"type": "domain"}
        ],

        [
            {"type": "xy"},
            {"type": "xy"}
        ],

        [
            {"type": "xy"},
            {"type": "xy"}
        ]
    ]
)


figures = [

    fig1,
    fig2,
    fig3,
    fig4,
    fig5,
    fig6

]


for i, fig in enumerate(figures):

    row = (
        i // 2
    ) + 1

    col = (
        i % 2
    ) + 1

    for trace in fig.data:

        dashboard.add_trace(

            trace,

            row=row,

            col=col
        )


# ============================================================
# DASHBOARD STYLE
# ============================================================

dashboard.update_layout(

    height=1250,

    paper_bgcolor=
        "rgba(0,0,0,0)",

    plot_bgcolor=
        "rgba(0,0,0,0)",

    title=dict(

        text=
            "Customer Portfolio",

        font=dict(

            color="#FFFFFF",

            size=22
        )
    ),

    font=dict(

        color=
            "rgba(255,255,255,0.88)"
    ),

    legend=dict(

        bgcolor=
            "rgba(0,0,0,0)"
    ),

    margin=dict(

        l=40,

        r=30,

        t=80,

        b=40
    )
)


dashboard.update_xaxes(

    showgrid=True,

    gridcolor=
        "rgba(255,255,255,0.07)",

    zerolinecolor=
        "rgba(255,255,255,0.10)",

    color=
        "rgba(255,255,255,0.72)"
)


dashboard.update_yaxes(

    showgrid=True,

    gridcolor=
        "rgba(255,255,255,0.07)",

    zerolinecolor=
        "rgba(255,255,255,0.10)",

    color=
        "rgba(255,255,255,0.72)"
)


st.plotly_chart(

    dashboard,

    use_container_width=True,

    config={
        "displayModeBar": True,
        "responsive": True
    }
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


seg_counts = (

    df["Group"]

    .value_counts()

    .reset_index()
)


seg_counts.columns = [

    "Group",
    "Count"

]


fig_seg = px.bar(

    seg_counts,

    x="Group",

    y="Count",

    color="Group",

    title="Customers per Segment"
)


fig_seg = make_glass_chart(
    fig_seg
)


st.plotly_chart(

    fig_seg,

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


importances = (

    pd.Series(

        model.feature_importances_,

        index=X.columns

    )

    .sort_values(
        ascending=False
    )

    .head(10)

    .reset_index()
)


importances.columns = [

    "Feature",
    "Importance"

]


fig_imp = px.bar(

    importances,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    color_continuous_scale="Cividis",

    title=
        "Top Features Driving Segment Prediction"
)


fig_imp.update_layout(

    yaxis=dict(
        autorange="reversed"
    )
)


fig_imp = make_glass_chart(
    fig_imp
)


st.plotly_chart(

    fig_imp,

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

    color_continuous_scale=
        "Cividis",

    labels={

        "x":
            "Predicted",

        "y":
            "Actual",

        "color":
            "Count"

    },

    title=
        "Prediction Accuracy Matrix"
)


fig_cm.update_layout(

    paper_bgcolor=
        "rgba(0,0,0,0)",

    plot_bgcolor=
        "rgba(0,0,0,0)",

    font=dict(

        color=
            "rgba(255,255,255,0.88)"
    )
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
        Live Customer Prediction
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


    input_encoded = pd.get_dummies(

        input_df_original,

        columns=
            categorical_cols,

        drop_first=True
    )


    input_encoded = (

        input_encoded

        .reindex(

            columns=X.columns,

            fill_value=0
        )
    )


    prediction = model.predict(

        input_encoded
    )


    predicted_label = (

        label_encoder

        .inverse_transform(
            prediction
        )[0]
    )


    probabilities = (

        model

        .predict_proba(
            input_encoded
        )[0]
    )


    confidence = (
        probabilities.max()
    )


    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    prediction_box = st.container(
        border=True
    )


    with prediction_box:

        st.markdown(
            "### Predicted Customer Segment"
        )

        st.markdown(
            f"## {predicted_label}"
        )

        st.caption(
            f"Confidence: {confidence:.1%}"
        )


    # --------------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------------

    prob_df = pd.DataFrame({

        "Segment":
            label_encoder.classes_,

        "Probability":
            probabilities

    })


    prob_df = prob_df.sort_values(

        "Probability",

        ascending=False
    )


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


    st.plotly_chart(

        fig_prob,

        use_container_width=True
    )


# ============================================================
# FOOTER
# IMPORTANT:
# FOOTER TEXT IS PLAIN STREAMLIT TEXT.
# ============================================================

st.markdown(
    "---"
)


with st.container():

    st.markdown(
        "CUSTOMER INSIGHTS • ANALYTICS DASHBOARD"
    )
```
