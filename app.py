import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Strategic Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide"
)

GOLD = "#FFD700"
NAVY = "#191970"
SLATE = "#2f4f4f"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "SEGMENTATION.csv")


# ============================================================
# LOAD / GENERATE DATA
# ============================================================

def generate_sample_data(path=CSV_PATH, n=500, seed=42):

    rng = np.random.default_rng(seed)

    genders = ["Male", "Female"]

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

    age = rng.integers(18, 70, n)

    gender = rng.choice(
        genders,
        n
    )

    satisfaction_factor = rng.choice(
        factors,
        n,
        p=[0.25, 0.25, 0.20, 0.15, 0.15]
    )

    loyalty_level = rng.choice(
        loyalty_levels,
        n,
        p=[0.35, 0.30, 0.25, 0.10]
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
        (annual_spend / 300) + rng.normal(0, 2, n),
        0.5,
        None
    ).round(1)

    satisfaction_score = np.clip(
        rng.normal(5.5, 2.0, n)
        + (annual_spend / 2000),
        1,
        10
    ).round(1)

    def assign_group(spend, score):

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

        "Group":
            group
    })

    data.to_csv(
        path,
        index=False
    )

    return data


# Always generate project dataset
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

missing = REQUIRED_COLS - set(df.columns)

if missing:

    st.error(
        f"Your CSV is missing required column(s): {missing}"
    )

    st.stop()


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

df_encoded = df.copy()

categorical_cols = df_encoded.select_dtypes(
    include=["object"]
).columns.tolist()

for col in [
    "Customer_ID",
    "Group"
]:

    if col in categorical_cols:
        categorical_cols.remove(col)


df_encoded = pd.get_dummies(
    df_encoded,
    columns=categorical_cols,
    drop_first=True
)


le = LabelEncoder()

df_encoded["Group_encoded"] = le.fit_transform(
    df_encoded["Group"]
)


X = df_encoded.drop(
    columns=[
        "Customer_ID",
        "Group",
        "Group_encoded"
    ]
)

y = df_encoded["Group_encoded"]


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
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
).transpose().round(3)


cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# LIQUID GLASS UI
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APP
       ======================================================== */

    .stApp {

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(255,215,0,0.10),
                transparent 30%
            ),

            radial-gradient(
                circle at 90% 20%,
                rgba(80,120,255,0.12),
                transparent 30%
            ),

            radial-gradient(
                circle at 50% 90%,
                rgba(0,180,255,0.08),
                transparent 35%
            ),

            linear-gradient(
                135deg,
                #030303 0%,
                #080d1b 45%,
                #030303 100%
            );

        background-attachment: fixed;
    }


    /* ========================================================
       REMOVE STREAMLIT DEFAULT BACKGROUNDS
       ======================================================== */

    [data-testid="stAppViewContainer"] {

        background: transparent !important;
    }

    [data-testid="stHeader"] {

        background: transparent !important;
    }

    [data-testid="stToolbar"] {

        background: transparent !important;
    }

    [data-testid="stSidebar"] {

        background: transparent !important;
    }

    .main {

        background: transparent !important;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .main .block-container {

        padding-top: 2rem;
        padding-bottom: 3rem;

        max-width: 1500px;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .main-title {

        text-align: center;

        color: #ffffff;

        padding: 30px 25px;

        border-radius: 24px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.12),
                rgba(255,255,255,0.035)
            );

        border:
            1px solid rgba(255,255,255,0.20);

        box-shadow:
            0 8px 32px rgba(0,0,0,0.45),
            inset 0 1px 1px rgba(255,255,255,0.25),
            0 0 30px rgba(255,215,0,0.08);

        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);

        letter-spacing: 2px;

        margin-bottom: 25px;
    }


    .main-title::after {

        content: "";

        display: block;

        width: 120px;

        height: 2px;

        margin: 15px auto 0;

        background:
            linear-gradient(
                90deg,
                transparent,
                #FFD700,
                transparent
            );

        box-shadow:
            0 0 12px rgba(255,215,0,0.7);
    }


    /* ========================================================
       SUMMARY
       ======================================================== */

    .summary {

        text-align: center;

        padding: 22px;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.11),
                rgba(255,255,255,0.035)
            );

        border:
            1px solid rgba(255,255,255,0.18);

        box-shadow:
            0 10px 35px rgba(0,0,0,0.35),
            inset 0 1px 1px rgba(255,255,255,0.18);

        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);

        margin: 20px 0;
    }


    .summary h2 {

        color: #FFD700;

        font-size: 1.35rem;

        margin-bottom: 10px;

        text-shadow:
            0 0 15px rgba(255,215,0,0.35);
    }


    .summary p {

        color:
            rgba(255,255,255,0.90);

        font-size: 1.15rem;

        margin: 0;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.10),
                rgba(255,255,255,0.025)
            ) !important;

        border:
            1px solid rgba(255,255,255,0.16);

        border-radius: 20px;

        padding: 20px;

        box-shadow:
            0 8px 30px rgba(0,0,0,0.35),
            inset 0 1px 1px rgba(255,255,255,0.18);

        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
    }


    [data-testid="stMetricLabel"] {

        color:
            rgba(255,255,255,0.65) !important;
    }


    [data-testid="stMetricValue"] {

        color: #ffffff !important;
    }


    /* ========================================================
       SELECT BOX
       ======================================================== */

    div[data-baseweb="select"] > div {

        background:
            rgba(255,255,255,0.07) !important;

        border:
            1px solid rgba(255,255,255,0.18) !important;

        border-radius: 16px !important;

        backdrop-filter: blur(20px);

        -webkit-backdrop-filter: blur(20px);

        box-shadow:
            inset 0 1px 1px rgba(255,255,255,0.15),
            0 5px 20px rgba(0,0,0,0.25);

        color: white !important;
    }


    div[data-baseweb="select"] span {

        color: white !important;
    }


    div[data-baseweb="popover"] {

        background:
            rgba(10,15,28,0.92) !important;

        border:
            1px solid rgba(255,255,255,0.18) !important;

        border-radius: 16px !important;

        backdrop-filter: blur(25px);

        -webkit-backdrop-filter: blur(25px);

        box-shadow:
            0 20px 50px rgba(0,0,0,0.55);
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {

        width: 100%;

        padding: 12px 20px;

        border-radius: 16px;

        border:
            1px solid rgba(255,255,255,0.20);

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.14),
                rgba(255,255,255,0.045)
            );

        color: white;

        font-weight: 600;

        letter-spacing: 0.5px;

        backdrop-filter: blur(20px);

        -webkit-backdrop-filter: blur(20px);

        box-shadow:
            0 8px 25px rgba(0,0,0,0.30),
            inset 0 1px 1px rgba(255,255,255,0.20);

        transition:
            all 0.25s ease;
    }


    .stButton > button:hover {

        border-color:
            rgba(255,215,0,0.75);

        color: #FFD700;

        transform:
            translateY(-2px);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.40),
            0 0 25px rgba(255,215,0,0.15);
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h2,
    h3 {

        color:
            #ffffff !important;

        letter-spacing:
            0.5px;
    }


    h2 {

        border-left:
            3px solid #FFD700;

        padding-left:
            12px;
    }


    /* ========================================================
       PLOTLY GLASS CONTAINER
       ======================================================== */

    div[data-testid="stPlotlyChart"] {

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.075),
                rgba(255,255,255,0.025)
            ) !important;

        border:
            1px solid rgba(255,255,255,0.14);

        border-radius:
            22px;

        padding:
            8px;

        box-shadow:
            0 10px 35px rgba(0,0,0,0.35),
            inset 0 1px 1px rgba(255,255,255,0.12);

        backdrop-filter:
            blur(18px);

        -webkit-backdrop-filter:
            blur(18px);

        margin-bottom:
            18px;

        overflow:
            hidden;
    }


    /* ========================================================
       FORCE PLOTLY IFRAME TRANSPARENCY
       ======================================================== */

    div[data-testid="stPlotlyChart"] iframe {

        background:
            transparent !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {

        border-radius:
            20px;

        overflow:
            hidden;

        border:
            1px solid rgba(255,255,255,0.15);

        box-shadow:
            0 10px 35px rgba(0,0,0,0.35);

        backdrop-filter:
            blur(20px);

        -webkit-backdrop-filter:
            blur(20px);
    }


    /* ========================================================
       PREDICTION CARD
       ======================================================== */

    .prediction-box {

        background:
            linear-gradient(
                135deg,
                rgba(255,215,0,0.12),
                rgba(255,255,255,0.035)
            );

        padding:
            22px;

        border-radius:
            22px;

        border:
            1px solid rgba(255,215,0,0.35);

        margin-top:
            18px;

        box-shadow:
            0 10px 35px rgba(0,0,0,0.40),
            0 0 25px rgba(255,215,0,0.08),
            inset 0 1px 1px rgba(255,255,255,0.18);

        backdrop-filter:
            blur(25px);

        -webkit-backdrop-filter:
            blur(25px);

        text-align:
            center;
    }


    .prediction-box h3 {

        color:
            #FFD700 !important;

        font-size:
            1.35rem;

        text-shadow:
            0 0 15px rgba(255,215,0,0.35);
    }


    .prediction-box p {

        color:
            rgba(255,255,255,0.85);

        font-size:
            1.05rem;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {

        border:
            none;

        height:
            1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(255,215,0,0.5),
                transparent
            );

        margin:
            35px 0;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        text-align:
            center;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.08),
                rgba(255,255,255,0.025)
            );

        color:
            #FFD700;

        padding:
            20px;

        border-radius:
            20px;

        border:
            1px solid rgba(255,255,255,0.15);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.35),
            inset 0 1px 1px rgba(255,255,255,0.15);

        backdrop-filter:
            blur(20px);

        -webkit-backdrop-filter:
            blur(20px);

        margin-top:
            30px;

        letter-spacing:
            1px;
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {

        width:
            8px;
    }


    ::-webkit-scrollbar-track {

        background:
            #030303;
    }


    ::-webkit-scrollbar-thumb {

        background:
            rgba(255,215,0,0.35);

        border-radius:
            10px;
    }


    ::-webkit-scrollbar-thumb:hover {

        background:
            rgba(255,215,0,0.65);
    }


    /* ========================================================
       AMBIENT GLASS LIGHT
       ======================================================== */

    .stApp::before {

        content:
            "";

        position:
            fixed;

        top:
            -200px;

        left:
            -200px;

        width:
            500px;

        height:
            500px;

        background:
            radial-gradient(
                circle,
                rgba(255,215,0,0.07),
                transparent 65%
            );

        pointer-events:
            none;

        z-index:
            0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PLOTLY LIQUID GLASS THEME
# ============================================================

def glass_plot(fig):

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="rgba(255,255,255,0.88)"
        ),

        title=dict(
            font=dict(
                color="#FFFFFF",
                size=18
            )
        ),

        legend=dict(
            font=dict(
                color="rgba(255,255,255,0.80)"
            ),

            bgcolor="rgba(0,0,0,0)"
        ),

        margin=dict(
            l=45,
            r=30,
            t=60,
            b=45
        ),

        xaxis=dict(

            color="rgba(255,255,255,0.75)",

            gridcolor="rgba(255,255,255,0.07)",

            zerolinecolor="rgba(255,255,255,0.10)"
        ),

        yaxis=dict(

            color="rgba(255,255,255,0.75)",

            gridcolor="rgba(255,255,255,0.07)",

            zerolinecolor="rgba(255,255,255,0.10)"
        )
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 class="main-title">
        STRATEGIC CUSTOMER SEGMENTATION ANALYTICS
    </h1>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    f"""
    <div class="summary">

        <h2>
            Customer Movement Prediction Summary
        </h2>

        <p>

            <b>Total Customers:</b>
            {len(df):,}

            &nbsp; | &nbsp;

            <b>Segments:</b>
            {df["Group"].nunique()}

            &nbsp; | &nbsp;

            <b>Model Accuracy:</b>
            {accuracy:.2%}

        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN DASHBOARD — 6 CHARTS
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


# ------------------------------------------------------------
# CHART 1
# ------------------------------------------------------------

fig1 = px.bar(

    factor_data,

    x="Satisfaction_Factor",

    y="Satisfaction_Score",

    color="Satisfaction_Score",

    color_continuous_scale="Cividis"
)

fig1 = glass_plot(fig1)


# ------------------------------------------------------------
# CHART 2
# ------------------------------------------------------------

fig2 = px.pie(

    factor_data,

    names="Satisfaction_Factor",

    values="Satisfaction_Score",

    hole=0.5
)

fig2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

fig2.update_traces(
    textfont=dict(
        color="white"
    )
)


# ------------------------------------------------------------
# CHART 3
# ------------------------------------------------------------

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

fig3 = glass_plot(fig3)


# ------------------------------------------------------------
# CHART 4
# ------------------------------------------------------------

fig4 = px.histogram(

    df,

    x="Satisfaction_Score"
)

fig4.update_traces(
    marker=dict(
        color=GOLD,
        opacity=0.75
    )
)

fig4 = glass_plot(fig4)


# ------------------------------------------------------------
# CHART 5
# ------------------------------------------------------------

fig5 = px.box(

    df,

    y="Satisfaction_Score"
)

fig5.update_traces(
    marker=dict(
        color=GOLD
    )
)

fig5 = glass_plot(fig5)


# ------------------------------------------------------------
# CHART 6
# ------------------------------------------------------------

fig6 = px.scatter(

    df,

    x="Age",

    y="Satisfaction_Score",

    color="Loyalty_Level",

    symbol="Gender"
)

fig6 = glass_plot(fig6)


# ============================================================
# COMBINED DASHBOARD
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


for i, fig in enumerate(
    [
        fig1,
        fig2,
        fig3,
        fig4,
        fig5,
        fig6
    ]
):

    row = (i // 2) + 1
    col = (i % 2) + 1

    for trace in fig.data:

        dashboard.add_trace(
            trace,
            row=row,
            col=col
        )


# ============================================================
# IMPORTANT — TRANSPARENT COMBINED PLOT
# ============================================================

dashboard.update_layout(

    height=1300,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    showlegend=True,

    title=dict(
        text="Integrated Customer Portfolio Analysis",
        font=dict(
            color="white",
            size=22
        )
    ),

    font=dict(
        color="rgba(255,255,255,0.88)"
    ),

    margin=dict(
        l=50,
        r=40,
        t=80,
        b=40
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="rgba(255,255,255,0.80)"
        )
    )
)


# ------------------------------------------------------------
# FORCE ALL SUBPLOTS TRANSPARENT
# ------------------------------------------------------------

dashboard.update_xaxes(

    showgrid=True,

    gridcolor="rgba(255,255,255,0.07)",

    zerolinecolor="rgba(255,255,255,0.10)",

    color="rgba(255,255,255,0.75)"
)


dashboard.update_yaxes(

    showgrid=True,

    gridcolor="rgba(255,255,255,0.07)",

    zerolinecolor="rgba(255,255,255,0.10)",

    color="rgba(255,255,255,0.75)"
)


st.plotly_chart(

    dashboard,

    use_container_width=True
)


# ============================================================
# SEGMENT BREAKDOWN
# ============================================================

st.subheader(
    "Segment Breakdown"
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

fig_seg = glass_plot(fig_seg)


st.plotly_chart(
    fig_seg,
    use_container_width=True
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

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

    title="Top Features Driving Segment Prediction"
)


fig_imp.update_layout(
    yaxis=dict(
        autorange="reversed"
    )
)


fig_imp = glass_plot(
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

    x=list(le.classes_),

    y=list(le.classes_),

    text_auto=True,

    color_continuous_scale="Cividis",

    labels={
        "x": "Predicted",
        "y": "Actual",
        "color": "Count"
    },

    title="Confusion Matrix (Test Set)"
)


fig_cm.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="rgba(255,255,255,0.88)"
    )
)


st.plotly_chart(

    fig_cm,

    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader(
    "Detailed Classification Report"
)


st.dataframe(

    report_df,

    use_container_width=True
)


# ============================================================
# LIVE PREDICTION
# ============================================================

st.subheader(
    "Live Prediction Tool"
)


selected_customer_id = st.selectbox(

    "Customer ID:",

    sorted(
        df["Customer_ID"].unique()
    )
)


if st.button(
    "Predict Segment",
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


    input_df = input_df_original.copy()


    input_encoded = pd.get_dummies(

        input_df,

        columns=categorical_cols,

        drop_first=True
    )


    input_encoded = input_encoded.reindex(

        columns=X.columns,

        fill_value=0
    )


    pred = model.predict(
        input_encoded
    )


    predicted_label = (
        le.inverse_transform(pred)[0]
    )


    proba = model.predict_proba(
        input_encoded
    )[0]


    confidence = proba.max()


    st.markdown(

        f"""
        <div class="prediction-box">

            <h3>
                Predicted Segment for
                {selected_customer_id}:
                {predicted_label}
            </h3>

            <p>
                Confidence:
                {confidence:.1%}
            </p>

        </div>
        """,

        unsafe_allow_html=True
    )


    prob_df = pd.DataFrame({

        "Segment":
            le.classes_,

        "Probability":
            proba

    }).sort_values(

        "Probability",

        ascending=False
    )


    fig_prob = px.bar(

        prob_df,

        x="Segment",

        y="Probability",

        color="Segment",

        title="Prediction Confidence by Segment"
    )


    fig_prob = glass_plot(
        fig_prob
    )


    st.plotly_chart(

        fig_prob,

        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(

    """
    <hr>

    <div class="footer">

        Strategic Customer Insights Dashboard

    </div>
    """,

    unsafe_allow_html=True
)
