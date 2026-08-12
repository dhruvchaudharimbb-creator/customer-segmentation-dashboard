import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "SEGMENTATION.csv")


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

    loyalty_spend = {
        "Bronze": 300,
        "Silver": 900,
        "Gold": 2200,
        "Platinum": 5000
    }

    annual_spend = np.array([
        max(
            20,
            rng.normal(
                loyalty_spend[level],
                loyalty_spend[level] * 0.30
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
        rng.normal(5.5, 2.0, n)
        + (annual_spend / 2000),
        1,
        10
    ).round(1)

    def assign_group(spend, score):

        if score < 4:
            return "At Risk"

        if spend > 1500 and score >= 6:
            return "Settled"

        return "Attention Required"

    group = [
        assign_group(spend, score)
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

        "Group": group
    })

    data.to_csv(
        path,
        index=False
    )

    return data


df = generate_sample_data()


# ============================================================
# SIMPLE ML — NO SKLEARN
# ============================================================

gender_map = {
    "Female": 0,
    "Male": 1
}

loyalty_map = {
    "Bronze": 0,
    "Silver": 1,
    "Gold": 2,
    "Platinum": 3
}

factor_map = {
    "Price": 0,
    "Service": 1,
    "Quality": 2,
    "Delivery": 3,
    "Support": 4
}

model_df = df.copy()

model_df["Gender_Code"] = (
    model_df["Gender"]
    .map(gender_map)
    .fillna(0)
)

model_df["Loyalty_Code"] = (
    model_df["Loyalty_Level"]
    .map(loyalty_map)
    .fillna(0)
)

model_df["Factor_Code"] = (
    model_df["Satisfaction_Factor"]
    .map(factor_map)
    .fillna(0)
)

FEATURE_COLUMNS = [
    "Age",
    "Satisfaction_Score",
    "Annual_Spend",
    "Visit_Frequency",
    "Gender_Code",
    "Loyalty_Code",
    "Factor_Code"
]

X = model_df[FEATURE_COLUMNS].astype(float).values
y = model_df["Group"].values

classes = sorted(
    model_df["Group"].unique()
)

feature_min = X.min(axis=0)
feature_max = X.max(axis=0)

feature_range = (
    feature_max - feature_min
)

feature_range[
    feature_range == 0
] = 1

X_normalized = (
    X - feature_min
) / feature_range


# ============================================================
# TRAIN / TEST
# ============================================================

rng = np.random.default_rng(42)

indices = np.arange(len(df))

rng.shuffle(indices)

split = int(
    len(indices) * 0.80
)

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X_normalized[train_indices]
X_test = X_normalized[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]


# ============================================================
# CENTROID CLASSIFIER
# ============================================================

centroids = {}

for cls in classes:

    rows = X_train[
        y_train == cls
    ]

    if len(rows) > 0:
        centroids[cls] = rows.mean(
            axis=0
        )


def predict_one(row):

    distances = {}

    for cls in classes:

        if cls in centroids:

            distances[cls] = np.linalg.norm(
                row - centroids[cls]
            )

    ordered = sorted(
        distances,
        key=distances.get
    )

    prediction = ordered[0]

    values = np.array(
        list(distances.values())
    )

    inverse = 1 / (
        values + 0.0001
    )

    probabilities = (
        inverse / inverse.sum()
    )

    probability_dict = dict(
        zip(
            distances.keys(),
            probabilities
        )
    )

    return prediction, probability_dict


predictions = []

for row in X_test:

    prediction, _ = predict_one(row)

    predictions.append(prediction)

predictions = np.array(predictions)

accuracy = np.mean(
    predictions == y_test
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_rows = []

for cls in classes:

    actual = y_test == cls
    predicted = predictions == cls

    tp = np.sum(
        actual & predicted
    )

    fp = np.sum(
        (~actual) & predicted
    )

    fn = np.sum(
        actual & (~predicted)
    )

    support = np.sum(actual)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    report_rows.append({

        "precision":
            round(precision, 3),

        "recall":
            round(recall, 3),

        "f1-score":
            round(f1, 3),

        "support":
            int(support)
    })


report_rows.append({

    "precision":
        round(accuracy, 3),

    "recall":
        round(accuracy, 3),

    "f1-score":
        round(accuracy, 3),

    "support":
        len(y_test)
})


report_df = pd.DataFrame(
    report_rows,
    index=classes + ["accuracy"]
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = np.zeros(
    (len(classes), len(classes)),
    dtype=int
)

for actual, predicted in zip(
    y_test,
    predictions
):

    a = classes.index(actual)
    p = classes.index(predicted)

    cm[a, p] += 1


# ============================================================
# 🍎 MAC / iOS LIQUID GLASS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   BACKGROUND
   ============================================================ */

.stApp {

    background:

        radial-gradient(
            circle at 5% 5%,
            rgba(255, 204, 80, 0.13),
            transparent 28%
        ),

        radial-gradient(
            circle at 95% 10%,
            rgba(90, 130, 255, 0.18),
            transparent 30%
        ),

        radial-gradient(
            circle at 50% 90%,
            rgba(0, 180, 255, 0.10),
            transparent 35%
        ),

        linear-gradient(
            135deg,
            #020305 0%,
            #080c16 45%,
            #020305 100%
        );

    background-attachment: fixed;
}


/* ============================================================
   STREAMLIT
   ============================================================ */

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

    max-width: 1500px;

    padding-top: 1.5rem;

    padding-bottom: 4rem;

    padding-left: 3rem;

    padding-right: 3rem;
}


/* ============================================================
   APPLE-STYLE LIGHT REFLECTION
   ============================================================ */

.main-title,
[data-testid="stMetric"],
div[data-testid="stPlotlyChart"],
.prediction-card,
.footer-box,
div[data-testid="stDataFrame"] {

    position: relative;

    overflow: hidden;
}


/* TOP EDGE LIGHT */

.main-title::before,
[data-testid="stMetric"]::before,
div[data-testid="stPlotlyChart"]::before,
.prediction-card::before,
.footer-box::before {

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
            rgba(255,255,255,0.70),
            rgba(255,255,255,0.25),
            transparent
        );

    opacity: 0.85;

    pointer-events: none;
}


/* SOFT GLASS GLOW */

.main-title::after,
[data-testid="stMetric"]::after,
div[data-testid="stPlotlyChart"]::after,
.prediction-card::after,
.footer-box::after {

    content: "";

    position: absolute;

    top: -80%;

    left: -40%;

    width: 70%;

    height: 220%;

    background:
        linear-gradient(
            105deg,
            transparent 35%,
            rgba(255,255,255,0.10) 47%,
            rgba(255,255,255,0.025) 52%,
            transparent 65%
        );

    transform:
        rotate(8deg);

    pointer-events: none;

    opacity: 0.55;
}


/* ============================================================
   MAIN TITLE
   ============================================================ */

.main-title {

    text-align: center;

    font-size:
        clamp(2rem, 5vw, 4rem);

    font-weight: 800;

    letter-spacing: 3px;

    color: #ffffff;

    padding: 30px 20px;

    margin-bottom: 30px;

    border-radius: 30px;

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.14),
            rgba(255,255,255,0.055) 40%,
            rgba(255,255,255,0.025)
        );

    border:
        1px solid
        rgba(255,255,255,0.22);

    box-shadow:

        0 25px 70px
        rgba(0,0,0,0.50),

        inset 0 1px 0
        rgba(255,255,255,0.35),

        inset 0 -1px 0
        rgba(255,255,255,0.06),

        0 0 45px
        rgba(120,150,255,0.07);

    backdrop-filter:
        blur(35px)
        saturate(160%);

    -webkit-backdrop-filter:
        blur(35px)
        saturate(160%);
}


/* ============================================================
   SECTION TITLE
   ============================================================ */

.section-title {

    position: relative;

    font-size: 1.45rem;

    font-weight: 700;

    color: #ffffff;

    margin-top: 30px;

    margin-bottom: 15px;

    padding: 14px 18px;

    border-radius: 16px;

    background:
        linear-gradient(
            90deg,
            rgba(255,255,255,0.07),
            transparent
        );

    border-left:
        3px solid #FFD700;

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,0.08);
}

.section-caption {

    color:
        rgba(255,255,255,0.48);

    font-size:
        0.75rem;

    font-weight:
        700;

    letter-spacing:
        2.5px;

    margin:
        22px 0 14px 4px;
}


/* ============================================================
   METRIC GLASS
   ============================================================ */

[data-testid="stMetric"] {

    min-height: 165px;

    padding: 24px !important;

    border-radius: 27px;

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.15),
            rgba(255,255,255,0.065) 45%,
            rgba(255,255,255,0.025)
        ) !important;

    border:
        1px solid
        rgba(255,255,255,0.20);

    box-shadow:

        0 20px 55px
        rgba(0,0,0,0.42),

        inset 0 1px 0
        rgba(255,255,255,0.28),

        inset 0 -1px 0
        rgba(255,255,255,0.05);

    backdrop-filter:
        blur(32px)
        saturate(155%);

    -webkit-backdrop-filter:
        blur(32px)
        saturate(155%);

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease,
        border-color 0.3s ease;
}

[data-testid="stMetric"]:hover {

    transform:
        translateY(-6px);

    border-color:
        rgba(255,255,255,0.34);

    box-shadow:

        0 28px 65px
        rgba(0,0,0,0.52),

        0 0 30px
        rgba(120,150,255,0.08),

        inset 0 1px 0
        rgba(255,255,255,0.38);
}

[data-testid="stMetricLabel"] {

    color:
        rgba(255,255,255,0.55)
        !important;

    font-size:
        0.75rem !important;

    font-weight:
        700 !important;

    letter-spacing:
        1.6px !important;
}

[data-testid="stMetricValue"] {

    color:
        #ffffff !important;

    font-size:
        2.3rem !important;

    font-weight:
        800 !important;

    text-shadow:
        0 2px 15px
        rgba(255,255,255,0.15);
}


/* ============================================================
   PLOTLY GLASS
   ============================================================ */

div[data-testid="stPlotlyChart"] {

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.115),
            rgba(255,255,255,0.045) 45%,
            rgba(255,255,255,0.018)
        ) !important;

    border:
        1px solid
        rgba(255,255,255,0.18);

    border-radius:
        27px;

    padding:
        10px;

    box-shadow:

        0 20px 55px
        rgba(0,0,0,0.40),

        inset 0 1px 0
        rgba(255,255,255,0.25),

        inset 0 -1px 0
        rgba(255,255,255,0.05);

    backdrop-filter:
        blur(30px)
        saturate(150%);

    -webkit-backdrop-filter:
        blur(30px)
        saturate(150%);

    transition:
        transform 0.3s ease,
        border-color 0.3s ease;
}

div[data-testid="stPlotlyChart"]:hover {

    transform:
        translateY(-3px);

    border-color:
        rgba(255,255,255,0.28);
}


/* ============================================================
   SELECT BOX
   ============================================================ */

div[data-baseweb="select"] > div {

    min-height: 50px;

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
        17px !important;

    color:
        white !important;

    box-shadow:

        inset 0 1px 0
        rgba(255,255,255,0.22),

        0 10px 30px
        rgba(0,0,0,0.30);

    backdrop-filter:
        blur(25px)
        saturate(160%);

    -webkit-backdrop-filter:
        blur(25px)
        saturate(160%);
}

div[data-baseweb="select"] span {

    color:
        #ffffff !important;
}

div[data-baseweb="popover"] {

    background:
        rgba(15,18,28,0.92)
        !important;

    border:
        1px solid
        rgba(255,255,255,0.18)
        !important;

    border-radius:
        18px !important;

    backdrop-filter:
        blur(30px);
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
            rgba(255,255,255,0.15),
            rgba(255,255,255,0.045)
        );

    color:
        #ffffff;

    font-weight:
        700;

    letter-spacing:
        0.5px;

    box-shadow:

        0 12px 35px
        rgba(0,0,0,0.35),

        inset 0 1px 0
        rgba(255,255,255,0.28);

    backdrop-filter:
        blur(25px)
        saturate(160%);

    -webkit-backdrop-filter:
        blur(25px)
        saturate(160%);

    transition:
        all 0.25s ease;
}

.stButton > button:hover {

    color:
        #FFD700;

    border-color:
        rgba(255,215,0,0.65);

    transform:
        translateY(-3px);

    box-shadow:

        0 18px 40px
        rgba(0,0,0,0.45),

        0 0 30px
        rgba(255,215,0,0.10),

        inset 0 1px 0
        rgba(255,255,255,0.35);
}


/* ============================================================
   DATAFRAME
   ============================================================ */

div[data-testid="stDataFrame"] {

    border-radius:
        24px;

    overflow:
        hidden;

    border:
        1px solid
        rgba(255,255,255,0.18);

    box-shadow:

        0 20px 55px
        rgba(0,0,0,0.42),

        inset 0 1px 0
        rgba(255,255,255,0.20);

    backdrop-filter:
        blur(25px);
}


/* ============================================================
   PREDICTION CARD
   ============================================================ */

.prediction-card {

    text-align:
        center;

    padding:
        30px;

    border-radius:
        28px;

    background:

        linear-gradient(
            145deg,
            rgba(255,215,0,0.14),
            rgba(255,255,255,0.065) 45%,
            rgba(255,255,255,0.025)
        );

    border:
        1px solid
        rgba(255,215,0,0.38);

    box-shadow:

        0 25px 65px
        rgba(0,0,0,0.48),

        0 0 45px
        rgba(255,215,0,0.09),

        inset 0 1px 0
        rgba(255,255,255,0.30);

    backdrop-filter:
        blur(35px)
        saturate(170%);

    -webkit-backdrop-filter:
        blur(35px)
        saturate(170%);
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer-box {

    text-align:
        center;

    color:
        rgba(255,255,255,0.65);

    font-size:
        0.78rem;

    letter-spacing:
        2px;

    padding:
        23px;

    margin-top:
        40px;

    border-radius:
        22px;

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.10),
            rgba(255,255,255,0.035)
        );

    border:
        1px solid
        rgba(255,255,255,0.15);

    box-shadow:

        0 15px 40px
        rgba(0,0,0,0.35),

        inset 0 1px 0
        rgba(255,255,255,0.22);

    backdrop-filter:
        blur(28px)
        saturate(150%);

    -webkit-backdrop-filter:
        blur(28px)
        saturate(150%);
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .main .block-container {

        padding-left:
            0.8rem;

        padding-right:
            0.8rem;

        padding-top:
            0.8rem;
    }

    .main-title {

        font-size:
            1.55rem;

        letter-spacing:
            1.5px;

        padding:
            25px 14px;

        border-radius:
            24px;
    }

    [data-testid="stMetric"] {

        min-height:
            140px;

        padding:
            18px !important;

        border-radius:
            23px;
    }

    [data-testid="stMetricValue"] {

        font-size:
            1.75rem !important;
    }

    div[data-testid="stPlotlyChart"] {

        border-radius:
            23px;

        padding:
            3px;
    }

    .footer-box {

        font-size:
            0.68rem;

        letter-spacing:
            1.2px;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        CUSTOMER SEGMENTATION ANALYTICS
    </div>
    """,
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


# ============================================================
# METRICS
# ============================================================

metric1, metric2, metric3 = st.columns(
    3,
    gap="medium"
)


with metric1:

    st.metric(
        "👥  TOTAL CUSTOMERS",
        f"{len(df):,}"
    )


with metric2:

    st.metric(
        "◈  CUSTOMER SEGMENTS",
        df["Group"].nunique()
    )


with metric3:

    st.metric(
        "⚡  MODEL ACCURACY",
        f"{accuracy:.2%}"
    )


# ============================================================
# PORTFOLIO
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
# CHART DATA
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


age_data = (
    df.groupby(
        "Age"
    )["Satisfaction_Score"]
    .sum()
    .reset_index()
)


# ============================================================
# CHART 1
# ============================================================

fig1 = px.bar(

    factor_data,

    x="Satisfaction_Factor",

    y="Satisfaction_Score",

    color="Satisfaction_Score",

    color_continuous_scale="Cividis",

    title="Factor Impact"
)


# ============================================================
# CHART 2
# ============================================================

fig2 = px.pie(

    factor_data,

    names="Satisfaction_Factor",

    values="Satisfaction_Score",

    hole=0.58,

    title="Score Distribution"
)


# ============================================================
# CHART 3
# ============================================================

fig3 = px.line(

    age_data,

    x="Age",

    y="Satisfaction_Score",

    markers=True,

    title="Age Trend"
)


# ============================================================
# CHART 4
# ============================================================

fig4 = px.histogram(

    df,

    x="Satisfaction_Score",

    title="Score Frequency"
)


fig4.update_traces(
    marker_color="#FFD700",
    opacity=0.78
)


# ============================================================
# CHART 5
# ============================================================

fig5 = px.box(

    df,

    y="Satisfaction_Score",

    title="Statistical Range"
)


fig5.update_traces(
    marker_color="#FFD700"
)


# ============================================================
# CHART 6
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
# CHART STYLE
# ============================================================

def style_chart(fig):

    fig.update_layout(

        paper_bgcolor=
            "rgba(0,0,0,0)",

        plot_bgcolor=
            "rgba(0,0,0,0)",

        font=dict(
            color=
                "rgba(255,255,255,0.88)"
        ),

        title=dict(
            font=dict(
                size=18,
                color="#FFFFFF"
            )
        ),

        legend=dict(
            bgcolor=
                "rgba(0,0,0,0)",

            font=dict(
                color=
                    "rgba(255,255,255,0.80)"
            )
        ),

        margin=dict(
            l=45,
            r=30,
            t=65,
            b=45
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


figures = [
    style_chart(fig1),
    style_chart(fig2),
    style_chart(fig3),
    style_chart(fig4),
    style_chart(fig5),
    style_chart(fig6)
]


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
        "Demographic Analysis"
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


for i, fig in enumerate(figures):

    row = i // 2 + 1
    col = i % 2 + 1

    for trace in fig.data:

        dashboard.add_trace(
            trace,
            row=row,
            col=col
        )


dashboard.update_layout(

    height=1250,

    paper_bgcolor=
        "rgba(0,0,0,0)",

    plot_bgcolor=
        "rgba(0,0,0,0)",

    font=dict(
        color=
            "rgba(255,255,255,0.88)"
    ),

    margin=dict(
        l=40,
        r=30,
        t=75,
        b=40
    ),

    showlegend=True
)


dashboard.update_xaxes(
    gridcolor=
        "rgba(255,255,255,0.07)",
    color=
        "rgba(255,255,255,0.72)"
)

dashboard.update_yaxes(
    gridcolor=
        "rgba(255,255,255,0.07)",
    color=
        "rgba(255,255,255,0.72)"
)


st.plotly_chart(
    dashboard,
    use_container_width=True,
    config={
        "responsive": True,
        "displayModeBar": True
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


fig_seg = style_chart(
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


feature_importance = []

for i, feature in enumerate(
    FEATURE_COLUMNS
):

    feature_importance.append({

        "Feature":
            feature,

        "Importance":
            np.var(
                X_normalized[:, i]
            )
    })


importance_df = (
    pd.DataFrame(
        feature_importance
    )
    .sort_values(
        "Importance",
        ascending=False
    )
)


fig_imp = px.bar(

    importance_df,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    color_continuous_scale="Cividis",

    title=
        "Features Driving Segment Prediction"
)


fig_imp.update_layout(
    yaxis=dict(
        autorange="reversed"
    )
)


fig_imp = style_chart(
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

    x=classes,

    y=classes,

    text_auto=True,

    color_continuous_scale="Cividis",

    labels={
        "x": "Predicted",
        "y": "Actual",
        "color": "Count"
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
    use_container_width=True
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


selected_customer = st.selectbox(

    "Select Customer",

    sorted(
        df["Customer_ID"].unique()
    )
)


if st.button(
    "⚡  Predict Customer Segment",
    type="primary"
):

    selected_row = model_df[
        model_df["Customer_ID"]
        == selected_customer
    ].iloc[0]


    input_values = (
        selected_row[
            FEATURE_COLUMNS
        ]
        .astype(float)
        .values
    )


    normalized_input = (
        input_values - feature_min
    ) / feature_range


    predicted_segment, probabilities = (
        predict_one(
            normalized_input
        )
    )


    confidence = max(
        probabilities.values()
    )


    st.markdown(
        f"""
        <div class="prediction-card">

            <div style="
                color:rgba(255,255,255,0.50);
                font-size:0.75rem;
                font-weight:700;
                letter-spacing:2px;
                margin-bottom:12px;
            ">
                PREDICTED CUSTOMER SEGMENT
            </div>

            <div style="
                color:#FFD700;
                font-size:2.2rem;
                font-weight:800;
                text-shadow:
                    0 0 25px
                    rgba(255,215,0,0.25);
            ">
                {predicted_segment}
            </div>

            <div style="
                color:rgba(255,255,255,0.72);
                margin-top:10px;
                font-size:1rem;
            ">
                Confidence: {confidence:.1%}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    probability_df = pd.DataFrame({

        "Segment":
            list(probabilities.keys()),

        "Probability":
            list(probabilities.values())
    })


    probability_df = (
        probability_df
        .sort_values(
            "Probability",
            ascending=False
        )
    )


    fig_prob = px.bar(

        probability_df,

        x="Segment",

        y="Probability",

        color="Segment",

        title=
            "Segment Prediction Confidence"
    )


    fig_prob = style_chart(
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
    <div class="footer-box">
        CUSTOMER INSIGHTS • ANALYTICS DASHBOARD
    </div>
    """,
    unsafe_allow_html=True
)
