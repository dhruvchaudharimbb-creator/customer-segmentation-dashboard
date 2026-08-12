import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots


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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "SEGMENTATION.csv"
)


# ============================================================
# SAMPLE DATA GENERATOR
# ============================================================

def generate_sample_data(path=CSV_PATH, n=500, seed=42):

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

        "Group": group
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
# VALIDATE DATA
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
        f"Missing required columns: {missing}"
    )

    st.stop()


# ============================================================
# SIMPLE MACHINE LEARNING
# No sklearn required
# ============================================================

FEATURE_COLUMNS = [
    "Age",
    "Satisfaction_Score",
    "Annual_Spend",
    "Visit_Frequency"
]


# Convert categorical columns into numerical values
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


X_all = model_df[FEATURE_COLUMNS].astype(float).values
y_all = model_df["Group"].values

classes = sorted(
    model_df["Group"].unique()
)


# ============================================================
# NORMALIZATION
# ============================================================

feature_min = X_all.min(axis=0)

feature_max = X_all.max(axis=0)

feature_range = feature_max - feature_min

feature_range[feature_range == 0] = 1

X_normalized = (
    X_all - feature_min
) / feature_range


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

rng = np.random.default_rng(42)

indices = np.arange(len(df))

rng.shuffle(indices)

split_point = int(
    len(indices) * 0.80
)

train_indices = indices[:split_point]

test_indices = indices[split_point:]


X_train = X_normalized[train_indices]

X_test = X_normalized[test_indices]

y_train = y_all[train_indices]

y_test = y_all[test_indices]


# ============================================================
# CENTROID CLASSIFIER
# ============================================================

centroids = {}

for cls in classes:

    class_rows = X_train[
        y_train == cls
    ]

    if len(class_rows) > 0:

        centroids[cls] = class_rows.mean(
            axis=0
        )


def predict_one(row):

    distances = {}

    for cls in classes:

        if cls in centroids:

            distance = np.linalg.norm(
                row - centroids[cls]
            )

            distances[cls] = distance

    sorted_classes = sorted(
        distances,
        key=distances.get
    )

    predicted = sorted_classes[0]

    # Convert distance to confidence
    dists = np.array(
        list(distances.values())
    )

    if len(dists) > 1:

        inverse = 1 / (
            dists + 0.0001
        )

        probabilities = (
            inverse / inverse.sum()
        )

    else:

        probabilities = np.array([1.0])

    probability_dict = dict(
        zip(
            distances.keys(),
            probabilities
        )
    )

    return predicted, probability_dict


# ============================================================
# TEST MODEL
# ============================================================

predictions = []

probability_list = []

for row in X_test:

    prediction, probs = predict_one(row)

    predictions.append(prediction)

    probability_list.append(probs)


predictions = np.array(predictions)


# ============================================================
# ACCURACY
# ============================================================

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
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    report_rows.append({
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1-score": round(f1, 3),
        "support": int(support)
    })


# Accuracy row
report_rows.append({
    "precision": round(accuracy, 3),
    "recall": round(accuracy, 3),
    "f1-score": round(accuracy, 3),
    "support": len(y_test)
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

    actual_index = classes.index(actual)

    predicted_index = classes.index(predicted)

    cm[
        actual_index,
        predicted_index
    ] += 1


# ============================================================
# LIQUID GLASS CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL APP
   ============================================================ */

.stApp {

    background:

        radial-gradient(
            circle at 8% 8%,
            rgba(255,215,0,0.10),
            transparent 28%
        ),

        radial-gradient(
            circle at 92% 15%,
            rgba(80,120,255,0.12),
            transparent 30%
        ),

        radial-gradient(
            circle at 50% 95%,
            rgba(0,180,255,0.07),
            transparent 35%
        ),

        linear-gradient(
            135deg,
            #030303 0%,
            #090d19 48%,
            #030303 100%
        );

    background-attachment: fixed;
}


/* ============================================================
   STREAMLIT BACKGROUND
   ============================================================ */

[data-testid="stAppViewContainer"] {

    background:
        transparent !important;
}

[data-testid="stHeader"] {

    background:
        transparent !important;
}

[data-testid="stToolbar"] {

    background:
        transparent !important;
}

.main {

    background:
        transparent !important;
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


/* ============================================================
   MAIN TITLE
   ============================================================ */

.main-title {

    text-align:
        center;

    font-size:
        clamp(2rem, 5vw, 4rem);

    font-weight:
        800;

    letter-spacing:
        3px;

    color:
        #ffffff;

    padding:
        28px 20px;

    margin-bottom:
        28px;

    border-radius:
        28px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.12),
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
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {

    font-size:
        1.45rem;

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


/* ============================================================
   METRIC CARDS
   ============================================================ */

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
        rgba(255,255,255,0.52) !important;

    font-size:
        0.75rem !important;

    font-weight:
        700 !important;

    letter-spacing:
        1.5px !important;
}

[data-testid="stMetricValue"] {

    color:
        #ffffff !important;

    font-size:
        2.2rem !important;

    font-weight:
        750 !important;
}


/* ============================================================
   PLOTLY GLASS
   ============================================================ */

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


/* ============================================================
   SELECT BOX
   ============================================================ */

div[data-baseweb="select"] > div {

    background:
        rgba(255,255,255,0.07)
        !important;

    border:
        1px solid
        rgba(255,255,255,0.18)
        !important;

    border-radius:
        16px !important;

    color:
        white !important;

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
        #ffffff !important;
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
        16px !important;

    backdrop-filter:
        blur(25px);
}


/* ============================================================
   BUTTON
   ============================================================ */

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

    transition:
        all 0.25s ease;
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


/* ============================================================
   DATAFRAME
   ============================================================ */

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


/* ============================================================
   PREDICTION CARD
   ============================================================ */

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
        22px;

    margin-top:
        35px;

    margin-bottom:
        10px;

    border-radius:
        20px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid
        rgba(255,255,255,0.12);

    box-shadow:
        inset 0 1px 1px
        rgba(255,255,255,0.12);

    backdrop-filter:
        blur(20px);
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
            1.65rem;

        letter-spacing:
            1.5px;

        padding:
            25px 14px;

        border-radius:
            22px;
    }

    .section-title {

        font-size:
            1.2rem;
    }

    [data-testid="stMetric"] {

        min-height:
            135px;

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

    .footer-box {

        font-size:
            0.68rem;

        letter-spacing:
            1.2px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CLEAN HEADER
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
# PORTFOLIO ANALYSIS
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
# CHART 1
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
# CHART 2
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
# CHART 3
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
# CHART 4
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
# CHART 5
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
# CHART 6
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


# Feature importance based on variance
feature_importance = []

for i, feature in enumerate(FEATURE_COLUMNS):

    importance = np.var(
        X_normalized[:, i]
    )

    feature_importance.append({
        "Feature": feature,
        "Importance": importance
    })


importances = (
    pd.DataFrame(feature_importance)
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(10)
)


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

    x=classes,

    y=classes,

    text_auto=True,

    color_continuous_scale="Cividis",

    labels={
        "x": "Predicted",
        "y": "Actual",
        "color": "Count"
    },

    title="Prediction Accuracy Matrix"
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
    hide_index=False
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

    selected_row = model_df[
        model_df["Customer_ID"]
        == selected_customer_id
    ].iloc[0]


    input_values = (
        selected_row[
            FEATURE_COLUMNS
        ]
        .astype(float)
        .values
    )


    input_normalized = (
        input_values - feature_min
    ) / feature_range


    predicted_label, probabilities = (
        predict_one(
            input_normalized
        )
    )


    confidence = max(
        probabilities.values()
    )


    # ========================================================
    # PREDICTION CARD
    # ========================================================

    st.markdown(
        f"""
        <div class="prediction-card">

            <div style="
                color: rgba(255,255,255,0.50);
                font-size: 0.75rem;
                letter-spacing: 2px;
                font-weight: 700;
                margin-bottom: 10px;
            ">
                PREDICTED CUSTOMER SEGMENT
            </div>

            <div style="
                color: #FFD700;
                font-size: 2rem;
                font-weight: 800;
                margin-bottom: 8px;
            ">
                {predicted_label}
            </div>

            <div style="
                color: rgba(255,255,255,0.70);
                font-size: 1rem;
            ">
                Confidence: {confidence:.1%}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    prob_df = pd.DataFrame({

        "Segment":
            list(probabilities.keys()),

        "Probability":
            list(probabilities.values())
    })


    prob_df = (
        prob_df
        .sort_values(
            "Probability",
            ascending=False
        )
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
# CLEAN FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-box">
        CUSTOMER INSIGHTS • ANALYTICS DASHBOARD
    </div>
    """,
    unsafe_allow_html=True
)
