import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- VS Code fix: Colab sets this automatically, VS Code needs it explicit ---
pio.renderers.default = "notebook"

GOLD, NAVY, SLATE = "#FFD700", "#191970", "#2f4f4f"
CSV_PATH = os.path.expanduser("~/SEGMENTATION.csv")

# ------------------------------------------------------------
# STEP 1: LOAD DATA (auto-generates a realistic sample dataset
# if you haven't uploaded your own SEGMENTATION.csv yet)
# ------------------------------------------------------------
def generate_sample_data(path=CSV_PATH, n=500, seed=42):
    rng = np.random.default_rng(seed)
    genders = ["Male", "Female"]
    factors = ["Price", "Service", "Quality", "Delivery", "Support"]
    loyalty_levels = ["Bronze", "Silver", "Gold", "Platinum"]

    age = rng.integers(18, 70, n)
    gender = rng.choice(genders, n)
    satisfaction_factor = rng.choice(factors, n, p=[0.25, 0.25, 0.2, 0.15, 0.15])
    loyalty_level = rng.choice(loyalty_levels, n, p=[0.35, 0.3, 0.25, 0.10])

    loyalty_spend_base = {"Bronze": 300, "Silver": 900, "Gold": 2200, "Platinum": 5000}
    annual_spend = np.array([
        max(20, rng.normal(loyalty_spend_base[lvl], loyalty_spend_base[lvl] * 0.3))
        for lvl in loyalty_level
    ]).round(2)

    visit_frequency = np.clip((annual_spend / 300) + rng.normal(0, 2, n), 0.5, None).round(1)
    satisfaction_score = np.clip(rng.normal(5.5, 2.0, n) + (annual_spend / 2000), 1, 10).round(1)

    def assign_group(spend, score):
        if score < 4:
            return "At Risk"
        elif spend > 1500 and score >= 6:
            return "Settled"
        return "Attention Required"

    group = [assign_group(s, sc) for s, sc in zip(annual_spend, satisfaction_score)]

    data = pd.DataFrame({
        "Customer_ID": [f"CUST{i+1:04d}" for i in range(n)],
        "Age": age, "Gender": gender,
        "Satisfaction_Factor": satisfaction_factor,
        "Satisfaction_Score": satisfaction_score,
        "Loyalty_Level": loyalty_level,
        "Annual_Spend": annual_spend,
        "Visit_Frequency": visit_frequency,
        "Group": group,
    })
    data.to_csv(path, index=False)
    return data


# Always regenerate data to ensure new segmentation logic is applied
print(f"Generating sample data with updated segmentation logic for '{CSV_PATH}'.")
print("To use YOUR real data instead: place your own SEGMENTATION.csv in this")
print("same folder as this notebook, then re-run this cell.")
df = generate_sample_data()

REQUIRED_COLS = {"Customer_ID", "Group", "Satisfaction_Factor", "Satisfaction_Score", "Age", "Loyalty_Level", "Gender"}
missing = REQUIRED_COLS - set(df.columns)
assert not missing, f"Your CSV is missing required column(s): {missing}"

# ------------------------------------------------------------
# STEP 2: TRAIN THE CUSTOMER SEGMENT PREDICTION MODEL
# ------------------------------------------------------------
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include=["object"]).columns.tolist()
for col in ("Customer_ID", "Group"):
    if col in categorical_cols:
        categorical_cols.remove(col)

df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
le = LabelEncoder()
df_encoded["Group_encoded"] = le.fit_transform(df_encoded["Group"])

X = df_encoded.drop(columns=["Customer_ID", "Group", "Group_encoded"])
y = df_encoded["Group_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report_df = pd.DataFrame(
    classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
).transpose().round(3)
cm = confusion_matrix(y_test, y_pred)

# ------------------------------------------------------------
# STEP 3: HEADER + PREDICTION SUMMARY BANNER
# ------------------------------------------------------------
header_html = f"""
<h1 style='text-align:center; color:{GOLD}; background:{NAVY}; padding:25px;
border-radius:10px; border:2px solid {GOLD};'>
STRATEGIC CUSTOMER SEGMENTATION ANALYTICS
</h1>"""

summary_html = f"""
<div style='text-align:center; background:{SLATE}; padding:15px; border-radius:8px;
border:1px solid {GOLD}; margin-top:20px; margin-bottom:20px;'>
<h2 style='color:{GOLD};'>Customer Movement Prediction Summary</h2>
<p style='color:white; font-size:1.2em;'>
<b>Total Customers:</b> {len(df):,} &nbsp;|&nbsp;
<b>Segments:</b> {df['Group'].nunique()} &nbsp;|&nbsp;
<b>Model Accuracy:</b> {accuracy:.2%}
</p></div>"""

display(HTML(header_html))
display(HTML(summary_html))

# ------------------------------------------------------------
# STEP 4: MAIN DASHBOARD — 6-CHART GRID
# ------------------------------------------------------------
factor_data = df.groupby("Satisfaction_Factor")["Satisfaction_Score"].sum().reset_index().sort_values("Satisfaction_Score", ascending=False)
fig1 = px.bar(factor_data, x="Satisfaction_Factor", y="Satisfaction_Score", color="Satisfaction_Score", color_continuous_scale="Cividis")
fig2 = px.pie(factor_data, names="Satisfaction_Factor", values="Satisfaction_Score", hole=0.5)
age_data = df.groupby("Age")["Satisfaction_Score"].sum().reset_index()
fig3 = px.line(age_data, x="Age", y="Satisfaction_Score", markers=True)
fig4 = px.histogram(df, x="Satisfaction_Score", color_discrete_sequence=[NAVY])
fig5 = px.box(df, y="Satisfaction_Score", color_discrete_sequence=[GOLD])
fig6 = px.scatter(df, x="Age", y="Satisfaction_Score", color="Loyalty_Level", symbol="Gender")

dashboard = make_subplots(
    rows=3, cols=2,
    subplot_titles=["Factor Impact", "Score Distribution", "Age Trend", "Score Frequency", "Statistical Range", "Demographic Map"],
    specs=[[{"type": "xy"}, {"type": "domain"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
)
for i, f in enumerate([fig1, fig2, fig3, fig4, fig5, fig6]):
    r, c = (i // 2) + 1, (i % 2) + 1
    for trace in f.data:
        dashboard.add_trace(trace, row=r, col=c)
dashboard.update_layout(height=1300, template="plotly_dark", showlegend=True, title_text="Integrated Customer Portfolio Analysis")
dashboard.show()

# ------------------------------------------------------------
# STEP 5: SEGMENT BREAKDOWN + MODEL PERFORMANCE CHARTS
# ------------------------------------------------------------
seg_counts = df["Group"].value_counts().reset_index()
seg_counts.columns = ["Group", "Count"]
px.bar(seg_counts, x="Group", y="Count", color="Group", title="Customers per Segment", template="plotly_dark").show()

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).reset_index()
importances.columns = ["Feature", "Importance"]
px.bar(importances, x="Importance", y="Feature", orientation="h", color="Importance",
       color_continuous_scale="Cividis", title="Top Features Driving Segment Prediction",
       template="plotly_dark").update_layout(yaxis=dict(autorange="reversed")).show()

px.imshow(cm, x=list(le.classes_), y=list(le.classes_), text_auto=True, color_continuous_scale="Cividis",
          labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion Matrix (Test Set)",
          template="plotly_dark").show()

print("\nDetailed Classification Report:")
display(report_df)

# ------------------------------------------------------------
# STEP 6: LIVE PREDICTION TOOL — pick a new customer's details,
# click Predict, see the segment prediction instantly
# ------------------------------------------------------------
customer_id_w = widgets.Dropdown(
    options=sorted(df['Customer_ID'].unique()),
    description='Customer ID:',
    value=df['Customer_ID'].iloc[0]
)

predict_btn = widgets.Button(description="Predict Segment", button_style="warning")
output_box = widgets.Output()


def on_predict_click(b):
    with output_box:
        clear_output()
        selected_customer_id = customer_id_w.value

        input_df_original = df[df['Customer_ID'] == selected_customer_id].drop(columns=['Customer_ID', 'Group'])

        input_df = input_df_original.copy()
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        pred = model.predict(input_encoded)
        predicted_label = le.inverse_transform(pred)[0]
        proba = model.predict_proba(input_encoded)[0]
        confidence = proba.max()

        display(HTML(f"""
        <div style='background:{SLATE}; padding:15px; border-radius:8px; border:1px solid {GOLD};'>
        <h3 style='color:{GOLD};'>Predicted Segment for {selected_customer_id}: {predicted_label}</h3>
        <p style='color:white;'>Confidence: {confidence:.1%}</p>
        </div>"""))

        prob_df = pd.DataFrame({"Segment": le.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
        px.bar(prob_df, x="Segment", y="Probability", color="Segment", template="plotly_dark",
               title="Prediction Confidence by Segment").show()


predict_btn.on_click(on_predict_click)

print("\nLive Prediction Tool — select a customer ID and click Predict:")
display(widgets.VBox([customer_id_w, predict_btn, output_box]))

# ------------------------------------------------------------
# STEP 7: FOOTER
# ------------------------------------------------------------
footer_html = f"""
<hr>
<h3 style='text-align:center; background:{NAVY}; color:{GOLD}; padding:20px; border-bottom:5px solid {GOLD};'>
Strategic Customer Insights Dashboard
</h3>"""
display(HTML(footer_html))
