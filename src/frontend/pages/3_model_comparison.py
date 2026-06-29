"""Model Comparison Dashboard page.

Demonstrates: complex 2D visualizations, interactive data exploration.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.title("📈 Model Comparison Dashboard")

st.markdown("""
Compare performance metrics across all trained models (classical ML and deep learning).
Data is loaded from the MLflow experiment runs via the API or shown as demo data.
""")

API_BASE = st.sidebar.text_input("API Base URL", value=os.getenv("API_BASE_URL", "http://localhost:8000"))

# ---------------------------------------------------------------------------
# Demo data (mirrors what notebooks 03/04 produce).
# In production, this would be fetched from MLflow or the serving layer.
# ---------------------------------------------------------------------------
rossmann_models = pd.DataFrame({
    "model": ["LinearRegression", "Ridge", "Lasso", "DecisionTree", "RandomForest",
              "GradientBoosting", "LightGBM", "XGBoost", "VotingEnsemble",
              "FNN_baseline", "FNN_deep", "FNN_sigmoid", "FNN_tanh",
              "FNN_sgd", "FNN_rmsprop", "FNN_mixed_precision"],
    "rmse": [2850, 2845, 2900, 1980, 1420, 1380, 1250, 1270, 1200,
             1650, 1580, 2100, 1900, 1750, 1620, 1660],
    "r2": [0.72, 0.72, 0.71, 0.87, 0.93, 0.94, 0.95, 0.95, 0.96,
           0.91, 0.92, 0.85, 0.88, 0.90, 0.91, 0.91],
    "mape": [0.32, 0.32, 0.33, 0.22, 0.14, 0.13, 0.11, 0.11, 0.10,
             0.18, 0.16, 0.25, 0.21, 0.19, 0.17, 0.18],
    "type": ["linear", "linear", "linear", "tree", "ensemble", "ensemble",
             "gbm", "gbm", "ensemble", "neural", "neural", "neural",
             "neural", "neural", "neural", "neural"],
})

nyc_models = pd.DataFrame({
    "model": ["LightGBM", "XGBoost", "FNN"],
    "rmse": [45.2, 47.8, 52.1],
    "r2": [0.88, 0.87, 0.83],
    "mape": [0.21, 0.23, 0.28],
    "type": ["gbm", "gbm", "neural"],
})

st.sidebar.markdown("---")
dataset = st.sidebar.radio("Dataset", ["Rossmann", "NYC Demand"])
df = rossmann_models if dataset == "Rossmann" else nyc_models

# ---------------------------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------------------------
st.subheader(f"{dataset} — Model Metrics")

metric_col = st.selectbox("Primary Metric", ["rmse", "r2", "mape"])
ascending = metric_col != "r2"
df_sorted = df.sort_values(metric_col, ascending=ascending)

col1, col2 = st.columns([2, 1])

with col1:
    color_map = {"linear": "#636EFA", "tree": "#EF553B", "ensemble": "#00CC96",
                  "gbm": "#AB63FA", "neural": "#FFA15A"}
    fig = px.bar(
        df_sorted, x=metric_col, y="model", orientation="h",
        color="type", color_discrete_map=color_map,
        title=f"{dataset} — {metric_col.upper()} by Model",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, len(df) * 30))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Best Model")
    best = df_sorted.iloc[0] if ascending else df_sorted.iloc[-1] if metric_col == "r2" else df_sorted.iloc[0]
    st.metric("Model", best["model"])
    st.metric("RMSE", f"{best['rmse']:.1f}")
    st.metric("R²", f"{best['r2']:.3f}")
    st.metric("MAPE", f"{best['mape']:.3f}")

# ---------------------------------------------------------------------------
# Scatter: RMSE vs R²
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("RMSE vs R² Scatter")

fig2 = px.scatter(
    df, x="rmse", y="r2", color="type", text="model",
    color_discrete_map=color_map, size="mape", size_max=20,
    title=f"{dataset} — RMSE vs R² (bubble size = MAPE)",
)
fig2.update_traces(textposition="top center")
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Model type aggregation
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Average Metrics by Model Type")

agg = df.groupby("type")[["rmse", "r2", "mape"]].mean().reset_index()
fig3 = make_subplots(rows=1, cols=3, subplot_titles=["Avg RMSE", "Avg R²", "Avg MAPE"])

for i, col in enumerate(["rmse", "r2", "mape"]):
    fig3.add_trace(
        go.Bar(x=agg["type"], y=agg[col], name=col,
               marker_color=["#636EFA", "#00CC96", "#AB63FA", "#FFA15A"][:len(agg)]),
        row=1, col=i + 1,
    )

fig3.update_layout(height=350, showlegend=False, title_text="Aggregated Performance by Model Family")
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Full table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Full Metrics Table")
st.dataframe(df_sorted.reset_index(drop=True), use_container_width=True)
