"""Model Comparison Dashboard page.

Static validation metrics copied from notebooks 03–05 (not live MLflow).
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.title("📈 Model Comparison Dashboard")

st.markdown(
    """
This page is **Demo only**: it does not call FastAPI and does not read MLflow at runtime.
The numbers are the **validation** metrics from notebooks `03` (Rossmann classical + ensemble),
`04` (Rossmann FNN experiments), and `05` (NYC LightGBM / XGBoost / FNN).

The **registered champion** is the model with the lowest validation RMSE — the same rule as
in those notebooks. Switching the chart to R² or MAPE can name a different “best” row;
that does not change who sits in Unity Catalog.
"""
)

API_BASE = st.sidebar.text_input(
    "API Base URL", value=os.getenv("API_BASE_URL", "http://localhost:8000")
)

# Validation metrics from notebook 03 section 9 (`print(comp_df.to_string())`),
# VotingEnsemble from 03 section 10 (`[VotingEnsemble] val_rmse=…`),
# FNN rows from notebook 04 section 10 (`print(dl_comp.to_string())`).
rossmann_models = pd.DataFrame(
    {
        "model": [
            "LinearRegression",
            "Ridge",
            "Lasso",
            "DecisionTree",
            "RandomForest",
            "GradientBoosting",
            "LightGBM",
            "XGBoost",
            "VotingEnsemble",
            "FNN_baseline",
            "FNN_deep",
            "FNN_sigmoid",
            "FNN_tanh",
            "FNN_sgd",
            "FNN_rmsprop",
            "FNN_mixed_precision",
        ],
        "rmse": [
            626.556421,
            626.556780,
            626.582488,
            134.617969,
            46.816420,
            55.558841,
            140.055711,
            147.148804,
            58.17,
            160.901404,
            101.150241,
            4436.755518,
            2408.232395,
            1005.415012,
            172.228373,
            7245.502826,
        ],
        "r2": [
            0.975367,
            0.975367,
            0.975365,
            0.998863,
            0.999862,
            0.999806,
            0.998769,
            0.998641,
            0.9998,
            0.998376,
            0.999358,
            -0.235173,
            0.636090,
            0.936571,
            0.998139,
            -2.294078,
        ],
        "mape": [
            0.062502,
            0.062501,
            0.062138,
            0.010714,
            0.000974,
            0.005735,
            0.010456,
            0.007408,
            0.0043,
            0.013359,
            0.007947,
            0.396230,
            0.107330,
            0.108313,
            0.015258,
            0.996788,
        ],
        "type": [
            "linear",
            "linear",
            "linear",
            "tree",
            "ensemble",
            "gbm",
            "gbm",
            "gbm",
            "ensemble",
            "neural",
            "neural",
            "neural",
            "neural",
            "neural",
            "neural",
            "neural",
        ],
    }
)

# Validation metrics from notebook 05 section 9 (`print(comp_df.to_string())`).
nyc_models = pd.DataFrame(
    {
        "model": ["LightGBM", "XGBoost", "FNN"],
        "rmse": [4.685446, 4.834012, 3.326853],
        "r2": [0.996302, 0.996064, 0.998136],
        "mape": [0.054255, 0.059535, 0.218530],
        "type": ["gbm", "gbm", "neural"],
    }
)

CHAMPIONS = {"Rossmann": "RandomForest", "NYC Demand": "FNN"}
CHAMPION_NOTES = {
    "Rossmann": (
        "Notebook 03 registered `rossmann_sales_champion` on lowest validation RMSE. "
        "VotingEnsemble (RMSE 58.17, members RF + GB + DT) and FNN_deep (RMSE 101.15) "
        "did not replace it."
    ),
    "NYC Demand": (
        "Notebook 05 registered `nyc_demand_champion` on lowest validation RMSE. "
        "LightGBM is better on MAPE, but MAPE is not the selection rule."
    ),
}

st.sidebar.markdown("---")
dataset = st.sidebar.radio("Dataset", ["Rossmann", "NYC Demand"])
df = (rossmann_models if dataset == "Rossmann" else nyc_models).copy()
champion_name = CHAMPIONS[dataset]
df["champion"] = df["model"].eq(champion_name)

# ---------------------------------------------------------------------------
# Registered champion (selection rule), separate from the chart's "best" row
# ---------------------------------------------------------------------------
champ_row = df.loc[df["model"] == champion_name].iloc[0]
st.success(
    f"**Registered champion:** `{champion_name}`  \n"
    f"RMSE {champ_row['rmse']:.2f} · R² {champ_row['r2']:.4f} · MAPE {champ_row['mape']:.4f}  \n"
    f"{CHAMPION_NOTES[dataset]}"
)

# ---------------------------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------------------------
st.subheader(f"{dataset} — Model Metrics")
st.markdown(
    """
Horizontal bars for the metric you pick. **RMSE** and **MAPE**: shorter is better.
**R²**: longer is better (closer to 1). Colors are model families — linear, tree,
GBM (Gradient Boosting: sklearn GB / LightGBM / XGBoost), ensemble, neural.
"""
)

metric_col = st.selectbox("Primary Metric", ["rmse", "r2", "mape"])
ascending = metric_col != "r2"
df_sorted = df.sort_values(metric_col, ascending=ascending)
df_plot = df_sorted.copy()
df_plot["label"] = df_plot.apply(
    lambda r: f"★ {r['model']}" if r["champion"] else r["model"], axis=1
)

col1, col2 = st.columns([2, 1])

color_map = {
    "linear": "#636EFA",
    "tree": "#EF553B",
    "ensemble": "#00CC96",
    "gbm": "#AB63FA",
    "neural": "#FFA15A",
}

with col1:
    fig = px.bar(
        df_plot,
        x=metric_col,
        y="label",
        orientation="h",
        color="type",
        color_discrete_map=color_map,
        title=f"{dataset} — {metric_col.upper()} by Model  (★ = registered champion)",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, len(df) * 30))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Best by this metric")
    st.caption(
        "First row after sorting the chart. This can differ from the registered champion "
        "if the primary metric is not RMSE."
    )
    best = df_sorted.iloc[0]
    st.metric("Model", best["model"])
    st.metric("RMSE", f"{best['rmse']:.2f}")
    st.metric("R²", f"{best['r2']:.4f}")
    st.metric("MAPE", f"{best['mape']:.4f}")
    if best["model"] == champion_name:
        st.caption("This row is also the registered champion.")
    else:
        st.caption(f"Registered champion remains `{champion_name}` (lowest RMSE).")

# ---------------------------------------------------------------------------
# Scatter: RMSE vs R²
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("RMSE vs R² Scatter")
st.markdown(
    """
Each point is one model. Left and up is better: low RMSE, high R². Bubble size is MAPE
(smaller is better). The registered champion is the larger outlined marker.
"""
)

fig2 = px.scatter(
    df,
    x="rmse",
    y="r2",
    color="type",
    text="model",
    color_discrete_map=color_map,
    size="mape",
    size_max=28,
    title=f"{dataset} — RMSE vs R² (bubble size = MAPE)",
)
fig2.update_traces(textposition="top center")
champ_scatter = df[df["champion"]]
fig2.add_trace(
    go.Scatter(
        x=champ_scatter["rmse"],
        y=champ_scatter["r2"],
        mode="markers",
        marker=dict(
            size=18,
            color="rgba(0,0,0,0)",
            line=dict(width=3, color="#02B8FA"),
        ),
        name="registered champion",
        hoverinfo="skip",
    )
)
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Model type aggregation
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Average Metrics by Model Type")
st.markdown(
    """
Family averages, not a champion rule. One bad FNN (negative R², huge RMSE) will pull the
neural average down — that is the point of showing the family, not a reason to ignore
the registered champion.
"""
)

agg = df.groupby("type")[["rmse", "r2", "mape"]].mean().reset_index()
fig3 = make_subplots(rows=1, cols=3, subplot_titles=["Avg RMSE", "Avg R²", "Avg MAPE"])

type_colors = {
    "linear": "#636EFA",
    "tree": "#EF553B",
    "ensemble": "#00CC96",
    "gbm": "#AB63FA",
    "neural": "#FFA15A",
}
bar_colors = [type_colors.get(t, "#888888") for t in agg["type"]]

for i, col in enumerate(["rmse", "r2", "mape"]):
    fig3.add_trace(
        go.Bar(x=agg["type"], y=agg[col], name=col, marker_color=bar_colors),
        row=1,
        col=i + 1,
    )

fig3.update_layout(
    height=350, showlegend=False, title_text="Aggregated Performance by Model Family"
)
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Full table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Full Metrics Table")
st.markdown(
    """
Same validation numbers as the charts. `champion` is true only for the model registered
in Unity Catalog (lowest RMSE). Sort here does not change that flag.
"""
)
table = df_sorted.reset_index(drop=True)[
    ["champion", "model", "type", "rmse", "r2", "mape"]
]
st.dataframe(table, use_container_width=True, hide_index=True)
