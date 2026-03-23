"""Streamlit main app — E2E ML Pipeline Frontend.

Run with: streamlit run workdir/src/frontend/app.py

Pages are auto-discovered from the pages/ directory.
"""
import streamlit as st

st.set_page_config(
    page_title="E2E ML Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("E2E ML Pipeline — Interactive Dashboard")

st.markdown("""
Welcome to the interactive frontend for the end-to-end ML pipeline project.

**Available pages** (see sidebar):
- **Rossmann Prediction** — Predict daily store sales using the champion model.
- **NYC Demand Heatmap** — Visualize zone-hour demand forecasts across NYC.
- **Model Comparison** — Compare metrics and charts across all trained models.

---

**Architecture:**
- This Streamlit app calls a FastAPI backend that serves champion models from the UC Model Registry.
- All models were trained on Databricks Free (Python + SQL Serverless).
""")

st.sidebar.success("Select a page above.")
