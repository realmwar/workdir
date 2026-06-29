"""Rossmann Sales Prediction page.

Demonstrates: interactive Streamlit UI, API integration, data visualization.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="Rossmann Prediction", page_icon="🏪", layout="wide")
st.title("🏪 Rossmann Store Sales Prediction")

API_BASE = st.sidebar.text_input("API Base URL", value=os.getenv("API_BASE_URL", "http://localhost:8000"))

st.sidebar.markdown("---")
st.sidebar.markdown("### Store Parameters")

store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1)
day_of_week = st.sidebar.selectbox("Day of Week", options=list(range(1, 8)),
                                     format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x - 1])
is_open = st.sidebar.checkbox("Store is open", value=True)
is_promo = st.sidebar.checkbox("Promo active", value=False)
is_school_holiday = st.sidebar.checkbox("School holiday", value=False)

store_type = st.sidebar.selectbox("Store Type", options=[0, 1, 2, 3],
                                    format_func=lambda x: ["a", "b", "c", "d"][x])
assortment_type = st.sidebar.selectbox("Assortment", options=[0, 1, 2],
                                         format_func=lambda x: ["a", "b", "c"][x])
competition_distance = st.sidebar.slider("Competition Distance (km)", 0.0, 75.0, 5.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Promo2 Parameters")
has_promo2 = st.sidebar.checkbox("Has Promo2", value=False)
is_promo2_active = st.sidebar.checkbox("Promo2 active now", value=False)
is_promo_interval_month = st.sidebar.checkbox("Promo interval month", value=False)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Single Prediction")

    if st.button("Predict Sales", type="primary"):
        payload = {
            "instances": [{
                "store_id": store_id,
                "day_of_week": day_of_week,
                "is_weekend": 1 if day_of_week >= 6 else 0,
                "is_open": int(is_open),
                "is_promo": int(is_promo),
                "state_holiday_code": 0,
                "is_school_holiday": int(is_school_holiday),
                "store_type": store_type,
                "assortment_type": assortment_type,
                "competition_distance_km": competition_distance,
                "has_promo2": int(has_promo2),
                "is_promo2_active": int(is_promo2_active),
                "is_promo_interval_month": int(is_promo_interval_month),
            }]
        }

        try:
            resp = requests.post(f"{API_BASE}/predict/rossmann", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pred = data["predictions"][0]
                st.metric("Predicted Daily Sales", f"€{pred['predicted_sales']:,.0f}")
                st.json(pred)
            else:
                st.error(f"API error: {resp.status_code} — {resp.text}")
        except requests.exceptions.ConnectionError:
            st.warning("Cannot connect to API. Showing demo mode with placeholder values.")
            st.metric("Predicted Daily Sales (demo)", f"€{5_432:,.0f}")

with col2:
    st.subheader("API Health")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.success(f"Status: {health['status']}")
        st.json(health["models_loaded"])
    except Exception:
        st.warning("API not reachable — running in demo mode.")

st.markdown("---")
st.subheader("Batch Prediction Scenario")
st.markdown("Simulate predictions across all 7 days for the selected store:")

if st.button("Run Weekly Forecast"):
    batch_instances = []
    for dow in range(1, 8):
        batch_instances.append({
            "store_id": store_id,
            "day_of_week": dow,
            "is_weekend": 1 if dow >= 6 else 0,
            "is_open": int(is_open),
            "is_promo": int(is_promo),
            "state_holiday_code": 0,
            "is_school_holiday": 0,
            "store_type": store_type,
            "assortment_type": assortment_type,
            "competition_distance_km": competition_distance,
            "has_promo2": int(has_promo2),
            "is_promo2_active": int(is_promo2_active),
            "is_promo_interval_month": int(is_promo_interval_month),
        })

    try:
        resp = requests.post(f"{API_BASE}/predict/rossmann", json={"instances": batch_instances}, timeout=15)
        if resp.status_code == 200:
            preds = resp.json()["predictions"]
            df = pd.DataFrame(preds)
            df["day_name"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            fig = px.bar(df, x="day_name", y="predicted_sales",
                         title=f"Weekly Sales Forecast — Store {store_id}",
                         labels={"predicted_sales": "Predicted Sales (€)", "day_name": "Day"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[["day_name", "predicted_sales"]], use_container_width=True)
        else:
            st.error(f"API error: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.warning("API not reachable. Connect the FastAPI backend to enable live predictions.")
        demo_sales = [4200, 4800, 5100, 4900, 5300, 6200, 3800]
        demo_df = pd.DataFrame({"day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                 "predicted_sales": demo_sales})
        fig = px.bar(demo_df, x="day_name", y="predicted_sales",
                     title=f"Weekly Sales Forecast (Demo) — Store {store_id}")
        st.plotly_chart(fig, use_container_width=True)
