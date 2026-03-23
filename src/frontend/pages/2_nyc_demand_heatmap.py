"""NYC Taxi Demand Heatmap page.

Demonstrates: interactive visualizations, heatmap, slider controls.
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NYC Demand Heatmap", page_icon="🚕", layout="wide")
st.title("🚕 NYC Taxi Demand — Zone-Hour Heatmap")

API_BASE = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

st.sidebar.markdown("---")
st.sidebar.markdown("### Prediction Parameters")

pickup_hour = st.sidebar.slider("Pickup Hour", 0, 23, 12)
num_zones = st.sidebar.slider("Number of Zones to Predict", 5, 50, 20)

# Default feature values for demonstration.
avg_distance = st.sidebar.number_input("Avg Trip Distance (mi)", 0.0, 50.0, 3.5, 0.5)
avg_duration = st.sidebar.number_input("Avg Trip Duration (min)", 0.0, 120.0, 15.0, 1.0)
avg_fare = st.sidebar.number_input("Avg Fare ($)", 0.0, 200.0, 15.0, 1.0)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Demand Forecast for Hour {pickup_hour}:00")

    if st.button("Generate Demand Heatmap", type="primary"):
        instances = []
        for zone_id in range(1, num_zones + 1):
            instances.append({
                "pickup_hour": pickup_hour,
                "pu_location_id": zone_id,
                "avg_trip_distance_miles": avg_distance,
                "avg_trip_duration_min": avg_duration,
                "avg_fare_amount": avg_fare,
                "avg_tip_pct": 0.15,
                "total_revenue": avg_fare * 50,
                "lag_trip_cnt_1h": np.random.uniform(10, 200),
                "lag_trip_cnt_24h": np.random.uniform(10, 200),
                "rolling_trip_cnt_mean_24h": np.random.uniform(20, 150),
                "rolling_fare_mean_24h": avg_fare * np.random.uniform(0.8, 1.2),
            })

        try:
            resp = requests.post(
                f"{API_BASE}/predict/nyc-demand",
                json={"instances": instances},
                timeout=15,
            )
            if resp.status_code == 200:
                preds = resp.json()["predictions"]
                df = pd.DataFrame(preds)
            else:
                st.error(f"API error: {resp.status_code}")
                df = None
        except requests.exceptions.ConnectionError:
            st.warning("API not reachable — showing demo data.")
            df = pd.DataFrame({
                "pu_location_id": list(range(1, num_zones + 1)),
                "pickup_hour": [pickup_hour] * num_zones,
                "predicted_trip_cnt": np.random.uniform(5, 300, num_zones),
            })

        if df is not None:
            # Bar chart of demand by zone.
            df_sorted = df.sort_values("predicted_trip_cnt", ascending=False)
            fig = px.bar(
                df_sorted, x="pu_location_id", y="predicted_trip_cnt",
                color="predicted_trip_cnt",
                color_continuous_scale="YlOrRd",
                title=f"Predicted Trip Count by Zone — Hour {pickup_hour}:00",
                labels={"predicted_trip_cnt": "Predicted Trips", "pu_location_id": "Zone ID"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_sorted[["pu_location_id", "predicted_trip_cnt"]].reset_index(drop=True),
                         use_container_width=True)

with col2:
    st.subheader("Top Zones")
    st.markdown("The busiest zones will appear here after running the forecast.")

st.markdown("---")
st.subheader("24-Hour Demand Curve (Demo)")
st.markdown("Hourly demand curve for a sample zone:")

demo_zone = st.selectbox("Select Zone ID", list(range(1, 11)), index=0)
hours = list(range(24))
# Simulate a realistic demand pattern: low at night, peaks at 8am and 6pm.
base = np.array([5, 3, 2, 2, 3, 8, 25, 60, 90, 70, 55, 50,
                  55, 50, 45, 55, 70, 95, 85, 60, 40, 25, 15, 8], dtype=float)
noise = np.random.normal(0, 5, 24)
demand = np.maximum(base + noise, 0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=hours, y=demand, mode="lines+markers", name=f"Zone {demo_zone}",
                          line=dict(color="steelblue", width=2)))
fig.update_layout(title=f"24h Demand Curve — Zone {demo_zone} (Demo)",
                   xaxis_title="Hour", yaxis_title="Trip Count")
st.plotly_chart(fig, use_container_width=True)
