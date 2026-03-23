-- ============================================================================
-- NYC Taxi Demand Dashboard — SQL queries for Databricks Dashboard widgets
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Widget 1: Hourly City-Level Demand Trend (line chart)
-- ---------------------------------------------------------------------------
SELECT
  pickup_date,
  pickup_hour,
  city_trip_cnt,
  rolling_city_trip_cnt_mean_24h
FROM demo.gold.nyc_dashboard_kpi_mart
ORDER BY pickup_date, pickup_hour;

-- ---------------------------------------------------------------------------
-- Widget 2: Total Demand KPIs (counter)
-- ---------------------------------------------------------------------------
SELECT
  SUM(city_trip_cnt) AS total_trips,
  AVG(city_trip_cnt) AS avg_hourly_trips,
  MAX(city_trip_cnt) AS peak_hourly_trips,
  SUM(city_total_revenue) AS total_revenue,
  AVG(city_avg_fare) AS avg_fare,
  AVG(city_avg_tip_pct) AS avg_tip_pct
FROM demo.gold.nyc_dashboard_kpi_mart;

-- ---------------------------------------------------------------------------
-- Widget 3: Demand by Hour of Day (bar chart)
-- ---------------------------------------------------------------------------
SELECT
  pickup_hour,
  AVG(city_trip_cnt) AS avg_trips,
  AVG(city_total_revenue) AS avg_revenue,
  AVG(city_avg_fare) AS avg_fare
FROM demo.gold.nyc_dashboard_kpi_mart
GROUP BY pickup_hour
ORDER BY pickup_hour;

-- ---------------------------------------------------------------------------
-- Widget 4: Daily Revenue Trend (area chart)
-- ---------------------------------------------------------------------------
SELECT
  pickup_date,
  SUM(city_total_revenue) AS daily_revenue,
  SUM(city_trip_cnt) AS daily_trips
FROM demo.gold.nyc_dashboard_kpi_mart
GROUP BY pickup_date
ORDER BY pickup_date;

-- ---------------------------------------------------------------------------
-- Widget 5: Hour-over-Hour Demand Change (line chart)
-- ---------------------------------------------------------------------------
SELECT
  pickup_date,
  pickup_hour,
  city_trip_cnt,
  lag_city_trip_cnt_1h,
  CASE
    WHEN lag_city_trip_cnt_1h IS NOT NULL AND lag_city_trip_cnt_1h > 0
    THEN ROUND((city_trip_cnt - lag_city_trip_cnt_1h) / lag_city_trip_cnt_1h * 100, 2)
    ELSE NULL
  END AS hoh_pct_change
FROM demo.gold.nyc_dashboard_kpi_mart
WHERE lag_city_trip_cnt_1h IS NOT NULL
ORDER BY pickup_date, pickup_hour;

-- ---------------------------------------------------------------------------
-- Widget 6: Top Demand Zones (table / bar chart from training mart)
-- ---------------------------------------------------------------------------
SELECT
  pu_location_id,
  SUM(label_trip_cnt) AS total_trips,
  AVG(avg_fare_amount) AS avg_fare,
  AVG(avg_trip_distance_miles) AS avg_distance_mi
FROM demo.gold.nyc_demand_training_mart
GROUP BY pu_location_id
ORDER BY total_trips DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Widget 7: Day-over-Day Demand Comparison (line chart)
-- ---------------------------------------------------------------------------
SELECT
  pickup_date,
  SUM(city_trip_cnt) AS daily_trips,
  LAG(SUM(city_trip_cnt), 1) OVER (ORDER BY pickup_date) AS prev_day_trips
FROM demo.gold.nyc_dashboard_kpi_mart
GROUP BY pickup_date
ORDER BY pickup_date;
