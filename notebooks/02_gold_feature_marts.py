# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Gold Feature Marts (Training / Serving / Dashboard)
# MAGIC
# MAGIC Goals:
# MAGIC - materialize business-ready gold marts on top of silver feature bases;
# MAGIC - define stable contracts for model training and serving interfaces;
# MAGIC - expose dashboard-friendly aggregates for Rossmann and NYC tracks.
# MAGIC
# MAGIC Constraints:
# MAGIC - no Spark DataFrame API;
# MAGIC - SQL-first transformations, Python only for lightweight orchestration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Runtime parameters

# COMMAND ----------

dbutils.widgets.dropdown("rebuild_all", "true", ["true", "false"])
REBUILD_ALL = dbutils.widgets.get("rebuild_all").lower() == "true"

CATALOG = "demo"

print("catalog:", CATALOG)
print("rebuild_all:", REBUILD_ALL)

if not REBUILD_ALL:
    dbutils.notebook.exit("rebuild_all=false -> gold rebuild skipped by parameter")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Gold prep: schema and audit sink

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ensure target schemas exist and create an audit table for gold row-count snapshots.
# MAGIC CREATE CATALOG IF NOT EXISTS demo;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.gold;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.audit;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS demo.audit.gold_snapshot (
# MAGIC   snapshot_ts TIMESTAMP,
# MAGIC   table_name STRING,
# MAGIC   row_count BIGINT
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Rossmann gold marts

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Training mart:
# MAGIC -- clean feature matrix + label with null-safe constraints.
# MAGIC CREATE OR REPLACE TABLE demo.gold.rossmann_training_mart AS
# MAGIC SELECT
# MAGIC   store_id,
# MAGIC   business_date,
# MAGIC   day_of_week,
# MAGIC   is_weekend,
# MAGIC   is_open,
# MAGIC   is_promo,
# MAGIC   state_holiday_code,
# MAGIC   is_school_holiday,
# MAGIC   store_type,
# MAGIC   assortment_type,
# MAGIC   competition_distance_km,
# MAGIC   has_promo2,
# MAGIC   is_promo2_active,
# MAGIC   is_promo_interval_month,
# MAGIC   lag_sales_1d,
# MAGIC   lag_sales_7d,
# MAGIC   rolling_sales_mean_7d,
# MAGIC   rolling_customers_mean_7d,
# MAGIC   promo_days_last_14d,
# MAGIC   customers,
# MAGIC   avg_ticket,
# MAGIC   sales AS label_sales,
# MAGIC   log1p_sales AS label_log1p_sales,
# MAGIC   silver_loaded_at,
# MAGIC   current_timestamp() AS gold_loaded_at
# MAGIC FROM demo.silver.rossmann_train_feature_base
# MAGIC WHERE business_date IS NOT NULL
# MAGIC   AND store_id IS NOT NULL
# MAGIC   AND sales IS NOT NULL
# MAGIC   AND sales >= 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Serving mart:
# MAGIC -- inference-ready feature contract (no label columns).
# MAGIC CREATE OR REPLACE TABLE demo.gold.rossmann_serving_mart AS
# MAGIC SELECT
# MAGIC   prediction_id,
# MAGIC   store_id,
# MAGIC   business_date,
# MAGIC   day_of_week,
# MAGIC   is_weekend,
# MAGIC   is_open,
# MAGIC   is_promo,
# MAGIC   state_holiday_code,
# MAGIC   is_school_holiday,
# MAGIC   store_type,
# MAGIC   assortment_type,
# MAGIC   competition_distance_km,
# MAGIC   has_promo2,
# MAGIC   is_promo2_active,
# MAGIC   is_promo_interval_month,
# MAGIC   silver_loaded_at,
# MAGIC   current_timestamp() AS gold_loaded_at
# MAGIC FROM demo.silver.rossmann_test_feature_base
# MAGIC WHERE prediction_id IS NOT NULL
# MAGIC   AND store_id IS NOT NULL
# MAGIC   AND business_date IS NOT NULL;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dashboard mart:
# MAGIC -- daily and rolling KPIs by store with trend markers.
# MAGIC CREATE OR REPLACE TABLE demo.gold.rossmann_dashboard_kpi_mart AS
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     business_date,
# MAGIC     store_id,
# MAGIC     label_sales AS sales,
# MAGIC     customers,
# MAGIC     avg_ticket,
# MAGIC     is_open,
# MAGIC     is_promo,
# MAGIC     is_weekend
# MAGIC   FROM demo.gold.rossmann_training_mart
# MAGIC ),
# MAGIC daily AS (
# MAGIC   SELECT
# MAGIC     business_date,
# MAGIC     COUNT(DISTINCT store_id) AS active_store_cnt,
# MAGIC     SUM(sales) AS total_sales,
# MAGIC     SUM(customers) AS total_customers,
# MAGIC     AVG(avg_ticket) AS avg_ticket_mean,
# MAGIC     AVG(CAST(is_open AS DOUBLE)) AS open_store_ratio,
# MAGIC     AVG(CAST(is_promo AS DOUBLE)) AS promo_ratio,
# MAGIC     AVG(CAST(is_weekend AS DOUBLE)) AS weekend_ratio
# MAGIC   FROM base
# MAGIC   GROUP BY business_date
# MAGIC )
# MAGIC SELECT
# MAGIC   business_date,
# MAGIC   active_store_cnt,
# MAGIC   total_sales,
# MAGIC   total_customers,
# MAGIC   avg_ticket_mean,
# MAGIC   open_store_ratio,
# MAGIC   promo_ratio,
# MAGIC   weekend_ratio,
# MAGIC   LAG(total_sales, 1) OVER (ORDER BY business_date) AS lag_total_sales_1d,
# MAGIC   LAG(total_sales, 7) OVER (ORDER BY business_date) AS lag_total_sales_7d,
# MAGIC   AVG(total_sales) OVER (
# MAGIC     ORDER BY business_date
# MAGIC     ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
# MAGIC   ) AS rolling_total_sales_mean_7d,
# MAGIC   current_timestamp() AS gold_loaded_at
# MAGIC FROM daily;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) NYC gold marts

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Training mart:
# MAGIC -- zone-hour demand/revenue features with demand as primary label.
# MAGIC CREATE OR REPLACE TABLE demo.gold.nyc_demand_training_mart AS
# MAGIC SELECT
# MAGIC   pickup_date,
# MAGIC   pickup_hour,
# MAGIC   pu_location_id,
# MAGIC   trip_cnt AS label_trip_cnt,
# MAGIC   avg_trip_distance_miles,
# MAGIC   avg_trip_duration_min,
# MAGIC   avg_fare_amount,
# MAGIC   avg_tip_pct,
# MAGIC   total_revenue,
# MAGIC   lag_trip_cnt_1h,
# MAGIC   lag_trip_cnt_24h,
# MAGIC   rolling_trip_cnt_mean_24h,
# MAGIC   rolling_fare_mean_24h,
# MAGIC   silver_loaded_at,
# MAGIC   current_timestamp() AS gold_loaded_at
# MAGIC FROM demo.silver.nyc_zone_hour_feature_base
# MAGIC WHERE pickup_date IS NOT NULL
# MAGIC   AND pu_location_id IS NOT NULL
# MAGIC   AND trip_cnt IS NOT NULL
# MAGIC   AND trip_cnt >= 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Serving mart:
# MAGIC -- latest available hour per pickup zone for "next-hour demand" inference workflows.
# MAGIC CREATE OR REPLACE TABLE demo.gold.nyc_serving_latest_zone_hour_mart AS
# MAGIC WITH ranked AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     ROW_NUMBER() OVER (
# MAGIC       PARTITION BY pu_location_id
# MAGIC       ORDER BY pickup_date DESC, pickup_hour DESC
# MAGIC     ) AS rn
# MAGIC   FROM demo.gold.nyc_demand_training_mart
# MAGIC )
# MAGIC SELECT
# MAGIC   pickup_date,
# MAGIC   pickup_hour,
# MAGIC   pu_location_id,
# MAGIC   avg_trip_distance_miles,
# MAGIC   avg_trip_duration_min,
# MAGIC   avg_fare_amount,
# MAGIC   avg_tip_pct,
# MAGIC   total_revenue,
# MAGIC   lag_trip_cnt_1h,
# MAGIC   lag_trip_cnt_24h,
# MAGIC   rolling_trip_cnt_mean_24h,
# MAGIC   rolling_fare_mean_24h,
# MAGIC   current_timestamp() AS gold_loaded_at
# MAGIC FROM ranked
# MAGIC WHERE rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dashboard mart:
# MAGIC -- city-level hourly demand/revenue and short-term trend indicators.
# MAGIC CREATE OR REPLACE TABLE demo.gold.nyc_dashboard_kpi_mart AS
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     pickup_date,
# MAGIC     pickup_hour,
# MAGIC     SUM(label_trip_cnt) AS city_trip_cnt,
# MAGIC     SUM(total_revenue) AS city_total_revenue,
# MAGIC     AVG(avg_fare_amount) AS city_avg_fare,
# MAGIC     AVG(avg_tip_pct) AS city_avg_tip_pct
# MAGIC   FROM demo.gold.nyc_demand_training_mart
# MAGIC   GROUP BY pickup_date, pickup_hour
# MAGIC ),
# MAGIC ts AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     TO_TIMESTAMP(
# MAGIC       CONCAT(
# MAGIC         CAST(pickup_date AS STRING), ' ',
# MAGIC         LPAD(CAST(pickup_hour AS STRING), 2, '0'),
# MAGIC         ':00:00'
# MAGIC       )
# MAGIC     ) AS city_hour_ts
# MAGIC   FROM base
# MAGIC )
# MAGIC SELECT
# MAGIC   pickup_date,
# MAGIC   pickup_hour,
# MAGIC   city_trip_cnt,
# MAGIC   city_total_revenue,
# MAGIC   city_avg_fare,
# MAGIC   city_avg_tip_pct,
# MAGIC   LAG(city_trip_cnt, 1) OVER (ORDER BY city_hour_ts) AS lag_city_trip_cnt_1h,
# MAGIC   LAG(city_trip_cnt, 24) OVER (ORDER BY city_hour_ts) AS lag_city_trip_cnt_24h,
# MAGIC   AVG(city_trip_cnt) OVER (
# MAGIC     ORDER BY city_hour_ts
# MAGIC     ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
# MAGIC   ) AS rolling_city_trip_cnt_mean_24h,
# MAGIC   current_timestamp() AS gold_loaded_at
# MAGIC FROM ts;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Unified contracts for downstream apps

# COMMAND ----------

# MAGIC %sql
# MAGIC -- A small routing table that helps API/dashboard layers discover mart contracts.
# MAGIC CREATE OR REPLACE TABLE demo.gold.contract_registry_mart AS
# MAGIC SELECT 'rossmann_training' AS contract_name, 'demo.gold.rossmann_training_mart' AS table_name, 'training' AS contract_type, current_timestamp() AS updated_at
# MAGIC UNION ALL
# MAGIC SELECT 'rossmann_serving', 'demo.gold.rossmann_serving_mart', 'serving', current_timestamp()
# MAGIC UNION ALL
# MAGIC SELECT 'rossmann_dashboard', 'demo.gold.rossmann_dashboard_kpi_mart', 'dashboard', current_timestamp()
# MAGIC UNION ALL
# MAGIC SELECT 'nyc_training', 'demo.gold.nyc_demand_training_mart', 'training', current_timestamp()
# MAGIC UNION ALL
# MAGIC SELECT 'nyc_serving', 'demo.gold.nyc_serving_latest_zone_hour_mart', 'serving', current_timestamp()
# MAGIC UNION ALL
# MAGIC SELECT 'nyc_dashboard', 'demo.gold.nyc_dashboard_kpi_mart', 'dashboard', current_timestamp();

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Gold audit snapshot + quick checks

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Persist table row counts for monitoring and pipeline regression detection.
# MAGIC INSERT INTO demo.audit.gold_snapshot
# MAGIC SELECT current_timestamp(), 'demo.gold.rossmann_training_mart', COUNT(*) FROM demo.gold.rossmann_training_mart
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.gold.rossmann_serving_mart', COUNT(*) FROM demo.gold.rossmann_serving_mart
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.gold.rossmann_dashboard_kpi_mart', COUNT(*) FROM demo.gold.rossmann_dashboard_kpi_mart
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.gold.nyc_demand_training_mart', COUNT(*) FROM demo.gold.nyc_demand_training_mart
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.gold.nyc_serving_latest_zone_hour_mart', COUNT(*) FROM demo.gold.nyc_serving_latest_zone_hour_mart
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.gold.nyc_dashboard_kpi_mart', COUNT(*) FROM demo.gold.nyc_dashboard_kpi_mart
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.gold.contract_registry_mart', COUNT(*) FROM demo.gold.contract_registry_mart;
# MAGIC
# MAGIC SELECT * FROM demo.audit.gold_snapshot ORDER BY snapshot_ts DESC, table_name;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Fast smoke checks for date ranges and basic numeric sanity.
# MAGIC SELECT
# MAGIC   'rossmann_training_mart' AS dataset,
# MAGIC   COUNT(*) AS rows,
# MAGIC   MIN(business_date) AS min_date,
# MAGIC   MAX(business_date) AS max_date
# MAGIC FROM demo.gold.rossmann_training_mart
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'nyc_demand_training_mart' AS dataset,
# MAGIC   COUNT(*) AS rows,
# MAGIC   MIN(pickup_date) AS min_date,
# MAGIC   MAX(pickup_date) AS max_date
# MAGIC FROM demo.gold.nyc_demand_training_mart;

# COMMAND ----------

# MAGIC %md
# MAGIC Completed. Recommended next step:
# MAGIC - `03_ml_training_and_registry.py` to train baseline models and track versions;
# MAGIC - `04_serving_contracts.py` to implement inference-ready outputs for API and dashboards.
