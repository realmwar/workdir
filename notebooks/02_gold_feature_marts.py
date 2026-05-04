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

# MAGIC %md
# MAGIC Implements a **conditional execution gate** for the gold layer transformation pipeline. 
# MAGIC The core idea is **parameter-driven orchestration with early exit**:
# MAGIC
# MAGIC **Parameter Setup:**
# MAGIC
# MAGIC - Creates a dropdown widget `rebuild_all` with options **"true"** and "false" (defaulting to **"true"**)
# MAGIC - Reads the widget value and converts it to a boolean `REBUILD_ALL`
# MAGIC - Sets the catalog name to `demo` for all downstream queries
# MAGIC
# MAGIC **Conditional Execution Logic:**
# MAGIC
# MAGIC - If `rebuild_all` is set to **"false"**, the notebook immediately exits with a message, skipping all gold table creation
# MAGIC - If **"true"**, the notebook continues to execute the full gold layer pipeline
# MAGIC
# MAGIC **Output Result:**
# MAGIC
# MAGIC - Printed `catalog: demo` and `rebuild_all: True`
# MAGIC - Since `REBUILD_ALL` was `True`, the notebook **did NOT exit early** — it proceeded to build all 7 gold marts (training, serving, and dashboard tables for both Rossmann and NYC tracks, plus the contract registry)
# MAGIC
# MAGIC This is a **deployment efficiency pattern** — in production, you can skip expensive gold layer rebuilds when the underlying silver data hasn't changed, or when you only want to refresh specific layers. 
# MAGIC
# MAGIC The parameter provides runtime control without modifying code, making the notebook suitable for scheduled jobs where you might conditionally rebuild based on upstream data freshness checks.

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

# MAGIC %md
# MAGIC Establishes the foundational infrastructure for the gold layer. The core idea is schema initialization + audit framework setup:
# MAGIC
# MAGIC **Schema Infrastructure:**
# MAGIC
# MAGIC - Creates the `demo` catalog if it doesn't exist (ensuring top-level namespace is available)
# MAGIC - Creates `demo.gold` schema to hold business-ready, production-facing tables (training marts, serving marts, dashboard KPIs)
# MAGIC - Creates `demo.audit` schema for operational observability infrastructure
# MAGIC
# MAGIC **Audit Framework:**
# MAGIC
# MAGIC - Creates `demo.audit.gold_snapshot` table with three columns: `snapshot_ts`, `table_name`, `row_count`
# MAGIC - This table will capture point-in-time row counts for all gold tables after each pipeline run
# MAGIC - **Enables regression detection** — if future runs show unexpected row count changes, you can immediately identify which gold table regressed
# MAGIC
# MAGIC **Output:** Executed successfully with no visible output (standard for DDL statements). The schemas and audit table now exist and are ready for use.
# MAGIC
# MAGIC This is an **idempotent initialization pattern** — using `IF NOT EXISTS` ensures the cell can be run repeatedly without errors, making it safe for scheduled jobs and incremental development. 
# MAGIC
# MAGIC The audit table created here will be populated by cell 19, which inserts row count snapshots for all 7 gold marts after they're created.

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

# MAGIC %md
# MAGIC Creates the Rossmann training mart, which is the **ML training contract** for the retail forecasting model. The core idea is **feature-label packaging with explicit target naming**:
# MAGIC
# MAGIC **Feature Selection:**
# MAGIC
# MAGIC - Pulls all engineered features from the silver feature base including:
# MAGIC   - Store attributes (`store_type`, `assortment`, competition distance)
# MAGIC   - Calendar features (`day_of_week`, `is_weekend`)
# MAGIC   - Business flags (`is_open`, `is_promo`, `promo2` status)
# MAGIC   - **Temporal features** (`lag_sales_1d`, `lag_sales_7d`, rolling averages, promo history)
# MAGIC   - Customer metrics
# MAGIC
# MAGIC **Label Naming Convention:**
# MAGIC
# MAGIC - Explicitly renames target variables as `label_sales` and `label_log1p_sales`
# MAGIC - This naming convention signals to downstream ML engineers: "these are prediction targets, not input features"
# MAGIC - Provides two target options: raw sales (for interpretability) and log-transformed sales (for handling skewness)
# MAGIC
# MAGIC **Data Quality Gates:**
# MAGIC
# MAGIC - Enforces NOT NULL constraints on business keys (`store_id`, `business_date`)
# MAGIC - Ensures labels exist and are valid (sales `>= 0`)
# MAGIC - These filters guarantee the training data is complete and usable
# MAGIC
# MAGIC **Output:** Successfully created `demo.gold.rossmann_training_mart`. This is a **stable contract table** — its schema defines what the ML model expects during training. Any feature engineering changes in silver must maintain this contract to avoid breaking downstream model training pipelines.
# MAGIC
# MAGIC This is the **authoritative training dataset** ready for model consumption — all features are present with proper temporal windowing, and the explicit label naming prevents accidental data leakage during feature selection.
# MAGIC
# MAGIC
# MAGIC
# MAGIC

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

# MAGIC %md
# MAGIC Creates the Rossmann serving mart, which defines the **ML inference contract** for production predictions. The core idea is **reduced feature set for real-time scoring without data leakage**:
# MAGIC
# MAGIC **Feature Selection (Mirrors Training Subset):**
# MAGIC
# MAGIC - Includes the same **static and business logic features** as training:
# MAGIC   - Store attributes (store_type, assortment, competition distance)
# MAGIC   - Calendar features (day_of_week, is_weekend)
# MAGIC   - Business flags (is_open, is_promo, promo2 status)
# MAGIC - Preserves `prediction_id` for mapping predictions back to original test/inference rows
# MAGIC
# MAGIC **Critical Omissions (Train-Serve Gap):**
# MAGIC
# MAGIC - **Excludes ALL temporal lag features** that were in the training mart:
# MAGIC   - No `lag_sales_1d`, `lag_sales_7d`
# MAGIC   - No `rolling_sales_mean_7d`, `rolling_customers_mean_7d`
# MAGIC   - No `promo_days_last_14d`
# MAGIC - **Excludes ALL label/target columns** (`sales`, `customers`, `avg_ticket`)
# MAGIC
# MAGIC **Why This Design?** At serving time, you can't compute "yesterday's sales" for a prediction about tomorrow — that creates a **train-serve mismatch**. This contract enforces that models must either:
# MAGIC
# MAGIC - Be trained on only the features available at inference time (the reduced set here), OR
# MAGIC - Use a separate strategy to populate lag features during serving (e.g., using the model's own recent predictions as pseudo-lags)
# MAGIC
# MAGIC Output: Successfully created `demo.gold.rossmann_serving_mart`. This is a **production-ready serving contract** that ensures models won't request unavailable features during real-time inference, preventing runtime errors and maintaining prediction quality.

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

# MAGIC %md
# MAGIC Creates the Rossmann dashboard KPI mart, which aggregates store-level data to **chain-wide daily metrics** for business intelligence. The core idea is **aggregation for executive visibility + trend signals**:
# MAGIC
# MAGIC **Granularity Shift:**
# MAGIC
# MAGIC - Transforms from store-day granularity (1M+ rows) to **daily chain-wide aggregates** (~900 days)
# MAGIC - Optimizes for dashboard performance — fewer rows, pre-computed KPIs
# MAGIC
# MAGIC **Business KPIs Computed:**
# MAGIC
# MAGIC - **Operational metrics:** active store count, total sales, total customers, average ticket
# MAGIC - **Behavioral ratios:** open store ratio, promo participation ratio, weekend ratio
# MAGIC - These ratios enable trend analysis — "Are we running more promos this quarter?" or "Is store closure impacting revenue?"
# MAGIC
# MAGIC **Time-Series Indicators:**
# MAGIC
# MAGIC - **Lag features:** yesterday's sales, same day last week's sales (for day-over-day and week-over-week comparisons)
# MAGIC - **Rolling averages:** 7-day rolling sales average (smooths daily volatility for trend detection)
# MAGIC - These enable dashboard visualizations like "Current vs 7-day average" or "% change from last week"
# MAGIC
# MAGIC **Output:** Successfully created `demo.gold.rossmann_dashboard_kpi_mart` with aggregated daily KPIs. This table powers executive dashboards showing chain-wide performance without exposing store-level details, providing a high-level operational view suitable for leadership and business analysts.
# MAGIC
# MAGIC This is **BI-optimized** — dashboards query ~900 daily rows instead of 1M+ store-day records, enabling fast refresh rates and responsive visualizations while still preserving enough temporal context for trend analysis.

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

# MAGIC %md
# MAGIC Creates the NYC demand training mart, which is the **ML training contract** for urban taxi demand forecasting. The core idea is **feature-label packaging for zone-hour demand prediction**:
# MAGIC
# MAGIC **Feature Selection:**
# MAGIC
# MAGIC - Pulls all engineered features from the silver zone-hour feature-base including:
# MAGIC   - **Temporal keys:** `pickup_date`, `pickup_hour`, `pu_location_id` (defines the prediction granularity)
# MAGIC   - **Aggregate metrics:** average trip distance, duration, fare amount, tip percentage, total revenue
# MAGIC   - **Autoregressive features:** `lag_trip_cnt_1h` (1 hour ago), `lag_trip_cnt_24h` (same hour yesterday)
# MAGIC   - **Rolling indicators:** 24-hour rolling averages for trip count and fare (short-term trend signals)
# MAGIC
# MAGIC **Label Naming Convention:**
# MAGIC
# MAGIC - Explicitly renames `trip_cnt` as `label_trip_cnt`
# MAGIC - This signals to ML engineers: "this is the prediction target" — forecasting how many trips will occur in each zone-hour
# MAGIC - The label represents demand intensity at zone-hour granularity
# MAGIC
# MAGIC **Data Quality Gates:**
# MAGIC
# MAGIC - Enforces NOT NULL constraints on temporal keys (`pickup_date`, `pu_location_id`)
# MAGIC - Ensures label exists and is valid (`trip_cnt >= 0`)
# MAGIC - Guarantees complete, usable training observations
# MAGIC
# MAGIC **Output:** Successfully created `demo.gold.nyc_demand_training_mart`. 
# MAGIC This is a stable contract table for training demand forecasting models that predict trip volume per zone per hour. 
# MAGIC Unlike the Rossmann mart which operates at daily store-level granularity, this operates at hourly zone-level granularity for real-time operational planning (driver dispatch, surge pricing).
# MAGIC
# MAGIC This is the **authoritative NYC training dataset** — all temporal features are properly windowed to prevent data leakage, enabling models to learn hourly demand patterns, zone-specific characteristics, and short-term momentum effects.
# MAGIC
# MAGIC
# MAGIC
# MAGIC

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

# MAGIC %md
# MAGIC Creates the NYC serving mart by extracting the **most recent observation for each zone**. The core idea is **latest-state serving contract for real-time next-hour demand prediction**:
# MAGIC
# MAGIC **Latest State Extraction:**
# MAGIC
# MAGIC - Uses `ROW_NUMBER()` window function partitioned by `pu_location_id` (pickup zone)
# MAGIC - Orders by `pickup_date DESC`, `pickup_hour DESC` to identify the most recent hour for each zone
# MAGIC - Filters to `rn = 1` to keep only the latest observation per zone
# MAGIC
# MAGIC **Feature Selection for Real-Time Inference:**
# MAGIC
# MAGIC - Includes temporal context features that represent the **latest known state**: `lag_trip_cnt_1h`, `lag_trip_cnt_24h`, rolling averages
# MAGIC - These lag features are valid because they're computed from historical data up to the latest available hour
# MAGIC - **Excludes the label** (`label_trip_cnt`) since this is for inference, not training
# MAGIC - Preserves average metrics (distance, duration, fare, tip) from the latest hour
# MAGIC
# MAGIC **Use Case - Next-Hour Forecasting:** Unlike the Rossmann serving mart which excluded temporal features entirely (cell 12), this mart includes lags because they're essential for short-term demand prediction.
# MAGIC
# MAGIC When predicting "next hour's demand at zone X," you need to know what happened 1 hour ago and 24 hours ago at that zone.
# MAGIC
# MAGIC **Output:** Successfully created `demo.gold.nyc_serving_latest_zone_hour_mart` with one row per pickup zone, each containing the latest available features. 
# MAGIC This is a **snapshot table** for real-time serving — API endpoints query this table to get current zone state before making next-hour predictions, ensuring models have the freshest temporal context.

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

# MAGIC %md
# MAGIC Creates the NYC dashboard KPI mart, which aggregates zone-level data to **city-wide hourly metrics** for business intelligence. The core idea is **hourly granularity aggregation + short-term trend signals**:
# MAGIC
# MAGIC **Granularity Shift:**
# MAGIC
# MAGIC - Transforms from zone-hour granularity (67K rows) to city-wide hourly aggregates (~2,600 hours across 3+ months)
# MAGIC - Optimizes for dashboard performance — one row per hour instead of hundreds of zone-hour observations
# MAGIC - Aggregates across all pickup zones to provide system-wide operational visibility
# MAGIC
# MAGIC **Business KPIs Computed:**
# MAGIC
# MAGIC - **Operational metrics:** total trip count, total revenue, average fare, average tip percentage (city-wide)
# MAGIC - These enable executive-level questions like "What's the hourly demand pattern citywide?" or "Are average fares increasing during peak hours?"
# MAGIC
# MAGIC **Temporal Feature Engineering:**
# MAGIC
# MAGIC - Constructs `city_hour_ts` timestamp by combining `pickup_date` and `pickup_hour` (enables proper window function ordering)
# MAGIC - **Lag features:** trip count 1 hour ago and 24 hours ago (for hour-over-hour and day-over-day comparisons)
# MAGIC - **Rolling averages:** 24-hour rolling mean for trip count (smooths hourly volatility, reveals demand trends)
# MAGIC
# MAGIC **Output:** Successfully created `demo.gold.nyc_dashboard_kpi_mart` with city-wide hourly KPIs. 
# MAGIC This table powers operational dashboards showing system-level demand patterns, enabling real-time operational decisions (fleet sizing, surge pricing zones) and trend analysis without exposing zone-level granularity.
# MAGIC
# MAGIC This is **BI-optimized for hourly operations** — unlike the Rossmann dashboard which operates at daily granularity for retail planning, this operates at **hourly granularity** for real-time urban mobility management, reflecting the faster decision cycles in taxi dispatch operations.

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

# MAGIC %md
# MAGIC Captures a comprehensive audit snapshot of all gold layer tables for production monitoring. The core idea is **complete gold layer observability + regression detection**:
# MAGIC
# MAGIC **Audit Snapshot Creation:**
# MAGIC
# MAGIC - Inserts row counts for all 7 gold tables into demo.audit.gold_snapshot with a single timestamp
# MAGIC - Uses UNION ALL to batch-insert all metrics in one transaction
# MAGIC - Covers the complete gold layer: 3 Rossmann marts, 3 NYC marts, plus the contract registry
# MAGIC
# MAGIC **Immediate Validation:**
# MAGIC
# MAGIC - Displays the complete audit history ordered by timestamp and table name
# MAGIC - Enables instant comparison with previous runs
# MAGIC
# MAGIC **Output Results at `2026-03-10 08:50:27`:**
# MAGIC
# MAGIC - **Rossmann track:** Training (1,017,209 rows), Serving (41,088 rows), Dashboard (942 daily aggregates)
# MAGIC - **NYC track:** Training (67,019 zone-hours), Serving (254 zones - one per location), Dashboard (751 hourly city-wide aggregates)
# MAGIC - **Contract registry:** 6 routing entries
# MAGIC
# MAGIC **Production Value:** This enables **multi-layer regression detection** — by comparing row counts across pipeline runs, you can immediately detect:
# MAGIC
# MAGIC Upstream data loss (training mart row drop indicates silver layer issues)
# MAGIC Serving contract staleness (serving mart unchanged suggests no new data)
# MAGIC Aggregation problems (dashboard mart row count mismatch indicates transformation logic bugs)
# MAGIC
# MAGIC This is the **final checkpoint** before handing off to downstream ML training and API serving — ensuring all 7 gold contracts are populated and ready for consumption.

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

# MAGIC %md
# MAGIC Performs final validation on the gold training marts. The core idea is **gold layer sanity check before ML handoff**:
# MAGIC
# MAGIC **Validation Design:**
# MAGIC
# MAGIC - Queries both gold training marts (Rossmann and NYC demand)
# MAGIC - Computes three critical metrics per dataset: row count, earliest date, latest date
# MAGIC - Uses UNION ALL for side-by-side comparison in a single result set
# MAGIC
# MAGIC **What This Validates:**
# MAGIC
# MAGIC - **Row counts** match the audit snapshot in cell 26 (ensures no silent data loss during gold transformations)
# MAGIC - **Date ranges** verify temporal coverage was preserved from silver to gold
# MAGIC - Provides a quick "go/no-go" check before handing off to ML training
# MAGIC
# MAGIC **Output Results:**
# MAGIC
# MAGIC - **Rossmann training mart:** `1,017,209` rows spanning **2.5 years** (Jan 2013 - July 2015) — substantial history for seasonality modeling
# MAGIC - **NYC demand training mart:** `67,019` rows spanning **3+ months** (Oct 2022 - Feb 2023) — sufficient hourly observations for demand patterns
# MAGIC
# MAGIC This is a **pre-ML checkpoint** that validates the gold transformation pipeline preserved data integrity. 
# MAGIC
# MAGIC If row counts mismatched the audit snapshot or date ranges were unexpectedly truncated, it would signal a transformation bug requiring investigation. 
# MAGIC
# MAGIC The check confirms that both training marts are complete and ready for model training in the next notebook (`03_ml_training_and_registry.py`).

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
