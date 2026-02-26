# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Silver Transformations (Rossmann + NYC TLC)
# MAGIC
# MAGIC Goals:
# MAGIC - build clean/enriched silver tables on top of bronze inputs;
# MAGIC - create feature-base tables for future ML training and serving;
# MAGIC - persist quality snapshots for silver-layer observability.
# MAGIC
# MAGIC Constraints:
# MAGIC - no Spark DataFrame API;
# MAGIC - SQL for transformations, Python only for lightweight orchestration.

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
    dbutils.notebook.exit("rebuild_all=false -> silver rebuild skipped by parameter")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Silver prep: schema and quality table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ensure core schemas exist and create a simple audit sink for silver row-count snapshots.
# MAGIC CREATE CATALOG IF NOT EXISTS demo;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.audit;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS demo.audit.silver_snapshot (
# MAGIC   snapshot_ts TIMESTAMP,
# MAGIC   table_name STRING,
# MAGIC   row_count BIGINT
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Rossmann: clean and enrich (silver)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Train table cleanup:
# MAGIC -- 1) robust type casting from raw CSV values,
# MAGIC -- 2) basic null/outlier filtering for target column,
# MAGIC -- 3) lightweight calendar and business-derived features.
# MAGIC CREATE OR REPLACE TABLE demo.silver.rossmann_train_clean AS
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     TRY_CAST(Store AS INT) AS store_id,
# MAGIC     TRY_CAST(DayOfWeek AS INT) AS day_of_week,
# MAGIC     TO_DATE(Date) AS business_date,
# MAGIC     TRY_CAST(Sales AS DOUBLE) AS sales,
# MAGIC     TRY_CAST(Customers AS DOUBLE) AS customers,
# MAGIC     COALESCE(TRY_CAST(Open AS INT), 1) AS is_open,
# MAGIC     COALESCE(TRY_CAST(Promo AS INT), 0) AS is_promo,
# MAGIC     CASE
# MAGIC       WHEN StateHoliday IS NULL OR StateHoliday IN ('0', '0.0', '') THEN '0'
# MAGIC       ELSE CAST(StateHoliday AS STRING)
# MAGIC     END AS state_holiday_code,
# MAGIC     COALESCE(TRY_CAST(SchoolHoliday AS INT), 0) AS is_school_holiday,
# MAGIC     ingest_ts,
# MAGIC     source_file
# MAGIC   FROM demo.bronze.rossmann_train
# MAGIC )
# MAGIC SELECT
# MAGIC   *,
# MAGIC   YEAR(business_date) AS year_num,
# MAGIC   MONTH(business_date) AS month_num,
# MAGIC   WEEKOFYEAR(business_date) AS week_num,
# MAGIC   DAYOFMONTH(business_date) AS day_num,
# MAGIC   CASE WHEN day_of_week IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
# MAGIC   CASE WHEN customers > 0 THEN sales / customers END AS avg_ticket,
# MAGIC   CASE WHEN sales >= 0 THEN LOG(1 + sales) END AS log1p_sales,
# MAGIC   current_timestamp() AS silver_loaded_at
# MAGIC FROM base
# MAGIC WHERE store_id IS NOT NULL
# MAGIC   AND business_date IS NOT NULL
# MAGIC   AND sales IS NOT NULL
# MAGIC   AND sales >= 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test table cleanup mirrors train cleanup, excluding target columns.
# MAGIC -- prediction_id is preserved for serving/inference joins later.
# MAGIC CREATE OR REPLACE TABLE demo.silver.rossmann_test_clean AS
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     TRY_CAST(Id AS BIGINT) AS prediction_id,
# MAGIC     TRY_CAST(Store AS INT) AS store_id,
# MAGIC     TRY_CAST(DayOfWeek AS INT) AS day_of_week,
# MAGIC     TO_DATE(Date) AS business_date,
# MAGIC     COALESCE(TRY_CAST(Open AS INT), 1) AS is_open,
# MAGIC     COALESCE(TRY_CAST(Promo AS INT), 0) AS is_promo,
# MAGIC     CASE
# MAGIC       WHEN StateHoliday IS NULL OR StateHoliday IN ('0', '0.0', '') THEN '0'
# MAGIC       ELSE CAST(StateHoliday AS STRING)
# MAGIC     END AS state_holiday_code,
# MAGIC     COALESCE(TRY_CAST(SchoolHoliday AS INT), 0) AS is_school_holiday,
# MAGIC     ingest_ts,
# MAGIC     source_file
# MAGIC   FROM demo.bronze.rossmann_test
# MAGIC )
# MAGIC SELECT
# MAGIC   *,
# MAGIC   YEAR(business_date) AS year_num,
# MAGIC   MONTH(business_date) AS month_num,
# MAGIC   WEEKOFYEAR(business_date) AS week_num,
# MAGIC   DAYOFMONTH(business_date) AS day_num,
# MAGIC   CASE WHEN day_of_week IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
# MAGIC   current_timestamp() AS silver_loaded_at
# MAGIC FROM base
# MAGIC WHERE store_id IS NOT NULL
# MAGIC   AND business_date IS NOT NULL;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Store dimension cleanup and median imputation for competition distance.
# MAGIC -- Median is used as a stable central tendency for missing competition values.
# MAGIC CREATE OR REPLACE TABLE demo.silver.rossmann_store_clean AS
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     TRY_CAST(Store AS INT) AS store_id,
# MAGIC     CAST(StoreType AS STRING) AS store_type,
# MAGIC     CAST(Assortment AS STRING) AS assortment_type,
# MAGIC     TRY_CAST(CompetitionDistance AS DOUBLE) AS competition_distance,
# MAGIC     TRY_CAST(CompetitionOpenSinceMonth AS INT) AS competition_open_since_month,
# MAGIC     TRY_CAST(CompetitionOpenSinceYear AS INT) AS competition_open_since_year,
# MAGIC     COALESCE(TRY_CAST(Promo2 AS INT), 0) AS has_promo2,
# MAGIC     TRY_CAST(Promo2SinceWeek AS INT) AS promo2_since_week,
# MAGIC     TRY_CAST(Promo2SinceYear AS INT) AS promo2_since_year,
# MAGIC     CAST(PromoInterval AS STRING) AS promo_interval_raw,
# MAGIC     ingest_ts,
# MAGIC     source_file
# MAGIC   FROM demo.bronze.rossmann_store
# MAGIC ),
# MAGIC stats AS (
# MAGIC   SELECT percentile_approx(competition_distance, 0.5) AS competition_distance_median
# MAGIC   FROM base
# MAGIC   WHERE competition_distance IS NOT NULL
# MAGIC )
# MAGIC SELECT
# MAGIC   b.store_id,
# MAGIC   b.store_type,
# MAGIC   b.assortment_type,
# MAGIC   COALESCE(b.competition_distance, s.competition_distance_median) AS competition_distance_km,
# MAGIC   b.competition_open_since_month,
# MAGIC   b.competition_open_since_year,
# MAGIC   b.has_promo2,
# MAGIC   b.promo2_since_week,
# MAGIC   b.promo2_since_year,
# MAGIC   b.promo_interval_raw,
# MAGIC   b.ingest_ts,
# MAGIC   b.source_file,
# MAGIC   current_timestamp() AS silver_loaded_at
# MAGIC FROM base b
# MAGIC CROSS JOIN stats s
# MAGIC WHERE b.store_id IS NOT NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Rossmann: feature-base for future model training/serving

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Feature base for training:
# MAGIC -- joins transactional train rows with static store attributes,
# MAGIC -- then builds promo flags + lag/rolling window features.
# MAGIC CREATE OR REPLACE TABLE demo.silver.rossmann_train_feature_base AS
# MAGIC WITH joined AS (
# MAGIC   SELECT
# MAGIC     t.store_id,
# MAGIC     t.business_date,
# MAGIC     t.day_of_week,
# MAGIC     t.is_weekend,
# MAGIC     t.is_open,
# MAGIC     t.is_promo,
# MAGIC     t.state_holiday_code,
# MAGIC     t.is_school_holiday,
# MAGIC     t.sales,
# MAGIC     t.customers,
# MAGIC     t.avg_ticket,
# MAGIC     t.log1p_sales,
# MAGIC     s.store_type,
# MAGIC     s.assortment_type,
# MAGIC     s.competition_distance_km,
# MAGIC     s.has_promo2,
# MAGIC     s.promo2_since_week,
# MAGIC     s.promo2_since_year,
# MAGIC     s.promo_interval_raw
# MAGIC   FROM demo.silver.rossmann_train_clean t
# MAGIC   LEFT JOIN demo.silver.rossmann_store_clean s
# MAGIC     ON t.store_id = s.store_id
# MAGIC ),
# MAGIC feature_enriched AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     CASE
# MAGIC       -- Promo2 activation depends on when promo2 started for this store.
# MAGIC       WHEN has_promo2 = 1
# MAGIC            AND promo2_since_year IS NOT NULL
# MAGIC            AND promo2_since_week IS NOT NULL
# MAGIC            AND (
# MAGIC              YEAR(business_date) > promo2_since_year
# MAGIC              OR (YEAR(business_date) = promo2_since_year AND WEEKOFYEAR(business_date) >= promo2_since_week)
# MAGIC            )
# MAGIC       THEN 1 ELSE 0
# MAGIC     END AS is_promo2_active,
# MAGIC     CASE
# MAGIC       -- If current month appears in PromoInterval (Jan/Feb/... strings), mark active month.
# MAGIC       WHEN promo_interval_raw IS NULL THEN 0
# MAGIC       WHEN ARRAY_CONTAINS(SPLIT(REPLACE(promo_interval_raw, ' ', ''), ','), DATE_FORMAT(business_date, 'MMM')) THEN 1
# MAGIC       ELSE 0
# MAGIC     END AS is_promo_interval_month,
# MAGIC     -- Temporal features for autoregressive signal and short-term trend.
# MAGIC     LAG(sales, 1) OVER (PARTITION BY store_id ORDER BY business_date) AS lag_sales_1d,
# MAGIC     LAG(sales, 7) OVER (PARTITION BY store_id ORDER BY business_date) AS lag_sales_7d,
# MAGIC     AVG(sales) OVER (
# MAGIC       PARTITION BY store_id
# MAGIC       ORDER BY business_date
# MAGIC       ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
# MAGIC     ) AS rolling_sales_mean_7d,
# MAGIC     AVG(customers) OVER (
# MAGIC       PARTITION BY store_id
# MAGIC       ORDER BY business_date
# MAGIC       ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
# MAGIC     ) AS rolling_customers_mean_7d,
# MAGIC     SUM(is_promo) OVER (
# MAGIC       PARTITION BY store_id
# MAGIC       ORDER BY business_date
# MAGIC       ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
# MAGIC     ) AS promo_days_last_14d
# MAGIC   FROM joined
# MAGIC )
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS silver_loaded_at
# MAGIC FROM feature_enriched;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Feature base for inference:
# MAGIC -- uses test rows + store attributes; excludes target-derived lag features by design.
# MAGIC CREATE OR REPLACE TABLE demo.silver.rossmann_test_feature_base AS
# MAGIC SELECT
# MAGIC   t.prediction_id,
# MAGIC   t.store_id,
# MAGIC   t.business_date,
# MAGIC   t.day_of_week,
# MAGIC   t.is_weekend,
# MAGIC   t.is_open,
# MAGIC   t.is_promo,
# MAGIC   t.state_holiday_code,
# MAGIC   t.is_school_holiday,
# MAGIC   s.store_type,
# MAGIC   s.assortment_type,
# MAGIC   s.competition_distance_km,
# MAGIC   s.has_promo2,
# MAGIC   CASE
# MAGIC     WHEN s.has_promo2 = 1
# MAGIC          AND s.promo2_since_year IS NOT NULL
# MAGIC          AND s.promo2_since_week IS NOT NULL
# MAGIC          AND (
# MAGIC            YEAR(t.business_date) > s.promo2_since_year
# MAGIC            OR (YEAR(t.business_date) = s.promo2_since_year AND WEEKOFYEAR(t.business_date) >= s.promo2_since_week)
# MAGIC          )
# MAGIC     THEN 1 ELSE 0
# MAGIC   END AS is_promo2_active,
# MAGIC   CASE
# MAGIC     WHEN s.promo_interval_raw IS NULL THEN 0
# MAGIC     WHEN ARRAY_CONTAINS(SPLIT(REPLACE(s.promo_interval_raw, ' ', ''), ','), DATE_FORMAT(t.business_date, 'MMM')) THEN 1
# MAGIC     ELSE 0
# MAGIC   END AS is_promo_interval_month,
# MAGIC   current_timestamp() AS silver_loaded_at
# MAGIC FROM demo.silver.rossmann_test_clean t
# MAGIC LEFT JOIN demo.silver.rossmann_store_clean s
# MAGIC   ON t.store_id = s.store_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) NYC TLC: clean and feature-base

# COMMAND ----------

# MAGIC %sql
# MAGIC -- NYC trip cleanup:
# MAGIC -- strict casting + range filters to remove invalid trips and extreme anomalies.
# MAGIC CREATE OR REPLACE TABLE demo.silver.nyc_taxi_trips_clean AS
# MAGIC WITH casted AS (
# MAGIC   SELECT
# MAGIC     TRY_CAST(VendorID AS INT) AS vendor_id,
# MAGIC     TRY_CAST(tpep_pickup_datetime AS TIMESTAMP) AS pickup_ts,
# MAGIC     TRY_CAST(tpep_dropoff_datetime AS TIMESTAMP) AS dropoff_ts,
# MAGIC     TRY_CAST(passenger_count AS DOUBLE) AS passenger_count,
# MAGIC     TRY_CAST(trip_distance AS DOUBLE) AS trip_distance_miles,
# MAGIC     TRY_CAST(RatecodeID AS INT) AS rate_code_id,
# MAGIC     CAST(store_and_fwd_flag AS STRING) AS store_and_fwd_flag,
# MAGIC     TRY_CAST(PULocationID AS INT) AS pu_location_id,
# MAGIC     TRY_CAST(DOLocationID AS INT) AS do_location_id,
# MAGIC     TRY_CAST(payment_type AS INT) AS payment_type,
# MAGIC     TRY_CAST(fare_amount AS DOUBLE) AS fare_amount,
# MAGIC     TRY_CAST(tip_amount AS DOUBLE) AS tip_amount,
# MAGIC     TRY_CAST(total_amount AS DOUBLE) AS total_amount,
# MAGIC     TRY_CAST(tolls_amount AS DOUBLE) AS tolls_amount,
# MAGIC     TRY_CAST(congestion_surcharge AS DOUBLE) AS congestion_surcharge,
# MAGIC     ingest_ts,
# MAGIC     source_file
# MAGIC   FROM demo.bronze.nyc_taxi_trips
# MAGIC ),
# MAGIC cleaned AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     TIMESTAMPDIFF(MINUTE, pickup_ts, dropoff_ts) AS trip_duration_min
# MAGIC   FROM casted
# MAGIC   WHERE pickup_ts IS NOT NULL
# MAGIC     AND dropoff_ts IS NOT NULL
# MAGIC     AND pickup_ts < dropoff_ts
# MAGIC     AND trip_distance_miles IS NOT NULL
# MAGIC     AND trip_distance_miles > 0
# MAGIC     AND trip_distance_miles < 100
# MAGIC     AND fare_amount IS NOT NULL
# MAGIC     AND fare_amount >= 0
# MAGIC     AND fare_amount < 1000
# MAGIC     AND pu_location_id IS NOT NULL
# MAGIC     AND do_location_id IS NOT NULL
# MAGIC )
# MAGIC SELECT
# MAGIC   vendor_id,
# MAGIC   pickup_ts,
# MAGIC   dropoff_ts,
# MAGIC   DATE(pickup_ts) AS pickup_date,
# MAGIC   HOUR(pickup_ts) AS pickup_hour,
# MAGIC   DAYOFWEEK(pickup_ts) AS pickup_day_of_week,
# MAGIC   MONTH(pickup_ts) AS pickup_month,
# MAGIC   CASE WHEN DAYOFWEEK(pickup_ts) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
# MAGIC   -- Rush hour indicator for demand and fare-pressure modeling.
# MAGIC   CASE WHEN HOUR(pickup_ts) IN (7, 8, 9, 16, 17, 18, 19) THEN 1 ELSE 0 END AS is_rush_hour,
# MAGIC   passenger_count,
# MAGIC   trip_distance_miles,
# MAGIC   trip_duration_min,
# MAGIC   -- Approximate average speed helps detect unusual trip patterns.
# MAGIC   CASE WHEN trip_duration_min > 0 THEN (trip_distance_miles / trip_duration_min) * 60 END AS speed_mph,
# MAGIC   pu_location_id,
# MAGIC   do_location_id,
# MAGIC   payment_type,
# MAGIC   fare_amount,
# MAGIC   tip_amount,
# MAGIC   total_amount,
# MAGIC   tolls_amount,
# MAGIC   congestion_surcharge,
# MAGIC   -- Tip ratio is a useful behavioral and revenue-related feature.
# MAGIC   CASE WHEN fare_amount > 0 THEN tip_amount / fare_amount END AS tip_pct,
# MAGIC   store_and_fwd_flag,
# MAGIC   ingest_ts,
# MAGIC   source_file,
# MAGIC   current_timestamp() AS silver_loaded_at
# MAGIC FROM cleaned
# MAGIC WHERE trip_duration_min BETWEEN 1 AND 180;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Zone-hour feature base:
# MAGIC -- aggregates at (pickup_date, pickup_hour, pickup_zone) and adds lag/rolling demand signals.
# MAGIC CREATE OR REPLACE TABLE demo.silver.nyc_zone_hour_feature_base AS
# MAGIC WITH agg AS (
# MAGIC   SELECT
# MAGIC     pickup_date,
# MAGIC     pickup_hour,
# MAGIC     pu_location_id,
# MAGIC     COUNT(*) AS trip_cnt,
# MAGIC     AVG(trip_distance_miles) AS avg_trip_distance_miles,
# MAGIC     AVG(trip_duration_min) AS avg_trip_duration_min,
# MAGIC     AVG(fare_amount) AS avg_fare_amount,
# MAGIC     AVG(tip_pct) AS avg_tip_pct,
# MAGIC     SUM(total_amount) AS total_revenue
# MAGIC   FROM demo.silver.nyc_taxi_trips_clean
# MAGIC   GROUP BY pickup_date, pickup_hour, pu_location_id
# MAGIC ),
# MAGIC ts_enriched AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     TO_TIMESTAMP(
# MAGIC       CONCAT(
# MAGIC         CAST(pickup_date AS STRING), ' ',
# MAGIC         LPAD(CAST(pickup_hour AS STRING), 2, '0'),
# MAGIC         ':00:00'
# MAGIC       )
# MAGIC     ) AS zone_hour_ts
# MAGIC   FROM agg
# MAGIC )
# MAGIC SELECT
# MAGIC   *,
# MAGIC   LAG(trip_cnt, 1) OVER (
# MAGIC     PARTITION BY pu_location_id
# MAGIC     ORDER BY zone_hour_ts
# MAGIC   ) AS lag_trip_cnt_1h,
# MAGIC   LAG(trip_cnt, 24) OVER (
# MAGIC     PARTITION BY pu_location_id
# MAGIC     ORDER BY zone_hour_ts
# MAGIC   ) AS lag_trip_cnt_24h,
# MAGIC   AVG(trip_cnt) OVER (
# MAGIC     PARTITION BY pu_location_id
# MAGIC     ORDER BY zone_hour_ts
# MAGIC     ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
# MAGIC   ) AS rolling_trip_cnt_mean_24h,
# MAGIC   AVG(avg_fare_amount) OVER (
# MAGIC     PARTITION BY pu_location_id
# MAGIC     ORDER BY zone_hour_ts
# MAGIC     ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
# MAGIC   ) AS rolling_fare_mean_24h,
# MAGIC   current_timestamp() AS silver_loaded_at
# MAGIC FROM ts_enriched;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Silver audit snapshot + quick checks

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO demo.audit.silver_snapshot
# MAGIC -- Persist row counts for operational visibility and pipeline regression checks.
# MAGIC SELECT current_timestamp(), 'demo.silver.rossmann_train_clean', COUNT(*) FROM demo.silver.rossmann_train_clean
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.silver.rossmann_store_clean', COUNT(*) FROM demo.silver.rossmann_store_clean
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.silver.rossmann_train_feature_base', COUNT(*) FROM demo.silver.rossmann_train_feature_base
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.silver.nyc_taxi_trips_clean', COUNT(*) FROM demo.silver.nyc_taxi_trips_clean
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.silver.nyc_zone_hour_feature_base', COUNT(*) FROM demo.silver.nyc_zone_hour_feature_base;
# MAGIC
# MAGIC SELECT * FROM demo.audit.silver_snapshot ORDER BY snapshot_ts DESC, table_name;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   'rossmann_train_feature_base' AS dataset,
# MAGIC   COUNT(*) AS rows,
# MAGIC   MIN(business_date) AS min_date,
# MAGIC   MAX(business_date) AS max_date
# MAGIC FROM demo.silver.rossmann_train_feature_base
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'nyc_zone_hour_feature_base' AS dataset,
# MAGIC   COUNT(*) AS rows,
# MAGIC   MIN(pickup_date) AS min_date,
# MAGIC   MAX(pickup_date) AS max_date
# MAGIC FROM demo.silver.nyc_zone_hour_feature_base;

# COMMAND ----------

# MAGIC %md
# MAGIC Completed. Recommended next step:
# MAGIC - `02_gold_feature_marts.py` (business/serving marts);
# MAGIC - `03_ml_training_and_registry.py` (baseline + versioning);
# MAGIC - `04_serving_contracts.py` (inference contracts for API/dashboard).
