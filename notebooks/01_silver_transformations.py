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

# MAGIC %md
# MAGIC Sets up runtime parameters and controls notebook execution. It:
# MAGIC
# MAGIC - Creates a dropdown widget named `rebuild_all` with options **"true"** and **"false"** (defaulting to **"true"**).
# MAGIC - Reads the widget value and converts it to a boolean variable `REBUILD_ALL`.
# MAGIC - Sets the catalog name to `demo` for all downstream queries.
# MAGIC - Prints the configuration values for visibility.
# MAGIC - Implements an early exit: if `rebuild_all` is set to "false", the notebook exits immediately with a message, skipping all silver layer transformations.
# MAGIC
# MAGIC This provides a parameter-driven way to control whether the entire silver transformation pipeline runs or is skipped, useful for scheduled jobs where you may want conditional execution based on data freshness or other logic.

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

# MAGIC %md
# MAGIC Sets up the core schema infrastructure for silver layer processing. It:
# MAGIC
# MAGIC - Creates the `demo` catalog if it doesn't exist.
# MAGIC - Creates the `demo.silver` schema to hold cleaned and enriched silver tables.
# MAGIC - Creates the `demo.audit` schema for operational observability.
# MAGIC - Creates the `demo.audit.silver_snapshot` table with columns for `snapshot_ts`, `table_name`, and `row_count` to track row counts over time.
# MAGIC
# MAGIC This is foundational setup that ensures all downstream silver transformations have the proper catalog structure and audit tracking table in place. The silver_snapshot table will later be used to record row counts for each silver table as a pipeline regression check.

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

# MAGIC %md
# MAGIC Transforms raw Rossmann store sales data from the bronze layer into a clean, ML-ready silver table. 
# MAGIC The core idea is **data quality** + **feature engineering for forecasting**:
# MAGIC
# MAGIC **Data Quality Layer:**
# MAGIC
# MAGIC - Safely converts CSV strings to proper types (INT, DOUBLE, DATE) using `TRY_CAST` to avoid pipeline breaks.
# MAGIC - Applies smart defaults for missing flags (assumes stores are open, no promo if blank).
# MAGIC - Filters out unusable rows (missing store/date/sales, negative sales outliers).
# MAGIC
# MAGIC **Feature Engineering Layer:**
# MAGIC
# MAGIC - Creates calendar features (year, month, week, day) for seasonality modeling.
# MAGIC - Builds business logic features (weekend flag, average ticket size, log-transformed sales for variance stabilization).
# MAGIC - Preserves audit lineage (original ingest timestamp and source file).
# MAGIC
# MAGIC **Output:** The execution created `demo.silver.rossmann_train_clean` successfully. This table now contains validated, feature-enriched training data ready for the next step where it gets joined with store attributes and time-series lag features are added.
# MAGIC
# MAGIC The transformation is **defensive by design** — it won't crash on bad data, and it explicitly documents what constitutes "clean" through the WHERE clause filters.

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

# MAGIC %md
# MAGIC Transforms the Rossmann test dataset using the same cleaning logic as the training data (cell 10), but adapted for inference. The core idea is feature parity without targets:
# MAGIC
# MAGIC **Mirrors Training Cleanup:**
# MAGIC
# MAGIC - Same defensive type casting with `TRY_CAST` to handle CSV inconsistencies.
# MAGIC - Same null handling defaults (stores assumed open, no promo if missing).
# MAGIC - Same `state_holiday_code` normalization.
# MAGIC - Same calendar features (year, month, week, day, weekend flag).
# MAGIC
# MAGIC **Inference-Specific Adaptations:**
# MAGIC
# MAGIC - Preserves `prediction_id` (from the original `Id` column) — critical for joining predictions back to the original test rows when scoring.
# MAGIC - **Excludes all target variables** (`sales`, `customers`, `avg_ticket`, `log1p_sales`) since test data doesn't have these.
# MAGIC - No sales-based filtering — only validates that `store_id` and `business_date` exist.
# MAGIC
# MAGIC **Output:** Successfully created `demo.silver.rossmann_test_clean`, providing test data with identical feature engineering to training data, ensuring model compatibility during inference.
# MAGIC
# MAGIC This design enforces **train-serve consistency** — the model will see the same feature names, types, and transformations at serving time as it did during training, preventing schema mismatches and prediction errors.

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

# MAGIC %md
# MAGIC Cleans the Rossmann store dimension table with a focus on **statistical imputation for missing competitive intelligence**. The core idea is **dimension enrichment with robust defaults**:
# MAGIC
# MAGIC **Type Safety & Structure:**
# MAGIC
# MAGIC - Casts store attributes (`store_type`, `assortment_type`, `competition` fields, `promo2` fields) to proper types using `TRY_CAST`
# MAGIC - Preserves audit lineage (`ingest_ts`, `source_file`)
# MAGIC - Applies smart defaults (`has_promo2` defaults to 0 if missing)
# MAGIC
# MAGIC **Statistical Imputation Strategy:**
# MAGIC
# MAGIC - Calculates the **median competition distance** across all stores in a separate `stats` CTE
# MAGIC - Uses `COALESCE` to fill missing `competition_distance` values with the median
# MAGIC - **Why median?** More robust than mean when dealing with geographic outliers—a few stores with extremely distant competitors won't skew the imputation
# MAGIC
# MAGIC **Output:** Successfully created `demo.silver.rossmann_store_clean`, a dimension table ready to be joined with transactional train/test data. The imputation ensures no store is excluded from analysis due to missing competition data, while the median provides a sensible, outlier-resistant default.
# MAGIC
# MAGIC This is a classic **slowly changing dimension (Type 1)** pattern — static store attributes that enrich time-series forecasting models with business context (store type affects sales patterns, competition affects performance).

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

# MAGIC %md
# MAGIC Creates the ML-ready training feature-base by combining transactional data with store dimensions and building time-series features. The core idea is **join + temporal feature engineering for forecasting models**:
# MAGIC
# MAGIC **Data Enrichment via Join:**
# MAGIC
# MAGIC - Joins cleaned train transactions with store dimension attributes (store type, assortment, competition distance, promo2 settings)
# MAGIC - Brings static store context into each daily transaction row
# MAGIC
# MAGIC **Business Logic Features:**
# MAGIC
# MAGIC - `is_promo2_active`: Determines whether Promo2 was active on this date based on when the store started participating (year/week comparison)
# MAGIC - `is_promo_interval_month`: Checks if the current month matches the store's promotional calendar (parses comma-separated month strings like "Jan,Apr,Jul,Oct")
# MAGIC
# MAGIC **Time-Series Features for Forecasting:**
# MAGIC
# MAGIC - **Lag features:** `lag_sales_1d` (yesterday's sales) and `lag_sales_7d` (same day last week) capture autoregressive patterns
# MAGIC - **Rolling windows:** 7-day rolling averages for sales and customers capture recent trends
# MAGIC - **Promo history:** `promo_days_last_14d` counts how many promotional days occurred in the past 2 weeks, capturing promo fatigue/momentum effects
# MAGIC
# MAGIC **Output:** Successfully created `demo.silver.rossmann_train_feature_base`, a wide-format ML training table ready for model consumption. This table includes everything a forecasting model needs: static store attributes, calendar features from cell 10, promotional intelligence, and temporal autoregressive signals.
# MAGIC
# MAGIC This is the **authoritative training dataset** — all features are computed from historical data using proper time-based windowing to prevent data leakage.

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

# MAGIC %md
# MAGIC Creates the ML-ready inference feature-base for the Rossmann test dataset. The core idea is **train-serve feature consistency with intentional target leakage prevention**:
# MAGIC
# MAGIC **Data Enrichment (Mirrors Training):**
# MAGIC
# MAGIC - Joins cleaned test transactions with store dimension attributes (store type, assortment, competition distance, promo2 settings)
# MAGIC - Includes the same business logic features as training: `is_promo2_active` and `is_promo_interval_month`
# MAGIC - Preserves `prediction_id` for mapping predictions back to original test rows
# MAGIC
# MAGIC **Critical Design Choice - Excludes Temporal Features:**
# MAGIC
# MAGIC - Deliberately omits all lag features (`lag_sales_1d`, `lag_sales_7d`) and rolling windows (`rolling_sales_mean_7d`, `promo_days_last_14d`) that were in the training feature-base
# MAGIC - **Why?** At inference time, future sales don't exist yet—you can't compute yesterday's sales for a prediction about tomorrow
# MAGIC - This prevents **target leakage** and ensures the model only uses features available at prediction time
# MAGIC - 
# MAGIC Output: Successfully created `demo.silver.rossmann_test_feature_base` with a **reduced feature set compared to training**. This is intentional: the model must be trained on a subset of features that will actually be available during serving, or you need a separate inference-time strategy to populate lag features (e.g., using the model's own recent predictions).
# MAGIC
# MAGIC This enforces **production realism** — the test feature-base reflects what you'll actually have when making real-world predictions, not idealized training conditions.

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

# MAGIC %md
# MAGIC Transforms raw NYC taxi trip data from the bronze layer into a clean, analysis-ready silver table with domain-specific feature engineering. The core idea is **aggressive outlier filtering** + **urban mobility feature engineering**:
# MAGIC
# MAGIC **Data Quality Layer (Multi-Stage Filtering):**
# MAGIC
# MAGIC - Safe type casting with `TRY_CAST` to handle raw parquet inconsistencies
# MAGIC -**Trip validity rules:** removes impossible trips (dropoff before pickup, missing timestamps)
# MAGIC - **Range-based anomaly detection:** filters extreme outliers (distances `<0` or `>100` miles, fares `>$1000`, durations `<1` or `>180` minutes)
# MAGIC - Ensures required location IDs exist (critical for zone-based analysis)
# MAGIC
# MAGIC **Feature Engineering for Urban Mobility Analysis:**
# MAGIC
# MAGIC - **Temporal features:** extracts pickup date, hour, day of week, month for demand patterns
# MAGIC - **Behavioral indicators:** weekend flag, rush hour flag (7-9 AM, 4-7 PM) for congestion/surge modeling
# MAGIC - **Trip characteristics:** calculates trip duration in minutes, average speed (mph) to detect unusual patterns
# MAGIC - **Revenue metrics:** computes tip percentage (`tip_pct`) as a behavioral and service quality signal
# MAGIC
# MAGIC **Output:** Successfully created `demo.silver.nyc_taxi_trips_clean`, a trip-level table ready for aggregation. The aggressive filtering removes bad sensor data and fraudulent/erroneous records common in taxi meter data, while the engineered features support downstream demand forecasting and revenue optimization models.
# MAGIC
# MAGIC This is **trip-level granularity** — each row is one taxi ride with enriched context, ready to be aggregated by zone/hour for demand modeling.

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

# MAGIC %md
# MAGIC Aggregates individual taxi trips into a zone-hour level time-series table for demand forecasting. The core idea is **granularity shift** + **temporal demand signals**:
# MAGIC
# MAGIC **Granularity Transformation:**
# MAGIC
# MAGIC - Aggregates trip-level data (cell 22) to (pickup_date, pickup_hour, pu_location_id) level
# MAGIC - Creates a much smaller table optimized for time-series modeling (zone-hour observations instead of individual trips)
# MAGIC - Computes aggregate metrics: trip count, average distance/duration/fare, average tip percentage, total revenue
# MAGIC
# MAGIC **Time-Series Feature Engineering:**
# MAGIC
# MAGIC - Constructs a proper timestamp (`zone_hour_ts`) from date + hour for window function ordering
# MAGIC - Autoregressive features:
# MAGIC   - `lag_trip_cnt_1h`: demand 1 hour ago (captures short-term momentum)
# MAGIC   - `lag_trip_cnt_24h`: demand at the same hour yesterday (captures daily seasonality)
# MAGIC - Rolling trend indicators:
# MAGIC   - `rolling_trip_cnt_mean_24h`: average demand over past 24 hours
# MAGIC   - `rolling_fare_mean_24h`: average fare over past 24 hours (for revenue forecasting)
# MAGIC
# MAGIC **Output:** Successfully created `demo.silver.nyc_zone_hour_feature_base`, a time-series table ready for **demand forecasting models** that predict trip volume per zone per hour. This granularity is ideal for operational planning (driver dispatch, surge pricing) and differs from the Rossmann feature-bases which operate at daily store-level granularity.
# MAGIC
# MAGIC This is an **ML-ready aggregation** — each row represents demand at a specific zone-hour with historical context, enabling models to learn hourly patterns, weekly cycles, and zone-specific demand characteristics.

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

# MAGIC %md
# MAGIC Captures a point-in-time snapshot of silver layer row counts for operational observability. The core idea is **audit trail + regression detection**:
# MAGIC
# MAGIC **Audit Trail Creation:**
# MAGIC
# MAGIC Inserts row counts for all 5 silver tables into `demo.audit.silver_snapshot` with a single timestamp
# MAGIC Uses `UNION ALL` to batch-insert all metrics in one transaction
# MAGIC Captures: `rossmann_train_clean`, `rossmann_store_clean`, `rossmann_train_feature_base`, `nyc_taxi_trips_clean`, `nyc_zone_hour_feature_base`
# MAGIC
# MAGIC **Immediate Validation:**
# MAGIC
# MAGIC Displays the complete audit history ordered by timestamp and table name
# MAGIC Shows the current run alongside any previous runs for comparison
# MAGIC
# MAGIC **Output:** Successfully recorded 5 table snapshots at `2026-02-26 19:05:11`:
# MAGIC
# MAGIC - **Rossmann train data:** `1,017,209` rows preserved through both clean and feature-base transformations
# MAGIC - **Rossmann stores:** `1,115` dimension records
# MAGIC - **NYC trips:** `2,987,067` cleaned trip records
# MAGIC - **NYC zone-hour aggregates:** `67,019` time-series observations
# MAGIC
# MAGIC This enables **pipeline regression detection** — if future runs show unexpected row count drops (indicating bad filters or data issues) or spikes (indicating data quality problems upstream), you can catch them immediately by comparing against historical snapshots. This is production-grade data engineering observability.

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

# MAGIC %md
# MAGIC Performs final validation checks on the ML-ready feature-base tables. 
# MAGIC The core idea is **data completeness verification + temporal coverage audit**:
# MAGIC
# MAGIC **Sanity Check Design:**
# MAGIC
# MAGIC - Queries both ML-ready feature-base tables (Rossmann training and NYC zone-hour)
# MAGIC - Computes three critical metrics per dataset: row count, earliest date, latest date
# MAGIC - Uses UNION ALL to present side-by-side comparison in a single result set
# MAGIC
# MAGIC **Business Value:**
# MAGIC
# MAGIC - Row counts confirm no catastrophic data loss during transformations
# MAGIC - Date ranges verify temporal coverage meets modeling requirements (sufficient history for training)
# MAGIC - Provides a quick "at-a-glance" health check before proceeding to model training
# MAGIC
# MAGIC **Output Results:**
# MAGIC
# MAGIC - **Rossmann training:** `1,017,209` rows spanning **2.5 years** (Jan 2013 - July 2015) — substantial history for capturing seasonality and trends
# MAGIC - **NYC zone-hour:** `67,019` rows spanning **3+ months** (Oct 2022 - Feb 2023) — sufficient hourly observations for short-term demand patterns
# MAGIC
# MAGIC This is a **final pre-flight check** before handing off to the modeling pipeline. 
# MAGIC If date ranges were truncated or row counts dramatically different from audit snapshots in cell 27, it would signal a transformation issue requiring investigation. 
# MAGIC This validates that the feature engineering preserved data integrity and the tables are ready for model training.

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
