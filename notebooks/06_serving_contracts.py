# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Serving Contracts & Batch Inference
# MAGIC
# MAGIC **Goals:**
# MAGIC - Load champion models from the UC Model Registry for both Rossmann and NYC.
# MAGIC - Run batch inference on serving marts.
# MAGIC - Persist results to `demo.serving.rossmann_predictions` and `demo.serving.nyc_demand_predictions`.
# MAGIC - Define and validate serving table schemas (input/output contracts).
# MAGIC - Create a serving metadata registry for downstream API/dashboard consumption.
# MAGIC
# MAGIC **Constraints:**
# MAGIC - No Spark DataFrame API — SQL + pandas + mlflow for inference.
# MAGIC - All code in English.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Imports & setup

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.pyfunc

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

CATALOG = "demo"
print("setup OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Ensure serving schema exists

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.serving;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Load champion models from UC Model Registry

# COMMAND ----------

# Load the Rossmann champion model.
ROSSMANN_MODEL_URI = f"models:/{CATALOG}.ml.rossmann_sales_champion/1"
try:
    rossmann_model = mlflow.pyfunc.load_model(ROSSMANN_MODEL_URI)
    print(f"Rossmann model loaded: {ROSSMANN_MODEL_URI}")
except Exception as e:
    print(f"Could not load Rossmann model from registry: {e}")
    print("Falling back to latest run artifact...")
    # Fallback: load from the latest MLflow run with the champion tag.
    runs = mlflow.search_runs(
        filter_string="tags.mlflow.runName LIKE '%champion_registration%'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) > 0:
        rossmann_model = mlflow.pyfunc.load_model(f"runs:/{runs.iloc[0].run_id}/champion_model")
        print(f"Loaded from run: {runs.iloc[0].run_id}")
    else:
        raise RuntimeError("No Rossmann champion model found.")

# COMMAND ----------

# Load the NYC demand champion model.
NYC_MODEL_URI = f"models:/{CATALOG}.ml.nyc_demand_champion/1"
try:
    nyc_model = mlflow.pyfunc.load_model(NYC_MODEL_URI)
    print(f"NYC model loaded: {NYC_MODEL_URI}")
except Exception as e:
    print(f"Could not load NYC model from registry: {e}")
    print("Falling back to latest run artifact...")
    runs = mlflow.search_runs(
        filter_string="tags.mlflow.runName LIKE '%NYC_%champion_registration%'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) > 0:
        nyc_model = mlflow.pyfunc.load_model(f"runs:/{runs.iloc[0].run_id}/champion_model")
        print(f"Loaded from run: {runs.iloc[0].run_id}")
    else:
        raise RuntimeError("No NYC demand champion model found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Define serving input contracts
# MAGIC
# MAGIC Each serving mart has a strict set of expected input columns. We validate
# MAGIC them before inference to enforce the contract and catch schema drift early.

# COMMAND ----------

ROSSMANN_INPUT_SCHEMA = [
    "store_id", "day_of_week", "is_weekend", "is_open", "is_promo",
    "state_holiday_code", "is_school_holiday", "store_type", "assortment_type",
    "competition_distance_km", "has_promo2", "is_promo2_active", "is_promo_interval_month",
]

NYC_INPUT_SCHEMA = [
    "pickup_hour", "pu_location_id",
    "avg_trip_distance_miles", "avg_trip_duration_min",
    "avg_fare_amount", "avg_tip_pct", "total_revenue",
    "lag_trip_cnt_1h", "lag_trip_cnt_24h",
    "rolling_trip_cnt_mean_24h", "rolling_fare_mean_24h",
]

ROSSMANN_OUTPUT_SCHEMA = {
    "prediction_id": "STRING",
    "store_id": "INT",
    "business_date": "DATE",
    "predicted_sales": "DOUBLE",
    "model_name": "STRING",
    "model_version": "STRING",
    "inference_ts": "TIMESTAMP",
}

NYC_OUTPUT_SCHEMA = {
    "pickup_date": "DATE",
    "pickup_hour": "INT",
    "pu_location_id": "INT",
    "predicted_trip_cnt": "DOUBLE",
    "model_name": "STRING",
    "model_version": "STRING",
    "inference_ts": "TIMESTAMP",
}


def validate_schema(df, expected_cols, name):
    """Verify that all expected columns are present in the dataframe."""
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")
    print(f"[{name}] Schema validation passed — {len(expected_cols)} columns OK.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Rossmann batch inference

# COMMAND ----------

# Load the serving mart.
rossmann_serving = spark.sql(f"SELECT * FROM {CATALOG}.gold.rossmann_serving_mart").toPandas()
print(f"Rossmann serving mart: {rossmann_serving.shape[0]:,} rows")

validate_schema(rossmann_serving, ROSSMANN_INPUT_SCHEMA, "rossmann_serving")

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# Prepare features: encode categoricals consistently with training.
CAT_COLS = ["store_type", "assortment_type", "state_holiday_code"]
serving_df = rossmann_serving.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    serving_df[col] = le.fit_transform(serving_df[col].astype(str))

serving_df = serving_df.fillna(0)

# The model expects features in the same order as training.
# Use the input schema columns for prediction.
X_serving = serving_df[ROSSMANN_INPUT_SCHEMA]

# Run batch inference.
predictions = rossmann_model.predict(X_serving)

# Build output dataframe.
out_ross = pd.DataFrame({
    "prediction_id": rossmann_serving["prediction_id"],
    "store_id": rossmann_serving["store_id"],
    "business_date": rossmann_serving["business_date"],
    "predicted_sales": predictions.flatten(),
    "model_name": "rossmann_sales_champion",
    "model_version": "1",
    "inference_ts": datetime.now(),
})

print(f"Rossmann predictions generated: {len(out_ross):,} rows")
print(out_ross.head(3))

# COMMAND ----------

# Persist to serving schema.
spark_ross = spark.createDataFrame(out_ross)
spark_ross.createOrReplaceTempView("tmp_rossmann_serving_preds")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.serving.rossmann_predictions AS
    SELECT * FROM tmp_rossmann_serving_preds
""")

cnt = spark.sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.serving.rossmann_predictions").collect()[0]["cnt"]
print(f"Written to {CATALOG}.serving.rossmann_predictions — {cnt:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) NYC batch inference

# COMMAND ----------

nyc_serving = spark.sql(f"SELECT * FROM {CATALOG}.gold.nyc_serving_latest_zone_hour_mart").toPandas()
print(f"NYC serving mart: {nyc_serving.shape[0]:,} rows")

validate_schema(nyc_serving, NYC_INPUT_SCHEMA, "nyc_serving")

# COMMAND ----------

serving_nyc_df = nyc_serving.copy().fillna(0)
X_nyc_serving = serving_nyc_df[NYC_INPUT_SCHEMA]

# Run batch inference.
nyc_predictions = nyc_model.predict(X_nyc_serving)

out_nyc = pd.DataFrame({
    "pickup_date": nyc_serving["pickup_date"],
    "pickup_hour": nyc_serving["pickup_hour"],
    "pu_location_id": nyc_serving["pu_location_id"],
    "predicted_trip_cnt": nyc_predictions.flatten(),
    "model_name": "nyc_demand_champion",
    "model_version": "1",
    "inference_ts": datetime.now(),
})

print(f"NYC predictions generated: {len(out_nyc):,} rows")
print(out_nyc.head(3))

# COMMAND ----------

spark_nyc = spark.createDataFrame(out_nyc)
spark_nyc.createOrReplaceTempView("tmp_nyc_serving_preds")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.serving.nyc_demand_predictions AS
    SELECT * FROM tmp_nyc_serving_preds
""")

cnt = spark.sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.serving.nyc_demand_predictions").collect()[0]["cnt"]
print(f"Written to {CATALOG}.serving.nyc_demand_predictions — {cnt:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Update contract registry

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Extend the contract registry with serving prediction tables.
# MAGIC INSERT INTO demo.gold.contract_registry_mart
# MAGIC SELECT 'rossmann_serving_predictions', 'demo.serving.rossmann_predictions', 'serving_output', current_timestamp()
# MAGIC UNION ALL
# MAGIC SELECT 'nyc_serving_predictions', 'demo.serving.nyc_demand_predictions', 'serving_output', current_timestamp();
# MAGIC
# MAGIC SELECT * FROM demo.gold.contract_registry_mart ORDER BY contract_type, contract_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Serving audit snapshot

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO demo.audit.gold_snapshot
# MAGIC SELECT current_timestamp(), 'demo.serving.rossmann_predictions', COUNT(*) FROM demo.serving.rossmann_predictions
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.serving.nyc_demand_predictions', COUNT(*) FROM demo.serving.nyc_demand_predictions;
# MAGIC
# MAGIC SELECT * FROM demo.audit.gold_snapshot WHERE table_name LIKE '%serving%' ORDER BY snapshot_ts DESC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Schema documentation
# MAGIC
# MAGIC ### Rossmann serving output schema
# MAGIC | Column | Type | Description |
# MAGIC |---|---|---|
# MAGIC | prediction_id | STRING | Unique ID from serving mart |
# MAGIC | store_id | INT | Rossmann store identifier |
# MAGIC | business_date | DATE | Prediction target date |
# MAGIC | predicted_sales | DOUBLE | Model sales prediction |
# MAGIC | model_name | STRING | Champion model name |
# MAGIC | model_version | STRING | Registry version |
# MAGIC | inference_ts | TIMESTAMP | When inference was executed |
# MAGIC
# MAGIC ### NYC serving output schema
# MAGIC | Column | Type | Description |
# MAGIC |---|---|---|
# MAGIC | pickup_date | DATE | Prediction target date |
# MAGIC | pickup_hour | INT | Hour of day |
# MAGIC | pu_location_id | INT | Taxi zone ID |
# MAGIC | predicted_trip_cnt | DOUBLE | Predicted demand |
# MAGIC | model_name | STRING | Champion model name |
# MAGIC | model_version | STRING | Registry version |
# MAGIC | inference_ts | TIMESTAMP | When inference was executed |
# MAGIC
# MAGIC **Next step:** `workdir/src/api/` — FastAPI backend to expose prediction endpoints.
