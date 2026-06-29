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
# MAGIC ## 0) Install dependencies (Serverless)

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks Free can ship an older MLflow build that is unreliable with Unity Catalog model
# MAGIC operations. We upgrade MLflow to the same Databricks-compatible version already needed in the
# MAGIC training notebooks. We also install the TensorFlow/Keras stack because the NYC champion may be
# MAGIC a neural network model, and this notebook has to be able to deserialize either sklearn or FNN
# MAGIC artifacts from the registry.

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall "mlflow[databricks]==2.22.0" "tensorflow==2.15.1" "keras==2.15.0" "protobuf==4.25.3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Imports & setup

# COMMAND ----------

# MAGIC %md
# MAGIC This cell prepares the batch inference runtime. Pandas carries the serving tables locally,
# MAGIC MLflow loads champion models from the registry, and the explicit Databricks tracking/registry URIs
# MAGIC make sure model resolution happens against the workspace tracking server and Unity Catalog registry.

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

CATALOG = "demo"
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _user = _ctx.userName().get()
    ROSSMANN_EXPERIMENT_NAME = f"/Users/{_user}/rossmann_baseline"
    NYC_EXPERIMENT_NAME = f"/Users/{_user}/nyc_demand_forecasting"
except Exception:
    ROSSMANN_EXPERIMENT_NAME = "/Shared/rossmann_baseline"
    NYC_EXPERIMENT_NAME = "/Shared/nyc_demand_forecasting"

client = MlflowClient()


def latest_registered_model_uri(model_name: str):
    """Return URI and version for the latest UC model version."""
    versions = list(client.search_model_versions(f"name = '{model_name}'"))
    if not versions:
        raise RuntimeError(f"No registered versions found for model: {model_name}")
    latest = max(versions, key=lambda mv: int(mv.version))
    return f"models:/{model_name}/{latest.version}", str(latest.version)


def load_latest_champion_run_artifact(experiment_name: str, run_name_like: str):
    """Fallback loader for the latest champion-registration artifact in a specific experiment."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.mlflow.runName LIKE '{run_name_like}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError(f"No champion registration run found in experiment: {experiment_name}")
    run_id = runs.iloc[0].run_id
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/champion_model"), run_id

print("setup OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Ensure serving schema exists

# COMMAND ----------

# MAGIC %md
# MAGIC The serving outputs from this notebook are written into `demo.serving`. Creating the schema
# MAGIC defensively here lets the notebook run in isolation without depending on earlier manual setup.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.serving;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Load champion models from UC Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC We first try to load the Rossmann champion from the Unity Catalog model registry. If that lookup
# MAGIC fails, the notebook falls back to the latest champion-registration run artifact so batch inference
# MAGIC can still proceed while the registry is being debugged or repopulated.

# COMMAND ----------

# Load the Rossmann champion model.
ROSSMANN_MODEL_NAME = f"{CATALOG}.ml.rossmann_sales_champion"
ROSSMANN_MODEL_URI, rossmann_model_version = latest_registered_model_uri(ROSSMANN_MODEL_NAME)
try:
    rossmann_model = mlflow.pyfunc.load_model(ROSSMANN_MODEL_URI)
    print(f"Rossmann model loaded: {ROSSMANN_MODEL_URI}")
except Exception as e:
    print(f"Could not load Rossmann model from registry: {e}")
    print("Falling back to latest run artifact...")
    rossmann_model, rossmann_fallback_run_id = load_latest_champion_run_artifact(
        ROSSMANN_EXPERIMENT_NAME,
        "%champion_registration%",
    )
    rossmann_model_version = "run_artifact"
    print(f"Loaded from run: {rossmann_fallback_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC The NYC demand champion follows the same pattern as Rossmann: prefer the stable registry URI, but
# MAGIC fall back to the latest matching MLflow run artifact if the registry entry is unavailable.

# COMMAND ----------

# Load the NYC demand champion model.
NYC_MODEL_NAME = f"{CATALOG}.ml.nyc_demand_champion"
NYC_MODEL_URI, nyc_model_version = latest_registered_model_uri(NYC_MODEL_NAME)
try:
    nyc_model = mlflow.pyfunc.load_model(NYC_MODEL_URI)
    print(f"NYC model loaded: {NYC_MODEL_URI}")
except Exception as e:
    print(f"Could not load NYC model from registry: {e}")
    print("Falling back to latest run artifact...")
    nyc_model, nyc_fallback_run_id = load_latest_champion_run_artifact(
        NYC_EXPERIMENT_NAME,
        "%NYC_%champion_registration%",
    )
    nyc_model_version = "run_artifact"
    print(f"Loaded from run: {nyc_fallback_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Define serving input contracts
# MAGIC
# MAGIC Each serving mart has a strict set of expected input columns. We validate
# MAGIC them before inference to enforce the contract and catch schema drift early.

# COMMAND ----------

# MAGIC %md
# MAGIC This cell defines the serving contracts explicitly. The input schemas describe the exact feature
# MAGIC columns each champion expects, the output schemas document what downstream systems will receive,
# MAGIC and `validate_schema` is the guard rail that catches missing columns before inference starts.

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


def align_to_model_signature(df, pyfunc_model, model_name, feature_columns=None):
    """Cast a pandas dataframe to the input schema declared in the MLflow model signature.

    For column-based schemas we:
    1. add any missing columns with safe defaults,
    2. cast dtypes to match the signature,
    3. reorder columns exactly as the model expects.

    For tensor-based schemas (common with Keras/TensorFlow models), we simply coerce
    every column to numeric float64 and preserve the existing column order.
    """
    schema = pyfunc_model.metadata.get_input_schema()
    aligned = df.copy()

    if schema is None:
        return aligned

    if hasattr(schema, "is_tensor_spec") and schema.is_tensor_spec():
        if feature_columns is not None:
            aligned = aligned[feature_columns].copy()
        for col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="raise").astype("float64")
        return aligned

    expected_cols = []
    for col_spec in schema.inputs:
        col_name = col_spec.name
        col_type = str(col_spec.type).lower()
        expected_cols.append(col_name)

        if col_name not in aligned.columns:
            if col_type in ("integer", "long"):
                aligned[col_name] = 0
            elif col_type in ("float", "double"):
                aligned[col_name] = 0.0
            elif col_type == "boolean":
                aligned[col_name] = False
            elif col_type == "string":
                aligned[col_name] = ""
            else:
                aligned[col_name] = 0.0
            print(f"[{model_name}] Added missing column with default value: {col_name}")

        if col_type in ("integer", "long"):
            aligned[col_name] = pd.to_numeric(aligned[col_name], errors="raise").astype("int64")
        elif col_type in ("float", "double"):
            aligned[col_name] = pd.to_numeric(aligned[col_name], errors="raise").astype("float64")
        elif col_type == "boolean":
            aligned[col_name] = aligned[col_name].astype("bool")
        elif col_type == "string":
            aligned[col_name] = aligned[col_name].astype("string")

    return aligned[expected_cols]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Rossmann batch inference

# COMMAND ----------

# MAGIC %md
# MAGIC We load the Rossmann serving mart from the gold layer, validate that it matches the expected input
# MAGIC contract, and print a quick row-count sanity check before any prediction code runs.

# COMMAND ----------

# Load the serving mart.
rossmann_serving = spark.sql(f"SELECT * FROM {CATALOG}.gold.rossmann_serving_mart").toPandas()
print(f"Rossmann serving mart: {rossmann_serving.shape[0]:,} rows")

validate_schema(rossmann_serving, ROSSMANN_INPUT_SCHEMA, "rossmann_serving")

# COMMAND ----------

# MAGIC %md
# MAGIC This cell prepares Rossmann features exactly the way the trained model expects them, runs batch
# MAGIC inference, and assembles a serving-friendly output table with identifiers, predictions, model
# MAGIC metadata, and the inference timestamp.

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# Prepare features: encode categoricals consistently with training.
CAT_COLS = ["store_type", "assortment_type", "state_holiday_code"]
serving_df = rossmann_serving.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    serving_df[col] = le.fit_transform(serving_df[col].astype(str))

serving_df = serving_df.fillna(0)

# Start from the full serving dataframe and let the MLflow signature decide the exact
# required feature order. Missing train-only lag features are backfilled with safe defaults.
X_serving = align_to_model_signature(serving_df, rossmann_model, "rossmann_model")

# Run batch inference.
predictions = rossmann_model.predict(X_serving)

# Build output dataframe.
out_ross = pd.DataFrame({
    "prediction_id": rossmann_serving["prediction_id"],
    "store_id": rossmann_serving["store_id"],
    "business_date": rossmann_serving["business_date"],
    "predicted_sales": predictions.flatten(),
    "model_name": "rossmann_sales_champion",
    "model_version": rossmann_model_version,
    "inference_ts": datetime.now(),
})

print(f"Rossmann predictions generated: {len(out_ross):,} rows")
print(out_ross.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC After local inference is complete, the Rossmann predictions are written back into Unity Catalog.
# MAGIC The temp view bridge keeps the write path simple while still ending with a managed serving table.

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

# MAGIC %md
# MAGIC The NYC serving mart is loaded from the latest gold serving table and validated against its input
# MAGIC contract before predictions are produced.

# COMMAND ----------

nyc_serving = spark.sql(f"SELECT * FROM {CATALOG}.gold.nyc_serving_latest_zone_hour_mart").toPandas()
print(f"NYC serving mart: {nyc_serving.shape[0]:,} rows")

validate_schema(nyc_serving, NYC_INPUT_SCHEMA, "nyc_serving")

# COMMAND ----------

# MAGIC %md
# MAGIC This cell prepares the NYC serving features, runs the champion demand model, and builds a tidy
# MAGIC output dataframe with zone-hour keys, predicted trip counts, model metadata, and inference time.

# COMMAND ----------

serving_nyc_df = nyc_serving.copy().fillna(0)
X_nyc_serving = align_to_model_signature(
    serving_nyc_df[NYC_INPUT_SCHEMA],
    nyc_model,
    "nyc_model",
    feature_columns=NYC_INPUT_SCHEMA,
)

# Run batch inference.
nyc_predictions = nyc_model.predict(X_nyc_serving)

out_nyc = pd.DataFrame({
    "pickup_date": nyc_serving["pickup_date"],
    "pickup_hour": nyc_serving["pickup_hour"],
    "pu_location_id": nyc_serving["pu_location_id"],
    "predicted_trip_cnt": nyc_predictions.flatten(),
    "model_name": "nyc_demand_champion",
    "model_version": nyc_model_version,
    "inference_ts": datetime.now(),
})

print(f"NYC predictions generated: {len(out_nyc):,} rows")
print(out_nyc.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC The NYC predictions are persisted into the serving schema so downstream APIs, dashboards, and QA
# MAGIC checks all read from the same contract table.

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

# MAGIC %md
# MAGIC The contract registry is extended with the two serving output tables created above. This keeps the
# MAGIC project-wide catalog of data contracts in sync with the actual assets now available to consumers.

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

# MAGIC %md
# MAGIC The audit snapshot records row counts for the newly written serving tables. These lightweight
# MAGIC inserts give us a simple operational breadcrumb for dashboards and job monitoring.

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
