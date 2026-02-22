# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - UC Bootstrap and Data Intake
# MAGIC
# MAGIC Goal:
# MAGIC - create base schemas in `demo` for the medallion/E2E setup;
# MAGIC - ingest raw datasets for Rossmann and a limited NYC TLC subset;
# MAGIC - register landing and bronze tables for the next phases.
# MAGIC
# MAGIC Constraints:
# MAGIC - no Spark DataFrame API;
# MAGIC - use SQL + Python for file-ingestion orchestration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Runtime parameters

# COMMAND ----------

dbutils.widgets.text("nyc_months", "2023-01,2023-02,2023-03,2023-04,2023-05,2023-06")
dbutils.widgets.dropdown("force_redownload", "false", ["false", "true"])

CATALOG = "demo"
NYC_MONTHS = [m.strip() for m in dbutils.widgets.get("nyc_months").split(",") if m.strip()]
FORCE_REDOWNLOAD = dbutils.widgets.get("force_redownload").lower() == "true"

print("catalog:", CATALOG)
print("nyc_months:", NYC_MONTHS)
print("force_redownload:", FORCE_REDOWNLOAD)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Create schemas and volumes (UC)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- If `demo` already exists, this section is idempotent.
# MAGIC -- If you don't have CREATE CATALOG privilege, keep existing catalog and only use schemas.
# MAGIC CREATE CATALOG IF NOT EXISTS demo;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.landing;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.gold;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.ml;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.serving;
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.audit;
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS demo.landing.raw_data;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Download public datasets into UC Volume

# COMMAND ----------

import os
import urllib.request
from pathlib import Path


def download_file(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"skip (exists): {destination}")
        return

    print(f"downloading: {url}")
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(destination)
    print(f"saved: {destination} ({destination.stat().st_size / (1024**2):.2f} MB)")


volume_root = Path("/Volumes") / CATALOG / "landing" / "raw_data"

rossmann_dir = volume_root / "rossmann"
nyc_dir = volume_root / "nyc_tlc"

rossmann_urls = {
    "train.csv": "https://raw.githubusercontent.com/lfaferreira/predict-rossmann-store-sales/master/data/train.csv",
    "test.csv": "https://raw.githubusercontent.com/lfaferreira/predict-rossmann-store-sales/master/data/test.csv",
    "store.csv": "https://raw.githubusercontent.com/lfaferreira/predict-rossmann-store-sales/master/data/store.csv",
}

for filename, url in rossmann_urls.items():
    download_file(url, rossmann_dir / filename, force=FORCE_REDOWNLOAD)


def nyc_url(month: str) -> str:
    # Official monthly parquet pattern for TLC data.
    return f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"


for month in NYC_MONTHS:
    file_name = f"yellow_tripdata_{month}.parquet"
    download_file(nyc_url(month), nyc_dir / file_name, force=FORCE_REDOWNLOAD)

print("raw dataset intake complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Register landing tables

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE demo.landing.rossmann_train_raw AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS ingest_ts,
# MAGIC   input_file_name() AS source_file
# MAGIC FROM read_files(
# MAGIC   '/Volumes/demo/landing/raw_data/rossmann/train.csv',
# MAGIC   format => 'csv',
# MAGIC   header => true,
# MAGIC   inferSchema => true
# MAGIC );
# MAGIC
# MAGIC CREATE OR REPLACE TABLE demo.landing.rossmann_test_raw AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS ingest_ts,
# MAGIC   input_file_name() AS source_file
# MAGIC FROM read_files(
# MAGIC   '/Volumes/demo/landing/raw_data/rossmann/test.csv',
# MAGIC   format => 'csv',
# MAGIC   header => true,
# MAGIC   inferSchema => true
# MAGIC );
# MAGIC
# MAGIC CREATE OR REPLACE TABLE demo.landing.rossmann_store_raw AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS ingest_ts,
# MAGIC   input_file_name() AS source_file
# MAGIC FROM read_files(
# MAGIC   '/Volumes/demo/landing/raw_data/rossmann/store.csv',
# MAGIC   format => 'csv',
# MAGIC   header => true,
# MAGIC   inferSchema => true
# MAGIC );
# MAGIC
# MAGIC CREATE OR REPLACE TABLE demo.landing.nyc_taxi_trips_raw AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS ingest_ts,
# MAGIC   input_file_name() AS source_file
# MAGIC FROM read_files(
# MAGIC   '/Volumes/demo/landing/raw_data/nyc_tlc/yellow_tripdata_*.parquet',
# MAGIC   format => 'parquet'
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Create bronze tables (minimal normalization + audit fields)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE demo.bronze.rossmann_train AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS bronze_loaded_at
# MAGIC FROM demo.landing.rossmann_train_raw;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE demo.bronze.rossmann_test AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS bronze_loaded_at
# MAGIC FROM demo.landing.rossmann_test_raw;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE demo.bronze.rossmann_store AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS bronze_loaded_at
# MAGIC FROM demo.landing.rossmann_store_raw;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE demo.bronze.nyc_taxi_trips AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS bronze_loaded_at
# MAGIC FROM demo.landing.nyc_taxi_trips_raw;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Basic validation and audit snapshot

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS demo.audit.intake_snapshot (
# MAGIC   snapshot_ts TIMESTAMP,
# MAGIC   table_name STRING,
# MAGIC   row_count BIGINT
# MAGIC );
# MAGIC
# MAGIC INSERT INTO demo.audit.intake_snapshot
# MAGIC SELECT current_timestamp(), 'demo.bronze.rossmann_train', COUNT(*) FROM demo.bronze.rossmann_train
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.bronze.rossmann_test', COUNT(*) FROM demo.bronze.rossmann_test
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.bronze.rossmann_store', COUNT(*) FROM demo.bronze.rossmann_store
# MAGIC UNION ALL
# MAGIC SELECT current_timestamp(), 'demo.bronze.nyc_taxi_trips', COUNT(*) FROM demo.bronze.nyc_taxi_trips;
# MAGIC
# MAGIC SELECT * FROM demo.audit.intake_snapshot ORDER BY snapshot_ts DESC, table_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Quick sanity checks

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'rossmann_train' AS dataset, COUNT(*) AS rows, MIN(Date) AS min_date, MAX(Date) AS max_date
# MAGIC FROM demo.bronze.rossmann_train
# MAGIC UNION ALL
# MAGIC SELECT 'nyc_taxi_trips' AS dataset, COUNT(*) AS rows, MIN(tpep_pickup_datetime) AS min_date, MAX(tpep_pickup_datetime) AS max_date
# MAGIC FROM demo.bronze.nyc_taxi_trips;

# COMMAND ----------

# MAGIC %md
# MAGIC Done. Next steps:
# MAGIC - silver transformations;
# MAGIC - baseline models (Rossmann as primary, NYC as phase 2 mini-track);
# MAGIC - feature + model versioning + serving + dashboards.
