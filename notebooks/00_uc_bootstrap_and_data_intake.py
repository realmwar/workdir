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

# MAGIC %md
# MAGIC This cell creates Databricks widgets for runtime parameters, parses their values, and prints them. The widgets allow users to specify which NYC TLC months to ingest and whether to force re-download of datasets. The parsed values are stored in variables for use in subsequent data intake steps.

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

# MAGIC %md
# MAGIC
# MAGIC This cell uses SQL to create schemas and a volume in the `demo` catalog for the medallion architecture. It ensures the catalog, schemas (landing, bronze, silver, gold, ml, serving, audit), and the `raw_data` volume are created if they do not already exist, making the setup idempotent.

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

# MAGIC %md
# MAGIC Defines a `download_file()` helper function that uses `urllib.request` to fetch remote files, with optional force re-download and skip-if-exists logic.
# MAGIC
# MAGIC Sets up paths to the volume at `/Volumes/demo/landing/raw_data/` with subdirectories for Rossmann and NYC TLC data.
# MAGIC
# MAGIC Downloads three Rossmann CSV files (`train.csv`, `test.csv`, `store.csv`) from a GitHub repository.
# MAGIC
# MAGIC Downloads NYC yellow taxi trip data as monthly parquet files based on the `NYC_MONTHS` parameter configured in cell 4.
# MAGIC
# MAGIC The cell respects the `FORCE_REDOWNLOAD` flag and prints download progress with file sizes. 
# MAGIC All files land in the Unity Catalog volume for subsequent table registration.
# MAGIC

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

# MAGIC %md
# MAGIC Registers four landing tables from the raw files downloaded into the UC Volume. It:
# MAGIC
# MAGIC - Creates `demo.landing.rossmann_train_raw` by reading `train.csv` with CSV format, header parsing, and schema inference.
# MAGIC - Creates `demo.landing.rossmann_test_raw` from `test.csv` with the same CSV settings.
# MAGIC - Creates `demo.landing.rossmann_store_raw` from `store.csv` with the same CSV settings.
# MAGIC - Creates `demo.landing.nyc_taxi_trips_raw` by reading all NYC taxi parquet files matching the wildcard pattern `yellow_tripdata_*.parquet`.
# MAGIC
# MAGIC Each table includes:
# MAGIC
# MAGIC - All columns from the source files (`*`).
# MAGIC - `ingest_ts` timestamp column capturing when the data was loaded.
# MAGIC - `source_file` column recording the file path via `input_file_name()`.
# MAGIC The cell uses `CREATE OR REPLACE TABLE` to make the registration idempotent, and leverages `read_files()` to load data directly from the volume paths.

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

# MAGIC %md
# MAGIC Promotes landing tables to the bronze layer in the medallion architecture. It:
# MAGIC
# MAGIC - Creates `demo.bronze.rossmann_train` from the landing table, selecting all columns plus adding a bronze_loaded_at timestamp audit field
# MAGIC - Creates `demo.bronze.rossmann_test` from the landing table with the same pattern
# MAGIC - Creates `demo.bronze.rossmann_store` from the landing table with the same pattern
# MAGIC - Creates `demo.bronze.nyc_taxi_trips` from the landing table with the same pattern
# MAGIC
# MAGIC Each bronze table is a straightforward copy of its corresponding landing table (which already has `ingest_ts` and `source_file` columns from cell 13), with one additional audit field capturing when the data was loaded into the bronze layer. The cell uses `CREATE OR REPLACE TABLE` to make the operation idempotent.

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

# MAGIC %md
# MAGIC Performs basic validation and creates an audit snapshot. It:
# MAGIC
# MAGIC - Creates the `demo.audit.intake_snapshot` table if it doesn't exist, with columns for snapshot timestamp, table name, and row count.
# MAGIC - Inserts four rows capturing the current timestamp and row count for each bronze table (`rossmann_train`, `rossmann_test`, `rossmann_store`, `nyc_taxi_trips`) using UNION ALL.
# MAGIC - Queries the audit table to display all snapshots ordered by timestamp descending and table name. 
# MAGIC
# MAGIC This provides a historical record of row counts after each data intake run, enabling tracking of data volume changes over time and validating that the bronze layer was successfully populated.

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

# MAGIC %md
# MAGIC Performs quick sanity checks on the bronze layer data. It:
# MAGIC
# MAGIC - Queries `demo.bronze.rossmann_train` to count rows and find the date range (MIN/MAX of the `Date` column).
# MAGIC - Queries `demo.bronze.nyc_taxi_trips` to count rows and find the date range (MIN/MAX of `tpep_pickup_datetime`).
# MAGIC - Uses UNION ALL to combine both results into a single output table with columns: `dataset`, `rows`, `min_date`, `max_date`.
# MAGIC This provides a quick validation of data completeness and temporal coverage for the two primary datasets after ingestion, making it easy to verify that the expected data landed and spot any date range anomalies.
# MAGIC

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
