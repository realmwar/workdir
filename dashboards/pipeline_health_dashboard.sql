-- ============================================================================
-- Pipeline Health Dashboard — SQL queries for Databricks Dashboard widgets
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Widget 1: Row Count Timeline (line chart per table)
-- ---------------------------------------------------------------------------
SELECT
  snapshot_ts,
  table_name,
  row_count
FROM demo.audit.gold_snapshot
ORDER BY snapshot_ts DESC, table_name;

-- ---------------------------------------------------------------------------
-- Widget 2: Latest Row Counts (table / bar chart)
-- ---------------------------------------------------------------------------
WITH latest AS (
  SELECT
    table_name,
    row_count,
    snapshot_ts,
    ROW_NUMBER() OVER (PARTITION BY table_name ORDER BY snapshot_ts DESC) AS rn
  FROM demo.audit.gold_snapshot
)
SELECT table_name, row_count, snapshot_ts AS last_updated
FROM latest
WHERE rn = 1
ORDER BY table_name;

-- ---------------------------------------------------------------------------
-- Widget 3: Row Count Delta (detect regressions)
-- ---------------------------------------------------------------------------
WITH ranked AS (
  SELECT
    table_name,
    row_count,
    snapshot_ts,
    LAG(row_count) OVER (PARTITION BY table_name ORDER BY snapshot_ts) AS prev_row_count
  FROM demo.audit.gold_snapshot
)
SELECT
  table_name,
  snapshot_ts,
  row_count,
  prev_row_count,
  row_count - prev_row_count AS delta,
  CASE
    WHEN prev_row_count IS NOT NULL AND prev_row_count > 0
    THEN ROUND((row_count - prev_row_count) * 100.0 / prev_row_count, 2)
    ELSE NULL
  END AS delta_pct
FROM ranked
WHERE prev_row_count IS NOT NULL
ORDER BY snapshot_ts DESC, table_name;

-- ---------------------------------------------------------------------------
-- Widget 4: Data Freshness (time since last snapshot per table)
-- ---------------------------------------------------------------------------
WITH latest AS (
  SELECT
    table_name,
    MAX(snapshot_ts) AS last_snapshot
  FROM demo.audit.gold_snapshot
  GROUP BY table_name
)
SELECT
  table_name,
  last_snapshot,
  TIMESTAMPDIFF(HOUR, last_snapshot, current_timestamp()) AS hours_since_refresh
FROM latest
ORDER BY hours_since_refresh DESC;

-- ---------------------------------------------------------------------------
-- Widget 5: Contract Registry Status (table)
-- ---------------------------------------------------------------------------
SELECT
  contract_name,
  table_name,
  contract_type,
  updated_at
FROM demo.gold.contract_registry_mart
ORDER BY contract_type, contract_name;

-- ---------------------------------------------------------------------------
-- Widget 6: Snapshot History Count (table health trend)
-- ---------------------------------------------------------------------------
SELECT
  table_name,
  COUNT(*) AS snapshot_count,
  MIN(snapshot_ts) AS first_snapshot,
  MAX(snapshot_ts) AS last_snapshot
FROM demo.audit.gold_snapshot
GROUP BY table_name
ORDER BY table_name;

-- ---------------------------------------------------------------------------
-- Widget 7: Date Range Sanity Check
-- ---------------------------------------------------------------------------
SELECT 'rossmann_training_mart' AS table_name, MIN(business_date) AS min_date, MAX(business_date) AS max_date, COUNT(*) AS rows
FROM demo.gold.rossmann_training_mart
UNION ALL
SELECT 'nyc_demand_training_mart', MIN(pickup_date), MAX(pickup_date), COUNT(*)
FROM demo.gold.nyc_demand_training_mart
UNION ALL
SELECT 'rossmann_predictions', MIN(business_date), MAX(business_date), COUNT(*)
FROM demo.ml.rossmann_predictions
UNION ALL
SELECT 'nyc_demand_predictions', MIN(pickup_date), MAX(pickup_date), COUNT(*)
FROM demo.ml.nyc_demand_predictions;
