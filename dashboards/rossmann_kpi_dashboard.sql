-- ============================================================================
-- Rossmann KPI Dashboard — SQL queries for Databricks Dashboard widgets
-- ============================================================================
-- Each query below corresponds to a single dashboard widget/tile.
-- Paste these into Databricks SQL Dashboard editor as separate queries.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Widget 1: Daily Sales Trend (line chart)
-- ---------------------------------------------------------------------------
SELECT
  business_date,
  total_sales,
  rolling_total_sales_mean_7d,
  lag_total_sales_7d
FROM demo.gold.rossmann_dashboard_kpi_mart
ORDER BY business_date;

-- ---------------------------------------------------------------------------
-- Widget 2: Total Sales KPI (counter)
-- ---------------------------------------------------------------------------
SELECT
  SUM(total_sales) AS grand_total_sales,
  AVG(total_sales) AS avg_daily_sales,
  MAX(total_sales) AS peak_daily_sales,
  MIN(total_sales) AS min_daily_sales
FROM demo.gold.rossmann_dashboard_kpi_mart;

-- ---------------------------------------------------------------------------
-- Widget 3: Promo Impact Analysis (grouped bar chart)
-- ---------------------------------------------------------------------------
SELECT
  CASE WHEN promo_ratio > 0.5 THEN 'Promo Day' ELSE 'Non-Promo Day' END AS day_type,
  COUNT(*) AS day_count,
  AVG(total_sales) AS avg_sales,
  AVG(total_customers) AS avg_customers,
  AVG(avg_ticket_mean) AS avg_ticket
FROM demo.gold.rossmann_dashboard_kpi_mart
GROUP BY CASE WHEN promo_ratio > 0.5 THEN 'Promo Day' ELSE 'Non-Promo Day' END;

-- ---------------------------------------------------------------------------
-- Widget 4: Weekend vs Weekday Performance (bar chart)
-- ---------------------------------------------------------------------------
SELECT
  CASE WHEN weekend_ratio > 0.5 THEN 'Weekend' ELSE 'Weekday' END AS period,
  COUNT(*) AS days,
  AVG(total_sales) AS avg_sales,
  AVG(total_customers) AS avg_customers
FROM demo.gold.rossmann_dashboard_kpi_mart
GROUP BY CASE WHEN weekend_ratio > 0.5 THEN 'Weekend' ELSE 'Weekday' END;

-- ---------------------------------------------------------------------------
-- Widget 5: Store Activity Rate Over Time (area chart)
-- ---------------------------------------------------------------------------
SELECT
  business_date,
  active_store_cnt,
  open_store_ratio
FROM demo.gold.rossmann_dashboard_kpi_mart
ORDER BY business_date;

-- ---------------------------------------------------------------------------
-- Widget 6: Week-over-Week Sales Change (line chart)
-- ---------------------------------------------------------------------------
SELECT
  business_date,
  total_sales,
  lag_total_sales_7d,
  CASE
    WHEN lag_total_sales_7d IS NOT NULL AND lag_total_sales_7d > 0
    THEN ROUND((total_sales - lag_total_sales_7d) / lag_total_sales_7d * 100, 2)
    ELSE NULL
  END AS wow_pct_change
FROM demo.gold.rossmann_dashboard_kpi_mart
WHERE lag_total_sales_7d IS NOT NULL
ORDER BY business_date;

-- ---------------------------------------------------------------------------
-- Widget 7: Monthly Sales Aggregation (bar chart)
-- ---------------------------------------------------------------------------
SELECT
  DATE_TRUNC('month', business_date) AS month,
  SUM(total_sales) AS monthly_sales,
  SUM(total_customers) AS monthly_customers,
  AVG(avg_ticket_mean) AS avg_ticket
FROM demo.gold.rossmann_dashboard_kpi_mart
GROUP BY DATE_TRUNC('month', business_date)
ORDER BY month;
