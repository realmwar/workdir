-- ============================================================================
-- Model Performance Dashboard — SQL queries for Databricks Dashboard widgets
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Widget 1: Rossmann Predictions — Actual vs Predicted Scatter
-- ---------------------------------------------------------------------------
SELECT
  actual_sales,
  predicted_sales,
  residual,
  model_name
FROM demo.ml.rossmann_predictions
ORDER BY actual_sales;

-- ---------------------------------------------------------------------------
-- Widget 2: Rossmann — Residual Distribution
-- ---------------------------------------------------------------------------
SELECT
  residual,
  model_name,
  CASE
    WHEN ABS(residual) < 500 THEN 'within_500'
    WHEN ABS(residual) < 1000 THEN 'within_1000'
    WHEN ABS(residual) < 2000 THEN 'within_2000'
    ELSE 'over_2000'
  END AS residual_bucket
FROM demo.ml.rossmann_predictions;

-- ---------------------------------------------------------------------------
-- Widget 3: Rossmann — Prediction Error by Store
-- ---------------------------------------------------------------------------
SELECT
  store_id,
  AVG(ABS(residual)) AS mae,
  SQRT(AVG(residual * residual)) AS rmse,
  COUNT(*) AS prediction_count
FROM demo.ml.rossmann_predictions
GROUP BY store_id
ORDER BY mae DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Widget 4: Rossmann — Daily Error Trend
-- ---------------------------------------------------------------------------
SELECT
  business_date,
  AVG(ABS(residual)) AS daily_mae,
  SQRT(AVG(residual * residual)) AS daily_rmse,
  AVG(predicted_sales) AS avg_prediction,
  AVG(actual_sales) AS avg_actual
FROM demo.ml.rossmann_predictions
GROUP BY business_date
ORDER BY business_date;

-- ---------------------------------------------------------------------------
-- Widget 5: NYC Demand — Actual vs Predicted
-- ---------------------------------------------------------------------------
SELECT
  actual_trip_cnt,
  predicted_trip_cnt,
  residual,
  model_name
FROM demo.ml.nyc_demand_predictions
ORDER BY actual_trip_cnt;

-- ---------------------------------------------------------------------------
-- Widget 6: NYC — Prediction Error by Zone (top 20)
-- ---------------------------------------------------------------------------
SELECT
  pu_location_id,
  AVG(ABS(residual)) AS mae,
  SQRT(AVG(residual * residual)) AS rmse,
  COUNT(*) AS prediction_count
FROM demo.ml.nyc_demand_predictions
GROUP BY pu_location_id
ORDER BY mae DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Widget 7: Serving Predictions — Latest Batch Stats
-- ---------------------------------------------------------------------------
SELECT
  'rossmann' AS domain,
  COUNT(*) AS prediction_count,
  AVG(predicted_sales) AS avg_prediction,
  MIN(inference_ts) AS batch_start,
  MAX(inference_ts) AS batch_end
FROM demo.serving.rossmann_predictions
UNION ALL
SELECT
  'nyc_demand',
  COUNT(*),
  AVG(predicted_trip_cnt),
  MIN(inference_ts),
  MAX(inference_ts)
FROM demo.serving.nyc_demand_predictions;
