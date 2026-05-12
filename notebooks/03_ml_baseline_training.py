# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — ML Baseline Training & Model Registry (Rossmann)
# MAGIC
# MAGIC **Goals:**
# MAGIC - Load the gold training mart into pandas and prepare a model-ready feature matrix.
# MAGIC - Implement a time-based train / validation / test split to prevent data leakage.
# MAGIC - Train multiple baseline models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting (LightGBM, XGBoost).
# MAGIC - Evaluate every model with MSE, RMSE, MAE, R², MAPE.
# MAGIC - Log all experiments, parameters, metrics, and artifacts to MLflow.
# MAGIC - Register the best model in the Unity Catalog Model Registry.
# MAGIC - Write predictions back to `demo.ml.rossmann_predictions`.
# MAGIC
# MAGIC **Constraints:**
# MAGIC - No Spark DataFrame API — SQL for reads, pandas/sklearn/mlflow for ML.
# MAGIC - All code in English.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Runtime parameters & imports

# COMMAND ----------

dbutils.widgets.dropdown("log_to_uc_registry", "true", ["true", "false"])
LOG_TO_UC_REGISTRY = dbutils.widgets.get("log_to_uc_registry").lower() == "true"

CATALOG = "demo"

# Resolve MLflow experiment path in a Serverless-safe way.
# spark.conf.get("spark.databricks.notebook.path") fails under Spark Connect
# (Databricks Free / Serverless), so we use dbutils context with a /Shared fallback.
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _user = _ctx.userName().get()
    EXPERIMENT_NAME = f"/Users/{_user}/mlflow_experiments/rossmann_baseline"
except Exception:
    EXPERIMENT_NAME = "/Shared/mlflow_experiments/rossmann_baseline"

print("catalog:", CATALOG)
print("log_to_uc_registry:", LOG_TO_UC_REGISTRY)
print("experiment_name:", EXPERIMENT_NAME)

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

print("imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Load gold training mart into pandas

# COMMAND ----------

# Pull the full training mart via SQL; convert to pandas immediately.
df_raw = spark.sql(f"SELECT * FROM {CATALOG}.gold.rossmann_training_mart").toPandas()

print(f"rows: {df_raw.shape[0]:,}  cols: {df_raw.shape[1]}")
df_raw.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Exploratory data checks & descriptive statistics

# COMMAND ----------

# Quick summary: nulls, types, basic stats — demonstrates descriptive-statistics coverage.
print("=== Null counts ===")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
print()
print("=== Numeric summary ===")
df_raw.describe()

# COMMAND ----------

# Distribution of the target variable (label_sales).
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_raw["label_sales"].dropna(), bins=80, edgecolor="k", alpha=0.7)
axes[0].set_title("label_sales distribution")
axes[0].set_xlabel("Sales")
axes[0].set_ylabel("Count")

axes[1].hist(df_raw["label_log1p_sales"].dropna(), bins=80, edgecolor="k", alpha=0.7, color="orange")
axes[1].set_title("label_log1p_sales (log1p) distribution")
axes[1].set_xlabel("log1p(Sales)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("/tmp/rossmann_target_dist.png", dpi=100)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Feature preparation
# MAGIC
# MAGIC Steps:
# MAGIC 1. Drop metadata columns (timestamps, raw IDs not needed as features).
# MAGIC 2. Encode categorical features (store_type, assortment_type, state_holiday_code) with LabelEncoder.
# MAGIC 3. Fill remaining nulls with sensible defaults.
# MAGIC 4. Define the feature list and target column.

# COMMAND ----------

TARGET = "label_sales"
LOG_TARGET = "label_log1p_sales"

# Columns to exclude from the feature matrix.
DROP_COLS = [
    "business_date",         # used for splitting only
    "silver_loaded_at",
    "gold_loaded_at",
    TARGET,
    LOG_TARGET,
]

# Categorical columns to label-encode (ordinal mapping is sufficient for tree-based models;
# linear models get regularized versions anyway).
CAT_COLS = ["store_type", "assortment_type", "state_holiday_code"]

df = df_raw.copy()

# Encode categoricals as integers — store the encoder for inverse lookup later.
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  encoded {col}: {list(le.classes_)}")

# Fill remaining nulls with 0 (lag/rolling features will be null at series start).
null_before = df.isnull().sum().sum()
df = df.fillna(0)
print(f"\nnulls filled: {null_before:,} → 0")

FEATURES = [c for c in df.columns if c not in DROP_COLS]
print(f"\n{len(FEATURES)} features: {FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Time-based train / validation / test split
# MAGIC
# MAGIC **Rationale**: Rossmann data is time-series, so we must respect temporal order to avoid leakage.
# MAGIC - **Train**: everything before cutoff-1
# MAGIC - **Validation**: between cutoff-1 and cutoff-2
# MAGIC - **Test**: after cutoff-2
# MAGIC
# MAGIC Approximate 70 / 15 / 15 split by date.

# COMMAND ----------

dates = df_raw["business_date"].sort_values().unique()
n = len(dates)
cut1 = dates[int(n * 0.70)]
cut2 = dates[int(n * 0.85)]

mask_train = df_raw["business_date"] < cut1
mask_val   = (df_raw["business_date"] >= cut1) & (df_raw["business_date"] < cut2)
mask_test  = df_raw["business_date"] >= cut2

X_train, y_train = df.loc[mask_train, FEATURES], df.loc[mask_train, TARGET]
X_val,   y_val   = df.loc[mask_val,   FEATURES], df.loc[mask_val,   TARGET]
X_test,  y_test  = df.loc[mask_test,  FEATURES], df.loc[mask_test,  TARGET]

print(f"Train : {X_train.shape[0]:>8,} rows  ({mask_train.sum()/len(df)*100:.1f}%)  up to {cut1}")
print(f"Val   : {X_val.shape[0]:>8,} rows  ({mask_val.sum()/len(df)*100:.1f}%)  {cut1} → {cut2}")
print(f"Test  : {X_test.shape[0]:>8,} rows  ({mask_test.sum()/len(df)*100:.1f}%)  from {cut2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) MLflow experiment setup

# COMMAND ----------

# Set the MLflow experiment; Databricks auto-creates it if missing.
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Helper: train, evaluate, and log a model

# COMMAND ----------

def evaluate_model(y_true, y_pred):
    """Return a dict of regression metrics."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # MAPE: guard against division by zero — exclude rows where actual == 0.
    mask_nonzero = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask_nonzero], y_pred[mask_nonzero]) if mask_nonzero.any() else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def train_and_log(model, model_name, X_tr, y_tr, X_v, y_v, params=None):
    """Fit a model, evaluate on validation, and log everything to MLflow.

    Returns (fitted_model, val_metrics_dict).
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters.
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_features", X_tr.shape[1])
        mlflow.log_param("n_train_rows", X_tr.shape[0])
        mlflow.log_param("n_val_rows", X_v.shape[0])
        if params:
            mlflow.log_params(params)

        # Fit.
        model.fit(X_tr, y_tr)

        # Predict on train & validation sets.
        y_tr_pred = model.predict(X_tr)
        y_v_pred  = model.predict(X_v)

        tr_metrics = evaluate_model(y_tr, y_tr_pred)
        val_metrics = evaluate_model(y_v, y_v_pred)

        # Log metrics with train_ / val_ prefixes.
        for k, v in tr_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        # Log the sklearn model artifact.
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"  [{model_name}]  val_rmse={val_metrics['rmse']:.2f}  val_r2={val_metrics['r2']:.4f}  val_mape={val_metrics['mape']:.4f}")

        return model, val_metrics, run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Train baseline models
# MAGIC
# MAGIC We train a progression from simplest to most complex:
# MAGIC 1. **Linear Regression** — baseline reference (bias-variance trade-off: high bias).
# MAGIC 2. **Ridge / Lasso** — L2 and L1 regularization to prevent overfitting.
# MAGIC 3. **Decision Tree** — non-linear, interpretable, prone to overfitting.
# MAGIC 4. **Random Forest** — bagging ensemble, reduces variance.
# MAGIC 5. **Gradient Boosting (sklearn)** — sequential boosting ensemble.
# MAGIC 6. **LightGBM** — efficient gradient boosting with histogram binning.
# MAGIC 7. **XGBoost** — regularized gradient boosting, widely used in competitions.

# COMMAND ----------

results = {}

# --- 8a) Linear Regression ---
lr = LinearRegression()
_, metrics_lr, _ = train_and_log(lr, "LinearRegression", X_train, y_train, X_val, y_val)
results["LinearRegression"] = metrics_lr

# COMMAND ----------

# --- 8b) Ridge Regression (L2 regularization) ---
ridge = Ridge(alpha=1.0)
_, metrics_ridge, _ = train_and_log(
    ridge, "Ridge", X_train, y_train, X_val, y_val,
    params={"alpha": 1.0, "regularization": "L2"},
)
results["Ridge"] = metrics_ridge

# COMMAND ----------

# --- 8c) Lasso Regression (L1 regularization) ---
lasso = Lasso(alpha=1.0, max_iter=5000)
_, metrics_lasso, _ = train_and_log(
    lasso, "Lasso", X_train, y_train, X_val, y_val,
    params={"alpha": 1.0, "regularization": "L1", "max_iter": 5000},
)
results["Lasso"] = metrics_lasso

# COMMAND ----------

# --- 8d) Decision Tree ---
dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42)
_, metrics_dt, _ = train_and_log(
    dt, "DecisionTree", X_train, y_train, X_val, y_val,
    params={"max_depth": 12, "min_samples_leaf": 20},
)
results["DecisionTree"] = metrics_dt

# COMMAND ----------

# --- 8e) Random Forest (bagging ensemble) ---
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=16,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)
_, metrics_rf, _ = train_and_log(
    rf, "RandomForest", X_train, y_train, X_val, y_val,
    params={"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 10},
)
results["RandomForest"] = metrics_rf

# COMMAND ----------

# --- 8f) Gradient Boosting (sklearn) ---
gb = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)
_, metrics_gb, _ = train_and_log(
    gb, "GradientBoosting", X_train, y_train, X_val, y_val,
    params={"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
)
results["GradientBoosting"] = metrics_gb

# COMMAND ----------

# --- 8g) LightGBM ---
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,       # L1 regularization
    reg_lambda=1.0,      # L2 regularization
    random_state=42,
    verbosity=-1,
)
_, metrics_lgb, run_id_lgb = train_and_log(
    lgb_model, "LightGBM", X_train, y_train, X_val, y_val,
    params={
        "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
        "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
    },
)
results["LightGBM"] = metrics_lgb

# COMMAND ----------

# --- 8h) XGBoost ---
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    verbosity=0,
)
_, metrics_xgb, run_id_xgb = train_and_log(
    xgb_model, "XGBoost", X_train, y_train, X_val, y_val,
    params={
        "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "tree_method": "hist",
    },
)
results["XGBoost"] = metrics_xgb

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Model comparison

# COMMAND ----------

# Build a comparison DataFrame sorted by validation RMSE.
comp_df = pd.DataFrame(results).T
comp_df.index.name = "model"
comp_df = comp_df.sort_values("rmse")
print(comp_df.to_string())

# COMMAND ----------

# Bar chart: validation RMSE by model.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

comp_sorted = comp_df.sort_values("rmse")

axes[0].barh(comp_sorted.index, comp_sorted["rmse"], color="steelblue", edgecolor="k")
axes[0].set_xlabel("RMSE")
axes[0].set_title("Validation RMSE (lower is better)")
axes[0].invert_yaxis()

axes[1].barh(comp_sorted.index, comp_sorted["r2"], color="seagreen", edgecolor="k")
axes[1].set_xlabel("R²")
axes[1].set_title("Validation R² (higher is better)")
axes[1].invert_yaxis()

axes[2].barh(comp_sorted.index, comp_sorted["mape"], color="coral", edgecolor="k")
axes[2].set_xlabel("MAPE")
axes[2].set_title("Validation MAPE (lower is better)")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("/tmp/rossmann_model_comparison.png", dpi=100)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10) Ensemble: Voting Regressor (weighted average)
# MAGIC
# MAGIC Combines the three best tree-based models to demonstrate ensemble techniques.

# COMMAND ----------

# Retrain top-3 for the ensemble (RF, LightGBM, XGBoost).
ensemble = VotingRegressor(
    estimators=[
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=10, n_jobs=-1, random_state=42)),
        ("lgb", lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=63,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                   random_state=42, verbosity=-1)),
        ("xgb", xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
                                   random_state=42, verbosity=0)),
    ],
    n_jobs=-1,
)
_, metrics_ens, run_id_ens = train_and_log(
    ensemble, "VotingEnsemble_RF_LGB_XGB", X_train, y_train, X_val, y_val,
    params={"ensemble_members": "RF+LightGBM+XGBoost", "strategy": "uniform_average"},
)
results["VotingEnsemble"] = metrics_ens

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11) Select champion model & evaluate on test set

# COMMAND ----------

# Pick the model with the lowest validation RMSE as champion.
comp_df_all = pd.DataFrame(results).T.sort_values("rmse")
champion_name = comp_df_all.index[0]
print(f"Champion model (best val RMSE): {champion_name}")

# Map names back to fitted objects for final test evaluation.
model_registry = {
    "LinearRegression": lr,
    "Ridge": ridge,
    "Lasso": lasso,
    "DecisionTree": dt,
    "RandomForest": rf,
    "GradientBoosting": gb,
    "LightGBM": lgb_model,
    "XGBoost": xgb_model,
    "VotingEnsemble": ensemble,
}

champion_model = model_registry[champion_name]

# Final evaluation on the held-out test set.
y_test_pred = champion_model.predict(X_test)
test_metrics = evaluate_model(y_test, y_test_pred)

print(f"\n=== Test-set metrics for {champion_name} ===")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# Actual vs Predicted scatter plot on test set.
fig, ax = plt.subplots(figsize=(7, 7))
sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(5000, len(y_test)), replace=False)
ax.scatter(y_test.values[sample_idx], y_test_pred[sample_idx], alpha=0.25, s=8, color="steelblue")
lims = [0, max(y_test.max(), y_test_pred.max()) * 1.05]
ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="perfect prediction")
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title(f"Test set: {champion_name} — Actual vs Predicted")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/rossmann_actual_vs_pred.png", dpi=100)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12) Feature importance (tree-based champion)

# COMMAND ----------

# Feature importance works for tree-based models.
if hasattr(champion_model, "feature_importances_"):
    fi = pd.Series(champion_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, len(fi) * 0.35)))
    fi.plot.barh(ax=ax, color="teal", edgecolor="k")
    ax.set_title(f"Feature Importance — {champion_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("/tmp/rossmann_feature_importance.png", dpi=100)
    plt.show()
elif hasattr(champion_model, "estimators_"):
    # VotingRegressor: show importances from the first estimator that has them.
    for name, est in champion_model.named_estimators_.items():
        if hasattr(est, "feature_importances_"):
            fi = pd.Series(est.feature_importances_, index=FEATURES).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(6, len(fi) * 0.35)))
            fi.plot.barh(ax=ax, color="teal", edgecolor="k")
            ax.set_title(f"Feature Importance — {champion_name} (via {name})")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            plt.savefig("/tmp/rossmann_feature_importance.png", dpi=100)
            plt.show()
            break
else:
    print("Champion model does not expose feature_importances_.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13) Register champion model in Unity Catalog Model Registry

# COMMAND ----------

if LOG_TO_UC_REGISTRY:
    # UC model registry uses three-level namespace: catalog.schema.model_name
    UC_MODEL_NAME = f"{CATALOG}.ml.rossmann_sales_champion"

    # Re-log the champion with its test metrics for a clean registry artifact.
    with mlflow.start_run(run_name=f"{champion_name}_champion_registration") as run:
        mlflow.log_param("champion_model", champion_name)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.sklearn.log_model(
            champion_model,
            artifact_path="champion_model",
            registered_model_name=UC_MODEL_NAME,
        )
        champion_run_id = run.info.run_id

    print(f"Champion registered as: {UC_MODEL_NAME}")
    print(f"Run ID: {champion_run_id}")
else:
    print("UC registry logging skipped (parameter log_to_uc_registry=false)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14) Write predictions to demo.ml.rossmann_predictions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.ml;

# COMMAND ----------

# Build a predictions dataframe from the test set.
pred_df = df_raw.loc[mask_test, ["store_id", "business_date"]].copy()
pred_df["actual_sales"] = y_test.values
pred_df["predicted_sales"] = y_test_pred
pred_df["residual"] = pred_df["actual_sales"] - pred_df["predicted_sales"]
pred_df["model_name"] = champion_name
pred_df["prediction_ts"] = pd.Timestamp.now()

# Write to Unity Catalog via a temp view + SQL INSERT.
spark_pred = spark.createDataFrame(pred_df)
spark_pred.createOrReplaceTempView("tmp_rossmann_preds")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.ml.rossmann_predictions AS
    SELECT * FROM tmp_rossmann_preds
""")

row_count = spark.sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.ml.rossmann_predictions").collect()[0]["cnt"]
print(f"Predictions written to {CATALOG}.ml.rossmann_predictions — {row_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15) Audit snapshot

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO demo.audit.gold_snapshot
# MAGIC SELECT current_timestamp(), 'demo.ml.rossmann_predictions', COUNT(*)
# MAGIC FROM demo.ml.rossmann_predictions;
# MAGIC
# MAGIC SELECT * FROM demo.audit.gold_snapshot
# MAGIC WHERE table_name LIKE '%rossmann_predictions%'
# MAGIC ORDER BY snapshot_ts DESC
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16) Summary
# MAGIC
# MAGIC | Step | Status |
# MAGIC |---|---|
# MAGIC | Gold mart → pandas | Done |
# MAGIC | Time-based split (70/15/15) | Done |
# MAGIC | Linear Regression | Done |
# MAGIC | Ridge (L2) | Done |
# MAGIC | Lasso (L1) | Done |
# MAGIC | Decision Tree | Done |
# MAGIC | Random Forest | Done |
# MAGIC | Gradient Boosting | Done |
# MAGIC | LightGBM | Done |
# MAGIC | XGBoost | Done |
# MAGIC | Voting Ensemble | Done |
# MAGIC | MLflow logging | Done |
# MAGIC | UC Model Registry | Done |
# MAGIC | Predictions persisted | Done |
# MAGIC
# MAGIC **Next steps:**
# MAGIC - `04_deep_learning_experiments.py` — FNN with Keras, optimizer comparison, mixed precision.
# MAGIC - `05_nyc_demand_model.py` — NYC TLC demand forecasting with GBM + NN.
