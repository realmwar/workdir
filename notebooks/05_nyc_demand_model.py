# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — NYC TLC Demand Forecasting (Phase 2 Mini-Project)
# MAGIC
# MAGIC **Goals:**
# MAGIC - Load `demo.gold.nyc_demand_training_mart` and build a demand forecasting pipeline.
# MAGIC - Train GBM baseline (LightGBM) + simple FNN for zone-hour demand prediction.
# MAGIC - Apply time-series validation strategy.
# MAGIC - Register best model in UC Model Registry.
# MAGIC - Write predictions to `demo.ml.nyc_demand_predictions`.
# MAGIC
# MAGIC **Constraints:**
# MAGIC - No Spark DataFrame API — SQL for reads, pandas + sklearn/keras/mlflow for ML.
# MAGIC - All code in English.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Install dependencies (Serverless)

# COMMAND ----------

# MAGIC %md
# MAGIC Serverless ships with `numpy`, `pandas`, `sklearn`, `matplotlib`, and `mlflow`, but the gradient
# MAGIC boosting libraries (`lightgbm`, `xgboost`) and TensorFlow are not preinstalled. The TensorFlow
# MAGIC stack is installed exactly the same way as in notebook 04 because that combination is already
# MAGIC verified to work on this Databricks setup.

# COMMAND ----------

# MAGIC %md
# MAGIC This cell installs the libraries that are not guaranteed to be present on Databricks Serverless.
# MAGIC We keep the TensorFlow/Keras/protobuf combination aligned with notebook 04, then install
# MAGIC LightGBM and XGBoost separately so the boosting packages do not interfere with the TensorFlow
# MAGIC resolver path.

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall "tensorflow==2.15.1" "keras==2.15.0" "tensorflow-model-optimization==0.8.0" "protobuf==4.25.3"

# COMMAND ----------

# MAGIC %pip install "lightgbm" "xgboost"

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks needs a Python restart after `%pip install` so the current notebook process can import
# MAGIC the newly installed packages. After this restart, run the imports cell again before continuing.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Imports & setup

# COMMAND ----------

# MAGIC %md
# MAGIC This cell prepares the runtime for the NYC demand modeling work. Pandas and NumPy handle the local
# MAGIC tabular data, matplotlib renders comparison charts, LightGBM/XGBoost/TensorFlow provide the model
# MAGIC families, and MLflow tracks every run under a Serverless-safe experiment path.

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers

import mlflow
import mlflow.sklearn
import mlflow.keras
from mlflow.models import infer_signature

CATALOG = "demo"
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _user = _ctx.userName().get()
    EXPERIMENT_NAME = f"/Users/{_user}/nyc_demand_forecasting"
except Exception:
    EXPERIMENT_NAME = "/Shared/nyc_demand_forecasting"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

keras = tf.keras

print("setup OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Load NYC demand training mart

# COMMAND ----------

# MAGIC %md
# MAGIC This is the handoff from the gold layer into model training. The table already contains the
# MAGIC zone-hour features produced by earlier notebooks, so this cell only loads it into pandas and
# MAGIC prints a small shape/head check before we start splitting and modeling.

# COMMAND ----------

df_raw = spark.sql(f"SELECT * FROM {CATALOG}.gold.nyc_demand_training_mart").toPandas()
print(f"rows: {df_raw.shape[0]:,}  cols: {df_raw.shape[1]}")
df_raw.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Feature preparation

# COMMAND ----------

# MAGIC %md
# MAGIC Here we define the prediction target and remove columns that should not become model inputs.
# MAGIC `pickup_date` is kept out of the feature matrix because it is only used to create a temporal split,
# MAGIC and the load timestamps are pipeline metadata rather than demand signals.

# COMMAND ----------

TARGET = "label_trip_cnt"

DROP_COLS = [
    "pickup_date",       # used for temporal split only
    "silver_loaded_at",
    "gold_loaded_at",
    TARGET,
]

df = df_raw.copy()
df = df.fillna(0)

FEATURES = [c for c in df.columns if c not in DROP_COLS]
print(f"{len(FEATURES)} features: {FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Time-based train / validation / test split

# COMMAND ----------

# MAGIC %md
# MAGIC Demand forecasting should be validated forward in time, not with a random split. This cell uses
# MAGIC the earliest 70% of dates for training, the next 15% for validation, and the most recent 15% for
# MAGIC the final test set so the evaluation resembles future demand prediction.

# COMMAND ----------

dates = df_raw["pickup_date"].sort_values().unique()
n = len(dates)
cut1, cut2 = dates[int(n * 0.70)], dates[int(n * 0.85)]

mask_train = df_raw["pickup_date"] < cut1
mask_val   = (df_raw["pickup_date"] >= cut1) & (df_raw["pickup_date"] < cut2)
mask_test  = df_raw["pickup_date"] >= cut2

X_train_raw = df.loc[mask_train, FEATURES].values
X_val_raw   = df.loc[mask_val,   FEATURES].values
X_test_raw  = df.loc[mask_test,  FEATURES].values
y_train = df.loc[mask_train, TARGET].values
y_val   = df.loc[mask_val,   TARGET].values
y_test  = df.loc[mask_test,  TARGET].values

print(f"Train: {X_train_raw.shape}  Val: {X_val_raw.shape}  Test: {X_test_raw.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Helper: evaluation metrics

# COMMAND ----------

# MAGIC %md
# MAGIC This helper keeps model evaluation consistent across LightGBM, XGBoost, and the neural network.
# MAGIC Every model gets the same regression scorecard: MSE/RMSE for error magnitude, MAE for average
# MAGIC absolute miss, R² for explained variance, and MAPE where the true value is non-zero.

# COMMAND ----------

def eval_metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mask_nz = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask_nz], y_pred[mask_nz]) if mask_nz.any() else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Model 1: LightGBM baseline

# COMMAND ----------

# MAGIC %md
# MAGIC LightGBM is the first serious baseline for the NYC zone-hour demand problem. It works well on
# MAGIC tabular data, handles non-linear interactions without manual feature crosses, and trains quickly
# MAGIC enough to be practical inside a notebook experiment.

# COMMAND ----------

results = {}

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
)

with mlflow.start_run(run_name="NYC_LightGBM"):
    lgb_params = {
        "model": "LightGBM", "n_estimators": 500, "max_depth": 8,
        "learning_rate": 0.05, "num_leaves": 63,
    }
    mlflow.log_params(lgb_params)

    lgb_model.fit(X_train_raw, y_train)

    y_val_lgb  = lgb_model.predict(X_val_raw)
    y_test_lgb = lgb_model.predict(X_test_raw)

    val_lgb  = eval_metrics(y_val, y_val_lgb)
    test_lgb = eval_metrics(y_test, y_test_lgb)
    for k, v in val_lgb.items():
        mlflow.log_metric(f"val_{k}", v)
    for k, v in test_lgb.items():
        mlflow.log_metric(f"test_{k}", v)

    mlflow.sklearn.log_model(lgb_model, artifact_path="model")
    run_id_lgb = mlflow.active_run().info.run_id

results["LightGBM"] = {**val_lgb, "test_rmse": test_lgb["rmse"]}
print(f"  [LightGBM] val_rmse={val_lgb['rmse']:.2f}  val_r2={val_lgb['r2']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Model 2: XGBoost

# COMMAND ----------

# MAGIC %md
# MAGIC XGBoost gives us a second gradient boosting implementation to compare against LightGBM. The setup is
# MAGIC intentionally close to the LightGBM run so the comparison is mostly about the algorithm and training
# MAGIC behavior, not a completely different feature set or validation strategy.

# COMMAND ----------

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    verbosity=0,
)

with mlflow.start_run(run_name="NYC_XGBoost"):
    mlflow.log_params({"model": "XGBoost", "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05})

    xgb_model.fit(X_train_raw, y_train)
    y_val_xgb  = xgb_model.predict(X_val_raw)
    y_test_xgb = xgb_model.predict(X_test_raw)

    val_xgb  = eval_metrics(y_val, y_val_xgb)
    test_xgb = eval_metrics(y_test, y_test_xgb)
    for k, v in val_xgb.items():
        mlflow.log_metric(f"val_{k}", v)
    for k, v in test_xgb.items():
        mlflow.log_metric(f"test_{k}", v)

    mlflow.sklearn.log_model(xgb_model, artifact_path="model")
    run_id_xgb = mlflow.active_run().info.run_id

results["XGBoost"] = {**val_xgb, "test_rmse": test_xgb["rmse"]}
print(f"  [XGBoost] val_rmse={val_xgb['rmse']:.2f}  val_r2={val_xgb['r2']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Model 3: Feedforward Neural Network (FNN)

# COMMAND ----------

# MAGIC %md
# MAGIC The neural network experiment tests whether a simple dense model can compete with the tree-based
# MAGIC baselines on the same gold mart. Unlike boosted trees, the FNN needs standardized inputs so the
# MAGIC optimizer sees features on comparable scales.

# COMMAND ----------

# NN requires standardized features.
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_raw)
X_val_nn   = scaler.transform(X_val_raw)
X_test_nn  = scaler.transform(X_test_raw)

nn_model = keras.Sequential([
    layers.Input(shape=(X_train_nn.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1),
], name="NYC_FNN")

nn_model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

with mlflow.start_run(run_name="NYC_FNN"):
    mlflow.log_params({
        "model": "FNN", "hidden_layers": "256-128-64",
        "activation": "relu", "optimizer": "adam", "lr": 1e-3,
    })

    history = nn_model.fit(
        X_train_nn, y_train,
        validation_data=(X_val_nn, y_val),
        epochs=100, batch_size=512, verbose=0,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ],
    )

    y_val_nn  = nn_model.predict(X_val_nn, verbose=0).flatten()
    y_test_nn = nn_model.predict(X_test_nn, verbose=0).flatten()

    val_nn  = eval_metrics(y_val, y_val_nn)
    test_nn = eval_metrics(y_test, y_test_nn)
    for k, v in val_nn.items():
        mlflow.log_metric(f"val_{k}", v)
    for k, v in test_nn.items():
        mlflow.log_metric(f"test_{k}", v)

    mlflow.keras.log_model(nn_model, artifact_path="model")

results["FNN"] = {**val_nn, "test_rmse": test_nn["rmse"]}
print(f"  [FNN] val_rmse={val_nn['rmse']:.2f}  val_r2={val_nn['r2']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Model comparison

# COMMAND ----------

# MAGIC %md
# MAGIC The three model runs have been collecting validation metrics in `results`. This cell turns that
# MAGIC dictionary into a sorted comparison table so the current champion is selected from validation
# MAGIC performance rather than from preference for a particular model family.

# COMMAND ----------

comp_df = pd.DataFrame(results).T.sort_values("rmse")
comp_df.index.name = "model"
print(comp_df.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC This chart is the visual version of the comparison table. RMSE shows the absolute demand error,
# MAGIC while R² shows how much of the variation in zone-hour trip demand each model explains.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

comp_sorted = comp_df.sort_values("rmse")
axes[0].barh(comp_sorted.index, comp_sorted["rmse"], color="steelblue", edgecolor="k")
axes[0].set_xlabel("RMSE"); axes[0].set_title("NYC Demand — Validation RMSE")
axes[0].invert_yaxis()

axes[1].barh(comp_sorted.index, comp_sorted["r2"], color="seagreen", edgecolor="k")
axes[1].set_xlabel("R²"); axes[1].set_title("NYC Demand — Validation R²")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("/tmp/nyc_model_comparison.png", dpi=100)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10) Feature importance (LightGBM)

# COMMAND ----------

# MAGIC %md
# MAGIC Feature importance is a quick sanity check for the tree-based demand model. If temporal, location,
# MAGIC and lag-style features dominate, the model is likely using the intended demand signals rather than
# MAGIC leaning on accidental metadata.

# COMMAND ----------

fi = pd.Series(lgb_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, max(5, len(fi) * 0.4)))
fi.plot.barh(ax=ax, color="teal", edgecolor="k")
ax.set_title("LightGBM Feature Importance — NYC Demand")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("/tmp/nyc_feature_importance.png", dpi=100)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11) Register best model in UC Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC This cell promotes the best validation model into the Unity Catalog Model Registry. The champion
# MAGIC can be LightGBM, XGBoost, or the FNN; we map the selected name back to the fitted object, log the
# MAGIC final test metrics, and register it under the stable `demo.ml.nyc_demand_champion` name.

# COMMAND ----------

champion_name = comp_df.index[0]
print(f"NYC champion: {champion_name}")

# Map to fitted model objects.
model_map = {"LightGBM": lgb_model, "XGBoost": xgb_model, "FNN": nn_model}
champion = model_map[champion_name]

UC_MODEL_NAME = f"{CATALOG}.ml.nyc_demand_champion"

with mlflow.start_run(run_name=f"NYC_{champion_name}_champion_registration"):
    mlflow.log_param("champion_model", champion_name)
    champion_test_pred = (
        y_test_lgb if champion_name == "LightGBM"
        else y_test_xgb if champion_name == "XGBoost"
        else y_test_nn
    )
    test_m = eval_metrics(y_test, champion_test_pred)
    champion_signature = infer_signature(
        X_test_nn if champion_name == "FNN" else X_test_raw,
        champion_test_pred,
    )
    for k, v in test_m.items():
        mlflow.log_metric(f"test_{k}", v)

    if isinstance(champion, keras.Model):
        mlflow.keras.log_model(
            champion,
            artifact_path="champion_model",
            registered_model_name=UC_MODEL_NAME,
            signature=champion_signature,
        )
    else:
        mlflow.sklearn.log_model(
            champion,
            artifact_path="champion_model",
            registered_model_name=UC_MODEL_NAME,
            signature=champion_signature,
        )

print(f"Registered: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12) Write predictions to demo.ml.nyc_demand_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC After selecting the champion, we persist its test-set predictions as a serving and dashboard table.
# MAGIC The table keeps the original zone-hour keys, the actual trip count, the prediction, the residual,
# MAGIC the model name, and a timestamp so downstream consumers do not have to recompute these fields.

# COMMAND ----------

# Use champion predictions on the test set.
y_pred_champion = (
    y_test_lgb if champion_name == "LightGBM"
    else y_test_xgb if champion_name == "XGBoost"
    else y_test_nn
)

pred_df = df_raw.loc[mask_test, ["pickup_date", "pickup_hour", "pu_location_id"]].copy()
pred_df["actual_trip_cnt"] = y_test
pred_df["predicted_trip_cnt"] = y_pred_champion
pred_df["residual"] = pred_df["actual_trip_cnt"] - pred_df["predicted_trip_cnt"]
pred_df["model_name"] = champion_name
pred_df["prediction_ts"] = pd.Timestamp.now()

spark_pred = spark.createDataFrame(pred_df)
spark_pred.createOrReplaceTempView("tmp_nyc_preds")

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.ml.nyc_demand_predictions AS
    SELECT * FROM tmp_nyc_preds
""")

row_count = spark.sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.ml.nyc_demand_predictions").collect()[0]["cnt"]
print(f"Predictions written: {CATALOG}.ml.nyc_demand_predictions — {row_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13) Audit snapshot

# COMMAND ----------

# MAGIC %md
# MAGIC The audit snapshot records the row count of the predictions table after this notebook writes it.
# MAGIC That gives us a simple pipeline-health breadcrumb for later dashboarding and job monitoring.

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO demo.audit.gold_snapshot
# MAGIC SELECT current_timestamp(), 'demo.ml.nyc_demand_predictions', COUNT(*)
# MAGIC FROM demo.ml.nyc_demand_predictions;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14) Summary
# MAGIC
# MAGIC | Model | Description |
# MAGIC |---|---|
# MAGIC | LightGBM | GBM baseline for zone-hour demand |
# MAGIC | XGBoost | Alternative GBM for comparison |
# MAGIC | FNN | Neural network for demand forecasting |
# MAGIC
# MAGIC **Next step:** `06_serving_contracts.py` — batch inference and serving table schemas.
