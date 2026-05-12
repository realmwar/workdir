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

# MAGIC %md
# MAGIC A few runtime knobs that make the notebook controllable without editing code.
# MAGIC The `log_to_uc_registry` widget lets us iterate on training logic without polluting the
# MAGIC Unity Catalog Model Registry with throwaway versions — we flip it off while experimenting
# MAGIC and back on for the actual champion run. The catalog name is pulled into a single constant
# MAGIC so the SQL below stays portable if we ever rename the workspace.
# MAGIC
# MAGIC The MLflow experiment path has a small wrinkle: on Databricks Free running under Spark Connect,
# MAGIC the usual `spark.conf.get("spark.databricks.notebook.path")` trick raises instead of returning the
# MAGIC default, so we resolve the path through the `dbutils` notebook context and fall back to `/Shared/`
# MAGIC if anything refuses to resolve. The end goal is simple — every Rossmann run we do here lands in
# MAGIC the same MLflow experiment, which is the only way to make models and versions actually comparable.

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

# MAGIC %md
# MAGIC The working set of libraries for the rest of the notebook. Pandas and NumPy carry the tabular
# MAGIC work, matplotlib draws the charts (forced into the non-interactive `Agg` backend because we're
# MAGIC inside Databricks, not a Jupyter window), and from sklearn we pull in the full spectrum of
# MAGIC regressors we want to benchmark side by side — plain linear regression, ridge and lasso for
# MAGIC regularization, a single decision tree, random forest, gradient boosting, and a voting ensemble
# MAGIC on top. MLflow is the recording layer that captures parameters, metrics, and model artifacts
# MAGIC for every run.
# MAGIC
# MAGIC The `warnings.filterwarnings("ignore")` line is purely cosmetic — sklearn and the boosting libs
# MAGIC are noisy by default and their deprecation chatter tends to drown out the actual training output.

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

# MAGIC %md
# MAGIC This is the entry point for everything that follows. We hit the gold training mart that notebook 02
# MAGIC already prepared — `demo.gold.rossmann_training_mart` — and pull the whole thing into pandas via a
# MAGIC one-shot SQL query. From this line on, no more Spark DataFrame API: every transformation, split,
# MAGIC model fit, and metric runs in pandas/numpy/sklearn.
# MAGIC
# MAGIC The reason we materialize the entire mart in memory is that classical ML on tabular data wins
# MAGIC nothing from distributed execution at this scale, and the gold contract is intentionally narrow
# MAGIC enough to fit comfortably. The `head(3)` at the end is a quick visual check that the columns and
# MAGIC types look like what notebook 02 declared.

# COMMAND ----------

# Pull the full training mart via SQL; convert to pandas immediately.
df_raw = spark.sql(f"SELECT * FROM {CATALOG}.gold.rossmann_training_mart").toPandas()

print(f"rows: {df_raw.shape[0]:,}  cols: {df_raw.shape[1]}")
df_raw.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Exploratory data checks & descriptive statistics

# COMMAND ----------

# MAGIC %md
# MAGIC Two quick sanity probes before we touch anything else. The null counts highlight any columns
# MAGIC where the lag/rolling features didn't fully populate — they shouldn't have meaningful gaps,
# MAGIC but if the upstream silver/gold rebuild went wrong, this is where we'd see it first.
# MAGIC `describe()` gives us the full numeric summary in one shot — means, mins, maxes, quartiles —
# MAGIC which makes it obvious if a column suddenly has unreasonable ranges or zero variance.
# MAGIC
# MAGIC These checks are cheap and they catch silent regressions early. We'd rather find out about
# MAGIC a broken pipeline here than after spending 20 minutes training a useless model on it.

# COMMAND ----------

# Quick summary: nulls, types, basic stats — demonstrates descriptive-statistics coverage.
print("=== Null counts ===")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
print()
print("=== Numeric summary ===")
df_raw.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC A side-by-side look at the target variable in raw and `log1p`-transformed space. Rossmann sales
# MAGIC are heavily right-skewed — a long tail of high-volume store-days — so a linear model would
# MAGIC struggle to fit both ends gracefully without some form of transformation. The `log1p` version
# MAGIC looks much closer to normal, which is why we kept it as an alternate label in the gold mart.
# MAGIC
# MAGIC For this notebook we still train on raw `label_sales` to keep the metric values in their
# MAGIC natural business units, but having the visual reminder here helps us interpret the residuals
# MAGIC later: if a particular model is systematically wrong on the big stores, that's a skew problem,
# MAGIC not a feature problem.

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

# MAGIC %md
# MAGIC This is the bridge between the SQL-built gold mart and a model-ready feature matrix. The
# MAGIC `DROP_COLS` list strips out columns that exist for bookkeeping rather than prediction —
# MAGIC timestamps, the raw target, and the log1p alternate. The three categorical columns
# MAGIC (`store_type`, `assortment_type`, `state_holiday_code`) get label-encoded into integer ordinals;
# MAGIC that's perfectly fine for tree-based models, and the linear models we train are mostly there
# MAGIC as a reference rather than to chase the best score.
# MAGIC
# MAGIC The `fillna(0)` is intentional and safe in this context — the only nulls left at this point
# MAGIC come from lag and rolling features at the very start of each store's history, where zero is a
# MAGIC reasonable proxy for "no prior data". The final `FEATURES` list is the canonical feature
# MAGIC contract that every model in this notebook will consume.

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

# MAGIC %md
# MAGIC Time-series data demands a chronological split — anything else leaks future information backward
# MAGIC and inflates validation scores beyond recognition. We sort all unique dates, take the 70th and
# MAGIC 85th percentile as cutoffs, and slice the dataframe into three windows: train is the earliest
# MAGIC 70% of dates, validation is the next 15%, test is the most recent 15%.
# MAGIC
# MAGIC The printed summary confirms the row counts and date boundaries, which is the first thing to
# MAGIC check when something looks suspicious downstream — a model with weirdly high validation scores
# MAGIC is almost always a split that quietly let some future data slip in.

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

# MAGIC %md
# MAGIC One line, but it's the line that ties the rest of the notebook together. `mlflow.set_experiment(...)`
# MAGIC makes sure every `start_run()` call below registers into the same experiment we resolved at the top.
# MAGIC If the path doesn't exist yet, Databricks creates it on the fly. From here on, every model we
# MAGIC train lands as a sibling run under the same tree — which is exactly what we need to compare
# MAGIC them side by side in the MLflow UI later.

# COMMAND ----------

# Set the MLflow experiment; Databricks auto-creates it if missing.
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Helper: train, evaluate, and log a model

# COMMAND ----------

# MAGIC %md
# MAGIC Two helpers that absorb all the repetition out of the next eight code cells. `evaluate_model`
# MAGIC packs MSE, RMSE, MAE, R², and MAPE into a single dict — MAPE gets a small guard to skip rows
# MAGIC where the actual value is zero, because dividing by zero is the fastest way to inject NaNs
# MAGIC into your metrics table.
# MAGIC
# MAGIC `train_and_log` is the real workhorse: open an MLflow run, log the hyperparameters, fit on train,
# MAGIC predict on train and validation, log both sets of metrics with `train_` and `val_` prefixes,
# MAGIC persist the fitted model as an artifact, and print a one-line summary. Every model below calls
# MAGIC into this function so the experiment tracking stays uniform — which is what makes the comparison
# MAGIC at the end of the notebook actually meaningful.

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

# MAGIC %md
# MAGIC The bias-variance baseline. Plain linear regression has no hyperparameters to tune and no
# MAGIC capacity to model interactions, so whatever score it gets is the floor — everything we do later
# MAGIC is judged against "is it actually better than a line?". On a heavily non-linear problem like
# MAGIC Rossmann sales, we fully expect this to be substantially worse than the tree-based models,
# MAGIC and that's exactly the point of keeping it in the lineup.

# COMMAND ----------

results = {}

# --- 8a) Linear Regression ---
lr = LinearRegression()
_, metrics_lr, _ = train_and_log(lr, "LinearRegression", X_train, y_train, X_val, y_val)
results["LinearRegression"] = metrics_lr

# COMMAND ----------

# MAGIC %md
# MAGIC Ridge is linear regression with L2 regularization — it pulls all coefficients toward zero
# MAGIC proportionally to their magnitude. For our feature set it's not really expected to outperform
# MAGIC plain linear regression by much (we don't have a high-multicollinearity problem), but it's
# MAGIC the textbook example of L2 and it costs us nothing to include it. The point is the explicit
# MAGIC demonstration that we know what L2 regularization does and how to apply it.

# COMMAND ----------

# --- 8b) Ridge Regression (L2 regularization) ---
ridge = Ridge(alpha=1.0)
_, metrics_ridge, _ = train_and_log(
    ridge, "Ridge", X_train, y_train, X_val, y_val,
    params={"alpha": 1.0, "regularization": "L2"},
)
results["Ridge"] = metrics_ridge

# COMMAND ----------

# MAGIC %md
# MAGIC Lasso is the L1 counterpart — same idea as Ridge, but the penalty drives some coefficients all
# MAGIC the way to zero, which makes it implicitly do feature selection. The `max_iter=5000` is just to
# MAGIC silence the convergence warnings on this particular feature scale; the default 1000 isn't always
# MAGIC enough for the solver to settle.
# MAGIC
# MAGIC Plain linear regression + Ridge + Lasso together cover the regularization spectrum cleanly,
# MAGIC which is one of the explicit asks in the model-pipeline competency: show that you understand
# MAGIC L1, L2, and no-penalty as three distinct points on the same continuum.

# COMMAND ----------

# --- 8c) Lasso Regression (L1 regularization) ---
lasso = Lasso(alpha=1.0, max_iter=5000)
_, metrics_lasso, _ = train_and_log(
    lasso, "Lasso", X_train, y_train, X_val, y_val,
    params={"alpha": 1.0, "regularization": "L1", "max_iter": 5000},
)
results["Lasso"] = metrics_lasso

# COMMAND ----------

# MAGIC %md
# MAGIC Our first non-linear model. A single decision tree can capture interactions and step-function
# MAGIC patterns that no linear model ever will, but it overfits aggressively if you let it grow
# MAGIC unconstrained. We cap depth at 12 and require at least 20 samples per leaf — enough to learn
# MAGIC meaningful structure without memorizing individual store-days.
# MAGIC
# MAGIC A single tree is rarely the best model in production, but it's an excellent diagnostic: if the
# MAGIC tree massively outperforms the linear family, we know the signal is non-linear; if the random
# MAGIC forest below outperforms the tree by a lot, we know variance is the bigger enemy than bias.

# COMMAND ----------

# --- 8d) Decision Tree ---
dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42)
_, metrics_dt, _ = train_and_log(
    dt, "DecisionTree", X_train, y_train, X_val, y_val,
    params={"max_depth": 12, "min_samples_leaf": 20},
)
results["DecisionTree"] = metrics_dt

# COMMAND ----------

# MAGIC %md
# MAGIC This is where things usually start to get respectable. 200 trees, each grown to depth 16 with a
# MAGIC minimum leaf size of 10, all trained on bootstrap samples of the data and averaged at prediction
# MAGIC time. The averaging dramatically reduces the variance that hurt the single tree, and the result
# MAGIC is typically much closer to what a gradient boosting model achieves — though it gets there by
# MAGIC a completely different mechanism.
# MAGIC
# MAGIC `n_jobs=-1` runs the trees in parallel across all cores. Random Forest is embarrassingly parallel
# MAGIC by construction (each tree is independent), so this is essentially a free speedup with no
# MAGIC accuracy trade-off.

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

# MAGIC %md
# MAGIC Gradient boosting takes a fundamentally different approach from Random Forest: instead of
# MAGIC averaging many strong learners trained independently, it builds trees sequentially where each
# MAGIC new tree corrects the errors of the previous ones. This usually gives a noticeable accuracy bump
# MAGIC over RF, at the cost of training time and a much higher sensitivity to hyperparameters.
# MAGIC
# MAGIC We use sklearn's reference implementation here mostly for completeness — the LightGBM and
# MAGIC XGBoost variants below are faster and stronger, but having the canonical gradient boosting in
# MAGIC the comparison shows where the gains actually come from (the algorithm itself) versus the
# MAGIC implementation tricks (histogram binning, leaf-wise growth) that the modern libraries add on top.

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

# MAGIC %md
# MAGIC LightGBM is gradient boosting reimagined for speed. The key trick is histogram-based feature
# MAGIC binning — instead of evaluating every possible split point, it groups feature values into a
# MAGIC fixed number of bins, which collapses the split search from O(n) to O(bins). On a dataset our
# MAGIC size this is dramatically faster than sklearn's gradient boosting, and the regularization
# MAGIC parameters (`reg_alpha` for L1, `reg_lambda` for L2) give us a knob for overfitting control
# MAGIC that the sklearn version lacks.
# MAGIC
# MAGIC `num_leaves=63` is the LightGBM way of controlling tree complexity — it grows leaf-wise rather
# MAGIC than depth-wise, so you tune the number of leaves directly rather than the depth. This is one
# MAGIC of the small things that makes LightGBM both faster and a bit easier to overfit if you're
# MAGIC not careful with the leaf count.

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

# MAGIC %md
# MAGIC XGBoost is the other major gradient boosting library. The algorithm is similar enough to LightGBM
# MAGIC that on a clean tabular problem like this one they tend to land within a couple of RMSE points
# MAGIC of each other, but they have different strengths in different scenarios — and the practical
# MAGIC reality is that one or the other is the right tool for whatever ML stack you join.
# MAGIC
# MAGIC `tree_method="hist"` enables the same histogram trick LightGBM uses by default. Without it,
# MAGIC XGBoost falls back to its exact greedy algorithm, which on our dataset is significantly slower
# MAGIC for no real accuracy gain.

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

# MAGIC %md
# MAGIC All the validation metrics we've been logging into the `results` dict get pivoted into a single
# MAGIC comparison table, sorted by RMSE so the best model floats to the top. This is the moment where
# MAGIC the eight separate training cells turn into an actual decision: which model family won, by how
# MAGIC much, and is the gap big enough to matter — or are two implementations of the same idea
# MAGIC essentially tied and we should pick on secondary criteria like training time or interpretability.

# COMMAND ----------

# Build a comparison DataFrame sorted by validation RMSE.
comp_df = pd.DataFrame(results).T
comp_df.index.name = "model"
comp_df = comp_df.sort_values("rmse")
print(comp_df.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC Three side-by-side bar charts: RMSE (lower is better), R² (higher is better), MAPE (lower is
# MAGIC better). Visualizing all three at once helps catch cases where a model wins on one metric and
# MAGIC loses on another — usually a sign that it's particularly good or bad at handling the
# MAGIC high-volume end of the distribution. MAPE in particular tends to disagree with RMSE when a
# MAGIC model is well-calibrated on small stores but biased on the few very large ones.

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

# MAGIC %md
# MAGIC The voting regressor takes the three strongest tree-based models (RF, LightGBM, XGBoost) and
# MAGIC averages their predictions. This is the simplest possible ensemble technique — no stacking, no
# MAGIC weighting, just a plain mean — and it works because the underlying models make different kinds
# MAGIC of mistakes. Where one model is biased low on a particular store, another is often biased high,
# MAGIC and the average is closer to truth than any individual prediction.
# MAGIC
# MAGIC For a serious production setup we'd weight the models by validation performance or fit a stacking
# MAGIC meta-model on top, but the uniform average is a clean demonstration of the ensemble idea and it
# MAGIC almost always nudges the metrics a bit better than the best single model.

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

# MAGIC %md
# MAGIC Champion selection is just "lowest validation RMSE wins". We pull the actual fitted estimator
# MAGIC out of the local registry dictionary and run it on the test set — this is the first time the
# MAGIC test set has been touched in the entire notebook, which is the whole point of holding it out
# MAGIC from validation. The test metrics are what we'd report externally; the validation metrics are
# MAGIC what we used to pick the model.
# MAGIC
# MAGIC If validation and test metrics diverge meaningfully, that's a signal the validation window was
# MAGIC unrepresentative — worth investigating before promoting anything to production.

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

# MAGIC %md
# MAGIC A scatter of actual versus predicted sales on the test set, with the diagonal drawn in for
# MAGIC reference. We subsample to 5000 points so the plot stays readable on a notebook canvas —
# MAGIC at full resolution everything turns into a solid blue cloud and you can't see anything.
# MAGIC
# MAGIC Points hugging the diagonal mean accurate predictions; systematic drift above or below the
# MAGIC line means the model has a bias we should investigate before deploying anything. A horizontal
# MAGIC band at the high end usually means the model is under-predicting the largest stores — a
# MAGIC classic symptom of training on raw sales without addressing the right-skew we saw in section 3.

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

# MAGIC %md
# MAGIC For tree-based champions we plot which features the model actually leaned on. The conditional
# MAGIC logic handles three cases: a direct estimator with `feature_importances_` (the common case for
# MAGIC RF/GBM/LightGBM/XGBoost), a voting ensemble where we dig into the first named estimator that
# MAGIC has the attribute, and a fallback message if neither applies (e.g. if a linear model somehow
# MAGIC ended up being the champion).
# MAGIC
# MAGIC The output is a quick reality check on the features: lag and rolling features almost always
# MAGIC dominate for time-series problems, and if `is_open` doesn't rank in the top three for Rossmann,
# MAGIC something is wrong with the feature engineering — closed stores have zero sales by definition.

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

# MAGIC %md
# MAGIC This is the handoff from training to serving. We open a fresh MLflow run dedicated to the
# MAGIC registration, attach the test-set metrics (so the registered version carries the production-quality
# MAGIC numbers, not validation), and log the fitted estimator into the Unity Catalog Model Registry under
# MAGIC the three-level namespace `demo.ml.rossmann_sales_champion`. Every run produces a new version, so
# MAGIC the registry naturally keeps a full lineage of champions over time.
# MAGIC
# MAGIC The `LOG_TO_UC_REGISTRY` widget from cell 1 lets us skip this whole block when we're iterating —
# MAGIC registering throwaway versions every run pollutes the model lineage in UC and makes it harder to
# MAGIC audit which version actually serves traffic.

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

# MAGIC %md
# MAGIC Idempotent guard rail. The `demo.ml` schema should already exist from earlier notebooks, but
# MAGIC creating it conditionally here means we can run notebook 03 in isolation against a fresh
# MAGIC workspace without it failing on a missing target. Cheap insurance.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS demo.ml;

# COMMAND ----------

# MAGIC %md
# MAGIC We build a tidy predictions table — store_id, business_date, actual, predicted, residual, the
# MAGIC model name that produced the row, and a timestamp — then push it back into Unity Catalog via a
# MAGIC temporary view and a SQL `CREATE OR REPLACE TABLE`. The temp view dance is what lets us go from
# MAGIC pandas back into UC cleanly without writing parquet files manually.
# MAGIC
# MAGIC These rows are what the model performance dashboard and the downstream monitoring queries will
# MAGIC read. Keeping the residual precomputed saves every consumer from having to compute it themselves
# MAGIC and avoids the chance of someone subtracting the columns in the wrong order.

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

# MAGIC %md
# MAGIC One row into the audit table per pipeline run — current timestamp, table name, row count.
# MAGIC This is the cheapest possible monitoring signal we can leave behind, but it's surprisingly
# MAGIC valuable: a sudden drop in row count between consecutive runs is almost always the first symptom
# MAGIC of a broken upstream pipeline, and you can spot it on a dashboard without writing any actual
# MAGIC data-quality framework. The follow-up `SELECT` just confirms the row landed where we expect.

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
