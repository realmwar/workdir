# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Deep Learning Experiments (Rossmann + NYC)
# MAGIC
# MAGIC **Goals:**
# MAGIC - Build a standard feedforward neural network (FNN) for Rossmann sales prediction using Keras/TensorFlow.
# MAGIC - Implement efficient training: learning rate scheduling, checkpointing, batch processing, early stopping.
# MAGIC - Experiment with network depth/width, activation functions, optimizers.
# MAGIC - Apply optimization techniques: mixed precision (FP16), pruning exploration, quantization awareness.
# MAGIC - Log all DL experiments to MLflow alongside classical models.
# MAGIC - Compare DL vs GBM performance on the same gold mart.
# MAGIC
# MAGIC **Constraints:**
# MAGIC - No Spark DataFrame API — SQL for reads, pandas + Keras/TF for DL.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Install dependencies (Serverless)

# COMMAND ----------

# MAGIC %md
# MAGIC Serverless ships with `numpy`, `pandas`, `sklearn`, `matplotlib`, and `mlflow`, but TensorFlow
# MAGIC (and the optional `tensorflow-model-optimization` package used in section 9 for pruning) are
# MAGIC not preinstalled. We pin TensorFlow and `protobuf` to compatible versions because newer protobuf
# MAGIC releases remove `google.protobuf.service`, which Databricks/MLflow imports still expect.

# COMMAND ----------

# MAGIC %md
# MAGIC This cell installs the DL-specific packages that are not guaranteed to exist on Databricks
# MAGIC Serverless. TensorFlow is the training backend for the feedforward networks, and
# MAGIC `tensorflow-model-optimization` is only needed later for the pruning experiment. The explicit
# MAGIC `protobuf` pin avoids the `cannot import name 'service' from 'google.protobuf'` import error.

# COMMAND ----------

# MAGIC %pip install "tensorflow==2.15.1" "tensorflow-model-optimization==0.8.0" "protobuf==4.25.3"

# COMMAND ----------

# MAGIC %md
# MAGIC After `%pip install`, Databricks needs a Python restart before the newly installed wheels are
# MAGIC visible to normal `import` statements. This is expected notebook behavior, not an error.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Imports & runtime config

# COMMAND ----------

# MAGIC %md
# MAGIC The imports set up the full deep learning workspace: pandas and NumPy for tabular data,
# MAGIC matplotlib for inline training curves, TensorFlow/Keras for model definition, sklearn for
# MAGIC preprocessing and metrics, and MLflow for experiment tracking. The experiment path follows the
# MAGIC same Serverless-safe pattern as notebook 03, and the explicit MLflow URIs make sure Databricks
# MAGIC tracking and Unity Catalog registry are used.

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, mixed_precision
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import mlflow
import mlflow.tensorflow
import mlflow.keras

CATALOG = "demo"
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _user = _ctx.userName().get()
    EXPERIMENT_NAME = f"/Users/{_user}/rossmann_deep_learning"
except Exception:
    EXPERIMENT_NAME = "/Shared/rossmann_deep_learning"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Load data & prepare features

# COMMAND ----------

# MAGIC %md
# MAGIC This cell turns the Rossmann gold mart into a neural-network-ready matrix. We keep the same target,
# MAGIC feature exclusions, categorical encoding, and time-based train/validation/test split as notebook 03
# MAGIC so the deep learning experiments are compared against the classical baselines on the same problem,
# MAGIC not on a silently different dataset.

# COMMAND ----------

df_raw = spark.sql(f"SELECT * FROM {CATALOG}.gold.rossmann_training_mart").toPandas()

TARGET = "label_sales"
LOG_TARGET = "label_log1p_sales"
DROP_COLS = ["business_date", "silver_loaded_at", "gold_loaded_at", TARGET, LOG_TARGET]
CAT_COLS = ["store_type", "assortment_type", "state_holiday_code"]

df = df_raw.copy()
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df = df.fillna(0)
FEATURES = [c for c in df.columns if c not in DROP_COLS]

# Time-based split (same as notebook 03).
dates = df_raw["business_date"].sort_values().unique()
n = len(dates)
cut1, cut2 = dates[int(n * 0.70)], dates[int(n * 0.85)]

mask_train = df_raw["business_date"] < cut1
mask_val   = (df_raw["business_date"] >= cut1) & (df_raw["business_date"] < cut2)
mask_test  = df_raw["business_date"] >= cut2

X_train_raw = df.loc[mask_train, FEATURES].values
X_val_raw   = df.loc[mask_val,   FEATURES].values
X_test_raw  = df.loc[mask_test,  FEATURES].values
y_train = df.loc[mask_train, TARGET].values
y_val   = df.loc[mask_val,   TARGET].values
y_test  = df.loc[mask_test,  TARGET].values

# Standardize features — critical for neural network convergence.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Helper utilities

# COMMAND ----------

# MAGIC %md
# MAGIC These helpers keep the experiment cells small. `eval_metrics` gives every model the same regression
# MAGIC scorecard, `build_fnn` creates a configurable feedforward network, and `train_nn_experiment` wraps
# MAGIC the repeatable training loop: build, compile, fit, evaluate, log to MLflow, and return the artifacts
# MAGIC we need for comparison.

# COMMAND ----------

def eval_metrics(y_true, y_pred):
    """Compute regression metrics and return as a dict."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mask_nz = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask_nz], y_pred[mask_nz]) if mask_nz.any() else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def build_fnn(n_features, hidden_layers, activation="relu", dropout_rate=0.2):
    """Build a configurable feedforward neural network.

    Args:
        n_features: number of input features.
        hidden_layers: list of ints, each element is the width of a hidden layer.
        activation: activation function name (relu, sigmoid, tanh, etc.).
        dropout_rate: dropout fraction after each hidden layer.
    """
    model = keras.Sequential(name="FNN")
    model.add(layers.Input(shape=(n_features,)))

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation=activation, name=f"dense_{i}"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{i}"))

    # Output layer: single neuron for regression.
    model.add(layers.Dense(1, name="output"))
    return model


def train_nn_experiment(
    name, hidden_layers, activation, optimizer_name, lr, epochs, batch_size,
    dropout_rate=0.2, use_lr_schedule=False, use_mixed_precision=False,
):
    """Train a single FNN experiment and log everything to MLflow.

    Returns (model, val_metrics, history).
    """
    # Mixed precision setup — FP16 for faster training on GPU.
    if use_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")

    model = build_fnn(X_train.shape[1], hidden_layers, activation, dropout_rate)

    # Select optimizer.
    opt_map = {
        "adam": optimizers.Adam(learning_rate=lr),
        "sgd": optimizers.SGD(learning_rate=lr, momentum=0.9),
        "rmsprop": optimizers.RMSprop(learning_rate=lr),
    }
    opt = opt_map.get(optimizer_name.lower(), optimizers.Adam(learning_rate=lr))

    model.compile(optimizer=opt, loss="mse", metrics=["mae"])

    # Callbacks: early stopping + model checkpointing + optional LR scheduling.
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint("/tmp/best_fnn.keras", monitor="val_loss", save_best_only=True, verbose=0),
    ]
    if use_lr_schedule:
        cb_list.append(
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        )

    with mlflow.start_run(run_name=name) as run:
        params = {
            "hidden_layers": str(hidden_layers),
            "activation": activation,
            "optimizer": optimizer_name,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "use_lr_schedule": use_lr_schedule,
            "use_mixed_precision": use_mixed_precision,
            "n_params": model.count_params(),
        }
        mlflow.log_params(params)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=0,
        )

        y_val_pred = model.predict(X_val, verbose=0).flatten()
        y_test_pred = model.predict(X_test, verbose=0).flatten()

        val_m = eval_metrics(y_val, y_val_pred)
        test_m = eval_metrics(y_test, y_test_pred)

        for k, v in val_m.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("best_epoch", int(np.argmin(history.history["val_loss"])) + 1)

        mlflow.keras.log_model(model, artifact_path="model")

        print(f"  [{name}] val_rmse={val_m['rmse']:.2f}  val_r2={val_m['r2']:.4f}  test_rmse={test_m['rmse']:.2f}")

    # Restore default precision policy.
    mixed_precision.set_global_policy("float32")
    return model, val_m, test_m, history

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Experiment 1 — Baseline FNN (ReLU, Adam)
# MAGIC
# MAGIC A simple 3-layer FNN to establish a DL baseline.

# COMMAND ----------

# MAGIC %md
# MAGIC The first experiment is intentionally plain: ReLU activations, Adam optimizer, three hidden layers,
# MAGIC and a moderate batch size. This gives us a clean deep learning baseline before changing architecture,
# MAGIC optimizer, precision, or compression settings.

# COMMAND ----------

dl_results = {}

model_1, val_1, test_1, hist_1 = train_nn_experiment(
    name="FNN_baseline_relu_adam",
    hidden_layers=[256, 128, 64],
    activation="relu",
    optimizer_name="adam",
    lr=1e-3,
    epochs=100,
    batch_size=512,
)
dl_results["FNN_baseline"] = {**val_1, "test_rmse": test_1["rmse"]}

# COMMAND ----------

# MAGIC %md
# MAGIC The training curves show whether the baseline network is learning smoothly or overfitting. Loss is
# MAGIC the objective the optimizer minimizes, while MAE is easier to read in the original sales units.
# MAGIC Plotting train and validation side by side makes the gap visible immediately.

# COMMAND ----------

# Training curves.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hist_1.history["loss"], label="train_loss")
axes[0].plot(hist_1.history["val_loss"], label="val_loss")
axes[0].set_title("FNN Baseline — Loss Curve")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
axes[0].legend()

axes[1].plot(hist_1.history["mae"], label="train_mae")
axes[1].plot(hist_1.history["val_mae"], label="val_mae")
axes[1].set_title("FNN Baseline — MAE Curve")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
axes[1].legend()

plt.tight_layout()
plt.savefig("/tmp/fnn_baseline_curves.png", dpi=100)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Experiment 2 — Deeper & wider network

# COMMAND ----------

# MAGIC %md
# MAGIC This run increases both depth and width to test whether the Rossmann feature mart benefits from a
# MAGIC larger neural network. The learning-rate scheduler is enabled because deeper networks are more
# MAGIC sensitive to optimizer settings and often need the step size reduced once validation loss plateaus.

# COMMAND ----------

model_2, val_2, test_2, hist_2 = train_nn_experiment(
    name="FNN_deep_512_256_128_64",
    hidden_layers=[512, 256, 128, 64],
    activation="relu",
    optimizer_name="adam",
    lr=5e-4,
    epochs=100,
    batch_size=512,
    use_lr_schedule=True,
)
dl_results["FNN_deep"] = {**val_2, "test_rmse": test_2["rmse"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Experiment 3 — Activation function comparison
# MAGIC
# MAGIC Same architecture, different activations: sigmoid vs tanh vs relu.

# COMMAND ----------

# MAGIC %md
# MAGIC Here the architecture stays fixed and only the activation function changes. That isolates the effect
# MAGIC of non-linearity choice: sigmoid can saturate, tanh is centered around zero, and ReLU is the baseline
# MAGIC from the first experiment.

# COMMAND ----------

for act_fn in ["sigmoid", "tanh"]:
    _, val_act, test_act, _ = train_nn_experiment(
        name=f"FNN_{act_fn}",
        hidden_layers=[256, 128, 64],
        activation=act_fn,
        optimizer_name="adam",
        lr=1e-3,
        epochs=80,
        batch_size=512,
    )
    dl_results[f"FNN_{act_fn}"] = {**val_act, "test_rmse": test_act["rmse"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Experiment 4 — Optimizer comparison (SGD, RMSprop)

# COMMAND ----------

# MAGIC %md
# MAGIC This cell keeps the ReLU architecture fixed and swaps the optimizer. SGD with momentum is the
# MAGIC classical baseline, while RMSprop adapts learning rates per parameter. Comparing them against Adam
# MAGIC shows whether optimizer choice matters more than network shape for this tabular task.

# COMMAND ----------

for opt_name, lr in [("sgd", 1e-2), ("rmsprop", 1e-3)]:
    _, val_opt, test_opt, _ = train_nn_experiment(
        name=f"FNN_relu_{opt_name}",
        hidden_layers=[256, 128, 64],
        activation="relu",
        optimizer_name=opt_name,
        lr=lr,
        epochs=80,
        batch_size=512,
        use_lr_schedule=True,
    )
    dl_results[f"FNN_{opt_name}"] = {**val_opt, "test_rmse": test_opt["rmse"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Experiment 5 — Mixed precision (FP16) training

# COMMAND ----------

# MAGIC %md
# MAGIC Mixed precision is mainly a systems experiment: on GPU-backed runtimes it can speed up training and
# MAGIC reduce memory pressure by using FP16 where safe. The model shape stays the same as the baseline so
# MAGIC any difference is easier to attribute to precision policy rather than architecture.

# COMMAND ----------

model_mp, val_mp, test_mp, hist_mp = train_nn_experiment(
    name="FNN_mixed_precision_fp16",
    hidden_layers=[256, 128, 64],
    activation="relu",
    optimizer_name="adam",
    lr=1e-3,
    epochs=80,
    batch_size=512,
    use_mixed_precision=True,
)
dl_results["FNN_mixed_precision"] = {**val_mp, "test_rmse": test_mp["rmse"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Experiment 6 — Pruning exploration
# MAGIC
# MAGIC Demonstrate magnitude-based weight pruning to reduce model size.

# COMMAND ----------

# MAGIC %md
# MAGIC Pruning explores model compression. We train a network while gradually forcing small-magnitude
# MAGIC weights toward zero, then strip the pruning wrappers before logging the final Keras model. If the
# MAGIC optional optimization package is unavailable, the notebook skips this experiment cleanly.

# COMMAND ----------

try:
    import tensorflow_model_optimization as tfmot

    # Build a base model, then wrap with pruning.
    base_model = build_fnn(X_train.shape[1], [256, 128, 64], "relu", dropout_rate=0.2)
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.20, final_sparsity=0.70,
        begin_step=0, end_step=int(len(X_train) / 512 * 40),  # ~40 epochs
    )
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(base_model, pruning_schedule=pruning_schedule)
    pruned_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    with mlflow.start_run(run_name="FNN_pruned_70pct"):
        mlflow.log_params({
            "technique": "magnitude_pruning", "initial_sparsity": 0.2, "final_sparsity": 0.7,
        })
        pruned_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=512, verbose=0,
            callbacks=[
                tfmot.sparsity.keras.UpdatePruningStep(),
                callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ],
        )
        y_val_pr = pruned_model.predict(X_val, verbose=0).flatten()
        val_pr = eval_metrics(y_val, y_val_pr)
        for k, v in val_pr.items():
            mlflow.log_metric(f"val_{k}", v)
        mlflow.keras.log_model(
            tfmot.sparsity.keras.strip_pruning(pruned_model), artifact_path="model"
        )
        print(f"  [FNN_pruned] val_rmse={val_pr['rmse']:.2f}  val_r2={val_pr['r2']:.4f}")
    dl_results["FNN_pruned"] = val_pr
except ImportError:
    print("tensorflow-model-optimization not installed — skipping pruning experiment.")
    print("Install with: %pip install tensorflow-model-optimization")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10) DL results comparison

# COMMAND ----------

# MAGIC %md
# MAGIC All deep learning experiments have been logging their validation metrics into `dl_results`. This
# MAGIC cell turns that dictionary into a sorted comparison table so the best neural-network configuration
# MAGIC is visible before we compare it with the tree-based champion from notebook 03.

# COMMAND ----------

dl_comp = pd.DataFrame(dl_results).T.sort_values("rmse")
dl_comp.index.name = "experiment"
print(dl_comp.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC The bar chart is the visual version of the comparison table. Sorting by validation RMSE keeps the
# MAGIC strongest experiment at the top and makes the spread between configurations easy to scan.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, max(4, len(dl_comp) * 0.5)))
dl_comp_sorted = dl_comp.sort_values("rmse")
ax.barh(dl_comp_sorted.index, dl_comp_sorted["rmse"], color="darkorange", edgecolor="k")
ax.set_xlabel("Validation RMSE")
ax.set_title("Deep Learning Experiments — Validation RMSE")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("/tmp/dl_experiments_comparison.png", dpi=100)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11) DL vs GBM comparison
# MAGIC
# MAGIC Compare the best DL model against the GBM champion from notebook 03.

# COMMAND ----------

# MAGIC %md
# MAGIC This is the practical checkpoint: the best neural network is compared against the production-style
# MAGIC predictions written by notebook 03. If the deep model does not beat the GBM champion, that is still
# MAGIC a useful result because it shows that tabular boosting remains the stronger baseline for this use case.

# COMMAND ----------

# Load GBM predictions from the ML table for head-to-head comparison.
gbm_preds = spark.sql(f"SELECT actual_sales, predicted_sales FROM {CATALOG}.ml.rossmann_predictions").toPandas()
gbm_metrics = eval_metrics(gbm_preds["actual_sales"].values, gbm_preds["predicted_sales"].values)

# Best DL model on test set — use the first experiment's test metrics as proxy.
best_dl_name = dl_comp.index[0]
best_dl_val_rmse = dl_comp.loc[best_dl_name, "rmse"]

comparison = pd.DataFrame({
    "GBM_champion (from 03)": gbm_metrics,
    f"Best_DL ({best_dl_name})": dl_comp.loc[best_dl_name].to_dict(),
}).T

print("=== DL vs GBM Head-to-Head ===")
print(comparison[["rmse", "r2", "mape"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12) Summary
# MAGIC
# MAGIC | Experiment | What it demonstrates |
# MAGIC |---|---|
# MAGIC | FNN Baseline | Standard multilayer FNN, ReLU, Adam |
# MAGIC | FNN Deep | Deeper/wider architecture exploration |
# MAGIC | FNN sigmoid/tanh | Activation function comparison |
# MAGIC | FNN SGD/RMSprop | Optimizer comparison |
# MAGIC | FNN mixed precision | FP16 training for speed/memory optimization |
# MAGIC | FNN pruned | Magnitude-based pruning for model compression |
# MAGIC | DL vs GBM | Head-to-head comparison with tree-based champion |
# MAGIC
# MAGIC **Competency coverage:**
# MAGIC - Deep Learning: Build standard NN, efficient training, optimize DNN
# MAGIC - ML Engineering: Simple NN with high-level APIs, complex NN with advanced APIs
# MAGIC
# MAGIC **Next step:** `05_nyc_demand_model.py`
