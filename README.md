# E2E ML Pipeline Project

End-to-end machine learning pipeline on **Databricks Free Account** (Python + SQL Serverless, no Spark DataFrame API).

## Project Structure

```
workdir/
├── databricks.yml                        # Databricks bundle config
├── README.md                             # This file
├── docker-compose.yml                    # Local FastAPI + Streamlit runtime
├── .env.example                          # Non-secret Databricks/local serving config template
├── notebooks/
│   ├── 00_uc_bootstrap_and_data_intake.py  # UC schemas, raw data intake, landing/bronze
│   ├── 01_silver_transformations.py        # Clean/enrich bronze → silver feature tables
│   ├── 02_gold_feature_marts.py            # Training/serving/dashboard gold marts
│   ├── 03_ml_baseline_training.py          # Rossmann: 7+ models, MLflow, registry
│   ├── 04_deep_learning_experiments.py     # FNN experiments: activations, optimizers, FP16, pruning
│   ├── 05_nyc_demand_model.py              # NYC TLC: GBM + FNN demand forecasting
│   ├── 06_serving_contracts.py             # Batch inference, schema contracts, serving tables
│   ├── 07_concurrency_experiments.py       # Threading, asyncio, process pools, benchmarks
│   └── 08_code_assessment.py               # DS&A, profiling, refactoring, design patterns
├── src/
│   ├── api/                                # FastAPI backend
│   │   ├── main.py                         #   App + endpoints (health, predict, models)
│   │   ├── models.py                       #   Model loading (Factory + Strategy patterns)
│   │   ├── services.py                     #   Prediction services (OOP, DI)
│   │   ├── schemas.py                      #   Pydantic request/response schemas
│   │   ├── Dockerfile                      #   Container image for local serving
│   │   └── requirements.txt                #   Python dependencies
│   └── frontend/                           # Streamlit frontend
│       ├── app.py                          #   Main app
│       ├── Dockerfile                      #   Container image for local UI
│       ├── requirements.txt                #   Python dependencies
│       └── pages/
│           ├── 1_rossmann_prediction.py    #   Store sales prediction page
│           ├── 2_nyc_demand_heatmap.py     #   Zone-hour demand visualization
│           └── 3_model_comparison.py       #   Model metrics dashboard
├── dashboards/                             # Databricks AI/BI dashboards + SQL
│   ├── README.md                           #   Board descriptions
│   ├── *_dashboard.sql                     #   Widget SQL libraries
│   └── *_dashboard.lvdash.json             #   Dashboard definitions
└── jobs/                                   # Databricks Job orchestration
    ├── e2e_pipeline_job.yml                #   Full DAG: intake → gold → training → serving
    ├── rossmann_retrain_job.yml            #   Scheduled weekly retrain
    └── nyc_incremental_job.yml             #   Monthly incremental ETL + retrain
```

## Architecture

```
                       Databricks Free Account
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  landing → bronze → silver → gold → ml → serving        │
│                                      │                  │
│                                      ├── MLflow         │
│                                      └── UC Registry    │
│                                                         │
│  audit (snapshots, row counts, data freshness)          │
│                                                         │
└───────────────────────┬─────────────────────────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
        FastAPI Backend    Databricks Dashboards
              │
        Streamlit Frontend
```

## Current Status

Notebooks `00` through `08` have been executed successfully in Databricks.

The core E2E ML pipeline path is `00` → `01` → `02` → `03` → `04` → `05` → `06`. It builds the UC medallion data layers, trains/registers Rossmann and NYC models, and writes batch inference outputs into `demo.serving`.

Notebooks `07_concurrency_experiments.py` and `08_code_assessment.py` are standalone competency labs. They support the project evidence matrix with concurrency, algorithms, profiling, refactoring, and design-pattern examples, but they are not required to produce serving predictions.

## Pipeline phases (notebooks 00–08)

Core chain (required for predictions and dashboards):

```text
00 intake → 01 silver → 02 gold → 03 Rossmann ML → 04 Rossmann DL → 05 NYC ML → 06 serving
```

Standalone labs (competency evidence only): `07` concurrency, `08` code assessment.

### 00 — UC bootstrap and data intake
`notebooks/00_uc_bootstrap_and_data_intake.py`

| | |
|---|---|
| **Goal** | Create Unity Catalog schemas for the medallion/E2E stack and ingest raw Rossmann + a limited NYC TLC month set. |
| **Outputs** | Schemas under `demo` (`landing`, `bronze`, `silver`, `gold`, `ml`, `serving`, `audit`); landing/bronze tables for Rossmann and NYC raw files. |
| **Used next by** | `01` reads bronze and builds cleaned silver feature bases. |

### 01 — Silver transformations
`notebooks/01_silver_transformations.py`

| | |
|---|---|
| **Goal** | Clean and enrich bronze into silver; materialize feature-base tables for later training/serving; write silver quality snapshots. |
| **Outputs** | `demo.silver.*` feature-base tables (Rossmann + NYC); audit snapshots for silver observability. |
| **Used next by** | `02` builds gold training, serving, and dashboard marts on top of silver. |

### 02 — Gold feature marts
`notebooks/02_gold_feature_marts.py`

| | |
|---|---|
| **Goal** | Publish business-ready gold marts with stable contracts for training, serving, and dashboards; maintain a contract registry. |
| **Outputs** | Training marts (`rossmann_training_mart`, `nyc_demand_training_mart`); inference-ready serving marts (no labels); dashboard KPI marts; `demo.gold.contract_registry_mart`; gold audit snapshots. |
| **Used next by** | `03`/`04` train Rossmann on the Rossmann training mart; `05` trains NYC on the NYC training mart; dashboards read KPI marts; `06` scores serving marts. |

### 03 — ML baseline training (Rossmann)
`notebooks/03_ml_baseline_training.py`

| | |
|---|---|
| **Goal** | Time-based split; train classical ML + ensemble baselines; track in MLflow; register the champion in UC; persist offline predictions. |
| **Outputs** | MLflow experiment runs; UC model `demo.ml.rossmann_sales_champion`; table `demo.ml.rossmann_predictions`. |
| **Used next by** | `04` compares deep models to the tree/GBM champion; `06` loads the UC champion for batch serving; Model Performance dashboard; local FastAPI Rossmann predict. |

### 04 — Deep learning experiments (Rossmann)
`notebooks/04_deep_learning_experiments.py`

| | |
|---|---|
| **Goal** | Train FNN variants (depth/width, activations, optimizers, mixed precision, pruning) on the same gold mart and compare against the notebook-`03` champion. |
| **Outputs** | DL experiment runs in MLflow; head-to-head metrics vs GBM; evidence that may keep or challenge the Rossmann champion. |
| **Used next by** | The active Rossmann champion in UC remains the artifact consumed by `06` and the local API; DL results support competency coverage and model-comparison narrative. |

### 05 — NYC demand model
`notebooks/05_nyc_demand_model.py`

| | |
|---|---|
| **Goal** | Second domain: zone-hour taxi demand with LightGBM / XGBoost / FNN, time-series validation, UC registration. |
| **Outputs** | UC model `demo.ml.nyc_demand_champion`; table `demo.ml.nyc_demand_predictions`; MLflow runs. |
| **Used next by** | `06` batch-serves the NYC champion; NYC Demand + Model Performance dashboards; local FastAPI `/predict/nyc-demand`. |

### 06 — Serving contracts and batch inference
`notebooks/06_serving_contracts.py`

| | |
|---|---|
| **Goal** | After registration, prove the Databricks serving path: load UC champions, validate serving contracts, align signatures, run batch predict, persist results. |
| **Outputs** | `demo.serving.rossmann_predictions`, `demo.serving.nyc_demand_predictions`; contract-registry rows for serving outputs; audit snapshots. |
| **Used next by** | Model Performance / Pipeline Health dashboards; the same UC model URIs are loaded by the local FastAPI backend for online/demo scoring (Streamlit + Swagger). |

### 07 — Concurrency experiments (standalone)
`notebooks/07_concurrency_experiments.py`

| | |
|---|---|
| **Goal** | Demonstrate threads, process pools, asyncio, synchronization, and sequential-vs-concurrent benchmarks for AI/batch workloads. |
| **Outputs** | Notebook benchmarks and concurrency examples (no production serving tables). |
| **Used next by** | Competency evidence for Programming (concurrency); not required by `06` or the local apps. |

### 08 — Code assessment / DS&A (standalone)
`notebooks/08_code_assessment.py`

| | |
|---|---|
| **Goal** | Profiling, refactoring patterns, design patterns, and core/advanced data structures & algorithms with Big O notes. |
| **Outputs** | Assessment examples and algorithm demos (no production serving tables). |
| **Used next by** | Competency evidence for Programming (algorithms / SE / code assessment); outside the predict chain. |

## Datasets

| Dataset | Source | Role |
|---|---|---|
| Rossmann Store Sales | Kaggle (public) | Primary: sales forecasting |
| NYC TLC Yellow Taxi | TLC.nyc.gov (public) | Phase 2: demand forecasting, incremental ETL |

## Medallion Architecture

| Layer | Schema | Description |
|---|---|---|
| Landing | `demo.landing` | Raw files registered as tables |
| Bronze | `demo.bronze` | Copied raw data, queryable |
| Silver | `demo.silver` | Cleaned, enriched, feature-base tables |
| Gold | `demo.gold` | Training/serving/dashboard marts + contracts |
| ML | `demo.ml` | Model predictions and experiment outputs |
| Serving | `demo.serving` | Batch inference results for API/dashboards |
| Audit | `demo.audit` | Row-count snapshots, data freshness tracking |

## Models Trained

### Rossmann Sales Forecasting
- Linear Regression, Ridge (L2), Lasso (L1)
- Decision Tree, Random Forest, Gradient Boosting
- LightGBM, XGBoost
- Voting Ensemble from the top-3 validation models
- FNN (multiple architectures, activations, optimizers)
- FNN with mixed precision (FP16) and pruning

### NYC Demand Forecasting
- LightGBM, XGBoost
- FNN (256-128-64)

## How to Run

### Notebooks (Databricks)
1. Import `workdir/` as a Databricks Repo or upload notebooks.
2. Run notebooks in order: `00` → `01` → `02` → `03` → `04` → `05` → `06`.
3. Run notebooks `07` and `08` separately when you want to review standalone competency evidence.

### Docker Compose (local FastAPI + Streamlit)
Local containers run FastAPI + Streamlit. The API does **not** call Databricks Model Serving REST endpoints. It loads champion model artifacts from **Unity Catalog Model Registry** via MLflow (`models:/...`) using workspace credentials, then runs inference inside the API container.

`DATABRICKS_HOST` comes from `databricks.yml` (`https://dbc-67156f8d-53eb.cloud.databricks.com`).

`DATABRICKS_TOKEN` is a Personal Access Token (PAT):
1. Open the workspace host above.
2. Profile / Settings → Developer → Access tokens (or User Settings → Access tokens).
3. Generate new token and paste it into `workdir/.env` as `DATABRICKS_TOKEN=...`.

```bash
cd workdir
cp .env.example .env   # if .env is missing
# paste DATABRICKS_TOKEN into .env
docker compose --env-file .env up --build
```

After startup:
- FastAPI Swagger UI: `http://localhost:8000/docs` (or open `http://localhost:8000/` — redirects to docs)
- ReDoc: `http://localhost:8000/redoc`
- In Swagger, use **Try it out** on `/predict/rossmann` and `/predict/nyc-demand` — request bodies ship with ready examples (single + small batch)
- FastAPI health: `http://localhost:8000/health` (expect `models_loaded.rossmann` and `nyc_demand` = `true`)
- Streamlit UI: `http://localhost:8501` (calls `http://api:8000` inside Docker)

Model URIs (env-overridable):

```bash
ROSSMANN_MODEL_URI=models:/demo.ml.rossmann_sales_champion/3
NYC_MODEL_URI=models:/demo.ml.nyc_demand_champion/2
```

### FastAPI Backend (local)
```bash
cd workdir
pip install -r src/api/requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

### Streamlit Frontend (local)
```bash
pip install -r src/frontend/requirements.txt
streamlit run src/frontend/app.py
```

### Databricks Dashboards

AI/BI boards under `dashboards/` (two-column layouts + English guide text).  
They read UC tables from notebooks `00`–`06`. Screenshots: `docs/presentation/pic/`.

#### Summary

| Dashboard | Purpose |
|---|---|
| Rossmann KPI | Business sales health from the gold KPI mart |
| NYC Demand | City-hour demand, revenue, top pickup zones |
| Model Performance | Offline actual-vs-predicted + UC serving batch stats |
| Pipeline Health | Audit freshness, contracts, date-range sanity |

#### Rossmann KPI — detail
- **Question answered:** Are Rossmann daily sales healthy, and what drives lifts (promo / weekend / openness)?
- **Primary source:** `demo.gold.rossmann_dashboard_kpi_mart` (built in notebook `02`).
- **Layout:** left = trends & segment charts; right = reading guide, KPI counters (grand / avg / peak sales), notes.
- **Widgets:** daily sales trend; open-store ratio; week-over-week %; promo vs non-promo; weekend vs weekday; monthly sales.
- **Not for:** model error analysis → use Model Performance.

#### NYC Demand — detail
- **Question answered:** When and where is NYC taxi demand strong, and what revenue does it generate?
- **Primary sources:** `demo.gold.nyc_dashboard_kpi_mart` (city-hour); `demo.gold.nyc_demand_training_mart` (top zones).
- **Layout:** left = KPI counters + hour-of-day / daily revenue / DoD trips; right = guide, notes, hour-over-hour % + top-20 zones.
- **Widgets:** total / avg hourly trips, total revenue; dual-peak hour bars; daily revenue area; DoD line; HoH spike view; top zones.
- **Not for:** prediction residuals → use Model Performance.

#### Model Performance — detail
- **Question answered:** Do registered models score well offline, and did Databricks serving batch succeed?
- **Primary sources:** `demo.ml.rossmann_predictions`, `demo.ml.nyc_demand_predictions`, `demo.serving.rossmann_predictions`, `demo.serving.nyc_demand_predictions`.
- **Layout:** left = Rossmann quality (scatter, worst stores, daily MAE, residual buckets) + serving batch table; right = guide, NYC quality (scatter + worst zones), notes.
- **Serving proof:** non-zero `prediction_count` and `inference_ts` after notebook `06` batch predict on UC champions.
- **Online twin:** same UC models via local FastAPI `/predict/*` + Streamlit / Swagger (separate runtime).

#### Pipeline Health — detail
- **Question answered:** Are gold/ML/serving tables fresh, populated, and covered by contracts?
- **Primary sources:** `demo.audit.gold_snapshot`, `demo.gold.contract_registry_mart`, plus date-range checks on gold/ML tables.
- **Layout:** left = ops guide, contract registry, date-range sanity, snapshot history, raw row-count timeline; right = latest row counts, hours-since-refresh bar, notes.
- **How to act:** zeros ⇒ failed writes; high hours-since-refresh ⇒ rerun jobs; thin date windows ⇒ intake/feature gaps.

### Jobs (Databricks)
Add job YAMLs from `jobs/` to `databricks.yml` under `resources.jobs` and deploy with:
```bash
databricks bundle deploy
```


