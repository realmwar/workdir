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
├── dashboards/                             # Databricks Dashboard SQL queries
│   ├── rossmann_kpi_dashboard.sql          #   Sales KPIs, promo impact, trends
│   ├── nyc_demand_dashboard.sql            #   Hourly demand, revenue, zone analysis
│   ├── model_performance_dashboard.sql     #   Actual vs predicted, residuals, errors
│   └── pipeline_health_dashboard.sql       #   Row counts, freshness, contract registry
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
The local containers still load models from Databricks Unity Catalog, so valid Databricks credentials are required.

```bash
cd workdir
cp .env.example .env
# edit .env with DATABRICKS_HOST and DATABRICKS_TOKEN
docker compose --env-file .env up --build
```

After startup:
- FastAPI docs: `http://localhost:8000/docs`
- FastAPI health: `http://localhost:8000/health`
- Streamlit UI: `http://localhost:8501`

The model URIs are configured through env vars:

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

### Jobs (Databricks)
Add job YAMLs from `jobs/` to `databricks.yml` under `resources.jobs` and deploy with:
```bash
databricks bundle deploy
```

## Competency Coverage

Full coverage tracking is maintained in `docs/html/README_competency_jobs.html` with per-file coverage blocks mapped to every job and competency.

### Coverage Matrix

| Competency Area | Jobs | Artifacts |
|---|---|---|
| Deep Learning | 3 | `04_deep_learning_experiments.py` |
| Machine Learning | 4 | `03_ml_baseline_training.py`, `05_nyc_demand_model.py` |
| MLOps | 1 | `03`, `06`, `jobs/`, `dashboards/` |
| Data Engineering | 2 | `00`, `01`, `02`, `dashboards/`, `jobs/` |
| ML Engineering | 5 | `03`, `04`, `05`, `06`, `src/frontend/`, `dashboards/` |
| Programming | 6 | `07`, `08`, `src/api/` |
| Statistics & Math | 3 | `01`, `02`, `03`, `04`, `dashboards/` |
| **Total** | **24 jobs** | **All artifacts** |
