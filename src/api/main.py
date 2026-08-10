"""FastAPI application — prediction endpoints for Rossmann and NYC demand models.

Demonstrates:
- FastAPI with typed request/response schemas (Pydantic)
- Dependency injection via app.state
- OOP service layer + Factory/Strategy design patterns
- Health check and model listing endpoints
- Error handling middleware
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import mlflow

from .models import LoadedModel
from .schemas import (
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    NYCDemandResponse,
    NYCDemandRequest,
    NYC_REQUEST_EXAMPLE,
    RossmannRequest,
    RossmannResponse,
    ROSSMANN_REQUEST_EXAMPLE,
)
from .services import (
    NYCDemandPredictionService,
    PredictionRouter,
    RossmannPredictionService,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model URIs — configurable via env vars for flexibility.
# ---------------------------------------------------------------------------
ROSSMANN_MODEL_URI = os.getenv(
    "ROSSMANN_MODEL_URI", "models:/demo.ml.rossmann_sales_champion/3"
)
NYC_MODEL_URI = os.getenv(
    "NYC_MODEL_URI", "models:/demo.ml.nyc_demand_champion/2"
)


def _model_version_from_uri(model_uri: str, fallback: str = "unknown") -> str:
    """Extract the trailing MLflow model version from models:/... URIs."""
    return model_uri.rstrip("/").split("/")[-1] if "/" in model_uri else fallback


def _databricks_credentials_ready() -> bool:
    """Return True when Databricks host/token look usable for UC model loading."""
    host = (os.getenv("DATABRICKS_HOST") or "").strip()
    token = (os.getenv("DATABRICKS_TOKEN") or "").strip()
    placeholder_tokens = {
        "",
        "replace_with_your_databricks_pat",
        "<your-databricks-token>",
    }
    if not host or "your-databricks-workspace-host" in host:
        logger.error(
            "DATABRICKS_HOST is missing or still a placeholder. "
            "Set it in workdir/.env (see workdir/databricks.yml)."
        )
        return False
    if token in placeholder_tokens:
        logger.error(
            "DATABRICKS_TOKEN is missing or still a placeholder. "
            "Create a PAT in Databricks: Settings → Developer → Access tokens, "
            "then paste it into workdir/.env."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Lifespan handler — load models on startup, clean up on shutdown.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and release on shutdown."""
    logger.info("Loading models...")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    rossmann_model = LoadedModel(
        "rossmann_sales_champion",
        _model_version_from_uri(ROSSMANN_MODEL_URI),
        ROSSMANN_MODEL_URI,
    )
    nyc_model = LoadedModel(
        "nyc_demand_champion",
        _model_version_from_uri(NYC_MODEL_URI),
        NYC_MODEL_URI,
    )

    if _databricks_credentials_ready():
        try:
            rossmann_model.load()
        except Exception as e:
            logger.warning("Rossmann model failed to load: %s", e)

        try:
            nyc_model.load()
        except Exception as e:
            logger.warning("NYC model failed to load: %s", e)
    else:
        logger.warning(
            "Skipping UC model loading — /health will report models_loaded=false "
            "until DATABRICKS_HOST and DATABRICKS_TOKEN are set."
        )

    # Wire up services (Dependency Injection).
    router = PredictionRouter()
    router.register("rossmann", RossmannPredictionService(rossmann_model))
    router.register("nyc_demand", NYCDemandPredictionService(nyc_model))

    app.state.rossmann_model = rossmann_model
    app.state.nyc_model = nyc_model
    app.state.router = router

    logger.info("Startup complete.")
    yield
    logger.info("Shutdown.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="E2E ML Pipeline — Prediction API",
    description=(
        "Serves Rossmann sales and NYC taxi demand predictions via champion models "
        "loaded from Databricks Unity Catalog (MLflow).\n\n"
        "**Interactive docs:** Swagger UI at `/docs`, ReDoc at `/redoc`.\n"
        "Use **Try it out** on predict endpoints — request bodies include ready-made examples."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    """Send browsers to Swagger UI (API has no HTML home page)."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check — confirms API is alive and reports model loading status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded={
            "rossmann": app.state.rossmann_model.is_loaded,
            "nyc_demand": app.state.nyc_model.is_loaded,
        },
    )


@app.get("/models", response_model=ModelsListResponse, tags=["system"])
async def list_models():
    """List registered model versions."""
    models = [
        ModelInfo(**app.state.rossmann_model.to_dict()),
        ModelInfo(**app.state.nyc_model.to_dict()),
    ]
    return ModelsListResponse(models=models)


@app.post("/predict/rossmann", response_model=RossmannResponse, tags=["predictions"])
async def predict_rossmann(
    request: RossmannRequest = Body(
        openapi_examples={
            "single_store_day": {
                "summary": "Single store-day",
                "description": "One Monday prediction for store 1 (matches Streamlit default).",
                "value": ROSSMANN_REQUEST_EXAMPLE,
            },
            "two_days": {
                "summary": "Batch: Mon + Fri",
                "description": "Two instances — same store, different day_of_week.",
                "value": {
                    "instances": [
                        {**ROSSMANN_REQUEST_EXAMPLE["instances"][0], "day_of_week": 1, "is_weekend": 0},
                        {**ROSSMANN_REQUEST_EXAMPLE["instances"][0], "day_of_week": 5, "is_weekend": 0},
                    ]
                },
            },
        },
    ),
):
    """Batch prediction for Rossmann store sales.

    Missing train-only lag/rolling columns are filled with defaults on the server
    (see `align_to_model_signature`).
    """
    try:
        service: RossmannPredictionService = app.state.router.get_service("rossmann")
        predictions = service.predict(request.instances)
        return RossmannResponse(predictions=predictions, count=len(predictions))
    except Exception as e:
        logger.error("Rossmann prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/nyc-demand", response_model=NYCDemandResponse, tags=["predictions"])
async def predict_nyc_demand(
    request: NYCDemandRequest = Body(
        openapi_examples={
            "single_zone_hour": {
                "summary": "Single zone-hour",
                "description": "Zone 1 at noon with demo lag/rolling values.",
                "value": NYC_REQUEST_EXAMPLE,
            },
            "two_zones": {
                "summary": "Batch: two zones",
                "description": "Same hour, zones 1 and 6.",
                "value": {
                    "instances": [
                        NYC_REQUEST_EXAMPLE["instances"][0],
                        {**NYC_REQUEST_EXAMPLE["instances"][0], "pu_location_id": 6},
                    ]
                },
            },
        },
    ),
):
    """Batch prediction for NYC taxi zone-hour demand."""
    try:
        service: NYCDemandPredictionService = app.state.router.get_service("nyc_demand")
        predictions = service.predict(request.instances)
        return NYCDemandResponse(predictions=predictions, count=len(predictions))
    except Exception as e:
        logger.error("NYC demand prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point for local execution: uvicorn workdir.src.api.main:app --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("workdir.src.api.main:app", host="0.0.0.0", port=8000, reload=True)
