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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import LoadedModel
from .schemas import (
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    NYCDemandResponse,
    NYCDemandRequest,
    RossmannRequest,
    RossmannResponse,
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
    "ROSSMANN_MODEL_URI", "models:/demo.ml.rossmann_sales_champion/1"
)
NYC_MODEL_URI = os.getenv(
    "NYC_MODEL_URI", "models:/demo.ml.nyc_demand_champion/1"
)


# ---------------------------------------------------------------------------
# Lifespan handler — load models on startup, clean up on shutdown.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and release on shutdown."""
    logger.info("Loading models...")

    rossmann_model = LoadedModel("rossmann_sales_champion", "1", ROSSMANN_MODEL_URI)
    nyc_model = LoadedModel("nyc_demand_champion", "1", NYC_MODEL_URI)

    try:
        rossmann_model.load()
    except Exception as e:
        logger.warning("Rossmann model failed to load: %s", e)

    try:
        nyc_model.load()
    except Exception as e:
        logger.warning("NYC model failed to load: %s", e)

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
    description="Serves Rossmann sales and NYC taxi demand predictions via champion models.",
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
async def predict_rossmann(request: RossmannRequest):
    """Batch prediction for Rossmann store sales."""
    try:
        service: RossmannPredictionService = app.state.router.get_service("rossmann")
        predictions = service.predict(request.instances)
        return RossmannResponse(predictions=predictions, count=len(predictions))
    except Exception as e:
        logger.error("Rossmann prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/nyc-demand", response_model=NYCDemandResponse, tags=["predictions"])
async def predict_nyc_demand(request: NYCDemandRequest):
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
