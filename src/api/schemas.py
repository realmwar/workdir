"""Pydantic request/response schemas for the prediction API.

Demonstrates: OOP, data validation, type safety, documentation.
"""
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Rossmann
# ---------------------------------------------------------------------------
ROSSMANN_INSTANCE_EXAMPLE = {
    "store_id": 1,
    "day_of_week": 1,
    "is_weekend": 0,
    "is_open": 1,
    "is_promo": 0,
    "state_holiday_code": 0,
    "is_school_holiday": 0,
    "store_type": 0,
    "assortment_type": 0,
    "competition_distance_km": 5.0,
    "has_promo2": 0,
    "is_promo2_active": 0,
    "is_promo_interval_month": 0,
}

ROSSMANN_REQUEST_EXAMPLE = {"instances": [ROSSMANN_INSTANCE_EXAMPLE]}


class RossmannFeatures(BaseModel):
    """Input features for a single Rossmann store-day prediction."""

    store_id: int = Field(..., ge=1, description="Store identifier", examples=[1])
    day_of_week: int = Field(..., ge=1, le=7, description="1=Mon … 7=Sun", examples=[1])
    is_weekend: int = Field(..., ge=0, le=1, examples=[0])
    is_open: int = Field(..., ge=0, le=1, examples=[1])
    is_promo: int = Field(..., ge=0, le=1, examples=[0])
    state_holiday_code: int = Field(0, ge=0, le=3, examples=[0])
    is_school_holiday: int = Field(0, ge=0, le=1, examples=[0])
    store_type: int = Field(0, ge=0, le=3, description="Encoded store type a/b/c/d → 0–3", examples=[0])
    assortment_type: int = Field(0, ge=0, le=2, description="Encoded assortment a/b/c → 0–2", examples=[0])
    competition_distance_km: float = Field(0.0, ge=0, examples=[5.0])
    has_promo2: int = Field(0, ge=0, le=1, examples=[0])
    is_promo2_active: int = Field(0, ge=0, le=1, examples=[0])
    is_promo_interval_month: int = Field(0, ge=0, le=1, examples=[0])

    model_config = ConfigDict(json_schema_extra={"examples": [ROSSMANN_INSTANCE_EXAMPLE]})


class RossmannRequest(BaseModel):
    """Batch prediction request for Rossmann stores."""

    instances: List[RossmannFeatures] = Field(..., min_length=1, max_length=1000)

    model_config = ConfigDict(json_schema_extra={"examples": [ROSSMANN_REQUEST_EXAMPLE]})

class RossmannPrediction(BaseModel):
    """Single Rossmann prediction result."""

    store_id: int
    predicted_sales: float
    model_name: str
    model_version: str
    inference_ts: datetime


class RossmannResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[RossmannPrediction]
    count: int


# ---------------------------------------------------------------------------
# NYC Demand
# ---------------------------------------------------------------------------
NYC_INSTANCE_EXAMPLE = {
    "pickup_hour": 12,
    "pu_location_id": 1,
    "avg_trip_distance_miles": 3.5,
    "avg_trip_duration_min": 15.0,
    "avg_fare_amount": 15.0,
    "avg_tip_pct": 0.15,
    "total_revenue": 750.0,
    "lag_trip_cnt_1h": 80.0,
    "lag_trip_cnt_24h": 95.0,
    "rolling_trip_cnt_mean_24h": 70.0,
    "rolling_fare_mean_24h": 14.5,
}

NYC_REQUEST_EXAMPLE = {"instances": [NYC_INSTANCE_EXAMPLE]}


class NYCDemandFeatures(BaseModel):
    """Input features for a single zone-hour demand prediction."""

    pickup_hour: int = Field(..., ge=0, le=23, examples=[12])
    pu_location_id: int = Field(..., ge=1, examples=[1])
    avg_trip_distance_miles: float = Field(0.0, ge=0, examples=[3.5])
    avg_trip_duration_min: float = Field(0.0, ge=0, examples=[15.0])
    avg_fare_amount: float = Field(0.0, ge=0, examples=[15.0])
    avg_tip_pct: float = Field(0.0, ge=0, examples=[0.15])
    total_revenue: float = Field(0.0, ge=0, examples=[750.0])
    lag_trip_cnt_1h: float = Field(0.0, examples=[80.0])
    lag_trip_cnt_24h: float = Field(0.0, examples=[95.0])
    rolling_trip_cnt_mean_24h: float = Field(0.0, examples=[70.0])
    rolling_fare_mean_24h: float = Field(0.0, examples=[14.5])

    model_config = ConfigDict(json_schema_extra={"examples": [NYC_INSTANCE_EXAMPLE]})


class NYCDemandRequest(BaseModel):
    """Batch prediction request for NYC demand."""

    instances: List[NYCDemandFeatures] = Field(..., min_length=1, max_length=5000)

    model_config = ConfigDict(json_schema_extra={"examples": [NYC_REQUEST_EXAMPLE]})


class NYCDemandPrediction(BaseModel):
    """Single NYC demand prediction result."""

    pu_location_id: int
    pickup_hour: int
    predicted_trip_cnt: float
    model_name: str
    model_version: str
    inference_ts: datetime


class NYCDemandResponse(BaseModel):
    """Batch prediction response for NYC demand."""

    predictions: List[NYCDemandPrediction]
    count: int


# ---------------------------------------------------------------------------
# Health / Metadata
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    timestamp: datetime
    models_loaded: dict


class ModelInfo(BaseModel):
    """Information about a registered model."""

    name: str
    version: str
    source: str
    status: str


class ModelsListResponse(BaseModel):
    """List of available models."""

    models: List[ModelInfo]
