"""Pydantic request/response schemas for the prediction API.

Demonstrates: OOP, data validation, type safety, documentation.
"""
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Rossmann
# ---------------------------------------------------------------------------
class RossmannFeatures(BaseModel):
    """Input features for a single Rossmann store-day prediction."""

    store_id: int = Field(..., ge=1, description="Store identifier")
    day_of_week: int = Field(..., ge=1, le=7)
    is_weekend: int = Field(..., ge=0, le=1)
    is_open: int = Field(..., ge=0, le=1)
    is_promo: int = Field(..., ge=0, le=1)
    state_holiday_code: int = Field(0, ge=0, le=3)
    is_school_holiday: int = Field(0, ge=0, le=1)
    store_type: int = Field(0, ge=0, le=3, description="Encoded store type")
    assortment_type: int = Field(0, ge=0, le=2, description="Encoded assortment")
    competition_distance_km: float = Field(0.0, ge=0)
    has_promo2: int = Field(0, ge=0, le=1)
    is_promo2_active: int = Field(0, ge=0, le=1)
    is_promo_interval_month: int = Field(0, ge=0, le=1)


class RossmannRequest(BaseModel):
    """Batch prediction request for Rossmann stores."""

    instances: List[RossmannFeatures] = Field(..., min_length=1, max_length=1000)


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
class NYCDemandFeatures(BaseModel):
    """Input features for a single zone-hour demand prediction."""

    pickup_hour: int = Field(..., ge=0, le=23)
    pu_location_id: int = Field(..., ge=1)
    avg_trip_distance_miles: float = Field(0.0, ge=0)
    avg_trip_duration_min: float = Field(0.0, ge=0)
    avg_fare_amount: float = Field(0.0, ge=0)
    avg_tip_pct: float = Field(0.0, ge=0)
    total_revenue: float = Field(0.0, ge=0)
    lag_trip_cnt_1h: float = Field(0.0)
    lag_trip_cnt_24h: float = Field(0.0)
    rolling_trip_cnt_mean_24h: float = Field(0.0)
    rolling_fare_mean_24h: float = Field(0.0)


class NYCDemandRequest(BaseModel):
    """Batch prediction request for NYC demand."""

    instances: List[NYCDemandFeatures] = Field(..., min_length=1, max_length=5000)


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
