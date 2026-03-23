"""Service layer — business logic for prediction endpoints.

Demonstrates:
- OOP: service classes with single responsibility
- Strategy pattern: prediction routing via PredictionService
- Dependency injection: services receive model containers
- Separation of concerns: services own the business logic, not the API layer
"""
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import logging

from .models import LoadedModel
from .schemas import (
    RossmannFeatures,
    RossmannPrediction,
    NYCDemandFeatures,
    NYCDemandPrediction,
)

logger = logging.getLogger(__name__)


class RossmannPredictionService:
    """Handles Rossmann sales predictions."""

    FEATURE_ORDER = [
        "store_id", "day_of_week", "is_weekend", "is_open", "is_promo",
        "state_holiday_code", "is_school_holiday", "store_type", "assortment_type",
        "competition_distance_km", "has_promo2", "is_promo2_active",
        "is_promo_interval_month",
    ]

    def __init__(self, model: LoadedModel):
        self._model = model

    def predict(self, instances: List[RossmannFeatures]) -> List[RossmannPrediction]:
        """Convert input features to DataFrame, run model, format output."""
        rows = [inst.model_dump() for inst in instances]
        df = pd.DataFrame(rows)[self.FEATURE_ORDER]

        preds = self._model.predict(df)
        now = datetime.now()

        return [
            RossmannPrediction(
                store_id=inst.store_id,
                predicted_sales=float(pred),
                model_name=self._model.name,
                model_version=self._model.version,
                inference_ts=now,
            )
            for inst, pred in zip(instances, preds)
        ]


class NYCDemandPredictionService:
    """Handles NYC taxi demand predictions."""

    FEATURE_ORDER = [
        "pickup_hour", "pu_location_id",
        "avg_trip_distance_miles", "avg_trip_duration_min",
        "avg_fare_amount", "avg_tip_pct", "total_revenue",
        "lag_trip_cnt_1h", "lag_trip_cnt_24h",
        "rolling_trip_cnt_mean_24h", "rolling_fare_mean_24h",
    ]

    def __init__(self, model: LoadedModel):
        self._model = model

    def predict(self, instances: List[NYCDemandFeatures]) -> List[NYCDemandPrediction]:
        rows = [inst.model_dump() for inst in instances]
        df = pd.DataFrame(rows)[self.FEATURE_ORDER]

        preds = self._model.predict(df)
        now = datetime.now()

        return [
            NYCDemandPrediction(
                pu_location_id=inst.pu_location_id,
                pickup_hour=inst.pickup_hour,
                predicted_trip_cnt=float(pred),
                model_name=self._model.name,
                model_version=self._model.version,
                inference_ts=now,
            )
            for inst, pred in zip(instances, preds)
        ]


class PredictionRouter:
    """Strategy pattern router — dispatches prediction requests to the right service.

    This enables adding new prediction domains (e.g., fraud, churn) without
    modifying existing service code (Open/Closed Principle).
    """

    def __init__(self):
        self._services: Dict[str, object] = {}

    def register(self, domain: str, service: object):
        self._services[domain] = service
        logger.info("Registered prediction service: %s", domain)

    def get_service(self, domain: str) -> object:
        if domain not in self._services:
            raise KeyError(f"No service registered for domain '{domain}'")
        return self._services[domain]

    @property
    def domains(self) -> List[str]:
        return list(self._services.keys())
