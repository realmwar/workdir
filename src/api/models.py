"""Model loading layer — Factory pattern for model acquisition.

Demonstrates:
- Factory design pattern (ModelLoaderFactory)
- Strategy pattern (different loaders for different model types)
- OOP: inheritance, encapsulation, abstraction
- Dependency management / loose coupling
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base — Strategy interface for model loaders
# ---------------------------------------------------------------------------
class BaseModelLoader(ABC):
    """Abstract base class for all model loaders (Strategy pattern)."""

    @abstractmethod
    def load(self, model_uri: str) -> Any:
        """Load a model from the given URI and return a predict-capable object."""
        ...

    @abstractmethod
    def predict(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        """Run inference on the loaded model."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------
class MLflowModelLoader(BaseModelLoader):
    """Loads models from MLflow (local or UC registry)."""

    def load(self, model_uri: str) -> Any:
        import mlflow.pyfunc
        logger.info("Loading model from MLflow: %s", model_uri)
        return mlflow.pyfunc.load_model(model_uri)

    def predict(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        return model.predict(features).flatten()


class PickleModelLoader(BaseModelLoader):
    """Loads models from local pickle files (fallback)."""

    def load(self, model_uri: str) -> Any:
        import pickle
        logger.info("Loading model from pickle: %s", model_uri)
        with open(model_uri, "rb") as f:
            return pickle.load(f)

    def predict(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        return np.array(model.predict(features)).flatten()


# ---------------------------------------------------------------------------
# Factory — creates the right loader based on URI scheme
# ---------------------------------------------------------------------------
class ModelLoaderFactory:
    """Factory pattern: returns the appropriate loader based on model URI scheme.

    URI formats:
        - "models:/catalog.schema.name/version" -> MLflow UC registry
        - "runs:/run_id/artifact_path" -> MLflow run artifact
        - "/path/to/model.pkl" -> local pickle fallback
    """

    _loaders: Dict[str, BaseModelLoader] = {
        "mlflow": MLflowModelLoader(),
        "pickle": PickleModelLoader(),
    }

    @classmethod
    def get_loader(cls, model_uri: str) -> BaseModelLoader:
        if model_uri.startswith("models:/") or model_uri.startswith("runs:/"):
            return cls._loaders["mlflow"]
        elif model_uri.endswith(".pkl") or model_uri.endswith(".pickle"):
            return cls._loaders["pickle"]
        else:
            # Default to MLflow for unknown schemes.
            return cls._loaders["mlflow"]

    @classmethod
    def register_loader(cls, scheme: str, loader: BaseModelLoader):
        """Extend the factory with custom loader strategies at runtime."""
        cls._loaders[scheme] = loader


# ---------------------------------------------------------------------------
# Model container — holds loaded model + metadata
# ---------------------------------------------------------------------------
class LoadedModel:
    """Encapsulates a loaded model with its metadata and prediction logic."""

    def __init__(self, name: str, version: str, model_uri: str):
        self.name = name
        self.version = version
        self.model_uri = model_uri
        self._loader = ModelLoaderFactory.get_loader(model_uri)
        self._model: Optional[Any] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> "LoadedModel":
        """Load the model (lazy initialization)."""
        if not self._loaded:
            self._model = self._loader.load(self.model_uri)
            self._loaded = True
            logger.info("Model '%s' v%s loaded successfully.", self.name, self.version)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Run prediction — loads model on first call if not already loaded."""
        if not self._loaded:
            self.load()
        return self._loader.predict(self._model, features)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.model_uri,
            "status": "loaded" if self._loaded else "not_loaded",
        }
