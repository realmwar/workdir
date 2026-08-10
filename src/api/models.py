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
        aligned = align_to_model_signature(features, model)
        return model.predict(aligned).flatten()


class PickleModelLoader(BaseModelLoader):
    """Loads models from local pickle files (fallback)."""

    def load(self, model_uri: str) -> Any:
        import pickle
        logger.info("Loading model from pickle: %s", model_uri)
        with open(model_uri, "rb") as f:
            return pickle.load(f)

    def predict(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        return np.array(model.predict(features)).flatten()


def align_to_model_signature(features: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Align request features to an MLflow pyfunc model signature before prediction.

    This mirrors the serving notebook behavior: pyfunc models validate input dtypes
    and column order before calling the underlying estimator. For column schemas we
    add missing train-only columns with safe defaults, cast dtypes, and reorder.
    For tensor schemas (typical Keras/FNN models), we coerce the supplied feature
    matrix to numeric values and preserve the service-defined feature order.
    """
    metadata = getattr(model, "metadata", None)
    if metadata is None:
        return features

    schema = metadata.get_input_schema()
    if schema is None:
        return features

    aligned = features.copy()

    if hasattr(schema, "is_tensor_spec") and schema.is_tensor_spec():
        for col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="raise").astype("float64")
        return aligned

    expected_cols = []
    for col_spec in schema.inputs:
        col_name = col_spec.name
        # MLflow may stringify types as "integer" or "DataType.integer".
        col_type = str(getattr(col_spec.type, "name", col_spec.type)).lower()
        col_type = col_type.replace("datatype.", "")
        expected_cols.append(col_name)

        if col_name not in aligned.columns:
            if col_type in ("integer", "long"):
                aligned[col_name] = 0
            elif col_type in ("float", "double"):
                aligned[col_name] = 0.0
            elif col_type == "boolean":
                aligned[col_name] = False
            elif col_type == "string":
                aligned[col_name] = ""
            else:
                aligned[col_name] = 0.0
            logger.info("Added missing model-signature column with default: %s", col_name)

        if col_type == "integer":
            # MLflow "integer" is int32; int64 fails schema enforcement.
            aligned[col_name] = pd.to_numeric(aligned[col_name], errors="raise").astype("int32")
        elif col_type == "long":
            aligned[col_name] = pd.to_numeric(aligned[col_name], errors="raise").astype("int64")
        elif col_type in ("float", "double"):
            aligned[col_name] = pd.to_numeric(aligned[col_name], errors="raise").astype("float64")
        elif col_type == "boolean":
            aligned[col_name] = aligned[col_name].astype("bool")
        elif col_type == "string":
            aligned[col_name] = aligned[col_name].astype("string")

    return aligned[expected_cols]


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
