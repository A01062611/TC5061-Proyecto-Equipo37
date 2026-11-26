"""Carga del modelo registrado en MLflow para el servicio de inferencia.

Orden de preferencia:
1. Modelo empacado en ``MODEL_DIR`` (por defecto ``models/<MODEL_NAME>``), exportado con
   ``scripts/export_model.py``. No requiere servidor MLflow: es lo que usa la imagen.
2. Model Registry de MLflow, con ``models:/<nombre>@<alias>`` (o ``models:/<nombre>/<versión>``),
   usando la variable de entorno ``MLFLOW_TRACKING_URI``.

Toda la configuración sale de variables de entorno.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_NAME = "mibici-equipo37-p2-riesgo"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_TAU = 0.15


def model_name() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", PROJ_ROOT / "models" / model_name()))


def registry_uri() -> str:
    """URI del modelo en el Registry: versión fija si ``MODEL_VERSION`` existe, si no el alias."""
    version = os.getenv("MODEL_VERSION")
    if version:
        return f"models:/{model_name()}/{version}"
    return f"models:/{model_name()}@{os.getenv('MODEL_ALIAS', DEFAULT_MODEL_ALIAS)}"


def tau() -> float:
    """Umbral de decisión: ``MODEL_TAU`` si existe, si no el del champion."""
    return float(os.getenv("MODEL_TAU", DEFAULT_TAU))


def model_version() -> str:
    """Versión del modelo en el Registry: ``MODEL_VERSION`` > metadatos del paquete > ``desconocida``."""
    env = os.getenv("MODEL_VERSION")
    if env:
        return env
    meta = model_dir() / "registered_model_meta"
    if meta.is_file():
        for line in meta.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model_version":
                return value.strip().strip("'\"")
    return "desconocida"


def load_model() -> mlflow.pyfunc.PyFuncModel:
    """Carga el modelo empacado si existe; si no, lo recupera del Registry."""
    local = model_dir()
    if (local / "MLmodel").is_file():
        logger.info("Cargando modelo empacado desde %s", local)
        return mlflow.pyfunc.load_model(local.as_posix())

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            f"No hay modelo empacado en {local} y MLFLOW_TRACKING_URI no está definida. "
            "Ejecuta scripts/export_model.py o define MLFLOW_TRACKING_URI."
        )
    mlflow.set_tracking_uri(tracking_uri)
    uri = registry_uri()
    logger.info("Cargando modelo %s desde %s", uri, tracking_uri)
    return mlflow.pyfunc.load_model(uri)


def feature_names(model: mlflow.pyfunc.PyFuncModel) -> list[str]:
    """Nombres de las variables de entrada, en el orden de la firma del modelo."""
    schema = model.metadata.get_input_schema()
    if schema is None:
        raise ValueError("El modelo no tiene firma de entrada")
    return list(schema.input_names())


_DTYPES = {"integer": "int32", "long": "int64", "float": "float32", "double": "float64", "boolean": "bool"}


def prepare_frame(model: mlflow.pyfunc.PyFuncModel, data: pd.DataFrame) -> pd.DataFrame:
    """Ordena las columnas y convierte los tipos a los que exige la firma (p. ej. ``hora`` -> int32)."""
    schema = model.metadata.get_input_schema()
    if schema is None:
        raise ValueError("El modelo no tiene firma de entrada")
    frame = data[schema.input_names()].copy()
    for col in schema.inputs:
        dtype = _DTYPES.get(col.type.name)
        if dtype:
            frame[col.name] = frame[col.name].astype(dtype)
    return frame


def aplicar_umbral(proba: np.ndarray, umbral: float) -> np.ndarray:
    """Regla de P2, igual a ``pred_umbral`` del entrenamiento: la clase de riesgo más probable
    (1 vaciado, 2 llenado) si su probabilidad llega al umbral; si no, 0 (normal)."""
    riesgo = proba[:, 1:].argmax(1) + 1
    return np.where(proba[:, 1:].max(1) >= umbral, riesgo, 0)


def predict(model: mlflow.pyfunc.PyFuncModel, rows: list[dict[str, Any]]) -> list[int]:
    """Firma de predicción para la API: lista de dicts (una fila por registro) -> lista de clases."""
    estimador = model.get_raw_model()
    if list(estimador.classes_) != [0, 1, 2]:
        raise ValueError(f"Clases inesperadas en el modelo: {list(estimador.classes_)}")
    proba = estimador.predict_proba(prepare_frame(model, pd.DataFrame(rows)))
    return aplicar_umbral(proba, tau()).tolist()
