"""
model.py

Cargador del modelo de MLflow para la API de diabetes.

Contiene el diccionario _state (modelo cargado y metadatos), la función
load_champion_model() que se ejecuta al arrancar, y dos funciones auxiliares
(get_model, get_model_info) que usan los endpoints para acceder al modelo.
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
from fastapi import HTTPException
from mlflow import MlflowClient

# Rutas y constantes: tracking URI e identificadores del registry

BASE_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

MODEL_NAME = "diabetes-progression-model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Logging: logger del módulo (basicConfig se configura en main.py)

logger = logging.getLogger(__name__)

# Estado: guarda el modelo cargado y sus metadatos entre peticiones

_state: dict = {
    "model": None,
    "model_version": None,
    "run_id": None,
    "model_type": None,
    "rmse": None,
}

# Cargador: rellena _state desde el registry de MLflow al arrancar


def load_champion_model() -> None:
    """Carga el modelo champion del registry de MLflow en _state.

    Fija el tracking URI, carga el modelo (pyfunc) y obtiene los metadatos de
    su versión con MlflowClient. Lanza RuntimeError si el alias no existe.
    """
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI.as_uri()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())

    logger.info("Cargando modelo del registry: %s", MODEL_URI)
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

        # La ruta del artefacto se construye con piezas conocidas en lugar de
        # seguir el artifact_uri guardado, que es una ruta absoluta.
        run = client.get_run(mv.run_id)
        model_path = (
            MLFLOW_TRACKING_URI / run.info.experiment_id / mv.run_id / "artifacts" / "model"
        )
        _state["model"] = mlflow.pyfunc.load_model(str(model_path))
        _state["model_version"] = str(mv.version)
        _state["run_id"] = mv.run_id
        _state["model_type"] = run.data.tags.get("model_type")
        _state["rmse"] = run.data.metrics.get("rmse")
        logger.info("Modelo cargado — versión=%s  run_id=%s", mv.version, mv.run_id)
    except Exception as exc:
        logger.error("No se pudo cargar el modelo: %s", exc)
        raise RuntimeError(
            f"No se pudo cargar '{MODEL_URI}'. "
            "Ejecuta antes evaluate_model.py para asignar el alias 'champion'."
        ) from exc

# Auxiliares: accesos seguros a _state para los endpoints


def get_model():
    """Devuelve el modelo cargado o lanza HTTP 503 si no hay modelo."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return _state["model"]


def get_model_info() -> dict:
    """Devuelve los metadatos del modelo cargado o lanza HTTP 503 si no hay modelo."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return {
        "model_name": MODEL_NAME,
        "version": _state["model_version"],
        "alias": MODEL_ALIAS,
        "run_id": _state["run_id"],
        "model_type": _state["model_type"],
        "rmse": _state["rmse"],
    }