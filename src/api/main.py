"""
main.py

Aplicación FastAPI que sirve predicciones del modelo champion de MLflow.

Carga el modelo registrado como 'diabetes-progression-model@champion' al arrancar
y expone tres endpoints: comprobación de estado, metadatos del modelo y predicción.
"""

# Concepto MLOps:
# El Model Serving hace que un modelo entrenado esté disponible para otros
# sistemas. En lugar de ejecutar un script a mano, exponemos el modelo detrás
# de una API HTTP: cualquier cliente puede pedir predicciones sin saber nada
# de scikit-learn ni de MLflow.
#
# Decisiones clave:
#   1. Carga única al arrancar: cargar un modelo es costoso, así que se hace una
#      sola vez (lifespan) y cada petición usa el modelo ya en memoria.
#   2. Validación con Pydantic: el esquema de entrada está declarado y FastAPI
#      rechaza (422) las peticiones mal formadas antes de llegar al modelo.
#   3. Carga por alias: se carga '@champion' y no un número de versión, así que
#      promocionar un nuevo champion basta para cambiar el modelo servido
#      (hay que reiniciar la API).
#   4. Errores controlados: sin modelo cargado -> 503; fallo al predecir -> 500.

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from .model import (
    MODEL_NAME,
    _state,
    get_model,
    get_model_info,
    load_champion_model,
)
from .schemas import FEATURE_COLUMNS, DiabetesFeatures, PredictionResponse

# Logging: se configura una sola vez en el punto de entrada

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Lifespan: hook de arranque y cierre; delega la carga del modelo en model.py


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Llama a load_champion_model() antes de aceptar peticiones.

    Si la carga falla, la API arranca igualmente y responde 503 hasta que
    haya un modelo champion disponible.
    """
    try:
        load_champion_model()
    except RuntimeError as exc:
        logger.error("La API arranca SIN modelo: %s", exc)
    yield  # la aplicación se ejecuta aquí
    logger.info("Cerrando — liberando el modelo de memoria.")
    _state["model"] = None

# App y endpoints: instancia de FastAPI y las tres rutas de la API

app = FastAPI(
    title="Diabetes Progression API",
    description="Sirve predicciones del modelo champion de MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Devuelve el estado del servicio y el nombre del modelo registrado."""
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/model-info")
def model_info() -> dict:
    """Devuelve los metadatos de la versión del modelo cargado."""
    return get_model_info()


@app.post("/predict")
def predict(features: DiabetesFeatures) -> PredictionResponse:
    """Devuelve la predicción de progresión de diabetes para un paciente.

    Args:
        features: Los 10 valores de entrada de un paciente.

    Returns:
        PredictionResponse con la predicción y la versión del modelo que respondió.
    """
    model = get_model()
    input_df = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)

    try:
        prediction = model.predict(input_df)
    except Exception as exc:
        logger.error("Fallo al predecir: %s", exc)
        raise HTTPException(status_code=500, detail="Error de predicción.") from exc

    result = float(prediction[0])
    logger.info("Predicción: %.4f", result)
    return PredictionResponse(prediction=result, model_version=_state["model_version"])