"""
schemas.py

Esquemas Pydantic de la API de predicción de progresión de diabetes.
Define el modelo de entrada (DiabetesFeatures) y el de salida (PredictionResponse),
que FastAPI usa para validar las peticiones y serializar las respuestas.
"""

from pydantic import BaseModel, ConfigDict

# Columnas de features: lista ordenada igual que diabetes_clean.csv (sin target)

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]

# Esquemas — cuerpo de la petición (entrada) y de la respuesta (salida)


class DiabetesFeatures(BaseModel):
    """Entrada: las 10 variables de un paciente (sin la columna target)."""

    model_config = ConfigDict(allow_inf_nan=False)  # rechaza NaN e infinitos

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class PredictionResponse(BaseModel):
    """Salida: progresión de la enfermedad predicha y versión del modelo que respondió."""

    prediction: float
    model_version: str