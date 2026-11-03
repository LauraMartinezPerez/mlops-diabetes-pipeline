"""
validate_data.py

Paso de validación de datos del pipeline MLOps.
Lee los datos crudos de data/raw/ y comprueba que cumplen el esquema
esperado: columnas obligatorias, columna objetivo, tipos numéricos y
ausencia de valores nulos. Si hay demasiados errores, el script falla
(Fail-Fast) para detener el pipeline.
"""

import sys

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

RAW_DATA_PATH = "data/raw/diabetes.csv"

# Porcentaje máximo de filas inválidas que toleramos antes de abortar
MAX_INVALID_RATIO = 0.05


class DiabetesRow(BaseModel):
    """Esquema que debe cumplir cada fila del dataset (contrato de datos)."""

    # No se permiten NaN ni infinitos en los campos float
    model_config = ConfigDict(allow_inf_nan=False)

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
    target: float


def load_dataset(path: str) -> pd.DataFrame:
    """Lee el CSV de datos crudos."""
    return pd.read_csv(path)


def check_columns(df: pd.DataFrame) -> None:
    """Comprueba que existen todas las columnas obligatorias, incluida target."""
    required = list(DiabetesRow.model_fields)
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")


def validate_rows(df: pd.DataFrame) -> list:
    """Valida cada fila con Pydantic y devuelve la lista de errores."""
    errors = []

    for index, record in enumerate(df.to_dict(orient="records")):
        try:
            DiabetesRow(**record)
        except ValidationError as error:
            # Guardamos la fila y el primer problema detectado en ella
            first = error.errors()[0]
            errors.append((index, first["loc"][0], first["msg"]))

    return errors


def main() -> None:
    df = load_dataset(RAW_DATA_PATH)
    print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas")

    # 1. Esquema a nivel de tabla: columnas obligatorias
    check_columns(df)

    # 2. Validación fila a fila: tipos y valores nulos
    errors = validate_rows(df)
    total = len(df)
    invalid = len(errors)
    print(f"Filas válidas: {total - invalid} | Filas inválidas: {invalid}")

    for index, column, message in errors[:5]:
        print(f"  Fila {index}, columna '{column}': {message}")

    # 3. Fail-Fast: si hay demasiados errores, abortamos el pipeline
    if invalid / total > MAX_INVALID_RATIO:
        print("Validación FALLIDA: demasiadas filas inválidas.")
        sys.exit(1)

    print("Validación superada.")


if __name__ == "__main__":
    main()