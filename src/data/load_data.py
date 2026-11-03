"""
load_data.py

Paso de extracción de datos del pipeline MLOps.
Carga el dataset de diabetes de scikit-learn y lo guarda como CSV
en data/raw/. Este script NO limpia ni transforma nada: su única
responsabilidad es obtener los datos crudos y dejarlos guardados.
"""

import os

import pandas as pd
from sklearn.datasets import load_diabetes

# Ruta de salida de los datos crudos (relativa a la raíz del proyecto,
# por eso el script se lanza siempre desde la raíz).
RAW_DATA_PATH = "data/raw/diabetes.csv"


def load_raw_data() -> pd.DataFrame:
    """Carga el dataset de diabetes y lo devuelve como DataFrame."""
    # as_frame=True devuelve ya un DataFrame con las features y la
    # columna 'target' (progresión de la enfermedad)
    dataset = load_diabetes(as_frame=True)

    # copy() para trabajar con una copia y no tocar el objeto original
    df = dataset.frame.copy()
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Guarda un DataFrame como CSV, creando las carpetas si hace falta.

    Args:
        df: DataFrame que se quiere guardar.
        path: Ruta del archivo de destino.
    """
    directory = os.path.dirname(path)

    # Si la ruta incluye carpeta, la crea (si ya existe, no pasa nada)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # index=False evita que pandas añada una columna extra con el índice
    df.to_csv(path, index=False)


def main() -> None:
    df = load_raw_data()
    print(f"Dataset extraído: {df.shape[0]} filas y {df.shape[1]} columnas")

    save_dataset(df, RAW_DATA_PATH)
    print(f"Datos crudos guardados en: {RAW_DATA_PATH}")


# Este bloque hace que main() solo se ejecute al lanzar el script
# directamente, y no cuando otro archivo lo importe.
if __name__ == "__main__":
    main()