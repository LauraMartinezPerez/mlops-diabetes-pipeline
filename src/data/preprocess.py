"""
preprocess.py

Paso de limpieza de datos del pipeline MLOps.
Lee los datos crudos de data/raw/, arregla los tipos, trata los valores
nulos, elimina duplicados y quita valores extremos (outliers).
Guarda el resultado en data/processed/. El dato crudo nunca se modifica.
"""

import pandas as pd
from pathlib import Path

# Raíz del proyecto (src/data/preprocess.py -> subimos dos niveles).
# Así el script funciona aunque se lance desde otra carpeta.
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "diabetes_clean.csv"

TARGET_COLUMN = "target"

# Columna sobre la que se aplica el filtro de outliers
OUTLIER_COLUMN = "bmi"


def load_data(path: Path) -> pd.DataFrame:
    """Lee un CSV y devuelve un DataFrame."""
    return pd.read_csv(path)


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte todas las columnas a numérico; lo inválido pasa a NaN."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena los nulos de las features con su mediana.

    Las filas sin target se eliminan: no se puede entrenar sin saber
    el valor que queremos predecir.
    """
    df = df.dropna(subset=[TARGET_COLUMN])

    for col in df.columns:
        if col != TARGET_COLUMN:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas duplicadas."""
    return df.drop_duplicates()


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas por encima del percentil 99 de OUTLIER_COLUMN."""
    if OUTLIER_COLUMN in df.columns:
        threshold = df[OUTLIER_COLUMN].quantile(0.99)
        df = df[df[OUTLIER_COLUMN] < threshold]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos los pasos de limpieza y devuelve el DataFrame limpio."""
    # copy() para no modificar el DataFrame original
    df = df.copy()

    # El orden importa: primero los tipos, para que un valor inválido
    # pase a NaN y el paso siguiente lo trate como un nulo más
    df = fix_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Guarda el DataFrame como CSV sin el índice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Ejecuta el pipeline de limpieza completo."""
    print("Cargando datos crudos...")
    df = load_data(RAW_PATH)
    rows_before = len(df)
    print(f"Filas antes de limpiar: {rows_before}")

    print("Limpiando datos...")
    df = clean_data(df)
    rows_after = len(df)
    print(f"Filas después de limpiar: {rows_after}")
    print(f"Filas eliminadas: {rows_before - rows_after}")

    print("Guardando datos limpios...")
    save_data(df, PROCESSED_PATH)

    print(f"Dimensiones finales: {df.shape}")
    print(f"Ruta de salida: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()