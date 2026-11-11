"""
evaluate_model.py

Evalúa la última versión registrada del modelo
"diabetes-progression-model" con mlflow.evaluate().

Los resultados se guardan en un run de MLflow propio. Si el modelo supera
el quality gate (rmse < RMSE_THRESHOLD) se le asigna el alias "champion".
"""

# Concepto MLOps:
# La evaluación responde a "¿este modelo es lo bastante bueno para servirlo?".
# mlflow.evaluate() genera un resultado estructurado (métricas + artefactos)
# asociado a un run, lo que permite un quality gate reproducible.
# Los alias sustituyen a los antiguos Stages: en lugar de pasar a "Production",
# asignamos un alias legible ("champion") que quien consume el modelo puede
# usar sin conocer el número de versión.

import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

# Rutas y constantes

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "diabetes_clean.csv"
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

EXPERIMENT_NAME = "diabetes-progression"
REGISTERED_MODEL_NAME = "diabetes-progression-model"
TARGET = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Quality gate: el alias "champion" solo se asigna si el RMSE es menor que este valor
RMSE_THRESHOLD = 60.0


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# Datos

def load_test_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset limpio y devuelve solo el conjunto de test.

    Usa los mismos parámetros de división que train.py, de modo que el
    conjunto de test sea idéntico al que se usó al entrenar.

    Lanza FileNotFoundError si el archivo no existe.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de entrada: {path}\n"
            "Ejecuta antes la limpieza: python src/data/preprocess.py"
        )
    df = pd.read_csv(path)
    logger.info("Cargadas %d filas de %s", len(df), path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Tamaño del conjunto de test: %d filas", len(X_test))
    return X_test, y_test


# Registry helpers

def get_latest_version(client: MlflowClient, model_name: str) -> str:
    """Devuelve el número de la última versión registrada del modelo.

    Lanza ValueError si el modelo no tiene ninguna versión.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(
            f"No hay versiones del modelo registrado '{model_name}'.\n"
            "Ejecuta antes register_model.py."
        )
    latest = max(versions, key=lambda v: int(v.version))
    logger.info(
        "Última versión de '%s': %s (run_id: %s)",
        model_name,
        latest.version,
        latest.run_id,
    )
    return latest.version


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate_registered_model(
    client: MlflowClient,
    model_version: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> mlflow.models.EvaluationResult:
    """Ejecuta mlflow.evaluate() sobre la versión registrada y devuelve el resultado.

    Abre un run nuevo con el tag stage="evaluation" para distinguir las
    evaluaciones de los entrenamientos en la interfaz de MLflow.
    """
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
    logger.info("Evaluando el modelo: %s", model_uri)

    # mlflow.evaluate() necesita un DataFrame que incluya la columna target
    eval_data = X_test.copy()
    eval_data[TARGET] = y_test.values

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evaluation_{model_version}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "stage": "evaluation",
            "model_version": model_version,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "dataset": INPUT_PATH.name,
        })

        # mlflow.evaluate() calcula un conjunto estándar de métricas de
        # regresión y las guarda automáticamente en el run activo.
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets=TARGET,
            model_type="regressor",
            evaluators="default",
        )

        logger.info("Métricas de la evaluación:")
        for key, value in sorted(result.metrics.items()):
            logger.info("  %-35s %.6f", key, value)

    return result


# Quality gate

def passes_quality_gate(metrics: dict) -> bool:
    """Devuelve True si el modelo cumple el umbral mínimo de calidad.

    Criterio: rmse < RMSE_THRESHOLD.

    mlflow.evaluate() guarda el MSE como "mean_squared_error"; el RMSE se
    obtiene con su raíz cuadrada.
    """
    mse = metrics.get("mean_squared_error", float("inf"))
    rmse = mse ** 0.5

    logger.info(
        "Quality gate — rmse=%.4f (umbral < %.1f)",
        rmse, RMSE_THRESHOLD,
    )
    return rmse < RMSE_THRESHOLD


def assign_champion_alias(
    client: MlflowClient,
    model_name: str,
    model_version: str,
) -> None:
    """Asigna el alias "champion" a una versión del modelo.

    Quien consuma el modelo podrá cargarlo con models:/nombre@champion sin
    conocer el número de versión. Si otra versión ya tenía el alias,
    MLflow lo mueve automáticamente a esta.
    """
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version,
    )
    logger.info(
        "Alias 'champion' asignado a '%s' versión %s.",
        model_name,
        model_version,
    )
    logger.info(
        "Carga el modelo champion con: mlflow.pyfunc.load_model('models:/%s@champion')",
        model_name,
    )


# Main

def main() -> None:
    """Evalúa la última versión registrada y la promociona si pasa el quality gate."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    logger.info("Modelo registrado   : %s", REGISTERED_MODEL_NAME)

    client = MlflowClient()

    # --- Localizar la versión a evaluar -------------------------------------
    model_version = get_latest_version(client, REGISTERED_MODEL_NAME)

    # --- Cargar datos de test -----------------------------------------------
    X_test, y_test = load_test_data(INPUT_PATH)

    # --- Evaluar ------------------------------------------------------------
    result = evaluate_registered_model(client, model_version, X_test, y_test)

    # --- Quality gate -------------------------------------------------------
    if passes_quality_gate(result.metrics):
        logger.info("El modelo ha superado el quality gate.")
        assign_champion_alias(client, REGISTERED_MODEL_NAME, model_version)
    else:
        logger.warning(
            "La versión %s NO ha superado el quality gate. "
            "El alias 'champion' NO se ha asignado.",
            model_version,
        )

    logger.info(
        "Terminado. Abre la interfaz de MLflow para revisar la evaluación:\n"
        "  mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )


if __name__ == "__main__":
    main()