"""
train.py

Paso de entrenamiento del pipeline MLOps.
Entrena tres modelos de regresión sobre el dataset de diabetes y registra
cada ejecución con MLflow Experiment Tracking: hiperparámetros, métricas
y el modelo serializado, para poder compararlos en la interfaz de MLflow.
"""

# Concepto MLOps:
# El Experiment Tracking responde a "qué probé y qué tal funcionó?".
# En lugar de apuntar resultados a mano, MLflow guarda cada ejecución:
# hiperparámetros, métricas y el propio modelo. Después, con `mlflow ui`,se comparan los runs uno al lado del otro.

import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Rutas y constantes

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "diabetes_clean.csv"
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

EXPERIMENT_NAME = "diabetes-progression"
TARGET = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# Datos

def load_features(path: Path) -> pd.DataFrame:
    """Carga el dataset limpio desde el CSV.

    Lanza FileNotFoundError si el archivo no existe.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de entrada: {path}\n"
            "Ejecuta antes la limpieza: python src/data/preprocess.py"
        )
    df = pd.read_csv(path)
    logger.info("Cargadas %d filas y %d columnas de %s", len(df), df.shape[1], path)
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa las features del target y divide en entrenamiento y test.

    Devuelve (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train: %d filas  |  Test: %d filas", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


# Métricas

def compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Calcula las métricas de regresión: rmse, mae y r2"""
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# Run de MLflow

def run_experiment(model, model_type: str, params: dict,
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series) -> None:
    """Entrena un modelo y registra todo en un único run de MLflow"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # --- Tags 
        mlflow.set_tags({
            "model_type": model_type,
            "dataset": INPUT_PATH.name,
            "stage": "training",
        })

        # Parámetros 
        mlflow.log_params(params)

        # Entrenamiento 
        logger.info("[%s] Entrenando...", model_type)
        model.fit(X_train, y_train)

        # Métricas
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        logger.info(
            "[%s] rmse=%.4f  mae=%.4f  r2=%.4f",
            model_type,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )

        # Artefacto (el modelo)
        # registered_model_name=None: el registro se hace en register_model.py
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
            input_example=X_train.head(5),
        )
        logger.info("[%s] Modelo guardado como artefacto.", model_type)


# Main

def main() -> None:
    """Ejecuta el entrenamiento con tracking para los tres modelos."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    logger.info("Experimento         : %s", EXPERIMENT_NAME)

    df = load_features(INPUT_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    # Modelo 1: LinearRegression
    run_experiment(
        model=LinearRegression(),
        model_type="LinearRegression",
        params={"fit_intercept": True},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )

    # Modelo 2: RandomForestRegressor
    run_experiment(
        model=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        model_type="RandomForestRegressor",
        params={"n_estimators": 100, "random_state": RANDOM_STATE},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )

    # Modelo 3: GradientBoostingRegressor
    run_experiment(
        model=GradientBoostingRegressor(random_state=RANDOM_STATE),
        model_type="GradientBoostingRegressor",
        params={"random_state": RANDOM_STATE},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )

    logger.info(
        "Terminado. Abre la interfaz de MLflow para comparar los runs:\n"
        "  mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )


if __name__ == "__main__":
    main()