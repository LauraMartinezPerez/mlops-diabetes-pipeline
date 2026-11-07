"""
register_model.py

Registra el mejor run del experimento 'diabetes-progression' en el Model Registry de MLflow.

Busca entre todos los runs del experimento el de menor RMSE y registra
su modelo con el nombre "diabetes-progression-model".
"""

# Concepto MLOps:
# El Model Registry es un almacén central de modelos versionados. Tras hacer
# el tracking de los experimentos, registramos el modelo del mejor run para
# poder referenciarlo por nombre (y más adelante por alias) sin depender de
# un run_id concreto.

import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"
EXPERIMENT_NAME = "diabetes-progression"
REGISTERED_MODEL_NAME = "diabetes-progression-model"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def get_best_run(client: MlflowClient, experiment_name: str) -> mlflow.entities.Run:
    """Devuelve el run terminado con menor RMSE del experimento.

    Lanza ValueError si el experimento no existe o no tiene runs terminados.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"No se encuentra el experimento '{experiment_name}'. "
            "Ejecuta antes train.py para crear los runs."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No hay runs terminados en el experimento '{experiment_name}'."
        )

    best = runs[0]
    logger.info(
        "Mejor run seleccionado — id: %s  rmse: %.4f  model_type: %s",
        best.info.run_id,
        best.data.metrics.get("rmse", float("nan")),
        best.data.tags.get("model_type", "desconocido"),
    )
    return best


def register_model(
    client: MlflowClient,
    run: mlflow.entities.Run,
    registered_model_name: str,
) -> mlflow.entities.model_registry.ModelVersion:
    """Registra en el Model Registry el modelo guardado como artefacto del run.

    Si el modelo registrado no existe todavía, MLflow lo crea automáticamente.
    Devuelve la versión creada.
    """
    model_uri = f"runs:/{run.info.run_id}/model"
    logger.info("Registrando modelo desde: %s", model_uri)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    logger.info(
        "Modelo registrado — nombre: '%s'  versión: %s",
        model_version.name,
        model_version.version,
    )
    return model_version


def add_model_description(
    client: MlflowClient,
    run: mlflow.entities.Run,
    model_version: mlflow.entities.model_registry.ModelVersion,
) -> None:
    """Añade a la versión registrada una descripción con sus métricas clave."""
    metrics = run.data.metrics
    model_type = run.data.tags.get("model_type", "desconocido")

    description = (
        f"Mejor run del experimento '{EXPERIMENT_NAME}'. "
        f"Tipo de modelo: {model_type}. "
        f"RMSE: {metrics.get('rmse', float('nan')):.4f}  "
        f"MAE: {metrics.get('mae', float('nan')):.4f}  "
        f"R2: {metrics.get('r2', float('nan')):.4f}"
    )

    client.update_model_version(
        name=model_version.name,
        version=model_version.version,
        description=description,
    )
    logger.info("Descripción añadida a la versión del modelo.")


def main() -> None:
    """Busca el mejor run del experimento y lo registra."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    client = MlflowClient()

    logger.info("Buscando el mejor run del experimento '%s'...", EXPERIMENT_NAME)
    best_run = get_best_run(client, EXPERIMENT_NAME)

    model_version = register_model(client, best_run, REGISTERED_MODEL_NAME)
    add_model_description(client, best_run, model_version)

    logger.info(
        "Terminado. El modelo '%s' versión %s ya está en el registry.",
        model_version.name,
        model_version.version,
    )


if __name__ == "__main__":
    main()