"""
run_pipeline.py

Orquesta el pipeline MLOps completo, ejecutando en orden los scripts del proyecto:
extracción -> validación -> limpieza -> entrenamiento -> registro -> evaluación.
Si un paso falla, el pipeline se detiene (Fail-Fast) y no se ejecutan los siguientes.
"""

# Concepto MLOps:
# Un pipeline encadena pasos que dependen unos de otros. En lugar de lanzar
# cada script a mano, un único comando los ejecuta siempre en el mismo orden. 
# # Cada paso se ejecuta como un proceso independiente
# (subprocess) con el mismo Python del entorno virtual.

import logging
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Pasos del pipeline, en orden de ejecución
STEPS = [
    ("Extracción de datos", "src/data/load_data.py"),
    ("Validación de datos", "src/data/validate_data.py"),
    ("Limpieza de datos", "src/data/preprocess.py"),
    ("Entrenamiento y tracking", "src/models/train.py"),
    ("Registro del mejor modelo", "src/models/register_model.py"),
    ("Evaluación y quality gate", "src/models/evaluate_model.py"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_step(name: str, script: str) -> None:
    """Ejecuta un script desde la raíz del proyecto.

    Si el script termina con error, lanza CalledProcessError y el pipeline se detiene.
    """
    logger.info("=== %s (%s) ===", name, script)
    # sys.executable: usa el mismo Python que ejecuta este script (el del venv)
    # cwd=BASE_DIR: los scripts usan rutas relativas a la raíz del proyecto
    subprocess.run([sys.executable, script], cwd=BASE_DIR, check=True)


def main() -> None:
    """Ejecuta todos los pasos del pipeline en orden."""
    for name, script in STEPS:
        run_step(name, script)
    logger.info("Pipeline completado.")


if __name__ == "__main__":
    main()