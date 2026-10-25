
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd

from contextlib import contextmanager
from loguru import logger

from MiBici.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR, PROJ_ROOT
from MiBici.dataset import MANIFEST_PATH

# Archivos con los que se ejecuta cualquier corrida; se guardan en config/ de cada run candidato
CONFIG_FILES = [PROJ_ROOT / "MLproject", PROJ_ROOT / "python_env.yaml", PROJ_ROOT / "MiBici" / "config.py"]

# skops (MLflow guarda asi los modelos de sklearn) requiere que se declare explícitamente qué clases son confiables,
# Estos modelos (tree y HGB) dan errror si no se declaran como confiables
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "functools.partial",
    "sklearn.utils.validation.check_array",
    "numpy.dtype",
]

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        if MLFLOW_TRACKING_URI.startswith("http"):
            mlflow.create_experiment(MLFLOW_EXPERIMENT)
        else:
            MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def convert_params_type(params: dict, default_params: dict) -> dict:
    """
    Convierte los tipos de los parámetros según el diccionario default_params, si no existe en default_params mostramos un warning.
    """
    converted_params = {}
    for key, value in params.items():
        # Valores 'none' los convertimos a None
        if value.lower() == "none":
            converted_params[key] = None
            continue

        if key in default_params:
            default_value = default_params[key]
            if isinstance(default_value, bool):
                converted_params[key] = value.lower() == "true"
            elif isinstance(default_value, int):
                converted_params[key] = int(value)
            elif isinstance(default_value, float):
                converted_params[key] = float(value)
            else:
                converted_params[key] = value
        else:
            logger.warning(f"Advertencia: El parámetro '{key}' no se encuentra en default_params y será ignorado.")
    return converted_params

def combinaciones(grid: dict) -> list[dict]:
    """Producto cartesiano del grid: {"a": [1, 2], "b": [3]} → [{"a": 1, "b": 3}, {"a": 2, "b": 3}]."""
    return [dict(zip(grid, valores)) for valores in itertools.product(*grid.values())]


def comando_candidato(pregunta: str, model: str, params: dict) -> str:
    """Comando de mlflow run que vuelve a entrenar una combinación como run candidate (con modelo y artefactos)."""
    opciones = " ".join(f"-P {k}={'none' if v is None else v}" for k, v in params.items())
    return (f"mlflow run . -e train -P question={pregunta} -P model={model} {opciones} "
            f"--experiment-name {MLFLOW_EXPERIMENT} --env-manager local")


@contextmanager
def search_run(pregunta: str, model_family: str, etiquetas: dict):
    """Run padre de una búsqueda; cada combinación se registra como run hijo dentro de él."""
    with mlflow.start_run(run_name=f"{pregunta} - {model_family} - search", nested=True) as parent:
        mlflow.set_tags(etiquetas | {"pregunta": pregunta, "model_family": model_family, "stage_of_work": "search"})
        yield parent


@contextmanager
def child_run(pregunta: str, model_family: str, params: dict, etiquetas: dict):
    """Run hijo de una búsqueda: una combinación de hiperparámetros."""
    nombre = ", ".join(f"{k}={v}" for k, v in params.items())
    with mlflow.start_run(run_name=f"{pregunta} - {model_family} - {nombre}", nested=True) as run:
        mlflow.set_tags(etiquetas | {"pregunta": pregunta, "model_family": model_family, "stage_of_work": "search"})
        mlflow.log_params(params)
        yield run


def log_run_artifacts(
    run_config: dict,
    train_df: pd.DataFrame | None = None,
    train_source: Path | None = None,
    target: str | None = None,
    model=None,
    input_example: pd.DataFrame | None = None,
    figures: dict | None = None,
    files: dict | None = None,
) -> None:
    """Registra en el run activo los artefactos que respaldan el resultado, organizados en subcarpetas.

    - data/: manifiesto de la ingesta (versión de los datos) y el conjunto de entrenamiento con log_input.
    - config/: MLproject, python_env.yaml, config.py y `run_config` (pregunta, modelo e hiperparámetros).
    - plots/: las figuras de `figures` ({"nombre.png": fig}).
    - `files`: archivos extra, por ejemplo tablas de resultados.
    - model/: el modelo con log_model, su firma y un ejemplo de entrada.

    `train_df` es la tabla agregada que entra al modelo.
    """
    mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")
    if train_df is not None:
        dataset = mlflow.data.from_pandas(train_df, source=str(train_source) if train_source else None,
                                          name=f"{run_config['pregunta']}_train", targets=target)
        mlflow.log_input(dataset, context="training")

    for archivo in CONFIG_FILES:
        mlflow.log_artifact(str(archivo), artifact_path="config")
    mlflow.log_dict(run_config, "config/run_config.json")

    for nombre, fig in (figures or {}).items():
        mlflow.log_figure(fig, f"plots/{nombre}")
        plt.close(fig)
    for archivo, carpeta in (files or {}).items():
        mlflow.log_artifact(str(archivo), artifact_path=carpeta)

    if model is not None and hasattr(model, "predict"):
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=input_example,
                                 skops_trusted_types=SKOPS_TRUSTED_TYPES)
    elif model is not None:
        logger.warning(f"{type(model).__name__} no tiene predict: el modelo no se registra con log_model.")