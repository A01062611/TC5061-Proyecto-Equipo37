from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import os
# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data
TRIPS_CSV = INTERIM_DATA_DIR / "mibici_2024_2025.csv"
STATIONS_CSV = INTERIM_DATA_DIR / "nomenclatura_2026_09.csv"
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2025-12-31"
TRAIN_FIN = "2025-06-30"
VALIDATION_FIN = "2025-09-30"

# Reproducibility
SEED = 42

# MLflow
MLFLOW_EXPERIMENT = "mibici-equipo37"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}")
MLFLOW_ARTIFACT_DIR = PROJ_ROOT / "mlruns" / MLFLOW_EXPERIMENT

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
