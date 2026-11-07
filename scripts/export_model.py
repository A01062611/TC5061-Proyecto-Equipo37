"""Descarga un modelo del Model Registry de MLflow a ``models/<nombre>`` para empacarlo en la imagen.

Uso:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python scripts/export_model.py                                  # P2 @champion
    python scripts/export_model.py --name mibici-equipo37-p1-demanda --version 1

Los modelos del equipo guardan sus artefactos con ``mlflow-artifacts:``, así que el tracking URI
debe ser un servidor MLflow (no un ``sqlite:///``). Para levantarlo localmente:
    mlflow server --backend-store-uri sqlite:///mlflow_equipo37/mlflow.db \\
        --artifacts-destination mlflow_equipo37/mlartifacts --serve-artifacts --port 5000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, PROJ_ROOT.as_posix())

from app.model_loader import DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_NAME, feature_names, prepare_frame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--alias", default=os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS))
    group.add_argument("--version", help="Versión fija; tiene prioridad sobre el alias")
    parser.add_argument("--output", type=Path, help="Destino (por defecto models/<nombre>)")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()

    if not args.tracking_uri:
        parser.error("Define MLFLOW_TRACKING_URI o usa --tracking-uri")
    mlflow.set_tracking_uri(args.tracking_uri)

    uri = f"models:/{args.name}/{args.version}" if args.version else f"models:/{args.name}@{args.alias}"
    destino = args.output or PROJ_ROOT / "models" / args.name
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    print(f"Descargando {uri} desde {args.tracking_uri}")
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=destino.as_posix())

    # Verifica con la carga del modelo y con una predicción del ejemplo de entrada guardado
    modelo = mlflow.pyfunc.load_model(destino.as_posix())
    columnas = feature_names(modelo)
    print(f"Modelo exportado a {destino} ({len(columnas)} variables de entrada)")
    ejemplo = destino / "serving_input_example.json"
    if ejemplo.is_file():
        datos = json.loads(ejemplo.read_text())
        frame = pd.DataFrame(**datos["dataframe_split"]) if "dataframe_split" in datos else pd.DataFrame(datos)
        print("Predicción de verificación:", modelo.predict(prepare_frame(modelo, frame)).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
