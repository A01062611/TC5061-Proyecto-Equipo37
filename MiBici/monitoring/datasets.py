"""Conjuntos de monitoreo del modelo P2: referencia y periodo actual.

Uso (desde la raíz del repositorio, después de `python -m MiBici.dataset monitoreo`):
    python -m MiBici.monitoring.datasets
    python -m MiBici.monitoring.datasets --tau 0.15 --model-uri models:/mibici-equipo37-p2-riesgo@champion

Salidas en `data/processed/monitoring/`:
- `reference.parquet`: partición train de `data/processed/p2/tabla.parquet` (2024-01 a 2025-06),
  con la que se entrenó el champion.
- `current.parquet`: 2026-07 a 2026-10, el "periodo de producción".
- `retrain.parquet`: 2026-07 a 2026-08, solo para entrenar el challenger.
- `ab_eval.parquet`: 2026-09 a 2026-10, solo para el A/B; ningún modelo la ve al entrenar.
- `s0_por_estacion.csv`: el ŝ₀ usado en el periodo actual.
- `info.json`: filas, estaciones, fechas y distribución de `clase` por conjunto, y el modelo usado.

Todos tienen las mismas columnas: `estacion`, `dia`, `periodo`, `estacion_nueva`, `clase`,
las 23 features de P2 (`features_de_p2`), y la predicción del champion (`pred`, `proba_*`).

- Se construye desde 2026-06-01 para que los lags (ayer, hace una semana, 7 días) de julio
  tengan datos, y después se recorta a partir de 2026-07-01.
- `pred` aplica la regla con la que se evaluó el champion en `train_p2`: la clase de riesgo más
  probable si su probabilidad llega a τ; si no, normal. τ se lee de los parámetros del run
  del modelo (o se pasa con `--tau`).
- `estacion_nueva` marca las estaciones que no aparecen en la referencia, para reportarlas aparte.
"""

import json
from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from MiBici.config import (
    MLFLOW_TRACKING_URI,
    MONITOREO_TRIPS_CSV,
    MONITORING_DATA_DIR,
    PROJ_ROOT,
    STATIONS_CSV,
)
from MiBici.features import CLASES_P2, OUT_DIR_P2, PARAMS_P2, construir_dataset_p2, features_de_p2

app = typer.Typer()

MODEL_NAME = "mibici-equipo37-p2-riesgo"
MODELO_EMPACADO = PROJ_ROOT / "models" / MODEL_NAME

# Periodo actual y sus dos ventanas (fechas de día operativo, inclusivas)
## un mes antes, solo para los lags
CONSTRUCCION_DESDE = "2026-06-01"  
ACTUAL_DESDE = "2026-07-01"
RETRAIN_HASTA = "2026-08-31"
AB_DESDE = "2026-09-01"
# último día operativo completo (el del 31 termina el 1 de noviembre)
ACTUAL_HASTA = "2026-10-30"  

PARAMS_MONITOREO = PARAMS_P2 | {"fecha_inicio": CONSTRUCCION_DESDE, "fecha_fin": ACTUAL_HASTA}

COLUMNAS_ID = ["estacion", "dia", "periodo", "estacion_nueva", "clase"]
COLUMNAS_PRED = ["pred"] + [f"proba_{c}" for c in CLASES_P2]


# --------------------------------------------------------------------------- modelo

def cargar_modelo(model_uri: str | None) -> tuple[mlflow.pyfunc.PyFuncModel, str]:
    """El modelo empacado en `models/` (la versión champion que sirve la API) o el de `model_uri`."""
    if model_uri is None:
        if not (MODELO_EMPACADO / "MLmodel").is_file():
            model_uri = f"models:/{MODEL_NAME}@champion"
        else:
            model_uri = MODELO_EMPACADO.as_posix()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Cargando modelo de {model_uri}")
    return mlflow.pyfunc.load_model(model_uri), model_uri


def tau_del_run(run_id: str) -> float:
    """τ con el que se evaluó el modelo, de los parámetros de su run en MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    params = mlflow.MlflowClient().get_run(run_id).data.params
    if "tau" not in params:
        raise typer.BadParameter(f"el run {run_id} no tiene el parámetro tau; pásalo con --tau")
    return float(params["tau"])


def predecir(modelo: mlflow.pyfunc.PyFuncModel, tabla: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Agrega las probabilidades por clase y la predicción con la regla de τ de `train_p2`."""
    entrada = modelo.metadata.get_input_schema().input_names()
    X = tabla[entrada].astype({"hora": "int32"})
    proba = modelo.get_raw_model().predict_proba(X)
    riesgo = proba[:, 1:].argmax(1) + 1
    tabla["pred"] = np.where(proba[:, 1:].max(1) >= tau, riesgo, 0).astype("int8")
    for i, c in enumerate(CLASES_P2):
        tabla[f"proba_{c}"] = proba[:, i].astype("float32")
    return tabla


# --------------------------------------------------------------------------- conjuntos

def construir_referencia() -> pd.DataFrame:
    """Partición train de la tabla de P2: los datos con los que se entrenó el champion."""
    tabla = pd.read_parquet(OUT_DIR_P2 / "tabla.parquet")
    ref = tabla[tabla["particion"] == "train"].drop(columns="particion").reset_index(drop=True)
    ref.insert(2, "periodo", "reference")
    return ref


def construir_actual(trips_path: Path, stations_path: Path, s0_ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tabla de P2 del periodo actual con el ŝ₀ del entrenamiento, recortada a julio–octubre."""
    tabla, s0_est, info = construir_dataset_p2(trips_path, stations_path, PARAMS_MONITOREO, s0_est=s0_ref)
    tabla = tabla[tabla["dia"] >= ACTUAL_DESDE].drop(columns="particion").reset_index(drop=True)
    tabla.insert(2, "periodo", np.where(tabla["dia"] <= RETRAIN_HASTA, "retrain", "ab_eval"))
    return tabla, s0_est, info


def ordenar(tabla: pd.DataFrame, estaciones_ref: set) -> pd.DataFrame:
    """Mismas columnas y tipos en todos los conjuntos."""
    tabla["estacion_nueva"] = ~tabla["estacion"].isin(estaciones_ref)
    tabla["periodo"] = tabla["periodo"].astype("string")
    features = [c for c in features_de_p2(tabla)]
    return tabla[COLUMNAS_ID + features + COLUMNAS_PRED]


def resumen(tabla: pd.DataFrame) -> dict:
    """Reporte de cada conjunto."""
    return {
        "desde": str(tabla["dia"].min().date()),
        "hasta": str(tabla["dia"].max().date()),
        "dias": int(tabla["dia"].nunique()),
        "filas": len(tabla),
        "estaciones": int(tabla["estacion"].nunique()),
        "estaciones_nuevas": int(tabla.loc[tabla["estacion_nueva"], "estacion"].nunique()),
        "filas_estaciones_nuevas": int(tabla["estacion_nueva"].sum()),
        "clase": tabla["clase"].map(dict(enumerate(CLASES_P2))).value_counts(normalize=True).round(4).to_dict(),
        "pred": tabla["pred"].map(dict(enumerate(CLASES_P2))).value_counts(normalize=True).round(4).to_dict(),
    }


# --------------------------------------------------------------------------- main

@app.command()
def main(
    trips_path: Path = MONITOREO_TRIPS_CSV,
    stations_path: Path = STATIONS_CSV,
    out_dir: Path = MONITORING_DATA_DIR,
    model_uri: str = typer.Option(None, help="Modelo a usar; por defecto el empacado en models/ o @champion"),
    tau: float = typer.Option(None, help="Umbral de la regla de predicción; por defecto, el del run del modelo"),
):
    """Genera reference, current, retrain y ab_eval con las features, la etiqueta y la predicción del champion."""
    if not trips_path.exists():
        raise typer.BadParameter(f"no existe {trips_path}; corre antes `python -m MiBici.dataset monitoreo`")

    modelo, model_uri = cargar_modelo(model_uri)
    run_id = modelo.metadata.run_id
    if tau is None:
        tau = tau_del_run(run_id)
    logger.info(f"Modelo del run {run_id}, τ = {tau}")

    logger.info("Referencia: partición train de la tabla de P2...")
    ref = construir_referencia()
    estaciones_ref = set(ref["estacion"])
    s0_ref = pd.read_csv(OUT_DIR_P2 / "s0_por_estacion.csv", index_col="estacion")

    logger.info(f"Actual: construyendo la tabla de P2 de {CONSTRUCCION_DESDE} a {ACTUAL_HASTA} con el ŝ₀ del entrenamiento...")
    act, s0_act, info_act = construir_actual(trips_path, stations_path, s0_ref)

    logger.info("Prediciendo con el champion...")
    ref = ordenar(predecir(modelo, ref, tau), estaciones_ref)
    act = ordenar(predecir(modelo, act, tau), estaciones_ref)

    conjuntos = {
        "reference": ref,
        "current": act,
        "retrain": act[act["periodo"] == "retrain"].reset_index(drop=True),
        "ab_eval": act[act["periodo"] == "ab_eval"].reset_index(drop=True),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in conjuntos.items():
        tabla.to_parquet(out_dir / f"{nombre}.parquet", index=False)
    s0_act.to_csv(out_dir / "s0_por_estacion.csv")

    info = {
        "modelo": {"uri": model_uri, "run_id": run_id, "tau": tau},
        "features": features_de_p2(ref),
        "conjuntos": {nombre: resumen(tabla) for nombre, tabla in conjuntos.items()},
        "construccion_actual": {k: v for k, v in info_act.items() if k != "params"},
        "params_actual": PARAMS_MONITOREO,
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n")
    for nombre, r in info["conjuntos"].items():
        logger.info(f"{nombre:>9}: {r['desde']} … {r['hasta']} · {r['filas']:,} filas · "
                    f"{r['estaciones']} estaciones ({r['estaciones_nuevas']} nuevas)")
    logger.success(f"Conjuntos de monitoreo en {out_dir}")


if __name__ == "__main__":
    app()
