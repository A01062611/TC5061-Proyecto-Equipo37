"""Reentrenamiento del modelo P2 para generar el challenger."""

import hashlib

import mlflow
import pandas as pd
import typer
from loguru import logger

from MiBici.config import MONITORING_DATA_DIR, SEED
from MiBici.modeling.helper import log_run_artifacts, setup_mlflow
from MiBici.modeling.train_p2 import (
  DEFAULT_HYPERPARAMS,
  K_VISITAS,
  git_commit,
  load_features,
  nombre_metrica,
  sha256,
  tabla_path,
  train_model,
)

RETRAIN_PATH = MONITORING_DATA_DIR / "retrain.parquet"


def preparar_challenger(historico, retrain, features, train_fin):
  """Prepara los conjuntos sin utilizar la ventana A/B."""
  try:
    fin = pd.Timestamp(train_fin)
  except (TypeError, ValueError) as exc:
    raise ValueError("train_fin debe ser una fecha válida") from exc

  if pd.isna(fin) or not pd.Timestamp("2026-08-01") <= fin <= pd.Timestamp("2026-08-31"):
    raise ValueError("train_fin debe estar entre 2026-08-01 y 2026-08-31")

  columnas = list(dict.fromkeys(["estacion", "dia", "clase", *features]))
  faltantes = set(columnas + ["periodo"]) - set(retrain.columns)

  if faltantes:
    raise ValueError(f"Columnas faltantes: {sorted(faltantes)}")

  reciente = retrain.copy()
  reciente["dia"] = pd.to_datetime(reciente["dia"], errors="coerce")

  if reciente["dia"].isna().any() or not reciente["dia"].between("2026-07-01", "2026-08-31").all():
    raise ValueError("Fechas fuera de julio-agosto de 2026")

  if not reciente["periodo"].eq("retrain").all():
    raise ValueError("Datos fuera de la ventana de reentrenamiento")

  if historico.empty or pd.to_datetime(historico["dia"]).max() > pd.Timestamp("2025-06-30"):
    raise ValueError("El histórico debe terminar en junio de 2025")

  reciente = reciente[reciente["dia"] <= fin]

  julio = reciente[reciente["dia"] < "2026-08-01"]
  agosto = reciente[reciente["dia"] >= "2026-08-01"]

  if julio.empty or agosto.empty:
    raise ValueError("Se necesitan datos de julio y agosto")

  train = pd.concat([historico[columnas], julio[columnas]], ignore_index=True)
  val = agosto[columnas].copy()
  train_final = pd.concat([historico[columnas], reciente[columnas]], ignore_index=True)

  return train, val, train_final


def main(
  train_fin: str = typer.Option("2026-08-31", help="Último día de reentrenamiento")
):
  if not RETRAIN_PATH.is_file():
    raise FileNotFoundError(f"No existe {RETRAIN_PATH}")

  _, historico, _, _, _, features, _ = load_features()
  retrain = pd.read_parquet(RETRAIN_PATH)

  train, val, train_final = preparar_challenger(
    historico, retrain, features, train_fin
  )

  model_name = "hist_gradient_boosting"
  params = DEFAULT_HYPERPARAMS[model_name].copy()

  logger.info("Entrenando challenger con histórico y julio...")

  resultado = train_model(
    model_name,
    train,
    val,
    train["clase"].to_numpy(),
    val["clase"].to_numpy(),
    features,
    K_VISITAS,
    SEED,
    params,
  )

  tau = resultado["params"]["tau"]

  logger.info("Reentrenando modelo final con histórico, julio y agosto...")

  estimador = resultado["estimador"]
  estimador.fit(
    train_final[features],
    train_final["clase"].to_numpy(),
  )

  version_datos = hashlib.sha256(
    f"{sha256(tabla_path)}:{sha256(RETRAIN_PATH)}".encode()
  ).hexdigest()

  commit, clean = git_commit()

  resumen = {
    "historico_hasta": "2025-06-30",
    "retrain_desde": "2026-07-01",
    "retrain_hasta": train_fin,
    "validacion": "Agosto de 2026 antes del reentrenamiento final",
    "evaluacion_ab": "Septiembre-octubre de 2026, no utilizados",
    "filas_historico": len(historico),
    "filas_julio": len(train) - len(historico),
    "filas_validacion": len(val),
    "filas_entrenamiento_final": len(train_final),
    "features": features,
    "tau": tau,
    "version_datos": version_datos,
  }

  setup_mlflow()

  with mlflow.start_run(run_name="P2 - Challenger 2026") as run:
    mlflow.log_params({
      "train_fin": train_fin,
      "tau": tau,
      "seed": SEED,
      **{f"hgb_{k}": v for k, v in params.items()},
    })

    mlflow.log_metrics({
      nombre_metrica(k): float(v)
      for k, v in resultado["res"].items()
    })

    mlflow.set_tags({
      "equipo": "37",
      "pregunta": "P2",
      "model_family": model_name,
      "stage_of_work": "candidate",
      "model_role": "challenger",
      "git_commit": commit,
      "git_worktree_clean": clean,
      "version_datos": version_datos,
      "validacion": "agosto-2026-pre-refit",
      "datos_ab_utilizados": "false",
    })

    mlflow.log_dict(resumen, "data/challenger_periodos.json")

    ejemplo = val[features].dropna().head(5)

    if ejemplo.empty:
      ejemplo = val[features].head(5)

    log_run_artifacts(
      run_config={
        "pregunta": "P2",
        "model": model_name,
        "stage_of_work": "candidate",
        "hyperparams": params,
        "tau": tau,
        "features": features,
        "periodos": resumen,
      },
      train_df=train_final[features + ["clase"]],
      target="clase",
      model=estimador,
      input_example=ejemplo,
    )

    logger.success(f"Challenger entrenado. Run ID: {run.info.run_id}")
    logger.info(f"Filas de entrenamiento final: {len(train_final):,}")
    logger.info(f"Tau calibrado: {tau}")
    logger.info("Las métricas de agosto son previas al ajuste final.")


if __name__ == "__main__":
  typer.run(main)