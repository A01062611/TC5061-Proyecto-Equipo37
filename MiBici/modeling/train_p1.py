"""P1: entrenamiento y registro MLflow desde código del repositorio.

Ubicación: MiBici/modeling/train_p1.py
Ejecución desde la raíz: python -m MiBici.modeling.train_p1
"""

from collections import Counter
from pathlib import Path
import hashlib
import subprocess
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from MiBici.modeling.helper import convert_params_type, setup_mlflow, combinaciones, search_run, child_run, log_run_artifacts
from MiBici.config import FECHA_INICIO, FECHA_FIN, PROJ_ROOT, TRAIN_FIN, TRIPS_CSV, SEED, VALIDATION_FIN

warnings.filterwarnings("ignore", message=r"IProgress not found.*")
import mlflow


FEATURE_COLUMNS = [
  "estacion_id",
  "dia_semana",
  "mes",
  "es_fin_semana",
  "viajes_lag_1",
  "viajes_lag_7",
  "promedio_7d"
]
DEFAULT_HYPERPARAMS = {
  "random_forest": {
    "n_estimators": 100,
    "max_depth": 18,
    "min_samples_leaf": 5,
    "random_state": SEED,
    "n_jobs": -1
  },
  "hist_gradient_boosting": {
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "random_state": SEED
  },
  "ridge": {
    "alpha": 1.0,
    "max_iter": 1000,
    "solver": "auto"
  }
}
SEARCH_GRIDS = {
  "random_forest": {
    "n_estimators": [100, 200],
    "max_depth": [10, 18, 25],
    "min_samples_leaf": [5, 15]
  },
  "hist_gradient_boosting": {
    "max_iter": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63]
  },
  "ridge": {
    "alpha": [0.1, 1.0, 10.0],
    "max_iter": [500, 1000, 2000],
    "solver": ["auto", "lsqr", "sparse_cg"]
  }
}

def preparar(df, dias_imputados, hash_csv, model) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
  """Partición temporal, parámetros comunes y etiquetas base."""
  train = df[df["fecha"] <= TRAIN_FIN].copy()
  val = df[
    (df["fecha"] > TRAIN_FIN) &
    (df["fecha"] <= VALIDATION_FIN)
  ].copy()
  test = df[df["fecha"] > VALIDATION_FIN].copy()
  if min(len(train), len(val), len(test)) == 0:
    raise ValueError("Algún conjunto temporal no contiene registros.")
  if not (train["fecha"].max() < val["fecha"].min() < test["fecha"].min()):
    raise ValueError("La partición temporal es inválida.")

  # Commit actual y valores de reproducibilidad
  git_commit, git_worktree_clean = commit_actual()
  comunes = {
    "horizonte_dias": 1,
    "inicio_entrenamiento": str(train["fecha"].min().date()),
    "fin_entrenamiento": str(train["fecha"].max().date()),
    "inicio_validacion": str(val["fecha"].min().date()),
    "fin_validacion": str(val["fecha"].max().date()),
    "filas_entrenamiento": int(len(train)),
    "filas_validacion": int(len(val)),
    "dias_imputados": dias_imputados,
    "datos_sha256": hash_csv,
    "variable_objetivo": "viajes_iniciados"
  }

  # Etiquetas para MLflow
  etiquetas = {
      "equipo": "37",
      "pregunta": "P1",
      "git_commit": git_commit,
      "git_worktree_clean": git_worktree_clean,
      "version_datos": hash_csv,
      "conjunto_metricas": "validacion",
      "model_family": model
    }

  return train, val, comunes, etiquetas


def leer_y_agregar_viajes(path: Path) -> tuple[pd.DataFrame, int, str]:
  """Valida los IDs y fechas, y cuenta viajes por estación y día."""
  if not path.is_file():
    raise FileNotFoundError("Generar el CSV de viajes primero con `python -m MiBici.dataset`.")

  conteos_diarios = Counter()
  ids_por_bloque = []
  dias_registrados = set()
  meses = set()
  total_viajes = 0
  fechas_invalidas = 0
  ids_viaje_invalidos = 0
  origenes_invalidos = 0
  fechas_fuera_rango = 0

  hasher = hashlib.sha256()
  with path.open("rb") as archivo:
    for bloque in iter(lambda: archivo.read(8 * 1024 * 1024), b""):
      hasher.update(bloque)

  for chunk in pd.read_csv(
    path,
    usecols=["Viaje_Id", "Inicio_del_viaje", "Origen_Id"],
    chunksize=200_000
  ):
    total_viajes += len(chunk)
    fechas = pd.to_datetime(chunk["Inicio_del_viaje"], errors="coerce")
    ids = pd.to_numeric(chunk["Viaje_Id"], errors="coerce")
    origenes = pd.to_numeric(chunk["Origen_Id"], errors="coerce")

    ids_validos = ids.notna() & (ids > 0) & (ids % 1 == 0)
    origenes_validos = origenes.notna() & (origenes > 0) & (origenes % 1 == 0)
    fechas_validas = fechas.notna()

    fechas_invalidas += int((~fechas_validas).sum())
    ids_viaje_invalidos += int((~ids_validos).sum())
    origenes_invalidos += int((~origenes_validos).sum())
    fechas_fuera_rango += int((
      fechas_validas & ~fechas.dt.normalize().between(FECHA_INICIO, FECHA_FIN)
    ).sum())

    ids_por_bloque.append(ids[ids_validos].to_numpy(dtype=np.int64))
    fechas_ok = fechas[fechas_validas]
    dias_registrados.update(fechas_ok.dt.normalize().unique())
    meses.update(fechas_ok.dt.to_period("M").astype(str).unique())

    filas_validas = fechas_validas & origenes_validos
    datos_validos = pd.DataFrame({
      "estacion_id": origenes[filas_validas].astype("int64"),
      "fecha": fechas[filas_validas].dt.normalize()
    })
    conteos_diarios.update(
      datos_validos.groupby(["estacion_id", "fecha"]).size().to_dict()
    )

  if not ids_por_bloque:
    raise ValueError("El CSV no contiene viajes.")

  ids_ordenados = np.concatenate(ids_por_bloque)
  ids_ordenados.sort()
  duplicados = int(np.count_nonzero(ids_ordenados[1:] == ids_ordenados[:-1]))
  dias_esperados = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
  dias_faltantes = dias_esperados.difference(pd.DatetimeIndex(dias_registrados))

  if any([
    fechas_invalidas,
    ids_viaje_invalidos,
    origenes_invalidos,
    fechas_fuera_rango,
    duplicados,
    len(dias_faltantes)
  ]):
    raise ValueError("Se encontraron registros inválidos, duplicados o días ausentes.")

  df_diario = pd.DataFrame(
    [
      (estacion, fecha, cantidad)
      for (estacion, fecha), cantidad in conteos_diarios.items()
    ],
    columns=["estacion_id", "fecha", "viajes_iniciados"]
  ).sort_values(["estacion_id", "fecha"]).reset_index(drop=True)

  if int(df_diario["viajes_iniciados"].sum()) != total_viajes:
    raise ValueError("La agregación no conservó todos los viajes.")

  print(f"Viajes: {total_viajes:,} | Meses: {len(meses)} | Estaciones: {df_diario['estacion_id'].nunique()}")
  return df_diario, total_viajes, hasher.hexdigest()


def construir_panel_y_features(df_diario: pd.DataFrame) -> tuple[pd.DataFrame, int]:
  """Completa días intermedios y produce variables usando datos anteriores."""
  rangos = df_diario.groupby("estacion_id")["fecha"].agg(["min", "max"])
  panel_fechas = pd.concat(
    [
      pd.DataFrame({
        "estacion_id": estacion_id,
        "fecha": pd.date_range(row["min"], row["max"], freq="D")
      })
      for estacion_id, row in rangos.iterrows()
    ],
    ignore_index=True
  )
  df = panel_fechas.merge(
    df_diario,
    on=["estacion_id", "fecha"],
    how="left",
    validate="one_to_one"
  )
  df["sin_viajes_registrados"] = df["viajes_iniciados"].isna()
  df["viajes_iniciados"] = df["viajes_iniciados"].fillna(0).astype(int)
  dias_imputados = int(df["sin_viajes_registrados"].sum())
  df = df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)

  df["dia_semana"] = df["fecha"].dt.dayofweek
  df["mes"] = df["fecha"].dt.month
  df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
  grupo = df.groupby("estacion_id")["viajes_iniciados"]
  df["viajes_lag_1"] = grupo.shift(1)
  df["viajes_lag_7"] = grupo.shift(7)
  df["promedio_7d"] = grupo.transform(
    lambda serie: serie.shift(1).rolling(window=7, min_periods=7).mean()
  )
  df = df.dropna(
    subset=["viajes_lag_1", "viajes_lag_7", "promedio_7d"]
  ).copy()
  return df, dias_imputados


def commit_actual() -> tuple[str, str]:
  """Identifica el commit y si hay cambios locales todavía no guardados."""
  result = subprocess.run(
    ["git", "-C", str(PROJ_ROOT), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=False
  )
  git_commit = result.stdout.strip() if result.returncode == 0 else "sin-commit"
  estado = subprocess.run(
    ["git", "-C", str(PROJ_ROOT), "status", "--porcelain", "--untracked-files=normal"],
    capture_output=True,
    text=True,
    check=False
  )
  trabajo_limpio = estado.returncode == 0 and not estado.stdout.strip()
  return git_commit, str(trabajo_limpio).lower()


def construir_modelo(model: str, params: dict):
  """Instancia el estimador de P1 con los hiperparámetros dados."""
  if model == "random_forest":
    return RandomForestRegressor(**params)
  if model == "hist_gradient_boosting":
    return HistGradientBoostingRegressor(**params)
  if model == "ridge":
    preprocesamiento = ColumnTransformer([
      ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["estacion_id", "dia_semana", "mes"]),
      ("numericas", StandardScaler(), ["viajes_lag_1", "viajes_lag_7", "promedio_7d"])
    ], remainder="passthrough")
    return Pipeline([("preprocesamiento", preprocesamiento), ("ridge", Ridge(**params))])
  raise ValueError(f"Modelo no reconocido: {model}")


def evaluar(modelo, train: pd.DataFrame, val: pd.DataFrame) -> dict:
  """Entrena con train y calcula MAE y RMSE en validación."""
  modelo.fit(train[FEATURE_COLUMNS], train["viajes_iniciados"])
  pred = modelo.predict(val[FEATURE_COLUMNS])
  real = val["viajes_iniciados"]
  return {
    "val_mae": float(mean_absolute_error(real, pred)),
    "val_rmse": float(np.sqrt(mean_squared_error(real, pred)))
  }


def graficas_desempeno(real: pd.Series, pred: np.ndarray, titulo: str) -> dict:
  """Real contra predicho y residuales en validación."""
  real = real.to_numpy()
  residuales = real - pred
  tope = float(max(real.max(), pred.max()))

  fig_rp, ax = plt.subplots(figsize=(6, 6))
  ax.hexbin(real, pred, gridsize=60, bins="log", mincnt=1, cmap="Blues")
  ax.plot([0, tope], [0, tope], c="k", lw=0.8, ls="--", label="predicción perfecta")
  ax.set(title=f"{titulo}: real contra predicho (validación)", xlabel="viajes reales por estación-día",
         ylabel="viajes predichos", xlim=(0, tope), ylim=(0, tope))
  ax.legend(loc="upper left")
  fig_rp.tight_layout()

  fig_res, axes = plt.subplots(1, 2, figsize=(12, 4))
  axes[0].hexbin(pred, residuales, gridsize=60, bins="log", mincnt=1, cmap="Blues")
  axes[0].axhline(0, c="k", lw=0.8, ls="--")
  axes[0].set(title="Residuales contra predicho", xlabel="viajes predichos", ylabel="real - predicho")
  axes[1].hist(residuales, bins=80, color="#4C72B0")
  axes[1].axvline(0, c="k", lw=0.8, ls="--")
  axes[1].set(title="Distribución de residuales", xlabel="real - predicho", ylabel="estación-días")
  fig_res.suptitle(f"{titulo} (validación)")
  fig_res.tight_layout()
  return {"real_vs_predicho.png": fig_rp, "residuales.png": fig_res}


def buscar_hiperparametros(train, val, model: str, comunes: dict, etiquetas: dict) -> None:
  """Una corrida padre por búsqueda y una corrida hija por combinación del grid."""
  grid = SEARCH_GRIDS[model]
  resultados = []
  with search_run("P1", model, etiquetas):
    mlflow.log_params(comunes)
    mlflow.log_dict(grid, "grid.json")
    for combinacion in combinaciones(grid):
      params = DEFAULT_HYPERPARAMS[model] | combinacion
      with child_run("P1", model, params, etiquetas):
        mlflow.log_params(comunes)
        metricas = evaluar(construir_modelo(model, params), train, val)
        mlflow.log_metrics(metricas)
      logger.info(f"{combinacion}: MAE={metricas['val_mae']:.4f}")
      resultados.append({**combinacion, **metricas})

    mejor = min(resultados, key=lambda r: r["val_mae"])
    mlflow.log_metrics({"best_val_mae": mejor["val_mae"], "best_val_rmse": mejor["val_rmse"]})
    mlflow.log_dict(mejor, "mejor_combinacion.json")
    mlflow.log_table(pd.DataFrame(resultados), "resultados_busqueda.json")


def search(model: str) -> None:
  """Realiza búsqueda de hiperparámetros para P1 y registra en MLflow."""
  if model == "none":
    logger.info("No se especificó modelo, se usará random_forest por defecto")
    model = "random_forest"
  df_diario, _, hash_csv = leer_y_agregar_viajes(TRIPS_CSV)
  df, dias_imputados = construir_panel_y_features(df_diario)
  train, val, comunes, etiquetas = preparar(df, dias_imputados, hash_csv, model)
  setup_mlflow()
  buscar_hiperparametros(train, val, model, comunes, etiquetas)


def registrar_experimentos(
  df: pd.DataFrame,
  dias_imputados: int,
  hash_csv: str,
  model: str,
  hyperparams: dict
) -> None:
  """Entrena y registra P1 en MLflow."""

  # Preparamos las particiones y etiquetas
  train, val, comunes, etiquetas = preparar(df, dias_imputados, hash_csv, model)

  real = val["viajes_iniciados"]

  # Si es baseline, no entrenamos modelo, solo usamos el lag de 7 días
  if model == "baseline":
    modelo = None
    params = {"lag_dias": 7}
    pred = val["viajes_lag_7"].to_numpy()
    score = {"run_name": "P1 - Baseline semanal", "modelo": "baseline_semanal"}
    etiquetas["model_family"] = "baseline"
    etiquetas["stage_of_work"] = "baseline"

  else:
    # Procesamos los hiperparametros y modelos
    # Convertimos default 'none' a 'random_forest'
    if model == "none":
      model = "random_forest"

    params = DEFAULT_HYPERPARAMS[model] | convert_params_type(hyperparams, DEFAULT_HYPERPARAMS[model])
    modelo = construir_modelo(model, params)
    evaluar(modelo, train, val)  
    pred = modelo.predict(val[FEATURE_COLUMNS])
    score = {"run_name": "P1 - " + model, "modelo": model}
    etiquetas["stage_of_work"] = "candidate"
    etiquetas["model_family"] = model

  score["mae"] = float(mean_absolute_error(real, pred))
  score["rmse"] = float(np.sqrt(mean_squared_error(real, pred)))

  setup_mlflow()

  with mlflow.start_run(run_name=score["run_name"]) as run:
    mlflow.log_params({**comunes, **params})
    mlflow.set_tags({**etiquetas, "modelo": score["modelo"]})
    mlflow.log_metrics({
      "val_mae": score["mae"],
      "val_rmse": score["rmse"]
    })
    mlflow.log_dict({
      "modelo": score["modelo"],
      "mae_validacion": score["mae"],
      "rmse_validacion": score["rmse"],
      "nota": "Datos agregados por estación y día"
    }, "results/resumen_validacion.json")
    log_run_artifacts(
      run_config={"pregunta": "P1", "model": score["modelo"], "stage_of_work": etiquetas["stage_of_work"],
                  "hyperparams": params, "features": FEATURE_COLUMNS},
      train_df=train[FEATURE_COLUMNS + ["viajes_iniciados"]],
      train_source=TRIPS_CSV,
      target="viajes_iniciados",
      model=modelo,
      input_example=val[FEATURE_COLUMNS].head(5),
      figures=graficas_desempeno(real, pred, score["run_name"]),
    )
    print(
      f"{score['run_name']}: MAE={score['mae']:.4f}, "
      f"RMSE={score['rmse']:.4f}, run_id={run.info.run_id}"
    )



def main(model: str, hyperparams: dict) -> None:
  """Punto de entrada: datos → modelos → experimentos."""
  df_diario, _, hash_csv = leer_y_agregar_viajes(TRIPS_CSV)
  df_modelo, dias_imputados = construir_panel_y_features(df_diario)
  print(f"Observaciones con historial suficiente: {len(df_modelo):,}")
  print(f"Días imputados: {dias_imputados}")
  registrar_experimentos(df_modelo, dias_imputados, hash_csv, model, hyperparams)


if __name__ == "__main__":
  main(model="random_forest", hyperparams={})
