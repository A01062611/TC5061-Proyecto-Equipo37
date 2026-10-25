"""P1: entrenamiento y registro MLflow desde código del repositorio.

Ubicación: MiBici/modeling/train_p1.py
Ejecución desde la raíz: python -m MiBici.modeling.train_p1
Datos locales: data/interim/mibici_2024_2025.csv
"""

from collections import Counter
from itertools import product
from pathlib import Path
import hashlib
import inspect
import os
import time
import subprocess
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=r"IProgress not found.*")
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from MiBici.config import (
  MLFLOW_EXPERIMENT,
  MLFLOW_TRACKING_URI,
  PROJ_ROOT,
  SEED,
  TRIPS_CSV
)


ROOT = PROJ_ROOT
CSV_PATH = TRIPS_CSV
FEATURE_COLUMNS = [
  "estacion_id",
  "dia_semana",
  "mes",
  "es_fin_semana",
  "viajes_lag_1",
  "viajes_lag_7",
  "promedio_7d"
]


def leer_y_agregar_viajes(path: Path) -> tuple[pd.DataFrame, int, str]:
  """Valida los IDs y fechas, y cuenta viajes por estación y día."""
  if not path.is_file():
    raise FileNotFoundError(f"No se encontró el CSV: {path}. Ejecuta la ingesta primero.")

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
      fechas_validas & ~fechas.between("2024-01-01", "2025-12-31 23:59:59")
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
  dias_esperados = pd.date_range("2024-01-01", "2025-12-31", freq="D")
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
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=False
  )
  git_commit = result.stdout.strip() if result.returncode == 0 else "sin-commit"
  estado = subprocess.run(
    ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
    capture_output=True,
    text=True,
    check=False
  )
  trabajo_limpio = estado.returncode == 0 and not estado.stdout.strip()
  return git_commit, str(trabajo_limpio).lower()


def dividir_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Mantiene la partición de la entrega anterior sin utilizar datos futuros."""
  train = df[df["fecha"] < "2025-07-01"].copy()
  val = df[(df["fecha"] >= "2025-07-01") & (df["fecha"] < "2025-10-01")].copy()
  test = df[df["fecha"] >= "2025-10-01"].copy()

  if min(len(train), len(val), len(test)) == 0:
    raise ValueError("La partición temporal contiene algún conjunto vacío.")
  if not (train["fecha"].max() < val["fecha"].min() < test["fecha"].min()):
    raise ValueError("La partición temporal presenta fuga entre conjuntos.")
  return train, val, test


def calcular_metricas(real: pd.Series, predicho: np.ndarray) -> dict[str, float]:
  return {
    "mae": float(mean_absolute_error(real, predicho)),
    "rmse": float(np.sqrt(mean_squared_error(real, predicho)))
  }


def generar_graficas(real: pd.Series, predicho: np.ndarray, prefijo: str = "val") -> None:
  """Registra imágenes en MLflow sin registrar datos personales ni el CSV."""
  valores = pd.DataFrame({
    "real": np.asarray(real, dtype=float),
    "predicho": np.asarray(predicho, dtype=float)
  })
  muestra = valores.sample(n=min(len(valores), 2500), random_state=SEED)

  fig, ax = plt.subplots(figsize=(6, 5))
  ax.scatter(muestra["real"], muestra["predicho"], s=9, alpha=0.25)
  limite = max(float(muestra[["real", "predicho"]].max().max()), 1.0)
  ax.plot([0, limite], [0, limite], color="black", linestyle="--", linewidth=1)
  ax.set(title="P1: real vs. predicho", xlabel="Viajes reales", ylabel="Viajes predichos")
  fig.tight_layout()
  mlflow.log_figure(fig, f"plots/p1_real_vs_predicho_{prefijo}.png")
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(6, 4))
  residuales = muestra["real"] - muestra["predicho"]
  ax.scatter(muestra["predicho"], residuales, s=9, alpha=0.25)
  ax.axhline(0, color="black", linestyle="--", linewidth=1)
  ax.set(title="P1: residuales", xlabel="Viajes predichos", ylabel="Real - predicho")
  fig.tight_layout()
  mlflow.log_figure(fig, f"plots/p1_residuales_{prefijo}.png")
  plt.close(fig)


def configurar_mlflow() -> None:
  """Permite el servidor del equipo mediante MLFLOW_TRACKING_URI."""
  mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
  mlflow.set_experiment(MLFLOW_EXPERIMENT)


def registrar_contexto(
  modelo: str,
  etapa: str,
  parametros: dict,
  contexto: dict,
  etiquetas: dict
) -> None:
  """Metadatos y artefactos requeridos para cada corrida candidata."""
  mlflow.log_params({**contexto, **parametros})
  mlflow.set_tags({**etiquetas, "model_family": modelo, "stage_of_work": etapa})
  manifest_path = ROOT / "references" / "data_manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(f"No existe el manifiesto obligatorio: {manifest_path}")
  mlflow.log_artifact(str(manifest_path), artifact_path="data")
  mlflow.log_dict({
    "pregunta": "p1",
    "modelo": modelo,
    "etapa": etapa,
    "parametros": parametros,
    "columnas": FEATURE_COLUMNS,
    "semilla": SEED,
    "split_train_fin_exclusivo": "2025-07-01",
    "split_val_fin_exclusivo": "2025-10-01"
  }, "config/run_config.json")


def entrenar_candidato(
  nombre: str,
  modelo,
  parametros: dict,
  train: pd.DataFrame,
  val: pd.DataFrame,
  contexto: dict,
  etiquetas: dict,
  nested: bool,
  etapa: str
) -> dict:
  """Entrena, evalúa, firma y registra el modelo de una corrida candidata."""
  X_train = train[FEATURE_COLUMNS]
  y_train = train["viajes_iniciados"]
  X_val = val[FEATURE_COLUMNS]
  y_val = val["viajes_iniciados"]

  with mlflow.start_run(run_name=f"P1 - {nombre}", nested=nested) as run:
    registrar_contexto(nombre, etapa, parametros, contexto, etiquetas)
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    segundos = time.perf_counter() - inicio
    predicciones = modelo.predict(X_val)
    metricas = calcular_metricas(y_val, predicciones)
    errores = np.abs(np.asarray(y_val, dtype=float) - np.asarray(predicciones, dtype=float))
    fines = val["es_fin_semana"].to_numpy(dtype=bool)
    metricas_segmento = {
      "val_p90_error_absoluto": float(np.percentile(errores, 90)),
      "val_mae_laboral": float(errores[~fines].mean()) if (~fines).any() else 0.0,
      "val_mae_fin_semana": float(errores[fines].mean()) if fines.any() else 0.0
    }
    mlflow.log_metrics({
      "val_mae": metricas["mae"],
      "val_rmse": metricas["rmse"],
      "train_seconds": segundos,
      **metricas_segmento
    })
    generar_graficas(y_val, predicciones)

    ejemplo = X_train.iloc[:5].copy()
    firma = infer_signature(ejemplo, modelo.predict(ejemplo))
    argumentos_modelo = {
      "sk_model": modelo,
      "signature": firma,
      "input_example": ejemplo,
      "serialization_format": "skops"
    }
    parametro_ruta = (
      "name" if "name" in inspect.signature(mlflow.sklearn.log_model).parameters
      else "artifact_path"
    )
    argumentos_modelo[parametro_ruta] = "model"
    if nombre == "hist_gradient_boosting":
      argumentos_modelo["skops_trusted_types"] = [
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
      ]
    elif nombre == "random_forest":
      argumentos_modelo["skops_trusted_types"] = [
        "sklearn.tree._tree.Tree"
      ]
    info_modelo = mlflow.sklearn.log_model(**argumentos_modelo)
    print(
      f"{nombre}: MAE={metricas['mae']:.4f}, "
      f"RMSE={metricas['rmse']:.4f}, "
      f"run_id={run.info.run_id}"
    )
    return {
      "modelo": nombre,
      "estimador": modelo,
      "mae": metricas["mae"],
      "rmse": metricas["rmse"],
      "seconds": segundos,
      "run_id": run.info.run_id,
      "model_uri": info_modelo.model_uri,
      "params": parametros
    }


def registrar_experimentos(
  df: pd.DataFrame,
  dias_imputados: int,
  hash_csv: str,
  model: str = "all",
  n_estimators: int = 100,
  max_depth: int = 18,
  min_samples_leaf: int = 5,
  search: bool = True,
  n_jobs: int = 2
) -> dict:
  """Ejecuta candidatos y una búsqueda anidada de hiperparámetros de RF."""
  if model not in {"all", "rf", "hgb", "ridge"}:
    raise ValueError("--model debe ser all, rf, hgb o ridge")
  if n_estimators < 1 or max_depth < 2 or min_samples_leaf < 1:
    raise ValueError("Hiperparámetros inválidos para Random Forest")

  train, val, test = dividir_temporal(df)
  configurar_mlflow()
  git_commit, git_worktree_clean = commit_actual()
  contexto = {
    "horizonte_dias": 1,
    "train_inicio": str(train["fecha"].min().date()),
    "train_fin": str(train["fecha"].max().date()),
    "val_inicio": str(val["fecha"].min().date()),
    "val_fin": str(val["fecha"].max().date()),
    "filas_train": len(train),
    "filas_val": len(val),
    "dias_imputados": dias_imputados,
    "datos_sha256": hash_csv,
    "seed": SEED
  }
  etiquetas = {
    "question": "p1",
    "equipo": "37",
    "git_commit": git_commit,
    "git_worktree_clean": git_worktree_clean,
    "data_version": hash_csv
  }

  candidatos = []
  with mlflow.start_run(run_name="P1 - comparacion de candidatos") as padre:
    mlflow.set_tags({**etiquetas, "model_family": "multi", "stage_of_work": "comparison"})
    mlflow.log_params({"model": model, "search": search})

    with mlflow.start_run(run_name="P1 - Baseline semanal", nested=True):
      registrar_contexto("baseline_semanal", "baseline", {"lag_dias": 7}, contexto, etiquetas)
      pred_baseline = val["viajes_lag_7"].to_numpy()
      met_baseline = calcular_metricas(val["viajes_iniciados"], pred_baseline)
      mlflow.log_metrics({"val_mae": met_baseline["mae"], "val_rmse": met_baseline["rmse"]})
      generar_graficas(val["viajes_iniciados"], pred_baseline)
    print(f"Baseline P1: MAE={met_baseline['mae']:.4f}, RMSE={met_baseline['rmse']:.4f}")

    if model in {"all", "ridge"}:
      candidatos.append(entrenar_candidato(
        "ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        {"alpha": 10.0}, train, val, contexto, etiquetas, True, "candidate"
      ))

    if model in {"all", "hgb"}:
      candidatos.append(entrenar_candidato(
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(
          max_iter=120, max_depth=8, learning_rate=0.1, random_state=SEED
        ),
        {"max_iter": 120, "max_depth": 8, "learning_rate": 0.1},
        train, val, contexto, etiquetas, True, "candidate"
      ))

    if model in {"all", "rf"}:
      combinaciones = [(n_estimators, max_depth)]
      if search:
        arboles = sorted({max(10, int(n_estimators * 0.7)), n_estimators})
        profundidades = sorted({max(2, max_depth - 6), max_depth})
        combinaciones = list(product(arboles, profundidades))

      # Un padre por búsqueda y un hijo por combinación. RF sin búsqueda queda
      # como corrida candidata normal dentro de la comparación.
      if search:
        with mlflow.start_run(run_name="P1 - Busqueda RF", nested=True):
          mlflow.set_tags({**etiquetas, "model_family": "random_forest", "stage_of_work": "search"})
          mlflow.log_param("total_combinaciones", len(combinaciones))
          for arboles, profundidad in combinaciones:
            parametros = {
              "n_estimators": arboles,
              "max_depth": profundidad,
              "min_samples_leaf": min_samples_leaf,
              "n_jobs": n_jobs,
              "random_state": SEED
            }
            candidatos.append(entrenar_candidato(
              "random_forest",
              RandomForestRegressor(**parametros),
              parametros, train, val, contexto, etiquetas, True, "search"
            ))
      else:
        parametros = {
          "n_estimators": n_estimators,
          "max_depth": max_depth,
          "min_samples_leaf": min_samples_leaf,
          "n_jobs": n_jobs,
          "random_state": SEED
        }
        candidatos.append(entrenar_candidato(
          "random_forest", RandomForestRegressor(**parametros),
          parametros, train, val, contexto, etiquetas, True, "candidate"
        ))

    ganador = min(candidatos, key=lambda c: (c["mae"], c["rmse"]))
    mejora = 100 * (met_baseline["mae"] - ganador["mae"]) / met_baseline["mae"]
    mlflow.log_metrics({"best_val_mae": ganador["mae"], "improvement_pct": mejora})
    mlflow.set_tag("best_run_id", ganador["run_id"])

    # Reservamos test para el ganador, una vez elegida la configuración por val.
    pred_test = ganador["estimador"].predict(test[FEATURE_COLUMNS])
    met_test = calcular_metricas(test["viajes_iniciados"], pred_test)

    resumen = {
      "baseline": met_baseline,
      "candidatos": [
        {clave: valor for clave, valor in c.items() if clave != "estimador"}
        for c in candidatos
      ],
      "ganador": {
        "modelo": ganador["modelo"],
        "run_id": ganador["run_id"],
        "model_uri": ganador["model_uri"],
        "val_mae": ganador["mae"],
        "val_rmse": ganador["rmse"],
        "test_mae": met_test["mae"],
        "test_rmse": met_test["rmse"],
        "mejora_pct_vs_baseline": mejora
      }
    }
    mlflow.log_dict(resumen, "config/p1_resultados.json")
  # Reabrir una corrida solo después de cerrar la corrida padre de comparación.
  # MLflow no permite reanudarla con start_run(run_id=...) cuando otra está activa.
  with mlflow.start_run(run_id=ganador["run_id"]):
    mlflow.log_metrics({"test_mae": met_test["mae"], "test_rmse": met_test["rmse"]})
    mlflow.set_tags({"selected_best_p1": "true", "selection_metric": "val_mae"})
    generar_graficas(test["viajes_iniciados"], pred_test, prefijo="test")

  print("\n=== RESULTADO FINAL P1 ===")
  print(f"Ganador: {ganador['modelo']} | val MAE: {ganador['mae']:.4f}")
  print(f"Mejora contra baseline: {mejora:.2f}%")
  print(f"Test MAE: {met_test['mae']:.4f} | Test RMSE: {met_test['rmse']:.4f}")
  print(f"Run ID ganador para Edgar: {ganador['run_id']}")
  print(f"Model URI ganador: {ganador['model_uri']}")
  print(f"Run ID de comparacion: {padre.info.run_id}")
  return resumen


def run_p1(
  model: str = "all",
  n_estimators: int = 100,
  max_depth: int = 18,
  min_samples_leaf: int = 5,
  search: bool = True,
  n_jobs: int = 2
) -> dict:
  """API del módulo para que Alan pueda despachar question=p1 desde train.py."""
  df_diario, _, hash_csv = leer_y_agregar_viajes(CSV_PATH)
  df_modelo, dias_imputados = construir_panel_y_features(df_diario)
  print(f"Observaciones con historial suficiente: {len(df_modelo):,}")
  print(f"Días imputados: {dias_imputados}")
  return registrar_experimentos(
    df_modelo, dias_imputados, hash_csv, model=model,
    n_estimators=n_estimators, max_depth=max_depth,
    min_samples_leaf=min_samples_leaf, search=search, n_jobs=n_jobs
  )


def main(
  model: str = typer.Option("all", help="all, rf, hgb o ridge"),
  n_estimators: int = typer.Option(100, help="Árboles del Random Forest"),
  max_depth: int = typer.Option(18, help="Profundidad máxima del Random Forest"),
  min_samples_leaf: int = typer.Option(5, help="Muestras mínimas por hoja"),
  search: bool = typer.Option(True, "--search/--no-search", help="Búsqueda nested para RF"),
  n_jobs: int = typer.Option(2, help="Procesos de Random Forest")
) -> None:
  run_p1(
    model=model,
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    search=search,
    n_jobs=n_jobs
  )


if __name__ == "__main__":
  typer.run(main)