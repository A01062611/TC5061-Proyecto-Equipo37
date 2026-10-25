"""P1: entrenamiento y registro MLflow desde código del repositorio.

Ubicación: MiBici/modeling/train_p1.py
Ejecución desde la raíz: python -m MiBici.modeling.train_p1
Datos locales: data/raw/mibici_2024_2025.csv
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", message=r"IProgress not found.*")
import mlflow


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "mibici_2024_2025.csv"
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
    raise FileNotFoundError("Coloca mibici_2024_2025.csv en data/raw/.")

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


def registrar_experimentos(
  df: pd.DataFrame,
  dias_imputados: int,
  hash_csv: str
) -> None:
  """Entrena y registra dos runs (baseline y Random Forest) en MLflow."""
  train = df[df["fecha"] < "2025-07-01"].copy()
  val = df[
    (df["fecha"] >= "2025-07-01") &
    (df["fecha"] < "2025-10-01")
  ].copy()
  test = df[df["fecha"] >= "2025-10-01"].copy()
  if min(len(train), len(val), len(test)) == 0:
    raise ValueError("Algún conjunto temporal no contiene registros.")
  if not (train["fecha"].max() < val["fecha"].min() < test["fecha"].min()):
    raise ValueError("La partición temporal es inválida.")

  modelo = RandomForestRegressor(
    n_estimators=100,
    max_depth=18,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
  )
  modelo.fit(train[FEATURE_COLUMNS], train["viajes_iniciados"])
  real = val["viajes_iniciados"]
  pred_baseline = val["viajes_lag_7"].to_numpy()
  pred_rf = modelo.predict(val[FEATURE_COLUMNS])

  scores = [
    {
      "run_name": "P1 - Baseline semanal",
      "modelo": "baseline_semanal",
      "params": {"lag_dias": 7},
      "mae": float(mean_absolute_error(real, pred_baseline)),
      "rmse": float(np.sqrt(mean_squared_error(real, pred_baseline)))
    },
    {
      "run_name": "P1 - Random Forest",
      "modelo": "random_forest_regressor",
      "params": {
        "n_estimators": 100,
        "max_depth": 18,
        "min_samples_leaf": 5,
        "random_state": 42
      },
      "mae": float(mean_absolute_error(real, pred_rf)),
      "rmse": float(np.sqrt(mean_squared_error(real, pred_rf)))
    }
  ]

  tabla = pd.DataFrame({
    "Modelo": [s["run_name"] for s in scores],
    "MAE": [s["mae"] for s in scores],
    "RMSE": [s["rmse"] for s in scores]
  })
  fig, ax = plt.subplots(figsize=(8, 4))
  tabla.set_index("Modelo")[["MAE", "RMSE"]].T.plot(kind="bar", ax=ax, rot=0)
  ax.set_title("P1 — Comparación de modelos en validación")
  ax.set_ylabel("Error en viajes por estación-día")
  ax.set_xlabel("Métrica (menor es mejor)")
  ax.legend(title="Modelo")
  fig.tight_layout()

  tracking_db = ROOT / "mlflow.db"
  artifact_dir = ROOT / "mlruns" / "mibici-equipo37"
  artifact_dir.mkdir(parents=True, exist_ok=True)
  mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
  experiment_name = "mibici-equipo37"
  if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(
      experiment_name,
      artifact_location=artifact_dir.resolve().as_uri()
    )
  mlflow.set_experiment(experiment_name)

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
  etiquetas = {
    "equipo": "37",
    "pregunta": "P1",
    "git_commit": git_commit,
    "git_worktree_clean": git_worktree_clean,
    "version_datos": hash_csv,
    "conjunto_metricas": "validacion"
  }

  for score in scores:
    with mlflow.start_run(run_name=score["run_name"]) as run:
      mlflow.log_params({**comunes, **score["params"]})
      mlflow.set_tags({**etiquetas, "modelo": score["modelo"]})
      mlflow.log_metrics({
        "val_mae": score["mae"],
        "val_rmse": score["rmse"]
      })
      mlflow.log_figure(fig, "graficas/comparacion_p1.png")
      mlflow.log_dict({
        "modelo": score["modelo"],
        "mae_validacion": score["mae"],
        "rmse_validacion": score["rmse"],
        "nota": "Datos agregados por estación y día"
      }, "resumen_validacion.json")
      print(
        f"{score['run_name']}: MAE={score['mae']:.4f}, "
        f"RMSE={score['rmse']:.4f}, run_id={run.info.run_id}"
      )
  plt.close(fig)


def main() -> None:
  """Punto de entrada: datos → modelos → experimentos."""
  df_diario, _, hash_csv = leer_y_agregar_viajes(CSV_PATH)
  df_modelo, dias_imputados = construir_panel_y_features(df_diario)
  print(f"Observaciones con historial suficiente: {len(df_modelo):,}")
  print(f"Días imputados: {dias_imputados}")
  registrar_experimentos(df_modelo, dias_imputados, hash_csv)


if __name__ == "__main__":
  main()
