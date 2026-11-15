"""Reporte de data drift de las features de P2 con Evidently (>= 0.7).

Compara un conjunto de referencia (con el que se entrenó el champion) contra uno
actual (periodo posterior). Genera un HTML completo y un JSON resumen por variable.

Uso (desde la raíz del repo):
    python -m MiBici.monitoring.drift                       # stand-in: train vs test de tabla.parquet
    python -m MiBici.monitoring.drift --reference data/monitoring/reference.parquet \
                                      --current data/monitoring/current.parquet --estaciones existentes
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from MiBici.features import features_de_p2

SEED = 42
TABLA_P2 = Path("data/processed/p2/tabla.parquet")
SALIDA = Path("reports/evidently")
CATEGORICAS = ["hora", "dia_semana", "fin_semana", "festivo", "riesgo_ayer"]

app = typer.Typer(add_completion=False)


def _cargar(reference: Path | None, current: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Lee los dos conjuntos. Sin rutas, usa train vs test de la tabla de P2 (stand-in)."""
    if reference and current:
        return pd.read_parquet(reference), pd.read_parquet(current), "reference vs current"
    tabla = pd.read_parquet(TABLA_P2)
    return (
        tabla[tabla["particion"] == "train"],
        tabla[tabla["particion"] == "test"],
        "stand-in: train vs test (2025)",
    )


def _filtrar_estaciones(cur: pd.DataFrame, subconjunto: str) -> pd.DataFrame:
    """Separa el actual por `estacion_nueva` (estaciones que no están en la referencia)."""
    if subconjunto == "todas":
        return cur
    if "estacion_nueva" not in cur:
        raise typer.BadParameter("el conjunto actual no tiene la columna estacion_nueva")
    return cur[cur["estacion_nueva"] == (subconjunto == "nuevas")]


def _muestrear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Misma semilla en ambos lados para que el reporte sea reproducible."""
    return df if n <= 0 or len(df) <= n else df.sample(n=n, random_state=SEED)


def _hubo_drift(metodo: str, score: float, umbral: float) -> bool:
    """Pruebas de p-valor marcan drift si score < umbral; las de distancia si score >= umbral."""
    if "p_value" in metodo.lower() or "p-value" in metodo.lower():
        return score < umbral
    return score >= umbral


def resumir(snapshot_dict: dict, n_ref: int, n_cur: int, descripcion: str) -> dict:
    """Convierte la salida de Evidently en un JSON plano: una fila por variable."""
    variables, global_ = {}, {}
    for m in snapshot_dict["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            global_ = {"columnas_con_drift": int(m["value"]["count"]),
                       "proporcion": float(m["value"]["share"]),
                       "umbral_dataset": cfg["drift_share"]}
        elif cfg["type"].endswith("ValueDrift"):
            score = float(m["value"])
            variables[cfg["column"]] = {
                "metodo": cfg["method"], "score": score, "umbral": cfg["threshold"],
                "drift": bool(_hubo_drift(cfg["method"], score, cfg["threshold"])),
            }
    return {"descripcion": descripcion, "seed": SEED, "filas_referencia": n_ref,
            "filas_actual": n_cur, "dataset": global_, "variables": variables}


@app.command()
def main(
    reference: Path = typer.Option(None, help="Parquet de referencia"),
    current: Path = typer.Option(None, help="Parquet actual"),
    muestra: int = typer.Option(10_000, help="Filas por conjunto (0 = todas)"),
    estaciones: str = typer.Option("todas", help="Filtro del actual: todas | existentes | nuevas"),
    salida: Path = typer.Option(SALIDA, help="Directorio de salida"),
):
    if estaciones not in ("todas", "existentes", "nuevas"):
        raise typer.BadParameter("estaciones debe ser todas, existentes o nuevas")
    ref, cur, descripcion = _cargar(reference, current)
    cur = _filtrar_estaciones(cur, estaciones)
    descripcion += f" · estaciones: {estaciones}"
    features = [c for c in features_de_p2(ref) if c in cur.columns]
    ref, cur = _muestrear(ref[features], muestra), _muestrear(cur[features], muestra)

    definicion = DataDefinition(
        numerical_columns=[c for c in features if c not in CATEGORICAS],
        categorical_columns=[c for c in CATEGORICAS if c in features],
    )
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(cur, data_definition=definicion),   # actual primero
        Dataset.from_pandas(ref, data_definition=definicion),
    )

    salida.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(salida / f"drift_report_{estaciones}.html"))
    resumen = resumir(snapshot.dict(), len(ref), len(cur), descripcion)
    (salida / f"drift_summary_{estaciones}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    con_drift = [v for v, d in resumen["variables"].items() if d["drift"]]
    typer.echo(f"{descripcion}: {len(con_drift)}/{len(resumen['variables'])} variables con drift {con_drift}")


if __name__ == "__main__":
    app()
