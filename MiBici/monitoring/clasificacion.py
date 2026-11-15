"""¿Ya se degradó el champion? Desempeño de P2 en el periodo actual con las etiquetas reales.

Compara el champion en cuatro conjuntos con las mismas métricas de `train_p2`
(AP, Brier y F2 por clase de riesgo, F2 macro) más recall por clase:

- `validacion`: partición val del entrenamiento. Es la línea base justa; el champion no entrenó con ella.
- `referencia`: partición train. Solo informativa, el champion la vio al ajustarse (optimista).
- `actual_*`: julio–octubre de 2026 en total y separado por estaciones existentes y nuevas.

Además genera un reporte HTML de Evidently (`ClassificationPreset`) por subconjunto del actual,
con la validación como referencia.

Uso (desde la raíz, después de `python -m MiBici.monitoring.datasets`):
    python -m MiBici.monitoring.clasificacion --tau 0.15
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score, log_loss, recall_score
import typer
from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.presets import ClassificationPreset

from MiBici.config import MONITORING_DATA_DIR
from MiBici.features import CLASES_P2, OUT_DIR_P2, features_de_p2
from MiBici.modeling.train_p2 import f2_macro, metricas
from MiBici.monitoring.datasets import cargar_modelo, predecir

SEED = 42
SALIDA = Path("reports/evidently")
PROBAS = [f"proba_{c}" for c in CLASES_P2]

app = typer.Typer(add_completion=False)


def metricas_conjunto(df: pd.DataFrame) -> dict:
    """Métricas de un conjunto con `clase`, `pred` y `proba_*`. Calculadas con todas las filas."""
    y, pred, proba = df["clase"].to_numpy(), df["pred"].to_numpy(), df[PROBAS].to_numpy()
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "filas": len(df),
        "estaciones": int(df["estacion"].nunique()),
        "clase": {c: round(float((y == i).mean()), 4) for i, c in enumerate(CLASES_P2)},
        "pred": {c: round(float((pred == i).mean()), 4) for i, c in enumerate(CLASES_P2)},
        "accuracy": accuracy_score(y, pred),
        "F2 macro": f2_macro(y, pred),
        "recall": {c: float(r) for c, r in zip(CLASES_P2, rec)},
        "logloss": log_loss(y, proba, labels=[0, 1, 2]),
    } | {k: float(v) for k, v in metricas(y, proba, pred).items()}


def reporte_evidently(actual: pd.DataFrame, referencia: pd.DataFrame, ruta: Path, muestra: int) -> None:
    """`ClassificationPreset` actual vs referencia. Evidently necesita clase y pred como texto
    y las columnas de probabilidad con el nombre de la clase ("0", "1", "2")."""
    def preparar(d: pd.DataFrame) -> pd.DataFrame:
        d = d if len(d) <= muestra else d.sample(n=muestra, random_state=SEED)
        d = d[["clase", "pred", *PROBAS]].rename(columns={p: str(i) for i, p in enumerate(PROBAS)})
        return d.astype({"clase": str, "pred": str})

    definicion = DataDefinition(classification=[MulticlassClassification(
        target="clase", prediction_labels="pred", prediction_probas=["0", "1", "2"])])
    snapshot = Report([ClassificationPreset()]).run(
        Dataset.from_pandas(preparar(actual), data_definition=definicion),   # actual primero
        Dataset.from_pandas(preparar(referencia), data_definition=definicion),
    )
    snapshot.save_html(str(ruta))


@app.command()
def main(
    tau: float = typer.Option(..., help="τ del champion (la regla de predicción de train_p2)"),
    model_uri: str = typer.Option(None, help="Modelo; por defecto el empacado en models/"),
    muestra: int = typer.Option(100_000, help="Filas por conjunto en el HTML de Evidently"),
    monitoreo: Path = typer.Option(MONITORING_DATA_DIR, help="Carpeta con reference/current.parquet"),
    salida: Path = typer.Option(SALIDA),
):
    modelo, _ = cargar_modelo(model_uri)

    logger.info("Validación: prediciendo con el champion (línea base)...")
    tabla = pd.read_parquet(OUT_DIR_P2 / "tabla.parquet")
    val = tabla[tabla["particion"] == "val"].reset_index(drop=True)
    val = predecir(modelo, val[["estacion", "clase", *features_de_p2(tabla)]].copy(), tau)

    ref = pd.read_parquet(monitoreo / "reference.parquet")
    cur = pd.read_parquet(monitoreo / "current.parquet")
    conjuntos = {
        "validacion": val,
        "referencia_train": ref,
        "actual_todas": cur,
        "actual_existentes": cur[~cur["estacion_nueva"]],
        "actual_nuevas": cur[cur["estacion_nueva"]],
    }
    resumen = {"tau": tau, "metricas": {n: metricas_conjunto(d) for n, d in conjuntos.items()}}

    salida.mkdir(parents=True, exist_ok=True)
    (salida / "classification_summary.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    for n in ("todas", "existentes", "nuevas"):
        logger.info(f"HTML de Evidently: actual {n} vs validación")
        reporte_evidently(conjuntos[f"actual_{n}"], val, salida / f"classification_report_{n}.html", muestra)

    cols = ["F2 macro", "F2 vaciado", "F2 llenado", "AP vaciado", "AP llenado", "Brier vaciado", "Brier llenado"]
    logger.info("\n" + pd.DataFrame({n: {c: m[c] for c in cols} for n, m in resumen["metricas"].items()}).round(3).to_string())


if __name__ == "__main__":
    app()
