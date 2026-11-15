"""Importancia por permutación del champion y su cruce con el drift.

Una variable con drift pesa distinto según cuánto la usa el modelo. Aquí se mide la caída de
F2 macro (la regla de predicción con τ de `train_p2`) al permutar cada feature en la partición
val del entrenamiento, que el champion no vio al ajustarse. Después se cruza con los
`drift_summary_*.json` de `MiBici.monitoring.drift`.

Salidas en `reports/evidently/`: `importancia_permutacion.csv` y `drift_vs_importancia.csv`.

Uso (desde la raíz, después de generar los resúmenes de drift):
    python -m MiBici.monitoring.importancia --tau 0.15
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from MiBici.features import OUT_DIR_P2, features_de_p2
from MiBici.modeling.train_p2 import f2_macro
from MiBici.monitoring.datasets import cargar_modelo

SEED = 42
SALIDA = Path("reports/evidently")

app = typer.Typer(add_completion=False)


def _f2(modelo, X: pd.DataFrame, y: np.ndarray, tau: float) -> float:
    proba = modelo.predict_proba(X)
    riesgo = proba[:, 1:].argmax(1) + 1
    pred = np.where(proba[:, 1:].max(1) >= tau, riesgo, 0)
    return f2_macro(y, pred)


def importancia_permutacion(modelo, X: pd.DataFrame, y: np.ndarray, tau: float, repeticiones: int) -> pd.DataFrame:
    """Caída de F2 macro al permutar cada columna (media y desviación de `repeticiones`)."""
    rng = np.random.default_rng(SEED)
    base = _f2(modelo, X, y, tau)
    filas = []
    for col in X.columns:
        caidas = []
        for _ in range(repeticiones):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            caidas.append(base - _f2(modelo, Xp, y, tau))
        filas.append({"variable": col, "caida_f2": float(np.mean(caidas)), "desv": float(np.std(caidas))})
        logger.info(f"{col:>16}: {filas[-1]['caida_f2']:+.4f}")
    df = pd.DataFrame(filas).sort_values("caida_f2", ascending=False).reset_index(drop=True)
    df["ranking"] = df.index + 1
    df.attrs["f2_base"] = base
    return df


def cruzar(imp: pd.DataFrame, salida: Path) -> pd.DataFrame:
    """Una fila por variable: importancia, y score y bandera de drift en cada subconjunto del actual."""
    tabla = imp.set_index("variable")[["ranking", "caida_f2"]]
    for s in ("todas", "existentes", "nuevas"):
        ruta = salida / f"drift_summary_{s}.json"
        if not ruta.exists():
            logger.warning(f"falta {ruta}; corre `python -m MiBici.monitoring.drift --estaciones {s}`")
            continue
        v = pd.DataFrame(json.loads(ruta.read_text())["variables"]).T
        tabla[f"score_{s}"] = v["score"].astype(float)
        tabla[f"drift_{s}"] = v["drift"].astype(bool)
    return tabla.reset_index()


@app.command()
def main(
    tau: float = typer.Option(..., help="τ del champion"),
    model_uri: str = typer.Option(None, help="Modelo; por defecto el empacado en models/"),
    muestra: int = typer.Option(100_000, help="Filas de validación (0 = todas)"),
    repeticiones: int = typer.Option(3, help="Permutaciones por variable"),
    salida: Path = typer.Option(SALIDA),
):
    modelo, _ = cargar_modelo(model_uri)
    tabla = pd.read_parquet(OUT_DIR_P2 / "tabla.parquet")
    val = tabla[tabla["particion"] == "val"]
    if 0 < muestra < len(val):
        val = val.sample(n=muestra, random_state=SEED)
    entrada = modelo.metadata.get_input_schema().input_names()
    X = val[entrada].astype({"hora": "int32"}).reset_index(drop=True)
    y = val["clase"].to_numpy()

    imp = importancia_permutacion(modelo.get_raw_model(), X, y, tau, repeticiones)
    salida.mkdir(parents=True, exist_ok=True)
    imp.to_csv(salida / "importancia_permutacion.csv", index=False)
    cruce = cruzar(imp, salida)
    cruce.to_csv(salida / "drift_vs_importancia.csv", index=False)
    logger.success(f"F2 macro base {imp.attrs['f2_base']:.4f} · {len(imp)} variables · {salida}")
    logger.info("\n" + cruce.round(3).to_string(index=False))


if __name__ == "__main__":
    app()
