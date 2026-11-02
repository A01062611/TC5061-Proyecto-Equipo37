"""Validación reproducible de drift de las features de P2 con DeepChecks.

Ejecutar desde la raíz del repositorio: python scripts/validate_drift.py
No entrena modelos ni modifica la tabla de entrada.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import FeatureDrift
from MiBici.features import features_de_p2

ENTRADA = Path("data/processed/p2/tabla.parquet")
SALIDA = Path("reports/deepchecks")
SEMILLA = 42
MUESTRA = 10_000
PARTICIONES = ("train", "val", "test")
CATEGORICAS = ["hora", "dia_semana", "fin_semana", "festivo", "riesgo_ayer"]


def _json_compatible(obj):
    """Convertir valores agregados de DeepChecks, nunca filas individuales."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return {str(k): _json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): _json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_compatible(v) for v in obj]
    if pd.isna(obj) if np.isscalar(obj) else False:
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main():
    if not ENTRADA.is_file():
        raise FileNotFoundError(f"Falta {ENTRADA}. Ejecuta python -m MiBici.features p2")

    tabla = pd.read_parquet(ENTRADA)
    if "particion" not in tabla:
        raise ValueError("Falta columna 'particion' en tabla.parquet")
    features = features_de_p2(tabla)
    if len(features) != 23:
        raise ValueError(f"Se esperaban 23 features de P2; se encontraron {len(features)}")
    if not all(pd.api.types.is_numeric_dtype(tabla[c]) for c in features):
        raise TypeError("Las features de P2 deben ser numéricas; revisar el esquema")

    if not set(CATEGORICAS).issubset(features):
        raise ValueError("Faltan features categóricas esperadas de P2")

    muestras = {}
    estadisticas = {}
    for p in PARTICIONES:
        subset = tabla.loc[tabla["particion"] == p, features]
        if len(subset) < MUESTRA:
            raise ValueError(f"La partición {p} contiene menos de {MUESTRA} filas")
        estadisticas[p] = {
            "registros": int(len(subset)),
            "faltantes_por_feature": {c: int(v) for c, v in subset.isna().sum().items()},
        }
        muestras[p] = subset.sample(n=MUESTRA, random_state=SEMILLA).reset_index(drop=True)
        print(f"{p}: {len(subset):,} filas; muestra {MUESTRA:,}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    comparaciones = {}
    for destino in ("val", "test"):
        suite = Suite(f"MiBici P2 - drift train vs {destino}", FeatureDrift(n_top_columns=len(features), ignore_na=True, random_state=SEMILLA))
        resultado = suite.run(
            Dataset(muestras["train"], features=features, cat_features=CATEGORICAS),
            Dataset(muestras[destino], features=features, cat_features=CATEGORICAS),
        )
        ruta_html = SALIDA / f"drift_p2_train_vs_{destino}.html"
        if ruta_html.exists():
            ruta_html.unlink()  # evita sufijos (1), (2) de DeepChecks
        resultado.save_as_html(str(ruta_html), as_widget=False)
        checks = []
        for r in resultado.results:
            checks.append({
                "nombre": str(getattr(r, "header", type(r).__name__)),
                "tipo": type(r).__name__,
                "resultado_agregado": _json_compatible(getattr(r, "value", None)),
                "excepcion": str(getattr(r, "exception", "")) if hasattr(r, "exception") else None,
            })
        comparaciones[f"train_vs_{destino}"] = {
            "html": str(ruta_html), "chequeos": checks,
        }
        print(f"Reporte generado: {ruta_html}")

    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "fuente": str(ENTRADA),
        "metodo": "FeatureDrift de DeepChecks 0.19.1",
        "semilla": SEMILLA,
        "tamano_muestra_por_particion": MUESTRA,
        "features": features,
        "categoricas": CATEGORICAS,
        "nota": "Drift calculado sobre muestras, no sobre todas las filas. Faltantes preservados; no se imputaron datos. FeatureDrift usa ignore_na=True; las tasas de faltantes se documentan por separado.",
        "particiones": estadisticas,
        "comparaciones": comparaciones,
    }
    ruta_json = SALIDA / "drift_p2_resumen.json"
    ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Resumen generado: {ruta_json}")


if __name__ == "__main__":
    main()
