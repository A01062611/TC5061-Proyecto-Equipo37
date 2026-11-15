"""Política de Continuous Training de P2: reentrenar, promover al challenger

Lee lo que generan los demás pasos del monitoreo y aplica umbrales fijados antes de ver los
resultados:

- `reports/evidently/drift_vs_importancia.csv` (drift por variable y su importancia),
- `reports/evidently/classification_summary.json` (desempeño del champion con etiquetas reales),
- `reports/ab_test/resultado_ab.json` (comparación champion vs challenger).

Reentrenar. Se distinguen dos niveles:
- **Urgente**: el champion ya se degradó en la red con la que se entrenó (estaciones existentes).
  Se dispara si el F2 macro cae más de `caida_f2_max` contra la validación, si el recall de
  alguna clase de riesgo baja de `recall_min`, o si las variables con drift concentran más de
  `drift_ponderado_max` de la importancia del modelo.
- **Programado**: el modelo no está fallando, pero los datos ya no representan la red. Se dispara
  si las estaciones nuevas pesan al menos `estaciones_nuevas_min` de las filas y su drift
  ponderado supera `drift_ponderado_max`, o si los datos de entrenamiento tienen más de
  `antiguedad_max_meses` meses.

El drift se pondera por importancia (caída de F2 al permutar cada variable) y no por el número de
variables: el criterio de Evidently (más de 50% de variables con drift) cambia de No a Sí con una
sola variable de importancia mínima, así que solo se reporta como alerta.

Promover. Se repite la regla escrita antes del A/B (`reports/ab_test/politica_evaluacion.md`):
F2 macro mayor, límite inferior del IC 95% de la diferencia mayor que 0 y ninguna clase de
riesgo pierde más de `recall_caida_max` de recall.

Uso (desde la raíz):
    python -m MiBici.monitoring.ct_policy
Salida: `reports/monitoring/ct_decision.json`.
"""

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from MiBici.config import REPORTS_DIR, TRAIN_FIN
from MiBici.monitoring.datasets import ACTUAL_HASTA

app = typer.Typer(add_completion=False)

EVIDENTLY = REPORTS_DIR / "evidently"
AB = REPORTS_DIR / "ab_test" / "resultado_ab.json"
SALIDA = REPORTS_DIR / "monitoring" / "ct_decision.json"

UMBRALES = {
    "caida_f2_max": 0.02,          # F2 macro de validación − F2 macro actual (estaciones existentes)
    "recall_min": 0.80,            # recall mínimo de vaciado y de llenado
    "drift_ponderado_max": 0.30,   # fracción de la importancia en variables con drift
    "estaciones_nuevas_min": 0.20, # fracción de filas del periodo actual en estaciones nuevas
    "antiguedad_max_meses": 12,    # meses entre el fin del entrenamiento y el fin del periodo actual
    "recall_caida_max": 0.02,      # regla de promoción del A/B
}
CLASES_RIESGO = ("vaciado", "llenado")


def drift_ponderado(cruce: pd.DataFrame, subconjunto: str) -> float:
    """Fracción de la importancia total que está en variables con drift."""
    imp = cruce["caida_f2"].clip(lower=0)
    return float(imp[cruce[f"drift_{subconjunto}"].astype(bool)].sum() / imp.sum())


def meses_entre(inicio: str, fin: str) -> int:
    a, b = pd.Timestamp(inicio), pd.Timestamp(fin)
    return (b.year - a.year) * 12 + b.month - a.month


def decidir_reentrenamiento(cruce: pd.DataFrame, metricas: dict, u: dict = UMBRALES) -> dict:
    """Evalúa los disparadores. Cada uno devuelve su valor, su umbral y si se activó."""
    val, ex = metricas["validacion"], metricas["actual_existentes"]
    filas_nuevas = metricas["actual_nuevas"]["filas"] / metricas["actual_todas"]["filas"]
    caida = val["F2 macro"] - ex["F2 macro"]
    recall_min = min(ex["recall"][c] for c in CLASES_RIESGO)
    dp_ex, dp_nu = drift_ponderado(cruce, "existentes"), drift_ponderado(cruce, "nuevas")
    meses = meses_entre(TRAIN_FIN, ACTUAL_HASTA)

    urgentes = {
        "caida_f2": {"valor": caida, "umbral": u["caida_f2_max"], "activo": caida > u["caida_f2_max"]},
        "recall_min": {"valor": recall_min, "umbral": u["recall_min"], "activo": recall_min < u["recall_min"]},
        "drift_ponderado_existentes": {"valor": dp_ex, "umbral": u["drift_ponderado_max"],
                                       "activo": dp_ex > u["drift_ponderado_max"]},
    }
    programados = {
        "estaciones_nuevas": {
            "valor": {"filas": filas_nuevas, "drift_ponderado": dp_nu},
            "umbral": {"filas": u["estaciones_nuevas_min"], "drift_ponderado": u["drift_ponderado_max"]},
            "activo": filas_nuevas >= u["estaciones_nuevas_min"] and dp_nu > u["drift_ponderado_max"],
        },
        "antiguedad_meses": {"valor": meses, "umbral": u["antiguedad_max_meses"],
                             "activo": meses > u["antiguedad_max_meses"]},
    }
    urgente = any(d["activo"] for d in urgentes.values())
    programado = any(d["activo"] for d in programados.values())
    return {
        "reentrenar": urgente or programado,
        "urgencia": "urgente" if urgente else "programado" if programado else "ninguna",
        "disparadores_urgentes": urgentes,
        "disparadores_programados": programados,
    }


def decidir_promocion(ab: dict, u: dict = UMBRALES) -> dict:
    """La regla del A/B, recalculada con los números de `resultado_ab.json`."""
    dif, ic = ab["diferencias"], ab["bootstrap"]
    criterios = {
        "f2_mayor": dif["F2 macro"] > 0,
        "ic95_inferior_positivo": ab["bootstrap"]["ic95_inferior"] > 0,
        "recall_protegido": all(dif[f"Recall {c}"] >= -u["recall_caida_max"] for c in CLASES_RIESGO),
    }
    promover = all(criterios.values())
    if promover != ab["decision"]["promover_challenger"]:
        logger.warning("la decisión no coincide con la de resultado_ab.json; revisa los umbrales")
    return {
        "promover_challenger": promover,
        "modelo_en_produccion": "challenger" if promover else "champion",
        "criterios": criterios,
        "delta_f2_macro": dif["F2 macro"],
        "ic95": [ic["ic95_inferior"], ic["ic95_superior"]],
        "versiones": ab["versiones"],
    }


@app.command()
def main(evidently: Path = EVIDENTLY, ab_path: Path = AB, salida: Path = SALIDA):
    """Lee los resultados del monitoreo y del A/B y escribe la decisión de Continuous Training."""
    cruce = pd.read_csv(evidently / "drift_vs_importancia.csv")
    metricas = json.loads((evidently / "classification_summary.json").read_text())["metricas"]
    ab = json.loads(ab_path.read_text())
    alerta = json.loads((evidently / "drift_summary_existentes.json").read_text())["dataset"]

    decision = {
        "umbrales": UMBRALES,
        "reentrenamiento": decidir_reentrenamiento(cruce, metricas),
        "promocion": decidir_promocion(ab),
        "alerta_evidently_existentes": {**alerta, "drift_del_conjunto": alerta["proporcion"] > alerta["umbral_dataset"]},
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(decision, indent=2, ensure_ascii=False, default=float) + "\n")

    r, p = decision["reentrenamiento"], decision["promocion"]
    for grupo in ("disparadores_urgentes", "disparadores_programados"):
        for nombre, d in r[grupo].items():
            logger.info(f"{'ACTIVO ' if d['activo'] else 'inactivo'} · {nombre}: {d['valor']} (umbral {d['umbral']})")
    logger.success(f"Reentrenar: {'sí' if r['reentrenar'] else 'no'} ({r['urgencia']}) · "
                   f"Promover challenger: {'sí' if p['promover_challenger'] else 'no'} · "
                   f"En producción: {p['modelo_en_produccion']} → {salida}")


if __name__ == "__main__":
    app()
