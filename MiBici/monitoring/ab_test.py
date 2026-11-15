"""Evaluación offline A/B de modelos P2 de MiBici."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, recall_score

from MiBici.config import MONITORING_DATA_DIR, SEED
from MiBici.modeling.helper import setup_mlflow
from MiBici.modeling.train_p2 import (
    f2_macro,
    metricas,
    pred_umbral,
)
from MiBici.monitoring.ab_statistics import (
    bootstrap_pareado,
    prueba_mcnemar,
)

MODEL_NAME = "mibici-equipo37-p2-riesgo"
CLASES = [0, 1, 2]


def cargar_version(alias):
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    run = client.get_run(version.run_id)

    if "tau" not in run.data.params:
        raise ValueError(f"{alias}: falta tau en MLflow")

    tau = float(run.data.params["tau"])
    modelo = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{alias}"
    )

    if not np.array_equal(modelo.classes_, CLASES):
        raise ValueError(f"{alias}: clases incompatibles")

    return modelo, version, tau


def evaluar(modelo, tau, datos, features):
    y = datos["clase"].to_numpy()
    proba = modelo.predict_proba(datos[features])
    pred = pred_umbral(proba, tau)

    res = metricas(y, proba, pred)
    res["F2 macro"] = f2_macro(y, pred)

    for clase, nombre in [(1, "vaciado"), (2, "llenado")]:
        res[f"Recall {nombre}"] = recall_score(
            y == clase,
            pred == clase,
            zero_division=0,
        )

    return res, pred, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--output",
        default="reports/ab_test",
    )
    args = parser.parse_args()

    setup_mlflow()

    datos = pd.read_parquet(
        MONITORING_DATA_DIR / "ab_eval.parquet"
    )

    if args.sample is not None:
        if not 1 <= args.sample <= len(datos):
            raise ValueError("Muestra inválida")
        datos = datos.sample(
            n=args.sample,
            random_state=SEED,
        )

    if not datos["clase"].isin(CLASES).all():
        raise ValueError("Clases inválidas")

    if datos[["estacion", "dia", "hora"]].duplicated().any():
        raise ValueError("Observaciones duplicadas")

    y = datos["clase"].to_numpy()
    resultados = {}
    predicciones = {}
    versiones = {}

    for alias in ["champion", "challenger"]:
        modelo, version, tau = cargar_version(alias)
        features = list(modelo.feature_names_in_)

        faltantes = set(features) - set(datos.columns)
        if faltantes:
            raise ValueError(
                f"{alias}: faltan {sorted(faltantes)}"
            )

        res, pred, _ = evaluar(
            modelo, tau, datos, features
        )

        resultados[alias] = {
            k: float(v) for k, v in res.items()
        }
        predicciones[alias] = pred
        versiones[alias] = {
            "version": str(version.version),
            "run_id": version.run_id,
            "tau": tau,
        }

        print(f"\n=== {alias.upper()} ===")
        print("Versión:", version.version)
        print("Tau:", tau)

        for metrica, valor in resultados[alias].items():
            print(f"{metrica}: {valor:.6f}")

        print("Matriz de confusión:")
        print(confusion_matrix(y, pred, labels=CLASES))

    diferencias = {
        k: resultados["challenger"][k]
        - resultados["champion"][k]
        for k in resultados["champion"]
    }

    print("\n=== DIFERENCIAS CHALLENGER - CHAMPION ===")
    for k, valor in diferencias.items():
        print(f"{k}: {valor:+.6f}")

    if args.sample is not None:
        print("\nModo diagnóstico: no se generan resultados finales.")
        return

    print("\n=== BOOTSTRAP PAREADO ===")
    bootstrap = bootstrap_pareado(
        y,
        predicciones["champion"],
        predicciones["challenger"],
        datos["estacion"].to_numpy(),
        n_bootstrap=args.bootstrap,
        seed=SEED,
    )
    print(json.dumps(bootstrap, indent=2))

    print("\n=== PRUEBA DE MCNEMAR ===")
    mcnemar = prueba_mcnemar(
        y,
        predicciones["champion"],
        predicciones["challenger"],
    )
    print(json.dumps(mcnemar, indent=2))

    # Política definida antes de evaluar.
    mejora_f2 = diferencias["F2 macro"] > 0
    ic_positivo = bootstrap["ic95_inferior"] > 0

    recall_protegido = all(
        diferencias[f"Recall {clase}"] >= -0.02
        for clase in ("vaciado", "llenado")
    )

    promover = bool(
        mejora_f2 and ic_positivo and recall_protegido
    )

    decision = {
        "mejora_f2": bool(mejora_f2),
        "ic95_positivo": bool(ic_positivo),
        "recall_protegido": bool(recall_protegido),
        "promover_challenger": promover,
        "modelo_recomendado": (
            "challenger" if promover else "champion"
        ),
    }

    salida = Path(args.output)
    salida.mkdir(parents=True, exist_ok=True)

    tabla = pd.DataFrame({
        "metrica": list(resultados["champion"]),
        "champion": list(resultados["champion"].values()),
        "challenger": list(resultados["challenger"].values()),
        "diferencia": list(diferencias.values()),
    })
    tabla.to_csv(salida / "comparacion_metricas.csv", index=False)

    reporte = {
        "modelo": MODEL_NAME,
        "tipo_evaluacion": "offline_pareada",
        "periodo": {
            "desde": str(datos["dia"].min()),
            "hasta": str(datos["dia"].max()),
        },
        "n_registros": int(len(datos)),
        "n_estaciones": int(datos["estacion"].nunique()),
        "versiones": versiones,
        "metricas": resultados,
        "diferencias": diferencias,
        "bootstrap": bootstrap,
        "mcnemar": mcnemar,
        "decision": decision,
    }

    with (salida / "resultado_ab.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print("\n=== DECISIÓN SEGÚN POLÍTICA ===")
    print(json.dumps(decision, indent=2))

    print("\nArchivos generados:")
    print(salida / "comparacion_metricas.csv")
    print(salida / "resultado_ab.json")


if __name__ == "__main__":
    main()
