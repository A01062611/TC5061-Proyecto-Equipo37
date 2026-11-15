"""Registra en MLflow los resultados reproducibles del A/B P2."""

import argparse
import json
from pathlib import Path

import mlflow

from MiBici.modeling.helper import setup_mlflow


def registrar_resultados(directorio: Path) -> str:
    resultado_path = directorio / "resultado_ab.json"
    tabla_path = directorio / "comparacion_metricas.csv"
    politica_path = directorio / "politica_evaluacion.md"

    for archivo in (resultado_path, tabla_path, politica_path):
        if not archivo.is_file():
            raise FileNotFoundError(f"Falta el artefacto: {archivo}")

    resultado = json.loads(
        resultado_path.read_text(encoding="utf-8")
    )

    if resultado["tipo_evaluacion"] != "offline_pareada":
        raise ValueError("Tipo de evaluación inesperado")

    if resultado["n_registros"] != 502260:
        raise ValueError("Se requiere la evaluación completa")

    versiones = resultado["versiones"]
    bootstrap = resultado["bootstrap"]

    if bootstrap["n_bootstrap"] != 2000:
        raise ValueError("Se esperaban 2000 réplicas")

    setup_mlflow()

    with mlflow.start_run(
        run_name="P2 - A/B champion vs challenger"
    ) as run:

        mlflow.set_tags({
            "pregunta": "P2",
            "equipo": "37",
            "stage_of_work": "ab_testing",
            "model_family": "hist_gradient_boosting",
            "evaluation_type": "offline_pareada",
            "modelo_recomendado":
                resultado["decision"]["modelo_recomendado"],
        })

        mlflow.log_params({
            "modelo": resultado["modelo"],
            "champion_version":
                versiones["champion"]["version"],
            "challenger_version":
                versiones["challenger"]["version"],
            "champion_run_id":
                versiones["champion"]["run_id"],
            "challenger_run_id":
                versiones["challenger"]["run_id"],
            "tau_champion":
                versiones["champion"]["tau"],
            "tau_challenger":
                versiones["challenger"]["tau"],
            "n_registros": resultado["n_registros"],
            "n_estaciones": resultado["n_estaciones"],
            "n_bootstrap": bootstrap["n_bootstrap"],
            "seed": bootstrap["seed"],
        })

        metricas = {}

        for alias in ("champion", "challenger"):
            for nombre, valor in resultado["metricas"][alias].items():
                clave = nombre.lower().replace(" ", "_")
                metricas[f"{alias}_{clave}"] = float(valor)

        metricas.update({
            "delta_f2": bootstrap["delta_f2"],
            "ic95_inferior": bootstrap["ic95_inferior"],
            "ic95_superior": bootstrap["ic95_superior"],
            "mcnemar_estadistico":
                resultado["mcnemar"]["estadistico"],
            "mcnemar_p_valor":
                resultado["mcnemar"]["p_valor"],
        })

        mlflow.log_metrics(metricas)

        mlflow.log_artifacts(
            str(directorio),
            artifact_path="ab_test"
        )

        run_id = run.info.run_id

    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/ab_test")
    )
    args = parser.parse_args()

    run_id = registrar_resultados(args.input)

    print("=== REGISTRO A/B COMPLETADO ===")
    print("Run ID:", run_id)
    print("Experimento: mibici-equipo37")


if __name__ == "__main__":
    main()
