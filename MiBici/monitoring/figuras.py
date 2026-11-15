"""Figuras PNG del monitoreo para el PDF, a partir de los JSON/CSV de `reports/evidently/`.

Uso (desde la raíz, después de `drift`, `clasificacion` e `importancia`):
    python -m MiBici.monitoring.figuras
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

SALIDA = Path("reports/evidently")
FIGURAS = SALIDA / "figuras"
UMBRAL = 0.1
COLORES = {"existentes": "#1f77b4", "nuevas": "#d95f02", "todas": "#555555"}
DPI = 200

app = typer.Typer(add_completion=False)


def _guardar(fig, ruta: Path) -> None:
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    logger.info(f"→ {ruta}")


def fig_drift_por_variable(cruce: pd.DataFrame, ruta: Path) -> None:
    """Score de drift por variable en estaciones existentes y nuevas, con el umbral de Evidently."""
    d = cruce.sort_values("score_existentes")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(y - 0.2, d["score_existentes"], 0.4, color=COLORES["existentes"], label="Estaciones existentes")
    ax.barh(y + 0.2, d["score_nuevas"], 0.4, color=COLORES["nuevas"], label="Estaciones nuevas")
    ax.axvline(UMBRAL, color="k", ls="--", lw=1, label=f"Umbral ({UMBRAL})")
    ax.set_yticks(y, d["variable"])
    ax.set_xlabel("Score de drift (distancia de Wasserstein normada / Jensen-Shannon)")
    ax.legend(loc="lower right")
    _guardar(fig, ruta)


def fig_drift_vs_importancia(cruce: pd.DataFrame, ruta: Path) -> None:
    """Dispersión importancia (caída de F2) vs drift. Arriba a la derecha = drift que sí importa."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, s in zip(axs, ("existentes", "nuevas")):
        ax.scatter(cruce["caida_f2"].clip(lower=0), cruce[f"score_{s}"], c=COLORES[s], s=28)
        ax.axhline(UMBRAL, color="k", ls="--", lw=1)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim(left=0, right=0.8)
        ax.set_xlabel("Importancia: caída de F2 macro al permutar (escala symlog)")
        ax.set_title(f"Estaciones {s}")
        destacadas = cruce[(cruce[f"drift_{s}"] & (cruce["caida_f2"] >= 0.015)) | (cruce["caida_f2"] > 0.1)]
        for i, (_, r) in enumerate(destacadas.sort_values(f"score_{s}").iterrows()):
            ax.annotate(r["variable"], (r["caida_f2"], r[f"score_{s}"]), fontsize=8,
                        xytext=(-6, 6 + 7 * (i % 2)), textcoords="offset points", ha="right")
    axs[0].set_ylabel("Score de drift")
    _guardar(fig, ruta)


def fig_clases(metricas: dict, ruta: Path) -> None:
    """Distribución real de `clase` por conjunto (prior drift)."""
    nombres = {"referencia_train": "Referencia\n(train)", "validacion": "Validación",
               "actual_existentes": "Actual\nexistentes", "actual_nuevas": "Actual\nnuevas"}
    clases = ["normal", "vaciado", "llenado"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, subset in zip(axs, (clases[:1], clases[1:])):
        x = np.arange(len(nombres))
        for i, c in enumerate(subset):
            vals = [metricas[k]["clase"][c] * 100 for k in nombres]
            b = ax.bar(x + (i - (len(subset) - 1) / 2) * 0.38, vals, 0.38, label=c)
            ax.bar_label(b, fmt="%.1f", fontsize=8)
        ax.set_xticks(x, nombres.values())
        ax.set_ylabel("% de filas")
        ax.legend()
    axs[0].set_title("Clase normal")
    axs[1].set_title("Clases de riesgo")
    _guardar(fig, ruta)


def fig_desempeno(metricas: dict, ruta: Path) -> None:
    """Métricas del champion en validación vs periodo actual."""
    nombres = {"validacion": "Validación", "actual_existentes": "Actual existentes", "actual_nuevas": "Actual nuevas"}
    cols = ["F2 macro", "AP vaciado", "AP llenado"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cols))
    for i, (k, n) in enumerate(nombres.items()):
        b = ax.bar(x + (i - 1) * 0.27, [metricas[k][c] for c in cols], 0.27, label=n)
        ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, cols)
    ax.set_ylim(0.6, 0.85)
    ax.set_ylabel("Métrica")
    ax.legend(loc="upper right")
    _guardar(fig, ruta)


@app.command()
def main(salida: Path = typer.Option(SALIDA), figuras: Path = typer.Option(FIGURAS)):
    figuras.mkdir(parents=True, exist_ok=True)
    cruce = pd.read_csv(salida / "drift_vs_importancia.csv")
    metricas = json.loads((salida / "classification_summary.json").read_text())["metricas"]
    fig_drift_por_variable(cruce, figuras / "drift_por_variable.png")
    fig_drift_vs_importancia(cruce, figuras / "drift_vs_importancia.png")
    fig_clases(metricas, figuras / "distribucion_clase.png")
    fig_desempeno(metricas, figuras / "desempeno_champion.png")


if __name__ == "__main__":
    app()
