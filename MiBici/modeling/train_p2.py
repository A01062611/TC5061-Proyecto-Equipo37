"""P2: baselines y modelo candidato de riesgo de vaciado/llenado, con runs en MLflow.

Genera tres runs en el experimento del equipo:
  - P2 - Baseline persistencia      (umbrales sobre la ocupación estimada actual)
  - P2 - Baseline perfil histórico  (frecuencia de clase por estación × día de la semana × hora)
  - P2 - HistGradientBoosting       (multiclase con pesos por clase)

Todo se evalúa en validación; la partición de prueba no se usa.

Ejecución desde la raíz:
  python -m MiBici.features p2          # construye data/processed/p2/tabla.parquet
  python -m MiBici.modeling.train_p2  # entrena, evalúa y registra en MLflow

Los hiperparámetros, la semilla y k se pasan como opciones, sin regenerar la tabla:
  python -m MiBici.modeling.train_p2 --max-iter 500 --learning-rate 0.05 --k-visitas 40
  python -m MiBici.modeling.train_p2 --help
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess

import joblib
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
)
import typer

from MiBici.config import (
    FIGURES_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROJ_ROOT,
    TRIPS_CSV,
)
from MiBici.features import CLASES_P2 as CLASES
from MiBici.features import OUT_DIR_P2, features_de_p2

app = typer.Typer()

FIG_DIR = FIGURES_DIR / "p2"
GRID_PERSISTENCIA = [(a, b) for a in np.arange(0.05, 0.45, 0.05) for b in np.arange(0.55, 0.96, 0.05)]
TAUS = np.arange(0.05, 0.6, 0.025)
# Valores por defecto del modelo y la evaluación; se pueden cambiar desde la línea de comandos
HGB_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": False,
}
SEED = 37
K_VISITAS = 20


# --------------------------------------------------------------------------- métricas

def metricas(y, proba, pred) -> dict:
    """AP y Brier por clase de riesgo, y F2 de vaciado y llenado."""
    out = {}
    for c in (1, 2):
        out[f"AP {CLASES[c]}"] = average_precision_score(y == c, proba[:, c])
        out[f"Brier {CLASES[c]}"] = brier_score_loss(y == c, proba[:, c])
    out["F2 vaciado"], out["F2 llenado"] = fbeta_score(y, pred, beta=2, labels=[1, 2], average=None)
    return out


def top_k(df: pd.DataFrame, score, k: int) -> dict:
    """Cada hora se visitan las k estaciones con mayor puntaje: recall y precisión por hora."""
    d = pd.DataFrame({"dia": df["dia"].to_numpy(), "hora": df["hora"].to_numpy(), "score": score, "riesgo": df["clase"].to_numpy() > 0})
    d["top"] = d.groupby(["dia", "hora"])["score"].rank(method="first", ascending=False) <= k
    d["cubiertos"] = d["top"] & d["riesgo"]
    g = d.groupby(["dia", "hora"])[["riesgo", "cubiertos"]].sum()
    g = g[g["riesgo"] > 0]
    return {
        f"recall@{k}": (g["cubiertos"] / g["riesgo"]).mean(),
        f"precision@{k}": (g["cubiertos"] / k).mean(),
        "eventos de riesgo cubiertos": int(g["cubiertos"].sum()),
        "eventos de riesgo totales": int(g["riesgo"].sum()),
    }


def techo_recall(df: pd.DataFrame, k: int) -> float:
    """Recall@k de un ranking perfecto: promedio por hora de min(riesgos, k) / riesgos."""
    r = (df["clase"] > 0).groupby([df["dia"], df["hora"]]).sum()
    r = r[r > 0]
    return float((np.minimum(r, k) / r).mean())


def f2_macro(y, pred) -> float:
    return fbeta_score(y, pred, beta=2, labels=[1, 2], average="macro")


# --------------------------------------------------------------------------- modelos

def pred_persistencia(oc, a, b):
    return np.select([oc <= a, oc >= b], [1, 2], 0)


def pred_umbral(proba, tau):
    """La clase de riesgo más probable, si su probabilidad llega a τ; si no, normal."""
    riesgo = proba[:, 1:].argmax(1) + 1
    return np.where(proba[:, 1:].max(1) >= tau, riesgo, 0)


def ajustar_tau(y, proba) -> float:
    return round(float(TAUS[np.argmax([f2_macro(y, pred_umbral(proba, t)) for t in TAUS])]), 3)


def baseline_persistencia(val: pd.DataFrame, y_val, k: int) -> dict:
    oc = val["ocup_est"].clip(0, 1).to_numpy()
    f2 = [f2_macro(y_val, pred_persistencia(oc, a, b)) for a, b in GRID_PERSISTENCIA]
    a, b = GRID_PERSISTENCIA[int(np.argmax(f2))]
    proba = np.column_stack([np.zeros_like(oc), 1 - oc, oc])  # puntajes, no probabilidades
    res = metricas(y_val, proba, pred_persistencia(oc, a, b)) | top_k(val, np.maximum(1 - oc, oc), k)
    return {"nombre": "P2 - Baseline persistencia", "modelo": "persistencia", "proba": proba, "res": res,
            "params": {"umbral_a": round(float(a), 2), "umbral_b": round(float(b), 2)}}


def baseline_perfil(train: pd.DataFrame, val: pd.DataFrame, y_train, y_val, k: int) -> dict:
    claves = ["estacion", "dia_semana", "hora"]
    perfil = pd.get_dummies(train["clase"]).astype(float).groupby([train[c] for c in claves]).mean()
    perfil.columns = [0, 1, 2]
    global_ = np.bincount(y_train, minlength=3) / len(y_train)
    proba = val[claves].join(perfil, on=claves)[[0, 1, 2]].fillna(dict(enumerate(global_))).to_numpy()
    tau = ajustar_tau(y_val, proba)
    res = metricas(y_val, proba, pred_umbral(proba, tau)) | top_k(val, proba[:, 1] + proba[:, 2], k)
    return {"nombre": "P2 - Baseline perfil histórico", "modelo": "perfil_historico", "proba": proba, "res": res,
            "params": {"tau": tau, "claves_perfil": ",".join(claves)}}


def entrenar_hgb(train: pd.DataFrame, val: pd.DataFrame, y_train, y_val, features: list[str], k: int,
                 seed: int, hgb_params: dict = HGB_PARAMS) -> dict:
    modelo = HistGradientBoostingClassifier(
        **hgb_params, categorical_features=[features.index("estado_est")], random_state=seed
    )
    modelo.fit(train[features], y_train)
    proba = modelo.predict_proba(val[features])
    tau = ajustar_tau(y_val, proba)
    res = metricas(y_val, proba, pred_umbral(proba, tau)) | top_k(val, proba[:, 1] + proba[:, 2], k)
    return {"nombre": "P2 - HistGradientBoosting", "modelo": "hist_gradient_boosting", "proba": proba, "res": res,
            "params": {"tau": tau} | {f"hgb_{p}": v for p, v in hgb_params.items()}, "estimador": modelo}


# --------------------------------------------------------------------------- gráficas

def plot_distribucion_clases(tabla: pd.DataFrame):
    """% de cada clase en train y val (prueba no se mira) y % en riesgo por hora en train."""
    tv = tabla[tabla["particion"].isin(["train", "val"])]
    dist = pd.crosstab(tv["particion"].astype(str), tv["clase"].map(dict(enumerate(CLASES))), normalize="index")
    dist = dist.reindex(index=["train", "val"], columns=CLASES)
    tr = tv[tv["particion"] == "train"]
    por_hora = pd.crosstab(tr["hora"], tr["clase"], normalize="index")
    orden = [h for h in [*range(5, 24), 0] if h in por_hora.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    (dist * 100).plot(kind="bar", stacked=True, ax=axes[0], color=["#B0B0B0", "#DD8452", "#4C72B0"], rot=0)
    axes[0].set(title="Distribución de clases por partición", ylabel="% de filas", xlabel="")
    (por_hora.reindex(orden)[[1, 2]] * 100).set_axis(["vaciado", "llenado"], axis=1).reset_index(drop=True).plot(
        ax=axes[1], marker="o", color=["#DD8452", "#4C72B0"])
    axes[1].set(title="% de filas en riesgo por hora (train)", xlabel="hora", ylabel="%", xticks=range(len(orden)))
    axes[1].set_xticklabels(orden)
    fig.tight_layout()
    return fig


def plot_curvas_pr(y_val, modelos: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, c in zip(axes, (1, 2)):
        for m in modelos:
            p, r, _ = precision_recall_curve(y_val == c, m["proba"][:, c])
            ax.plot(r, p, label=f"{m['nombre'].removeprefix('P2 - ')} (AP={m['res'][f'AP {CLASES[c]}']:.3f})")
        ax.axhline((y_val == c).mean(), ls="--", c="grey", lw=0.8, label="prevalencia")
        ax.set(title=f"Curva PR · {CLASES[c]} (validación)", xlabel="recall", ylabel="precisión", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- MLflow

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def git_commit() -> tuple[str, str]:
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False).stdout.strip()

    return run("rev-parse", "HEAD") or "desconocido", str(run("status", "--porcelain") == "").lower()


def nombre_metrica(nombre: str) -> str:
    """MLflow no acepta "@" en los nombres: "recall@20" → "val_recall_at_20"."""
    return "val_" + re.sub(r"[^a-z0-9]+", "_", nombre.lower().replace("@", "_at_")).strip("_")


def registrar_runs(modelos: list[dict], comunes: dict, tags: dict, metricas_comunes: dict, artefactos: list[Path]):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for m in modelos:
        with mlflow.start_run(run_name=m["nombre"]) as run:
            mlflow.log_params(comunes | m["params"])
            mlflow.set_tags(tags | {"modelo": m["modelo"]})
            mlflow.log_metrics({nombre_metrica(k): float(v) for k, v in m["res"].items()} | metricas_comunes)
            for archivo in artefactos:
                mlflow.log_artifact(str(archivo), artifact_path="artefactos")
            logger.info(f"{m['nombre']}: AP vaciado={m['res']['AP vaciado']:.3f}, "
                        f"AP llenado={m['res']['AP llenado']:.3f}, run_id={run.info.run_id}")


# --------------------------------------------------------------------------- punto de entrada

@app.command()
def main(
    tabla_path: Path = OUT_DIR_P2 / "tabla.parquet",
    info_path: Path = OUT_DIR_P2 / "info.json",
    s0_path: Path = OUT_DIR_P2 / "s0_por_estacion.csv",
    trips_path: Path = TRIPS_CSV,
    max_iter: int = HGB_PARAMS["max_iter"],
    learning_rate: float = HGB_PARAMS["learning_rate"],
    max_leaf_nodes: int = HGB_PARAMS["max_leaf_nodes"],
    min_samples_leaf: int = HGB_PARAMS["min_samples_leaf"],
    l2_regularization: float = HGB_PARAMS["l2_regularization"],
    class_weight: str = typer.Option(HGB_PARAMS["class_weight"], help='"balanced" o "none"'),
    seed: int = SEED,
    k_visitas: int = typer.Option(K_VISITAS, help="Estaciones que se pueden visitar por hora (top-k)"),
    registrar: bool = typer.Option(True, help="Registrar los runs en MLflow"),
):
    info = json.loads(info_path.read_text())
    params = info["params"]
    k = k_visitas
    hgb_params = HGB_PARAMS | {
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "class_weight": None if class_weight.lower() == "none" else class_weight,
    }

    tabla = pd.read_parquet(tabla_path)
    features = features_de_p2(tabla)
    train = tabla[tabla["particion"] == "train"]
    val = tabla[tabla["particion"] == "val"]
    y_train, y_val = train["clase"].to_numpy(), val["clase"].to_numpy()
    logger.info(f"train: {len(train):,} filas · val: {len(val):,} filas · {len(features)} features")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_distribucion_clases(tabla)
    fig.savefig(FIG_DIR / "distribucion_clases.png", dpi=150)
    plt.close(fig)
    del tabla

    logger.info("Baselines...")
    modelos = [baseline_persistencia(val, y_val, k), baseline_perfil(train, val, y_train, y_val, k)]
    logger.info("Gradient boosting...")
    modelos.append(entrenar_hgb(train, val, y_train, y_val, features, k, seed, hgb_params))

    fig = plot_curvas_pr(y_val, modelos)
    fig.savefig(FIG_DIR / "curvas_pr_val.png", dpi=150)
    plt.close(fig)

    techo = techo_recall(val, k)
    resultados = pd.DataFrame([{"modelo": m["nombre"]} | m["res"] for m in modelos]).set_index("modelo")
    resultados[f"% del techo recall@{k}"] = resultados[f"recall@{k}"] / techo
    resultados.to_csv(FIG_DIR / "resultados_val.csv")
    logger.info(f"Resultados en validación (techo de recall@{k} = {techo:.3f}):\n{resultados.round(3).to_string()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelos[-1]["estimador"], MODELS_DIR / "p2_hgb.joblib")

    if not registrar:
        return
    datos_sha = sha256(trips_path)
    commit, clean = git_commit()
    comunes = params | {
        "k_visitas": k,
        "seed": seed,
        "datos_sha256": datos_sha,
        "estaciones_version": "nomenclatura_2026_09",
        "n_filas_train": len(train),
        "n_filas_val": len(val),
        "n_features": len(features),
    }
    tags = {
        "equipo": "37",
        "pregunta": "P2",
        "git_commit": commit,
        "git_worktree_clean": clean,
        "version_datos": datos_sha,
    }
    metricas_comunes = {f"val_techo_recall_at_{k}": techo} | {n: float(v) for n, v in info["proxy"].items()}
    artefactos = [FIG_DIR / "distribucion_clases.png", FIG_DIR / "curvas_pr_val.png",
                  FIG_DIR / "resultados_val.csv", s0_path, info_path]
    registrar_runs(modelos, comunes, tags, metricas_comunes, artefactos)
    logger.success("Runs de P2 registrados en MLflow.")


if __name__ == "__main__":
    app()
