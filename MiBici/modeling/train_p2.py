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
import sys

import joblib
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from MiBici.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROJ_ROOT,
    TRIPS_CSV,
    SEED
)
from MiBici.modeling.helper import convert_params_type, combinaciones, search_run, child_run, setup_mlflow, log_run_artifacts
from MiBici.features import CLASES_P2 as CLASES
from MiBici.features import OUT_DIR_P2, features_de_p2

app = typer.Typer()

FIG_DIR = FIGURES_DIR / "p2"
GRID_PERSISTENCIA = [(a, b) for a in np.arange(0.05, 0.45, 0.05) for b in np.arange(0.55, 0.96, 0.05)]
TAUS = np.arange(0.05, 0.6, 0.025)
# Valores por defecto del modelo y la evaluación; se pueden cambiar desde la línea de comandos
DEFAULT_HYPERPARAMS = {
    "hist_gradient_boosting": {
        "max_iter": 300,
        "learning_rate": 0.1,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
        "early_stopping": False,
    },
    "logistic_regression": {
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 500,
        "solver": "lbfgs",
    }
}

SEARCH_GRIDS = {
    "hist_gradient_boosting": {
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [31, 63],
        "class_weight": ["balanced", None],
    },
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "class_weight": ["balanced", None]
    }
}

K_VISITAS = 20


tabla_path = OUT_DIR_P2 / "tabla.parquet"
info_path = OUT_DIR_P2 / "info.json"
s0_path = OUT_DIR_P2 / "s0_por_estacion.csv"
trips_path = TRIPS_CSV

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

def init_common(params: dict, k: int, SEED: int, trips_path: str, len_train: int, len_val: int, len_features: int) -> tuple[dict, dict]:
    datos_sha = sha256(trips_path)
    commit, clean = git_commit()
    comunes = params | {
        "k_visitas": k,
        "seed": SEED,
        "datos_sha256": datos_sha,
        "estaciones_version": "nomenclatura_2026_09",
        "n_filas_train": len_train,
        "n_filas_val": len_val,
        "n_features": len_features,
    }
    tags = {
        "equipo": "37",
        "pregunta": "P2",
        "git_commit": commit,
        "git_worktree_clean": clean,
        "version_datos": datos_sha,
    }

    return comunes, tags

def load_features() -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    ## Revisar si existe la tabla de features, si no, generarla con MiBici.features p2
    if not tabla_path.exists():
        logger.warning(f"No existe {tabla_path}, generando con MiBici.features p2...")
        subprocess.run([sys.executable, "-m", "MiBici.features", "p2"], cwd=PROJ_ROOT, check=True)

    info = json.loads(info_path.read_text())

    tabla = pd.read_parquet(tabla_path)
    features = features_de_p2(tabla)
    train = tabla[tabla["particion"] == "train"]
    val = tabla[tabla["particion"] == "val"]
    y_train, y_val = train["clase"].to_numpy(), val["clase"].to_numpy()
    logger.info(f"train: {len(train):,} filas · val: {len(val):,} filas · {len(features)} features")
    
    return info, train, val, y_train, y_val, features, tabla

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
    # puntajes, no probabilidxades
    proba = np.column_stack([np.zeros_like(oc), 1 - oc, oc])
    pred = pred_persistencia(oc, a, b)
    res = metricas(y_val, proba, pred) | top_k(val, np.maximum(1 - oc, oc), k)
    return {"nombre": "P2 - Baseline persistencia", "modelo": "persistencia", "proba": proba, "pred": pred, "res": res,
            "params": {"umbral_a": round(float(a), 2), "umbral_b": round(float(b), 2)}}


def baseline_perfil(train: pd.DataFrame, val: pd.DataFrame, y_train, y_val, k: int) -> dict:
    claves = ["estacion", "dia_semana", "hora"]
    perfil = pd.get_dummies(train["clase"]).astype(float).groupby([train[c] for c in claves]).mean()
    perfil.columns = [0, 1, 2]
    global_ = np.bincount(y_train, minlength=3) / len(y_train)
    proba = val[claves].join(perfil, on=claves)[[0, 1, 2]].fillna(dict(enumerate(global_))).to_numpy()
    tau = ajustar_tau(y_val, proba)
    pred = pred_umbral(proba, tau)
    res = metricas(y_val, proba, pred) | top_k(val, proba[:, 1] + proba[:, 2], k)
    return {"nombre": "P2 - Baseline perfil histórico", "modelo": "perfil_historico", "proba": proba, "pred": pred, "res": res,
            "params": {"tau": tau, "claves_perfil": ",".join(claves)}}


def train_model(model, train, val, y_train, y_val, features: list[str], k: int,
                 seed: int, model_params: dict = DEFAULT_HYPERPARAMS) -> dict:
    """Función genérica para entrenar un modelo con los hiperparámetros dados."""
    if model == "hist_gradient_boosting":
        estimador = HistGradientBoostingClassifier(
            **model_params, categorical_features=[features.index("estado_est")], random_state=seed
        )
    elif model == "logistic_regression":
        estimador = Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", StandardScaler()),
            ("logistica", LogisticRegression(**model_params, random_state=seed)),
        ])
    else:
        raise ValueError(f"Modelo no reconocido: {model}.")
    estimador.fit(train[features], y_train)
    proba = estimador.predict_proba(val[features])
    tau = ajustar_tau(y_val, proba)
    pred = pred_umbral(proba, tau)
    res = metricas(y_val, proba, pred) | top_k(val, proba[:, 1] + proba[:, 2], k)
    return {"nombre": f"P2 - {model}", "modelo": model, "proba": proba, "pred": pred, "res": res,
            "params": {"tau": tau} | {f"{model}_{p}": v for p, v in model_params.items()}, "estimador": estimador}


def evaluate_and_metrics(modelos, y_val, val, k):
    fig = plot_curvas_pr(y_val, modelos)
    fig.savefig(FIG_DIR / "curvas_pr_val.png", dpi=150)
    plt.close(fig)

    techo = techo_recall(val, k)
    resultados = pd.DataFrame([{"modelo": m["nombre"]} | m["res"] for m in modelos]).set_index("modelo")
    resultados[f"% del techo recall@{k}"] = resultados[f"recall@{k}"] / techo
    resultados.to_csv(FIG_DIR / "resultados_val.csv")
    for m in modelos:
        disp = ConfusionMatrixDisplay.from_predictions(y_val, m["pred"], display_labels=CLASES, normalize="true", cmap="Blues")
        disp.figure_.savefig(FIG_DIR / f"matriz_confusion_{m['modelo']}.png", dpi=150)
        plt.close(disp.figure_)

    logger.info(f"Resultados en validación (techo de recall@{k} = {techo:.3f}):\n{resultados.round(3).to_string()}")
    return techo


def search_model(model, train, val, tags, comunes, grid: dict = SEARCH_GRIDS, features: list[str] = features_de_p2, k: int = K_VISITAS) -> None:
    """Función genérica para buscar hiperparámetros de un modelo con el grid dado."""
    grid = SEARCH_GRIDS[model]
    resultados = []
    y_train, y_val = train["clase"].to_numpy(), val["clase"].to_numpy()
    with search_run("P2", model, tags):
        mlflow.log_params(comunes)
        mlflow.log_dict(grid, "grid.json")
        for combinacion in combinaciones(grid):
            params = DEFAULT_HYPERPARAMS[model] | combinacion
            with child_run("P2", model, params, tags):
                mlflow.log_params(comunes)
                models = train_model(model, train, val, y_train, y_val, features, K_VISITAS, SEED, params)
                valores = {nombre_metrica(n): float(v) for n, v in models["res"].items()}
                mlflow.log_metrics(valores)
                mlflow.log_param("tau", models["params"]["tau"])
            resultados.append({**combinacion, **valores})

        mejor = max(resultados, key=lambda r: r[f"val_recall_at_{k}"])
        mlflow.log_metrics({f"best_val_recall_at_{k}": mejor[f"val_recall_at_{k}"]})
        mlflow.log_dict(mejor, "mejor_combinacion.json")
        mlflow.log_table(pd.DataFrame(resultados), "resultados_busqueda.json")

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


def registrar_runs(modelos: list[dict], comunes: dict, tags: dict, metricas_comunes: dict, artefactos: dict[Path, str],
                   train: pd.DataFrame, val: pd.DataFrame, features: list[str]):
    """Un run por modelo; `artefactos` son los archivos comunes como {ruta: subcarpeta}."""
    setup_mlflow()
    # Ejemplo de entrada sin NaN para la firma del modelo
    sin_nan = val[features].dropna()
    input_example = (sin_nan if len(sin_nan) else val[features]).head(5)

    for m in modelos:
        with mlflow.start_run(run_name=m["nombre"]) as run:
            mlflow.log_params(comunes | m["params"])
            mlflow.set_tags(tags | {"modelo": m["modelo"], "model_family": m["modelo"]})
            mlflow.log_metrics({nombre_metrica(k): float(v) for k, v in m["res"].items()} | metricas_comunes)
            log_run_artifacts(
                run_config={"pregunta": "P2", "model": m["modelo"], "stage_of_work": tags["stage_of_work"],
                            "hyperparams": m["params"], "k_visitas": K_VISITAS, "features": features},
                train_df=train[features + ["clase"]],
                train_source=tabla_path,
                target="clase",
                model=m.get("estimador"),
                input_example=input_example,
                files=artefactos | {FIG_DIR / f"matriz_confusion_{m['modelo']}.png": "plots"},
            )
            logger.info(f"{m['nombre']}: AP vaciado={m['res']['AP vaciado']:.3f}, "
                        f"AP llenado={m['res']['AP llenado']:.3f}, run_id={run.info.run_id}")


# --------------------------------------------------------------------------- punto de entrada

def search(model: str = typer.Option("hist_gradient_boosting", help="Nombre del modelo a entrenar")):
    """Realiza una búsqueda de hiperparámetros para el modelo especificado."""
    logger.info(f"Realizando búsqueda de hiperparámetros para {model}")
    if model == "none":
        logger.info("No se especificó modelo, se usará hist_gradient_boosting por defecto")
        model = "hist_gradient_boosting"

     ## Cargar tabla de features y particiones train/val
    info, train, val, y_train, y_val, features, tabla = load_features()
    params = info["params"]
    del tabla

    ## Variables comunes y tags para todos los runs
    comunes, tags = init_common(params, K_VISITAS, SEED, trips_path, len(train), len(val), len(features))
               
    setup_mlflow()
    search_model(model, train, val, tags, comunes, SEARCH_GRIDS, features, K_VISITAS)
    


@app.command()
def main(
    model: str = typer.Option("hist_gradient_boosting", help="Nombre del modelo a entrenar"),
    hyperparams: dict = typer.Option({}, help="Diccionario de hiperparámetros para el modelo"),
):

    ## Cargar tabla de features y particiones train/val
    info, train, val, y_train, y_val, features, tabla = load_features()
    params = info["params"]

    ## Variables comunes y tags para todos los runs
    comunes, tags = init_common(params, K_VISITAS, SEED, trips_path, len(train), len(val), len(features))
        
    ## Grafica de distribución de clases
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_distribucion_clases(tabla)
    fig.savefig(FIG_DIR / "distribucion_clases.png", dpi=150)
    plt.close(fig)
    del tabla

    if model == "baseline":
        logger.info("Baselines...")
        modelos = [baseline_persistencia(val, y_val, K_VISITAS), baseline_perfil(train, val, y_train, y_val, K_VISITAS)]
        tags["stage_of_work"] = "baseline"
    else:
        logger.info(f"Entrenando modelo {model}...")
        # Si no seleccionan modelo, se entrena el HistGradientBoosting con los hiperparámetros por defecto
        if model == "none":
            model = "hist_gradient_boosting"

        model_params = DEFAULT_HYPERPARAMS[model] | convert_params_type(hyperparams, DEFAULT_HYPERPARAMS[model])
        modelos = [train_model(model, train, val, y_train, y_val, features, K_VISITAS, SEED, model_params)]

        tags["model_family"] = model
        tags["stage_of_work"] = "candidate"


    techo = evaluate_and_metrics(modelos, y_val, val, K_VISITAS)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if "estimador" in modelos[-1]:
        joblib.dump(modelos[-1]["estimador"], MODELS_DIR / f"p2_{model}.joblib")

    metricas_comunes = {f"val_techo_recall_at_{K_VISITAS}": techo} | {n: float(v) for n, v in info["proxy"].items()}
    artefactos = {
        FIG_DIR / "distribucion_clases.png": "plots",
        FIG_DIR / "curvas_pr_val.png": "plots",
        FIG_DIR / "resultados_val.csv": "results",
        s0_path: "data",
        info_path: "data",
    }
    registrar_runs(modelos, comunes, tags, metricas_comunes, artefactos, train, val, features)
    logger.success("Runs de P2 registrados en MLflow.")


if __name__ == "__main__":
    app()
