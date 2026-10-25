"""P3: clustering de estaciones por patrón de uso, registrado en MLflow.

Se ejecuta con el MLflow Project desde la raíz del repo:
  mlflow run . -e train -P question=p3 -P model=baseline
  mlflow run . -e train -P question=p3 -P model=pca_kmeans -P n_clusters=6
  mlflow run . -e search -P question=p3 -P model=pca_kmeans

Modelos (`model`):
- baseline: K-Means con k=4 sobre el perfil estandarizado, sin PCA.
- pca_kmeans: PCA + K-Means. Con n_clusters=0, k se elige por silueta, Davies-Bouldin
  y concordancia con Ward (el candidato de la entrega anterior).
- agglomerative: clustering jerárquico sobre el perfil estandarizado.
- gaussian_mixture: mezcla de gaussianas sobre el perfil estandarizado.

Las métricas de todos los runs se calculan sobre el perfil estandarizado completo
para que sean comparables entre sí, aunque el modelo se ajuste en el espacio PCA.
"""

import hashlib
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from MiBici.config import (
    PROJ_ROOT,
    SEED,
    STATIONS_CSV,
    TRIPS_CSV,
)
from MiBici.modeling.helper import (
    child_run,
    combinaciones,
    comando_candidato,
    convert_params_type,
    log_run_artifacts,
    search_run,
    setup_mlflow,
)
from MiBici.features import (
    filter_valid_trips,
    load_trips,
    n_service_days,
    station_hourly_counts,
    station_usage_profile,
    station_usage_summary,
)

DEFAULT_HYPERPARAMS = {
    "pca_kmeans": {
        "n_components": 0.9,
        "n_clusters": 5,
    },
    "agglomerative": {
        "n_clusters": 5,
        "linkage": "ward",
    },
    "gaussian_mixture": {
        "n_components": 5,
        "covariance_type": "full",
    },
}

SEARCH_GRIDS = {
    "pca_kmeans": {
        "n_components": [0.8, 0.9, 0.95],
        "n_clusters": [3, 4, 5, 6, 7],
    },
    "agglomerative": {
        "n_clusters": [3, 4, 5, 6, 7],
        "linkage": ["ward", "complete", "average"],
    },
    "gaussian_mixture": {
        "n_components": [3, 4, 5, 6, 7],
        "covariance_type": ["full", "tied", "diag", "spherical"],
    }
}

MIN_TRIPS = 500  # estaciones con menos viajes en el periodo no se agrupan
K_BASELINE = 4
K_RANGE = (3, 8)  # k permitido cuando pca_kmeans lo elige solo (n_clusters=0)

CITY_CENTER = (20.6767, -103.3475)  # Plaza de Armas


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> tuple[str, str]:
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True).stdout.strip()

    return run("rev-parse", "HEAD") or "desconocido", str(run("status", "--porcelain") == "").lower()


def build_dataset(trips_path: Path, stations_path: Path, min_trips: int):
    trips = load_trips(trips_path)
    n_total = len(trips)
    dates = trips["Inicio_del_viaje"].agg(["min", "max"]).dt.date.astype(str)
    trips = filter_valid_trips(trips)
    counts = station_hourly_counts(trips)
    summary = station_usage_summary(counts, n_service_days(trips))

    stations = pd.read_csv(stations_path, encoding="utf-8").set_index("id")
    keep = summary.index[summary.index.isin(stations.index) & (summary["viajes"] >= min_trips)]
    summary = summary.loc[keep].join(stations[["name", "latitude", "longitude", "dpcapacity"]])
    summary["municipio"] = summary["name"].str.extract(r"\(([A-Z]+)-")[0].fillna("OTRO")

    info = {
        "fecha_inicio": dates["min"],
        "fecha_fin": dates["max"],
        "viajes_crudos": n_total,
        "viajes_limpios": len(trips),
        "n_estaciones": len(keep),
    }
    return station_usage_profile(counts.loc[keep], summary), summary, info


def evaluate(X: np.ndarray, labels: np.ndarray, k: int) -> dict:
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    return {
        "train_silhouette": silhouette_score(X, labels),
        "train_davies_bouldin": davies_bouldin_score(X, labels),
        "train_ari_vs_ward": adjusted_rand_score(labels, ward),
        "train_min_cluster_size": int(np.bincount(labels).min()),
    }


def score(df: pd.DataFrame) -> pd.Series:
    """Silueta + ARI vs Ward - Davies-Bouldin, cada una normalizada a [0, 1] dentro de `df`."""
    def norm(s):
        spread = s.max() - s.min()
        return (s - s.min()) / spread if spread > 0 else s * 0
    return norm(df["train_silhouette"]) + norm(df["train_ari_vs_ward"]) - norm(df["train_davies_bouldin"])


def select_k(X: np.ndarray, k_min: int, k_max: int) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        rows.append({"k": k, "inercia": km.inertia_, **evaluate(X, km.labels_, k)})
    k_eval = pd.DataFrame(rows).set_index("k")
    # El puntaje se normaliza solo dentro del rango de k permitido
    scores = score(k_eval.loc[k_min:k_max])
    k_eval["puntaje"] = scores
    return int(scores.idxmax()), k_eval


def fit_model(model: str, X_std: np.ndarray, params: dict):
    """Entrena el modelo de P3.

    Regresa etiquetas, parámetros a registrar, métricas extra, la tabla de selección de k 
    y el estimador ajustado.
    """
    if model == "pca_kmeans":
        pca = PCA(n_components=params["n_components"], random_state=SEED).fit(X_std)
        X = pca.transform(X_std)
        k, k_eval = params["n_clusters"], None
        if k == 0:
            k, k_eval = select_k(X, *K_RANGE)
        km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
        aris = [adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X))
                for s in range(10)]
        logged_params = {"k": k, "pca": True, "pca_varianza": params["n_components"], "pca_componentes": int(pca.n_components_)}
        if k_eval is not None:
            logged_params["k_rango_busqueda"] = f"{K_RANGE[0]}-{K_RANGE[1]}"
        estimator = Pipeline([("pca", pca), ("kmeans", km)])
        return km.labels_, logged_params, {"train_ari_semillas_media": float(np.mean(aris))}, k_eval, estimator
    if model == "agglomerative":
        # No tiene predict: asigna grupos solo a las estaciones con que se ajustó
        agg = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params["linkage"]).fit(X_std)
        return agg.labels_, {"k": params["n_clusters"], "pca": False, "linkage": params["linkage"]}, {}, None, agg
    if model == "gaussian_mixture":
        gm = GaussianMixture(n_components=params["n_components"], covariance_type=params["covariance_type"],
                             n_init=5, random_state=SEED).fit(X_std)
        logged_params = {"k": params["n_components"], "pca": False, "covariance_type": params["covariance_type"]}
        return gm.predict(X_std), logged_params, {"train_bic": float(gm.bic(X_std))}, None, gm
    raise ValueError(f"Modelo no reconocido: {model}.")


def label_clusters(summary: pd.DataFrame) -> pd.Series:
    # Umbrales relativos a la mediana de todas las estaciones
    ref = summary[["viajes_dia", "pct_finde"]].median()
    g = summary.groupby("cluster").agg(
        viajes_dia=("viajes_dia", "median"),
        balance_am=("balance_am", "mean"),
        balance_pm=("balance_pm", "mean"),
        pct_finde=("pct_finde", "mean"),
    )

    def label(r):
        if r["balance_am"] > 0.15 and r["balance_pm"] < -0.15:
            parts = ["laboral/escolar"]
        elif r["balance_am"] < -0.15 and r["balance_pm"] > 0.15:
            parts = ["residencial"]
        else:
            parts = ["mixto"]
        if r["pct_finde"] > ref["pct_finde"] * 1.15:
            parts.append("recreativo")
        if r["viajes_dia"] < ref["viajes_dia"] * 0.6:
            parts.append("baja demanda")
        return " / ".join(parts)

    return g.apply(label, axis=1)


def plot_map(summary: pd.DataFrame, labels: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = plt.cm.tab10(np.arange(len(labels)))
    size = 8 + 60 * summary["viajes_dia"] / summary["viajes_dia"].max()
    for c, name in labels.items():
        sub = summary[summary["cluster"] == c]
        ax.scatter(sub["longitude"], sub["latitude"], s=size[sub.index], color=colors[c], alpha=0.8,
                   edgecolor="k", linewidth=0.3, label=f"G{c} {name} (n={len(sub)})")
    ax.scatter(CITY_CENTER[1], CITY_CENTER[0], marker="*", s=250, c="k", label="Plaza de Armas")
    ax.set(title=title, xlabel="longitud", ylabel="latitud")
    ax.set_aspect(1 / np.cos(np.radians(CITY_CENTER[0])))
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_profiles(profile: pd.DataFrame, summary: pd.DataFrame, labels: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    colors = plt.cm.tab10(np.arange(len(labels)))
    for ax, day in zip(axes, ["semana", "finde"]):
        for c, name in labels.items():
            sub = profile[summary["cluster"] == c]
            ax.plot(range(24), sub.filter(like=f"sal_{day}").mean().values, c=colors[c], label=f"G{c} {name} salidas")
            ax.plot(range(24), sub.filter(like=f"lle_{day}").mean().values, c=colors[c], ls=":", label=f"G{c} llegadas")
        ax.set(title="Entre semana" if day == "semana" else "Fin de semana", xlabel="hora", xticks=range(0, 24, 2))
    axes[0].set_ylabel("proporción de viajes del día")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def log_run(run_name, cluster_labels, X, profile, summary, params, base_params, tags, extra_metrics=None, k_eval=None,
            model=None):
    """Registra un run de P3"""
    summary = summary.assign(cluster=cluster_labels)
    labels = label_clusters(summary)
    summary["etiqueta"] = summary["cluster"].map(labels)
    metrics = evaluate(X, cluster_labels, params["k"]) | (extra_metrics or {})

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(base_params | params)
        mlflow.set_tags(tags | {"modelo": params["modelo"]})
        mlflow.log_metrics(metrics)
        mlflow.log_table(
            summary[["name", "municipio", "latitude", "longitude", "cluster", "etiqueta", "viajes_dia", "balance"]]
            .reset_index(names="estacion_id"),
            "results/estaciones_cluster.json",
        )
        if k_eval is not None:
            mlflow.log_table(k_eval.reset_index(), "results/seleccion_k.json")
        log_run_artifacts(
            run_config={"pregunta": "P3", "model": params["modelo"], "stage_of_work": tags["stage_of_work"],
                        "hyperparams": params, "min_viajes": MIN_TRIPS},
            train_df=profile,
            train_source=TRIPS_CSV,
            model=model,
            input_example=profile.head(5),
            figures={
                "mapa_clusters.png": plot_map(summary, labels, f"{run_name} (k={params['k']})"),
                "perfiles_cluster.png": plot_profiles(profile, summary, labels),
            },
        )
        logger.info(f"{run_name}: {', '.join(f'{m}={v:.3f}' for m, v in metrics.items())} run_id={run.info.run_id}")


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler, dict, dict]:
    """Perfil por estación, perfil estandarizado con su scaler, parámetros comunes y etiquetas base."""
    logger.info("Construyendo perfiles por estación...")
    profile, summary, info = build_dataset(TRIPS_CSV, STATIONS_CSV, MIN_TRIPS)
    scaler = StandardScaler().fit(profile)
    X_std = scaler.transform(profile)

    commit, clean = git_commit()
    data_hash = sha256(TRIPS_CSV)
    base_params = info | {
        "datos_sha256": data_hash,
        "seed": SEED,
        "min_viajes": MIN_TRIPS,
        "duracion_min_valida": "(1, 180)",
        "excluye_origen_igual_destino": True,
        "n_features": profile.shape[1],
    }
    tags = {
        "equipo": "37",
        "pregunta": "P3",
        "git_commit": commit,
        "git_worktree_clean": clean,
        "version_datos": data_hash,
    }
    return profile, summary, X_std, scaler, base_params, tags


def search(model: str = "none") -> None:
    """Búsqueda de hiperparámetros con nested runs: una corrida padre y una hija por combinación del grid."""
    if model == "none":
        model = "pca_kmeans"
    if model not in SEARCH_GRIDS:
        raise ValueError(f"No hay grid de búsqueda para {model}. Opciones: {list(SEARCH_GRIDS)}")

    profile, summary, X_std, _, base_params, tags = prepare_data()
    setup_mlflow()
    grid = SEARCH_GRIDS[model]
    results = []
    with search_run("P3", model, tags):
        mlflow.log_params(base_params)
        mlflow.log_dict(grid, "grid.json")
        for combination in combinaciones(grid):
            params = DEFAULT_HYPERPARAMS[model] | combination
            with child_run("P3", model, params, tags) as child:
                mlflow.log_params(base_params)
                cluster_labels, logged_params, extra, _, _ = fit_model(model, X_std, params)
                metrics = evaluate(X_std, cluster_labels, logged_params["k"]) | extra
                mlflow.log_metrics(metrics)
            logger.info(f"{combination}: silueta={metrics['train_silhouette']:.3f}, "
                        f"DB={metrics['train_davies_bouldin']:.3f}, ARI={metrics['train_ari_vs_ward']:.3f}")
            results.append({"run_id": child.info.run_id} | combination | metrics)

        # La mejor combinación se elige con el mismo puntaje que select_k, no solo con la silueta
        table = pd.DataFrame(results)
        table["puntaje"] = score(table)
        # Se toma la fila de `results` para que los enteros no salgan como float
        best_idx = int(table["puntaje"].idxmax())
        best = results[best_idx] | {"puntaje": float(table.loc[best_idx, "puntaje"])}
        mlflow.log_metrics({"best_puntaje": best["puntaje"], "best_train_silhouette": best["train_silhouette"]})
        mlflow.set_tags({
            "best_run_id": best["run_id"],
            # las hijas no guardan modelo: la mejor combinación se vuelve a correr como candidate para registrarla
            "comando_candidato": comando_candidato("p3", model, {p: best[p] for p in grid}),
        })
        mlflow.log_dict(best, "mejor_combinacion.json")
        mlflow.log_table(table, "resultados_busqueda.json")
    logger.success(f"Búsqueda de P3 registrada en MLflow. Mejor combinación: { {p: best[p] for p in grid} }")


def main(model: str = "none", hyperparams: dict | None = None) -> None:
    """Entrena un modelo de P3 (o el baseline) y lo registra en MLflow."""
    if model == "none":
        model = "pca_kmeans"
    if model != "baseline" and model not in DEFAULT_HYPERPARAMS:
        raise ValueError(f"Modelo no reconocido: {model}. Opciones: baseline, {', '.join(DEFAULT_HYPERPARAMS)}")

    profile, summary, X_std, scaler, base_params, tags = prepare_data()
    setup_mlflow()

    if model == "baseline":
        # k fijo, sin reducción de dimensión
        km = KMeans(n_clusters=K_BASELINE, n_init=20, random_state=SEED).fit(X_std)
        log_run(
            f"P3 - Baseline KMeans k={K_BASELINE}",
            km.labels_, X_std, profile, summary,
            {"modelo": "kmeans", "k": K_BASELINE, "pca": False},
            base_params, tags | {"model_family": "kmeans", "stage_of_work": "baseline"},
            model=Pipeline([("scaler", scaler), ("kmeans", km)]),
        )
    else:
        params = DEFAULT_HYPERPARAMS[model] | convert_params_type(hyperparams or {}, DEFAULT_HYPERPARAMS[model])
        cluster_labels, logged_params, extra, k_eval, estimator = fit_model(model, X_std, params)
        log_run(
            f"P3 - {model} k={logged_params['k']}",
            cluster_labels, X_std, profile, summary,
            {"modelo": model} | logged_params,
            base_params, tags | {"model_family": model, "stage_of_work": "candidate"},
            extra_metrics=extra,
            k_eval=k_eval,
            model=Pipeline([("scaler", scaler), ("modelo", estimator)]),
        )
    logger.success("Run de P3 registrado en MLflow.")


if __name__ == "__main__":
    main()
