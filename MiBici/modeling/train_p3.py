"""P3: clustering de estaciones por patrón de uso, registrado en MLflow.

Ejecución desde la raíz del repo: python -m MiBici.modeling.train_p3

Genera dos runs en el experimento del equipo:
- Baseline: K-Means con k fijo sobre el perfil estandarizado, sin PCA.
- Candidato: PCA (90 % de varianza) + K-Means, con k elegido por silueta,
  Davies-Bouldin y concordancia con Ward.

Las métricas de ambos runs se calculan sobre el perfil estandarizado completo
para que sean comparables entre sí.
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
from sklearn.preprocessing import StandardScaler
import typer

from MiBici.config import (
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    SEED,
    STATIONS_CSV,
    TRIPS_CSV,
)
from MiBici.features import (
    filter_valid_trips,
    load_trips,
    n_service_days,
    station_hourly_counts,
    station_usage_profile,
    station_usage_summary,
)

app = typer.Typer()

CENTRO = (20.6767, -103.3475)  # Plaza de Armas


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


def build_dataset(trips_path: Path, stations_path: Path, min_viajes: int):
    trips = load_trips(trips_path)
    n_total = len(trips)
    fechas = trips["Inicio_del_viaje"].agg(["min", "max"]).dt.date.astype(str)
    trips = filter_valid_trips(trips)
    counts = station_hourly_counts(trips)
    summary = station_usage_summary(counts, n_service_days(trips))

    stations = pd.read_csv(stations_path, encoding="utf-8").set_index("id")
    keep = summary.index[summary.index.isin(stations.index) & (summary["viajes"] >= min_viajes)]
    summary = summary.loc[keep].join(stations[["name", "latitude", "longitude", "dpcapacity"]])
    summary["municipio"] = summary["name"].str.extract(r"\(([A-Z]+)-")[0].fillna("OTRO")

    info = {
        "fecha_inicio": fechas["min"],
        "fecha_fin": fechas["max"],
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


def select_k(X: np.ndarray, k_min: int, k_max: int) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        rows.append({"k": k, "inercia": km.inertia_, **evaluate(X, km.labels_, k)})
    k_eval = pd.DataFrame(rows).set_index("k")
    # Puntaje = silueta + ARI - DB, normalizados dentro del rango de k permitido
    cand = k_eval.loc[k_min:k_max]
    norm = lambda s: (s - s.min()) / (s.max() - s.min())
    score = norm(cand["train_silhouette"]) + norm(cand["train_ari_vs_ward"]) - norm(cand["train_davies_bouldin"])
    k_eval["puntaje"] = score
    return int(score.idxmax()), k_eval


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
    ax.scatter(CENTRO[1], CENTRO[0], marker="*", s=250, c="k", label="Plaza de Armas")
    ax.set(title=title, xlabel="longitud", ylabel="latitud")
    ax.set_aspect(1 / np.cos(np.radians(CENTRO[0])))
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_profiles(profile: pd.DataFrame, summary: pd.DataFrame, labels: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    colors = plt.cm.tab10(np.arange(len(labels)))
    for ax, dia in zip(axes, ["semana", "finde"]):
        for c, name in labels.items():
            sub = profile[summary["cluster"] == c]
            ax.plot(range(24), sub.filter(like=f"sal_{dia}").mean().values, c=colors[c], label=f"G{c} {name} salidas")
            ax.plot(range(24), sub.filter(like=f"lle_{dia}").mean().values, c=colors[c], ls=":", label=f"G{c} llegadas")
        ax.set(title="Entre semana" if dia == "semana" else "Fin de semana", xlabel="hora", xticks=range(0, 24, 2))
    axes[0].set_ylabel("proporción de viajes del día")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def log_run(run_name, model, X, profile, summary, params, base_params, tags, extra_metrics=None, k_eval=None):
    summary = summary.assign(cluster=model.labels_)
    labels = label_clusters(summary)
    summary["etiqueta"] = summary["cluster"].map(labels)
    metrics = evaluate(X, model.labels_, params["k"]) | (extra_metrics or {})

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(base_params | params)
        mlflow.set_tags(tags | {"modelo": params["modelo"]})
        mlflow.log_metrics(metrics)
        mlflow.log_figure(plot_map(summary, labels, f"{run_name} (k={params['k']})"), "graficas/mapa_clusters.png")
        mlflow.log_figure(plot_profiles(profile, summary, labels), "graficas/perfiles_cluster.png")
        mlflow.log_table(
            summary[["name", "municipio", "latitude", "longitude", "cluster", "etiqueta", "viajes_dia", "balance"]]
            .reset_index(names="estacion_id"),
            "estaciones_cluster.json",
        )
        if k_eval is not None:
            mlflow.log_table(k_eval.reset_index(), "seleccion_k.json")
        plt.close("all")
        logger.info(f"{run_name}: {', '.join(f'{m}={v:.3f}' for m, v in metrics.items())} run_id={run.info.run_id}")


@app.command()
def main(
    trips_path: Path = TRIPS_CSV,
    stations_path: Path = STATIONS_CSV,
    min_viajes: int = 500,
    k_baseline: int = 4,
    k_min: int = 3,
    k_max: int = 8,
    pca_varianza: float = 0.90,
):
    logger.info("Construyendo perfiles por estación...")
    profile, summary, info = build_dataset(trips_path, stations_path, min_viajes)
    X_std = StandardScaler().fit_transform(profile)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    commit, clean = git_commit()
    data_hash = sha256(trips_path)
    base_params = info | {
        "datos_sha256": data_hash,
        "seed": SEED,
        "min_viajes": min_viajes,
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

    # Baseline: k fijo, sin reducción de dimensión
    km = KMeans(n_clusters=k_baseline, n_init=20, random_state=SEED).fit(X_std)
    log_run(
        f"P3 - Baseline KMeans k={k_baseline}",
        km, X_std, profile, summary,
        {"modelo": "kmeans", "k": k_baseline, "pca": False},
        base_params, tags,
    )

    # Candidato: PCA + K-Means con k seleccionado
    pca = PCA(n_components=pca_varianza, random_state=SEED).fit(X_std)
    X = pca.transform(X_std)
    k, k_eval = select_k(X, k_min, k_max)
    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
    aris = [adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X))
            for s in range(10)]
    log_run(
        f"P3 - PCA + KMeans k={k}",
        km, X_std, profile, summary,
        {
            "modelo": "pca_kmeans",
            "k": k,
            "pca": True,
            "pca_varianza": pca_varianza,
            "pca_componentes": pca.n_components_,
            "k_rango_busqueda": f"{k_min}-{k_max}",
        },
        base_params, tags,
        extra_metrics={"train_ari_semillas_media": float(np.mean(aris))},
        k_eval=k_eval,
    )
    logger.success("Runs de P3 registrados en MLflow.")


if __name__ == "__main__":
    app()
