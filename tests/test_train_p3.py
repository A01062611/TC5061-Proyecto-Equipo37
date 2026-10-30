"""Pruebas de los modelos de clustering de P3: casos normales, límite y reproducibilidad."""

import numpy as np
import pandas as pd
import pytest

from MiBici.modeling.train_p3 import (
    DEFAULT_HYPERPARAMS,
    evaluate,
    fit_model,
    label_clusters,
    score,
    select_k,
)

MODELOS = list(DEFAULT_HYPERPARAMS)


def _params(model, k):
    p = dict(DEFAULT_HYPERPARAMS[model])
    p["n_components" if model == "gaussian_mixture" else "n_clusters"] = k
    return p


# --- casos normales ---------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELOS)
def test_fit_model_asigna_una_etiqueta_por_estacion(model, X_std):
    labels, logged, _, _, _ = fit_model(model, X_std, _params(model, 3))
    assert len(labels) == len(X_std)
    assert len(set(labels)) == 3
    assert logged["k"] == 3


def test_pca_kmeans_separa_los_tres_ritmos_de_uso(X_std):
    labels, *_ = fit_model("pca_kmeans", X_std, _params("pca_kmeans", 3))
    # las estaciones se generaron en bloques de 4 por arquetipo
    for i in range(0, 12, 4):
        assert len(set(labels[i:i + 4])) == 1


def test_evaluate_regresa_metricas_validas(X_std):
    labels, *_ = fit_model("pca_kmeans", X_std, _params("pca_kmeans", 3))
    m = evaluate(X_std, labels, 3)
    assert -1 <= m["train_silhouette"] <= 1
    assert m["train_davies_bouldin"] >= 0
    assert -1 <= m["train_ari_vs_ward"] <= 1
    assert m["train_min_cluster_size"] >= 1


def test_select_k_respeta_el_rango(X_std):
    k, tabla = select_k(X_std, 3, 6)
    assert 3 <= k <= 6
    assert {"k", "puntaje"} <= set(tabla.reset_index().columns)


def test_score_normaliza_y_premia_buena_silueta():
    df = pd.DataFrame({
        "train_silhouette": [0.1, 0.5], "train_ari_vs_ward": [0.2, 0.8], "train_davies_bouldin": [2.0, 0.5],
    })
    s = score(df)
    assert s.iloc[1] > s.iloc[0]
    # mejor en las tres métricas: 1 + 1 - 0; peor: 0 + 0 - 1
    assert np.isclose(s.iloc[1], 2.0)
    assert np.isclose(s.iloc[0], -1.0)


def test_label_clusters_distingue_residencial_y_laboral():
    summary = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "viajes_dia": [10, 10, 10, 10],
        "pct_finde": [0.2] * 4,
        "balance_am": [-0.5, -0.5, 0.5, 0.5],
        "balance_pm": [0.5, 0.5, -0.5, -0.5],
    })
    etiquetas = label_clusters(summary)
    assert etiquetas[0] == "residencial"
    assert etiquetas[1] == "laboral/escolar"


# --- casos límite -----------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELOS)
def test_dataset_vacio_falla(model):
    with pytest.raises(ValueError):
        fit_model(model, np.empty((0, 5)), _params(model, 3))


@pytest.mark.parametrize("model", MODELOS)
def test_k_mayor_que_estaciones_falla(model, X_std):
    with pytest.raises(ValueError):
        fit_model(model, X_std, _params(model, len(X_std) + 1))


@pytest.mark.parametrize("model", ["agglomerative", "gaussian_mixture"])
@pytest.mark.parametrize("k", [0, -1])
def test_k_no_positivo_falla(model, k, X_std):
    with pytest.raises(ValueError):
        fit_model(model, X_std, _params(model, k))


@pytest.mark.parametrize("model", MODELOS)
def test_nans_en_el_perfil_fallan(model, X_std):
    X = X_std.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_model(model, X, _params(model, 3))


def test_modelo_desconocido_falla(X_std):
    with pytest.raises(ValueError, match="no reconocido"):
        fit_model("dbscan", X_std, {})


def test_evaluate_con_un_solo_cluster_falla(X_std):
    with pytest.raises(ValueError):
        evaluate(X_std, np.zeros(len(X_std), dtype=int), 2)
