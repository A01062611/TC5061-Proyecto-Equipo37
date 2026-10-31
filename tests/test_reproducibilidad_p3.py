"""Dos entrenamientos con la misma semilla deben dar las mismas etiquetas y métricas."""

import numpy as np
import pytest

from MiBici.modeling.train_p3 import DEFAULT_HYPERPARAMS, evaluate, fit_model


@pytest.mark.parametrize("model", list(DEFAULT_HYPERPARAMS))
def test_mismo_seed_mismas_etiquetas_y_metricas(model, X_std):
    params = {**DEFAULT_HYPERPARAMS[model]}
    labels_a, logged_a, extra_a, _, _ = fit_model(model, X_std, params)
    labels_b, logged_b, extra_b, _, _ = fit_model(model, X_std, params)

    np.testing.assert_array_equal(labels_a, labels_b)
    assert logged_a == logged_b
    assert extra_a == extra_b
    assert evaluate(X_std, labels_a, logged_a["k"]) == evaluate(X_std, labels_b, logged_b["k"])


def test_pipeline_pca_kmeans_predice_igual_en_dos_entrenamientos(X_std):
    params = {**DEFAULT_HYPERPARAMS["pca_kmeans"]}
    *_, est_a = fit_model("pca_kmeans", X_std, params)
    *_, est_b = fit_model("pca_kmeans", X_std, params)
    np.testing.assert_array_equal(est_a.predict(X_std), est_b.predict(X_std))
