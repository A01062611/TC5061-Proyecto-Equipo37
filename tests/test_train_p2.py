"""Pruebas del entrenamiento y la evaluación de P2: baselines, modelos, umbrales y métricas."""

import numpy as np
import pandas as pd
import pytest

from MiBici.config import SEED
from MiBici.modeling.train_p2 import (
    DEFAULT_HYPERPARAMS,
    TAUS,
    ajustar_tau,
    baseline_perfil,
    baseline_persistencia,
    metricas,
    nombre_metrica,
    pred_persistencia,
    pred_umbral,
    techo_recall,
    top_k,
    train_model,
)

K = 2
MODELOS = list(DEFAULT_HYPERPARAMS)


def _entrenar(model, p2_split, **cambios):
    train, val, y_train, y_val, features = p2_split
    return train_model(model, train, val, y_train, y_val, features, K, SEED, DEFAULT_HYPERPARAMS[model] | cambios)


# --- reglas de decisión -----------------------------------------------------------------

def test_pred_persistencia_umbrales_inclusivos():
    oc = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    np.testing.assert_array_equal(pred_persistencia(oc, 0.10, 0.90), [1, 1, 0, 2, 2])


def test_pred_umbral_elige_el_riesgo_mas_probable_si_llega_a_tau():
    proba = np.array([[0.90, 0.05, 0.05], [0.40, 0.50, 0.10], [0.50, 0.20, 0.30]])
    np.testing.assert_array_equal(pred_umbral(proba, 0.3), [0, 1, 2])
    # con τ alto todo queda como normal aunque normal no sea la clase más probable
    np.testing.assert_array_equal(pred_umbral(proba, 0.6), [0, 0, 0])

# --- métricas ---------------------------------------------------------------------------

def _dia_hora(clases, hora=8):
    return pd.DataFrame({"dia": pd.Timestamp("2025-07-01"), "hora": hora, "clase": clases})


def test_top_k_recall_y_precision_por_hora():
    df = _dia_hora([1, 0, 2])
    res = top_k(df, np.array([0.9, 0.8, 0.1]), k=1)
    # cubre 1 de los 2 riesgos
    assert res["recall@1"] == 0.5  
    assert res["precision@1"] == 1.0
    assert (res["eventos de riesgo cubiertos"], res["eventos de riesgo totales"]) == (1, 2)


def test_top_k_ignora_horas_sin_riesgo():
    df = pd.concat([_dia_hora([1, 0], hora=8), _dia_hora([0, 0], hora=9)])
    res = top_k(df, np.array([0.9, 0.1, 0.9, 0.8]), k=1)
    assert res["recall@1"] == 1.0
    assert res["eventos de riesgo totales"] == 1


def test_techo_recall_limitado_por_k():
    assert techo_recall(_dia_hora([1, 2, 1, 0]), k=1) == pytest.approx(1 / 3)
    assert techo_recall(_dia_hora([1, 2, 1, 0]), k=5) == 1.0


def test_metricas_con_prediccion_perfecta():
    y = np.array([0, 1, 2, 0, 1, 2])
    res = metricas(y, np.eye(3)[y], y)
    assert res["AP vaciado"] == res["AP llenado"] == 1.0
    assert res["Brier vaciado"] == res["Brier llenado"] == 0.0
    assert res["F2 vaciado"] == res["F2 llenado"] == 1.0

# --- modelos ----------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELOS)
def test_train_model_regresa_probabilidades_validas(model, p2_split):
    _, val, _, y_val, _ = p2_split
    m = _entrenar(model, p2_split)
    assert m["proba"].shape == (len(val), 3)
    np.testing.assert_allclose(m["proba"].sum(1), 1)
    assert set(m["pred"]) <= {0, 1, 2}
    assert np.isclose(TAUS, m["params"]["tau"]).any()
    assert 0 <= m["res"][f"recall@{K}"] <= techo_recall(val, K) + 1e-9


def test_hist_gradient_boosting_supera_al_azar(p2_split):
    # los datos sintéticos tienen un ritmo claro (vaciado en la mañana, llenado en la tarde)
    _, _, _, y_val, _ = p2_split
    m = _entrenar("hist_gradient_boosting", p2_split, max_iter=50)
    assert m["res"]["AP vaciado"] > (y_val == 1).mean()
    assert m["res"]["AP llenado"] > (y_val == 2).mean()

def test_modelo_desconocido_falla(p2_split):
    train, val, y_train, y_val, features = p2_split
    with pytest.raises(ValueError, match="no reconocido"):
        train_model("random_forest", train, val, y_train, y_val, features, K, SEED, {})


@pytest.mark.parametrize("model", MODELOS)
def test_train_vacio_falla(model, p2_split):
    train, val, y_train, y_val, features = p2_split
    with pytest.raises(ValueError):
        train_model(model, train.iloc[:0], val, y_train[:0], y_val, features, K, SEED, DEFAULT_HYPERPARAMS[model])
