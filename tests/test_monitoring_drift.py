"""Reporte de drift con Evidently (`MiBici/monitoring/drift.py`) y cruce con la importancia.

Todo corre con datos sintéticos: no depende de `data/` ni del modelo.
"""

import json

import numpy as np
import pandas as pd
import pytest
import typer
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from MiBici.monitoring.drift import SEED, _filtrar_estaciones, _hubo_drift, _muestrear, resumir
from MiBici.monitoring.importancia import cruzar


@pytest.fixture(scope="module")
def snapshot_y_tamanos():
    """Evidently sobre dos conjuntos: `x` se desplaza en el actual, `y` no."""
    rng = np.random.default_rng(SEED)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "y": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"x": rng.normal(3, 1, 2000), "y": rng.normal(0, 1, 2000)})
    definicion = DataDefinition(numerical_columns=["x", "y"])
    snap = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(cur, data_definition=definicion),
        Dataset.from_pandas(ref, data_definition=definicion),
    )
    return snap, len(ref), len(cur)


def test_resumir_marca_solo_la_variable_desplazada(snapshot_y_tamanos):
    snap, n_ref, n_cur = snapshot_y_tamanos
    r = resumir(snap.dict(), n_ref, n_cur, "sintetico")
    assert set(r["variables"]) == {"x", "y"}
    assert r["variables"]["x"]["drift"] is True
    assert r["variables"]["y"]["drift"] is False
    assert r["dataset"]["columnas_con_drift"] == 1
    assert r["dataset"]["proporcion"] == pytest.approx(0.5)
    assert (r["filas_referencia"], r["filas_actual"], r["seed"]) == (n_ref, n_cur, SEED)
    json.dumps(r)  # serializable


@pytest.mark.parametrize(
    "metodo, score, esperado",
    [
        ("Wasserstein distance (normed)", 0.15, True),
        ("Wasserstein distance (normed)", 0.05, False),
        ("Jensen-Shannon distance", 0.1, True),
        ("chi-square p_value", 0.01, True),   # p-valor: drift si es menor al umbral
        ("chi-square p_value", 0.50, False),
    ],
)
def test_hubo_drift_segun_tipo_de_prueba(metodo, score, esperado):
    umbral = 0.05 if "p_value" in metodo else 0.1
    assert _hubo_drift(metodo, score, umbral) is esperado


def test_filtrar_estaciones():
    cur = pd.DataFrame({"estacion_nueva": [False, False, True], "v": [1, 2, 3]})
    assert len(_filtrar_estaciones(cur, "todas")) == 3
    assert _filtrar_estaciones(cur, "existentes")["v"].tolist() == [1, 2]
    assert _filtrar_estaciones(cur, "nuevas")["v"].tolist() == [3]
    with pytest.raises(typer.BadParameter):
        _filtrar_estaciones(cur.drop(columns="estacion_nueva"), "nuevas")


def test_muestrear_es_reproducible_y_respeta_el_tamano():
    df = pd.DataFrame({"a": np.arange(1000)})
    pd.testing.assert_frame_equal(_muestrear(df, 100), _muestrear(df, 100))
    assert len(_muestrear(df, 100)) == 100
    assert _muestrear(df, 0) is df       # 0 = todas
    assert _muestrear(df, 5000) is df    # menos filas que la muestra


def test_cruzar_une_importancia_y_drift(tmp_path):
    imp = pd.DataFrame({"variable": ["a", "b"], "ranking": [1, 2], "caida_f2": [0.4, 0.001]})
    for s, drift_a in (("existentes", False), ("nuevas", True)):
        resumen = {"variables": {
            "a": {"score": 0.3 if drift_a else 0.05, "drift": drift_a},
            "b": {"score": 0.5, "drift": True},
        }}
        (tmp_path / f"drift_summary_{s}.json").write_text(json.dumps(resumen))
    cruce = cruzar(imp, tmp_path).set_index("variable")
    assert bool(cruce.loc["a", "drift_nuevas"]) and not bool(cruce.loc["a", "drift_existentes"])
    assert cruce.loc["b", "score_existentes"] == pytest.approx(0.5)
    assert "score_todas" not in cruce   # sin JSON de `todas`, se omite sin fallar
