"""Política de Continuous Training (`MiBici/monitoring/ct_policy.py`) con entradas sintéticas."""

import pandas as pd
import pytest

from MiBici.monitoring.ct_policy import (
    UMBRALES,
    decidir_promocion,
    decidir_reentrenamiento,
    drift_ponderado,
)


def cruce(drift_existentes: dict, drift_nuevas: dict | None = None) -> pd.DataFrame:
    """Tres variables: `a` concentra casi toda la importancia."""
    df = pd.DataFrame({"variable": ["a", "b", "c"], "caida_f2": [0.40, 0.05, 0.05]})
    df["drift_existentes"] = [drift_existentes.get(v, False) for v in df["variable"]]
    df["drift_nuevas"] = [(drift_nuevas or {}).get(v, False) for v in df["variable"]]
    return df


def metricas(f2_val=0.74, f2_ex=0.73, recall=(0.83, 0.85), filas_nuevas=100, filas_todas=1000) -> dict:
    return {
        "validacion": {"F2 macro": f2_val},
        "actual_existentes": {"F2 macro": f2_ex, "recall": {"vaciado": recall[0], "llenado": recall[1]}},
        "actual_nuevas": {"filas": filas_nuevas},
        "actual_todas": {"filas": filas_todas},
    }


def ab(delta=-0.004, ic=(-0.006, -0.002), recall=(0.03, 0.03), promover=False) -> dict:
    return {
        "diferencias": {"F2 macro": delta, "Recall vaciado": recall[0], "Recall llenado": recall[1]},
        "bootstrap": {"ic95_inferior": ic[0], "ic95_superior": ic[1]},
        "decision": {"promover_challenger": promover},
        "versiones": {},
    }


def test_drift_ponderado_cuenta_importancia_no_variables():
    # 2 de 3 variables con drift, pero son las menos importantes
    assert drift_ponderado(cruce({"b": True, "c": True}), "existentes") == pytest.approx(0.2)
    assert drift_ponderado(cruce({"a": True}), "existentes") == pytest.approx(0.8)


def test_sin_disparadores_no_reentrena(monkeypatch):
    monkeypatch.setattr("MiBici.monitoring.ct_policy.meses_entre", lambda *_: 3)
    r = decidir_reentrenamiento(cruce({"b": True}), metricas())
    assert r == r | {"reentrenar": False, "urgencia": "ninguna"}


@pytest.mark.parametrize("args, disparador", [
    ({"f2_ex": 0.70}, "caida_f2"),
    ({"recall": (0.75, 0.85)}, "recall_min"),
])
def test_degradacion_es_urgente(monkeypatch, args, disparador):
    monkeypatch.setattr("MiBici.monitoring.ct_policy.meses_entre", lambda *_: 3)
    r = decidir_reentrenamiento(cruce({}), metricas(**args))
    assert r["urgencia"] == "urgente"
    assert r["disparadores_urgentes"][disparador]["activo"]


def test_drift_en_variable_importante_es_urgente(monkeypatch):
    monkeypatch.setattr("MiBici.monitoring.ct_policy.meses_entre", lambda *_: 3)
    assert decidir_reentrenamiento(cruce({"a": True}), metricas())["urgencia"] == "urgente"


def test_estaciones_nuevas_o_antiguedad_son_programado(monkeypatch):
    monkeypatch.setattr("MiBici.monitoring.ct_policy.meses_entre", lambda *_: 3)
    r = decidir_reentrenamiento(cruce({}, {"a": True}), metricas(filas_nuevas=300))
    assert r["urgencia"] == "programado"

    monkeypatch.setattr("MiBici.monitoring.ct_policy.meses_entre", lambda *_: UMBRALES["antiguedad_max_meses"] + 1)
    assert decidir_reentrenamiento(cruce({}), metricas())["urgencia"] == "programado"


def test_promocion_requiere_los_tres_criterios():
    assert decidir_promocion(ab(delta=0.01, ic=(0.002, 0.02), promover=True))["promover_challenger"]
    assert not decidir_promocion(ab())["promover_challenger"]                                    # F2 peor
    assert not decidir_promocion(ab(delta=0.01, ic=(-0.001, 0.02)))["promover_challenger"]       # IC incluye 0
    assert not decidir_promocion(ab(delta=0.01, ic=(0.002, 0.02), recall=(-0.05, 0.0)))["promover_challenger"]
