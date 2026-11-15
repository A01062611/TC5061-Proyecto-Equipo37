"""Pruebas para las estadísticas del A/B de P2."""

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from MiBici.modeling.train_p2 import f2_macro
from MiBici.monitoring.ab_statistics import (
    f2_desde_matriz,
    bootstrap_pareado,
    prueba_mcnemar,
)


@pytest.fixture
def datos():
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    a = np.array([0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 2])
    b = np.array([0, 1, 1, 1, 2, 0, 0, 0, 2, 0, 1, 2])
    estaciones = np.repeat(["A", "B", "C"], 4)
    return y, a, b, estaciones


def test_f2_coincide_con_sklearn(datos):
    y, a, _, _ = datos
    matriz = confusion_matrix(y, a, labels=[0, 1, 2])
    obtenido = f2_desde_matriz(matriz)

    assert obtenido == pytest.approx(f2_macro(y, a))


def test_bootstrap_reproducible(datos):
    y, a, b, estaciones = datos

    primero = bootstrap_pareado(
        y, a, b, estaciones, n_bootstrap=100, seed=42
    )
    segundo = bootstrap_pareado(
        y, a, b, estaciones, n_bootstrap=100, seed=42
    )

    assert primero == segundo
    assert primero["n_estaciones"] == 3
    assert primero["ic95_inferior"] <= primero["ic95_superior"]


def test_bootstrap_modelos_identicos(datos):
    y, a, _, estaciones = datos

    resultado = bootstrap_pareado(
        y, a, a, estaciones, n_bootstrap=100, seed=42
    )

    assert resultado["delta_f2"] == pytest.approx(0.0)
    assert resultado["ic95_inferior"] == pytest.approx(0.0)
    assert resultado["ic95_superior"] == pytest.approx(0.0)


def test_mcnemar_discordantes(datos):
    y, a, b, _ = datos
    resultado = prueba_mcnemar(y, a, b)

    assert resultado["solo_champion_acierta"] == 3
    assert resultado["solo_challenger_acierta"] == 2
    assert resultado["discordantes"] == 5


def test_mcnemar_modelos_identicos(datos):
    y, a, _, _ = datos
    resultado = prueba_mcnemar(y, a, a)

    assert resultado["discordantes"] == 0
    assert resultado["p_valor"] == pytest.approx(1.0)


def test_bootstrap_rechaza_longitudes_distintas(datos):
    y, a, b, estaciones = datos

    with pytest.raises(ValueError):
        bootstrap_pareado(
            y[:-1], a, b, estaciones,
            n_bootstrap=100,
        )
