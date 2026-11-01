"""Pruebas rápidas de reglas globales sin requerir el CSV real."""
import pandas as pd
import pytest

from scripts.validate_data import COLUMNAS, analizar_bloque


def _filas() -> pd.DataFrame:
    return pd.DataFrame([
        ["1", "10", "M", "1990", "2025-01-01 08:00:00", "2025-01-01 08:15:00", "101", "102"],
        ["2", "11", "", "", "2025-01-02 08:00:00", "2025-01-02 08:20:00", "102", "103"],
    ], columns=COLUMNAS)


def test_bloque_normal_y_faltantes():
    r = analizar_bloque(_filas())
    assert r["registros"] == 2
    assert r["faltantes"]["Genero"] == 1
    assert r["faltantes"]["Anio_de_nacimiento"] == 1
    assert r["inicio_posterior_fin"] == 0


def test_fechas_e_identificadores_invalidos():
    df = _filas()
    df.loc[0, "Inicio_del_viaje"] = "2025-02-30 08:00:00"
    df.loc[1, "Origen_Id"] = "ABC"
    r = analizar_bloque(df)
    assert r["invalidos"]["Inicio_del_viaje"] == 1
    assert r["invalidos"]["Origen_Id"] == 1


def test_esquema_incorrecto():
    with pytest.raises(ValueError):
        analizar_bloque(_filas().drop(columns=["Destino_Id"]))


def test_inicio_posterior_fin():
    df = _filas()
    df.loc[0, "Fin_del_viaje"] = "2025-01-01 07:00:00"
    assert analizar_bloque(df)["inicio_posterior_fin"] == 1
