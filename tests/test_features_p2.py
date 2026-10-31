"""Pruebas del feature engineering de P2 (inventario reconstruido y etiqueta de riesgo)."""

import numpy as np
import pandas as pd
import pytest

from MiBici.features import (
    CLASES_P2,
    NO_FEATURES_P2,
    PARAMS_P2,
    calcular_cotas,
    calcular_s0,
    construir_dataset_p2,
    construir_eventos,
    features_de_p2,
    indices_malla,
    limpiar_viajes_p2,
)

ESTACIONES = pd.DataFrame({"id": [1, 2, 3]})


def _viajes(*filas):
    """Filas (inicio, fin, origen, destino) → DataFrame con las columnas de P2."""
    df = pd.DataFrame(filas, columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"])
    for col in ["Inicio_del_viaje", "Fin_del_viaje"]:
        df[col] = pd.to_datetime(df[col])
    return df

def _eventos(deltas, estacion=1, dia="2025-06-02", sd=0):
    acum = np.cumsum(deltas)
    return pd.DataFrame({"sd": sd, "estacion": estacion, "dia": pd.Timestamp(dia), "delta": deltas, "acum": acum})

def _c_train(cotas_LU, C=6, estacion=1):
    return pd.DataFrame({"estacion": estacion, "C": C, "L": [l for l, _ in cotas_LU], "U": [u for _, u in cotas_LU]})


# --- limpieza ---------------------------------------------------------------------------

def test_limpiar_viajes_p2_aplica_cada_regla():
    trips = _viajes(
        ("2025-06-02 08:00:00", "2025-06-02 08:20:00", 1, 2),  # válido
        ("2025-06-02 08:00:00", "2025-06-02 07:59:00", 1, 2),  # duración negativa
        ("2025-06-02 08:00:00", "2025-06-02 08:00:00", 1, 2),  # duración cero
        ("2025-06-02 08:00:00", "2025-06-02 08:00:30", 1, 2),  # < 1 min entre estaciones distintas
        ("2025-06-02 08:00:00", "2025-06-03 09:00:00", 1, 2),  # más de 24 h
        ("2025-06-02 08:00:00", "2025-06-02 08:20:00", 1, 99), # estación fuera del catálogo
    )
    limpios, conteo = limpiar_viajes_p2(trips, ESTACIONES, dur_max_h=24)
    assert len(limpios) == 1
    assert conteo == {
        "duracion_negativa": 1, "duracion_cero": 1, "menos_1min_entre_estaciones": 1,
        "mas_dur_max": 1, "estacion_fuera_catalogo": 1, "total_eliminado": 5,
    }

def test_limpiar_viajes_p2_conserva_casos_en_el_limite():
    # < 1 min, pero en la misma estación o exactamente 24 h
    trips = _viajes(
        ("2025-06-02 08:00:00", "2025-06-02 08:00:30", 1, 1),  
        ("2025-06-02 08:00:00", "2025-06-03 08:00:00", 1, 2),  
    )
    limpios, conteo = limpiar_viajes_p2(trips, ESTACIONES, dur_max_h=24)
    assert len(limpios) == 2
    assert conteo["total_eliminado"] == 0

# --- eventos y cotas --------------------------------------------------------------------

def test_construir_eventos_neto_por_segundo_y_dia_operativo():
    trips = _viajes(
        ("2025-06-02 08:00:00", "2025-06-02 08:10:00", 1, 2),
        ("2025-06-02 07:50:00", "2025-06-02 08:10:00", 3, 2),  # llega a 2 en el mismo segundo
        ("2025-06-03 00:30:00", "2025-06-03 00:50:00", 1, 3),  # después de medianoche: día 2
        ("2025-06-03 05:00:00", "2025-06-03 05:10:00", 1, 3),  # al reinicio: día 3
    )
    ev = construir_eventos(trips, PARAMS_P2 | {"fecha_inicio": "2025-06-01", "fecha_fin": "2025-06-30"})

    llegada_doble = ev[(ev["estacion"] == 2) & (ev["ts"] == pd.Timestamp("2025-06-02 08:10:00"))]
    assert len(llegada_doble) == 1
    assert llegada_doble[["llegada", "salida", "delta"]].iloc[0].tolist() == [2, 0, 2]

    salidas_1 = ev[(ev["estacion"] == 1) & (ev["salida"] == 1)].set_index("ts")["dia"]
    assert salidas_1[pd.Timestamp("2025-06-03 00:30:00")] == pd.Timestamp("2025-06-02")
    assert salidas_1[pd.Timestamp("2025-06-03 05:00:00")] == pd.Timestamp("2025-06-03")

    # cada viaje genera una salida y una llegada
    assert ev["salida"].sum() == ev["llegada"].sum() == len(trips)

def test_calcular_cotas_dia_factible_e_infactible():
    ev = pd.concat([
        # mínimo −2 y máximo +2 → necesita al menos 2 bicis y caben a lo más 3
        _eventos([-2, 4], estacion=1, sd=0),
        # necesita 4 bicis al inicio, pero luego llegan 6: no cabe ningún inventario
        _eventos([-4, 6], estacion=2, sd=1),
    ])
    cotas = calcular_cotas(ev, pd.Series({1: 5, 2: 5}), PARAMS_P2)
    assert (cotas.loc[0, "L"], cotas.loc[0, "U"], cotas.loc[0, "infactible"]) == (2, 3, False)
    assert (cotas.loc[1, "L"], cotas.loc[1, "U"], cotas.loc[1, "infactible"]) == (4, 3, True)

# --- S0 ---------------------------------------------------------------------------------

def test_calcular_s0_elige_el_mas_factible_cercano_a_la_mitad():
    # s = 4 y s = 5 son factibles los dos días; gana 4 por estar más cerca de C/2 = 3
    s0 = calcular_s0(_c_train([(3, 5), (4, 6)]), [1], pd.Series({1: 6}), min_dias=2)
    assert s0.loc[1, "s0"] == 4
    assert s0.loc[1, "metodo"] == "max_factible"
    assert s0.loc[1, "pct_factible_s0"] == 1.0

def test_calcular_s0_estacion_sin_dias_de_train():
    s0 = calcular_s0(_c_train([(3, 5)]), [1, 9], pd.Series({1: 6, 9: 11}), min_dias=1)
    assert s0.loc[9, "s0"] == 5
    assert s0.loc[9, "dias_train"] == 0
    assert s0.loc[9, "metodo"] == "fallback_cs2"

def test_indices_malla_ventana_fuera_del_dia_falla():
    with pytest.raises(ValueError, match="ventana"):
        indices_malla(PARAMS_P2 | {"n_horas_prediccion": 24})


# --- tabla completa ---------------------------------------------------------------------

def test_tabla_p2_una_fila_por_estacion_dia_y_hora(p2_dataset):
    tabla, _, info = p2_dataset
    assert len(tabla) == info["filas_tabla"]
    assert not tabla.duplicated(["estacion", "dia", "hora"]).any()
    assert (tabla.groupby(["estacion", "dia"]).size() == PARAMS_P2["n_horas_prediccion"]).all()
    assert set(tabla["hora"]) == {*range(5, 24), 0}

def test_tabla_p2_etiqueta_y_particiones_validas(p2_dataset):
    tabla, _, _ = p2_dataset
    assert set(tabla["clase"]) == set(range(len(CLASES_P2)))
    assert set(tabla["particion"]) == {"train", "val", "test"}
    # partición temporal: todo train antes de val y todo val antes de test
    ultimo = tabla.groupby("particion", observed=True)["dia"].agg(["min", "max"])
    assert ultimo.loc["train", "max"] < ultimo.loc["val", "min"]
    assert ultimo.loc["val", "max"] < ultimo.loc["test", "min"]

def test_tabla_p2_primeros_dias_sin_historial(p2_dataset):
    # sin días anteriores, los lags de hace una semana quedan como NaN y no como 0
    tabla, _, _ = p2_dataset
    primer_dia = tabla[tabla["dia"] == tabla["dia"].min()]
    assert primer_dia["sal_sem_2h"].isna().all()
    assert primer_dia["riesgo_ayer"].isna().all()

def test_construir_dataset_p2_sin_viajes_da_tabla_vacia(tmp_path, p2_csvs, p2_params, p2_dataset):
    _, stations_path = p2_csvs
    vacio = tmp_path / "vacio.csv"
    pd.DataFrame(columns=["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]).to_csv(
        vacio, index=False)
    tabla, _, info = construir_dataset_p2(vacio, stations_path, p2_params)
    assert tabla.empty
    assert info["filas_tabla"] == 0
    # mismas columnas que con datos, para que el entrenamiento falle con un mensaje claro y no por un KeyError
    assert tabla.columns.tolist() == p2_dataset[0].columns.tolist()
