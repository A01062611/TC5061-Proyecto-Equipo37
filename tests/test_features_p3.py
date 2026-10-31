"""Pruebas de las funciones de features de P3 (perfiles de uso por estación)."""

import numpy as np
import pandas as pd

from MiBici.features import (
    filter_valid_trips,
    n_service_days,
    station_hourly_counts,
    station_usage_summary,
)


def _viaje(inicio, fin, origen=1, destino=2):
    return {"Inicio_del_viaje": pd.Timestamp(inicio), "Fin_del_viaje": pd.Timestamp(fin),
            "Origen_Id": origen, "Destino_Id": destino}


def test_filter_valid_trips_quita_duraciones_fuera_de_rango():
    trips = pd.DataFrame([
        _viaje("2025-06-02 08:00", "2025-06-02 08:20"),  # válido
        _viaje("2025-06-02 08:00", "2025-06-02 08:00"),  # 0 min
        _viaje("2025-06-02 08:00", "2025-06-02 12:00"),  # 240 min
    ])
    assert len(filter_valid_trips(trips)) == 1


def test_filter_valid_trips_limites_son_estrictos():
    trips = pd.DataFrame([
        _viaje("2025-06-02 08:00", "2025-06-02 08:01"),  # exactamente 1 min
        _viaje("2025-06-02 08:00", "2025-06-02 11:00"),  # exactamente 180 min
    ])
    assert filter_valid_trips(trips).empty


def test_filter_valid_trips_quita_viajes_circulares():
    trips = pd.DataFrame([_viaje("2025-06-02 08:00", "2025-06-02 08:20", origen=5, destino=5)])
    assert filter_valid_trips(trips).empty
    assert len(filter_valid_trips(trips, drop_round_trips=False)) == 1


def test_filter_valid_trips_dataset_vacio():
    trips = pd.DataFrame({
        "Inicio_del_viaje": pd.to_datetime([]), "Fin_del_viaje": pd.to_datetime([]),
        "Origen_Id": [], "Destino_Id": [],
    })
    assert filter_valid_trips(trips).empty


def test_conteo_horario_tiene_96_columnas_y_conserva_viajes(trips_sample):
    counts = station_hourly_counts(trips_sample)
    assert counts.shape[1] == 96
    assert counts.sum().sum() == 2 * len(trips_sample)  # una salida y una llegada por viaje


def test_conteo_horario_separa_semana_y_finde():
    # 2025-06-07 es sábado
    trips = pd.DataFrame([_viaje("2025-06-07 10:05", "2025-06-07 10:25")])
    counts = station_hourly_counts(trips)
    assert counts.loc[1, "sal_finde_10"] == 1
    assert counts.loc[2, "lle_finde_10"] == 1
    assert counts.filter(like="semana").sum().sum() == 0


def test_resumen_balance_en_los_extremos():
    trips = pd.DataFrame([_viaje("2025-06-02 08:00", "2025-06-02 08:20")])
    resumen = station_usage_summary(station_hourly_counts(trips), n_dias=1)
    assert resumen.loc[1, "balance"] == -1  # solo salidas
    assert resumen.loc[2, "balance"] == 1  # solo llegadas


def test_n_service_days_cuenta_dias_distintos():
    trips = pd.DataFrame([
        _viaje("2025-06-02 08:00", "2025-06-02 08:20"),
        _viaje("2025-06-02 18:00", "2025-06-02 18:20"),
        _viaje("2025-06-03 08:00", "2025-06-03 08:20"),
    ])
    assert n_service_days(trips) == 2


def test_perfil_normaliza_cada_bloque_horario(profile):
    for bloque in ("sal_semana", "lle_semana", "sal_finde", "lle_finde"):
        sumas = profile.filter(like=bloque).sum(axis=1)
        assert np.allclose(sumas[sumas > 0], 1.0)


def test_perfil_sin_nulos(profile):
    assert not profile.isna().any().any()
