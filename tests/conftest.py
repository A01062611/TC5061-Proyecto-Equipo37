"""Fixtures compartidas: muestras sintéticas y chicas de viajes (P2 y P3), sin depender de `data/`."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from MiBici.features import (
    PARAMS_P2,
    construir_dataset_p2,
    features_de_p2,
    filter_valid_trips,
    n_service_days,
    station_hourly_counts,
    station_usage_profile,
    station_usage_summary,
)

SEED = 42
N_PER_ARCHETYPE = 4  # estaciones por tipo de uso
TRIPS_PER_STATION = 80

# Horas de salida por tipo de uso; el recreativo viaja solo en fin de semana
ARCHETYPES = {
    "residencial": range(6, 10),
    "laboral": range(17, 21),
    "recreativo": range(11, 16),
}


def _synthetic_trips(rng: np.random.Generator) -> pd.DataFrame:
    """Viajes entre estaciones con tres ritmos de uso distintos (14 días desde un lunes)."""
    rows = []
    station_id = 1
    for name, hours in ARCHETYPES.items():
        for _ in range(N_PER_ARCHETYPE):
            for _ in range(TRIPS_PER_STATION):
                weekend = name == "recreativo"
                day = rng.integers(0, 14)
                while ((pd.Timestamp("2025-06-02") + pd.Timedelta(days=int(day))).dayofweek >= 5) != weekend:
                    day = rng.integers(0, 14)
                start = (
                    pd.Timestamp("2025-06-02")
                    + pd.Timedelta(days=int(day), hours=int(rng.choice(list(hours))), minutes=int(rng.integers(0, 50)))
                )
                dest = int(rng.integers(100, 103))  # destinos compartidos para no sesgar el perfil
                rows.append((start, start + pd.Timedelta(minutes=int(rng.integers(5, 40))), station_id, dest))
            station_id += 1
    return pd.DataFrame(rows, columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"])


@pytest.fixture(scope="session")
def trips_sample() -> pd.DataFrame:
    return _synthetic_trips(np.random.default_rng(SEED))


@pytest.fixture(scope="session")
def profile(trips_sample) -> pd.DataFrame:
    """Perfil de uso por estación, igual al que consume `train_p3`."""
    trips = filter_valid_trips(trips_sample)
    counts = station_hourly_counts(trips)
    origenes = counts.index[counts.index <= N_PER_ARCHETYPE * len(ARCHETYPES)]
    counts = counts.loc[origenes]
    return station_usage_profile(counts, station_usage_summary(counts, n_service_days(trips)))


@pytest.fixture(scope="session")
def X_std(profile) -> np.ndarray:
    return StandardScaler().fit_transform(profile)


# --- P2 ---
# Cuatro semanas, 18 días de train, 5 de val y 5 de test
PARAMS_P2_PRUEBA = PARAMS_P2 | {
    "fecha_inicio": "2025-04-28",
    "fecha_fin": "2025-05-25",
    "train_fin": "2025-05-15",
    "val_fin": "2025-05-20",
    "min_dias_s0": 5,
}


def _viajes_p2(rng: np.random.Generator) -> pd.DataFrame:
    """Las estaciones 1 y 2 se vacían en la mañana (sus bicis van a 3 y 4, que se llenan) y se
    recuperan en la tarde; 5 y 6 tienen poco uso y casi siempre quedan en clase normal."""
    rows = []
    for dia in pd.date_range(PARAMS_P2_PRUEBA["fecha_inicio"], PARAMS_P2_PRUEBA["fecha_fin"]):
        # Estacion 1,2,3,4: 14 viajes por día, 7-10 y 17-20
        for _ in range(14):
            casa, trabajo = int(rng.choice([1, 2])), int(rng.choice([3, 4]))
            for horas, o, d in [(range(7, 10), casa, trabajo), (range(17, 20), trabajo, casa)]:
                inicio = dia + pd.Timedelta(hours=int(rng.choice(horas)), minutes=int(rng.integers(0, 60)))
                rows.append((inicio, inicio + pd.Timedelta(minutes=int(rng.integers(5, 30))), o, d))
        # Estacion 5,6: 6 viajes por día, horarios aleatorios
        for _ in range(6):
            o, d = (int(e) for e in rng.choice([5, 6], 2, replace=False))
            inicio = dia + pd.Timedelta(hours=int(rng.integers(6, 23)), minutes=int(rng.integers(0, 60)))
            rows.append((inicio, inicio + pd.Timedelta(minutes=int(rng.integers(5, 30))), o, d))
    trips = pd.DataFrame(rows, columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"])
    trips.insert(0, "Viaje_Id", np.arange(1, len(trips) + 1))
    return trips


@pytest.fixture(scope="session")
def p2_params() -> dict:
    return PARAMS_P2_PRUEBA


@pytest.fixture(scope="session")
def p2_csvs(tmp_path_factory) -> tuple:
    """CSV de viajes y de estaciones con el mismo formato que `data/interim/`."""
    carpeta = tmp_path_factory.mktemp("p2")
    trips_path, stations_path = carpeta / "viajes.csv", carpeta / "estaciones.csv"
    _viajes_p2(np.random.default_rng(SEED)).to_csv(trips_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    pd.DataFrame({
        "id": range(1, 7),
        "name": [f"Estación {i}" for i in range(1, 7)],
        "latitude": 20.66 + np.arange(6) / 100,
        "longitude": -103.35 - np.arange(6) / 100,
        "dpcapacity": [10, 10, 8, 8, 10, 10],
    }).to_csv(stations_path, index=False)
    return trips_path, stations_path


@pytest.fixture(scope="session")
def p2_dataset(p2_csvs) -> tuple:
    """Salida de `construir_dataset_p2` sobre los CSV sintéticos."""
    return construir_dataset_p2(*p2_csvs, PARAMS_P2_PRUEBA)


@pytest.fixture(scope="session")
def p2_split(p2_dataset) -> tuple:
    """train, val, y_train, y_val y features, igual que `load_features` de `train_p2`."""
    tabla = p2_dataset[0]
    train = tabla[tabla["particion"] == "train"]
    val = tabla[tabla["particion"] == "val"]
    return train, val, train["clase"].to_numpy(), val["clase"].to_numpy(), features_de_p2(tabla)
