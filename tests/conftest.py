"""Fixtures compartidas: una muestra sintética y chica de viajes, sin depender de `data/`."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from MiBici.features import (
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
