"""Pruebas de P1 independientes del CSV completo de MiBici."""

import numpy as np
import pandas as pd

from MiBici.modeling.train_p1 import (
  FEATURE_COLUMNS,
  calcular_metricas,
  construir_panel_y_features,
  dividir_temporal
)


def test_metricas_exactas():
  met = calcular_metricas(pd.Series([10, 20, 30]), np.array([11, 18, 33]))
  assert np.isclose(met["mae"], 2.0)
  assert np.isclose(met["rmse"], np.sqrt(14 / 3))


def test_features_no_usan_el_valor_futuro():
  fechas = pd.date_range("2025-06-01", periods=12, freq="D")
  viajes = list(range(1, 13))
  datos = pd.DataFrame({
    "estacion_id": [100] * 12,
    "fecha": fechas,
    "viajes_iniciados": viajes
  })
  panel, dias_imputados = construir_panel_y_features(datos)
  assert dias_imputados == 0
  assert not panel[FEATURE_COLUMNS].isna().any().any()
  fila = panel.loc[panel["fecha"] == pd.Timestamp("2025-06-08")].iloc[0]
  assert fila["viajes_lag_1"] == 7
  assert fila["viajes_lag_7"] == 1
  assert fila["promedio_7d"] == 4


def test_particion_estrictamente_cronologica():
  df = pd.DataFrame({
    "fecha": pd.to_datetime(["2025-06-30", "2025-07-01", "2025-09-30", "2025-10-01"]),
    "viajes_iniciados": [1, 2, 3, 4]
  })
  train, val, test = dividir_temporal(df)
  assert len(train) == 1
  assert len(val) == 2
  assert len(test) == 1
  assert train["fecha"].max() < val["fecha"].min() < test["fecha"].min()
