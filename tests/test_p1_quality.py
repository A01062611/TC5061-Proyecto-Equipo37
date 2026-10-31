"""Pruebas de calidad para P1 (MiBici), sin depender del dataset real."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from MiBici.config import SEED
from MiBici.modeling.train_p1 import (
  FEATURE_COLUMNS,
  calcular_metricas,
  construir_panel_y_features,
  dividir_temporal,
  leer_y_agregar_viajes
)


@pytest.fixture
def viajes_csv(tmp_path):
  """CSV sintético válido con un viaje diario entre 2024 y 2025."""
  fechas = pd.date_range("2024-01-01", "2025-12-31", freq="D")
  datos = pd.DataFrame({
    "Viaje_Id": np.arange(1, len(fechas) + 1),
    "Inicio_del_viaje": fechas.strftime("%Y-%m-%d %H:%M:%S"),
    "Origen_Id": 100
  })
  ruta = tmp_path / "viajes_prueba.csv"
  datos.to_csv(ruta, index=False)
  return ruta


@pytest.fixture
def panel_diario():
  """Dos estaciones con 35 días de viajes cada una."""
  fechas = pd.date_range("2025-05-01", periods=35, freq="D")
  filas = []
  for estacion in (100, 200):
    for i, fecha in enumerate(fechas):
      filas.append({
        "estacion_id": estacion,
        "fecha": fecha,
        "viajes_iniciados": 3 + (i * 3 + estacion // 100) % 12
      })
  return pd.DataFrame(filas)


def test_agregacion_con_csv_valido(viajes_csv):
  diario, total, sha256 = leer_y_agregar_viajes(viajes_csv)

  assert total == 731  # 2024 es año bisiesto; 2025 tiene 365 días.
  assert len(diario) == 731
  assert diario["viajes_iniciados"].eq(1).all()
  assert diario["estacion_id"].eq(100).all()
  assert sha256 == hashlib.sha256(viajes_csv.read_bytes()).hexdigest()


@pytest.mark.parametrize(
  ("columna", "valor"),
  [
    ("Inicio_del_viaje", "fecha-invalida"),
    ("Origen_Id", -5),
    ("Viaje_Id", -1)
  ]
)
def test_agregacion_rechaza_datos_invalidos(viajes_csv, columna, valor):
  datos = pd.read_csv(viajes_csv)
  datos.loc[0, columna] = valor
  datos.to_csv(viajes_csv, index=False)

  with pytest.raises(ValueError):
    leer_y_agregar_viajes(viajes_csv)


def test_agregacion_rechaza_columnas_faltantes(viajes_csv):
  datos = pd.read_csv(viajes_csv).drop(columns=["Origen_Id"])
  datos.to_csv(viajes_csv, index=False)

  with pytest.raises(ValueError):
    leer_y_agregar_viajes(viajes_csv)


def test_agregacion_rechaza_ids_duplicados(viajes_csv):
  datos = pd.read_csv(viajes_csv)
  datos.loc[1, "Viaje_Id"] = datos.loc[0, "Viaje_Id"]
  datos.to_csv(viajes_csv, index=False)

  with pytest.raises(ValueError):
    leer_y_agregar_viajes(viajes_csv)


def test_features_imputan_dia_ausente_sin_fuga_temporal():
  fechas = pd.date_range("2025-06-01", periods=12, freq="D")
  datos = pd.DataFrame({
    "estacion_id": 100,
    "fecha": fechas,
    "viajes_iniciados": np.arange(1, 13)
  })
  datos = datos[datos["fecha"] != pd.Timestamp("2025-06-05")]

  panel, dias_imputados = construir_panel_y_features(datos)
  fila = panel.loc[panel["fecha"] == "2025-06-08"].iloc[0]

  assert dias_imputados == 1
  assert fila["viajes_lag_1"] == 7
  assert fila["viajes_lag_7"] == 1
  assert fila["promedio_7d"] == pytest.approx(23 / 7)
  assert not panel[FEATURE_COLUMNS].isna().any().any()


def test_features_no_mezclan_estaciones(panel_diario):
  panel, dias_imputados = construir_panel_y_features(panel_diario)

  assert dias_imputados == 0
  assert panel["estacion_id"].nunique() == 2
  for estacion in (100, 200):
    original = panel_diario[panel_diario["estacion_id"] == estacion]
    esperado = original.iloc[6]["viajes_iniciados"]
    fila = panel[
      (panel["estacion_id"] == estacion)
      & (panel["fecha"] == pd.Timestamp("2025-05-08"))
    ].iloc[0]
    assert fila["viajes_lag_1"] == esperado


def test_division_temporal_rechaza_conjunto_vacio():
  datos = pd.DataFrame({
    "fecha": pd.to_datetime(["2025-06-30", "2025-07-01"]),
    "viajes_iniciados": [5, 6]
  })

  with pytest.raises(ValueError, match="vacío"):
    dividir_temporal(datos)


def test_metricas_rechazan_longitudes_diferentes():
  with pytest.raises(ValueError):
    calcular_metricas(pd.Series([3, 5, 7]), np.array([3, 5]))


def test_random_forest_p1_es_reproducible(panel_diario):
  panel, _ = construir_panel_y_features(panel_diario)
  X = panel[FEATURE_COLUMNS]
  y = panel["viajes_iniciados"]
  parametros = {
    "n_estimators": 20,
    "max_depth": 5,
    "min_samples_leaf": 1,
    "random_state": SEED,
    "n_jobs": 1
  }

  modelo_1 = RandomForestRegressor(**parametros)
  modelo_2 = RandomForestRegressor(**parametros)
  modelo_1.fit(X, y)
  modelo_2.fit(X, y)

  pred_1 = modelo_1.predict(X)
  pred_2 = modelo_2.predict(X)

  np.testing.assert_array_equal(pred_1, pred_2)
  assert calcular_metricas(y, pred_1) == calcular_metricas(y, pred_2)