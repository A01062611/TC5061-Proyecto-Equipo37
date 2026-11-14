import pandas as pd
import pytest

from MiBici.modeling.challenger_p2 import preparar_challenger


FEATURES = ["hora", "ocup_est", "estado_est"]


def crear_datos(fechas, periodo="retrain"):
  return pd.DataFrame({
    "estacion": [1] * len(fechas),
    "dia": pd.to_datetime(fechas),
    "clase": [i % 3 for i in range(len(fechas))],
    "hora": [8] * len(fechas),
    "ocup_est": [0.2] * len(fechas),
    "estado_est": [0] * len(fechas),
    "periodo": [periodo] * len(fechas),
  })


@pytest.fixture
def conjuntos():
  historico = crear_datos(["2024-01-02", "2025-06-30"])
  retrain = crear_datos([
    "2026-07-10",
    "2026-07-20",
    "2026-08-05",
    "2026-08-25",
  ])
  return historico, retrain


def test_preparar_challenger(conjuntos):
  historico, retrain = conjuntos

  train, val, final = preparar_challenger(
    historico, retrain, FEATURES, "2026-08-31"
  )

  assert len(train) == 4
  assert len(val) == 2
  assert len(final) == 6
  assert train["dia"].max() < val["dia"].min()
  assert final["dia"].max() <= pd.Timestamp("2026-08-31")


def test_rechaza_datos_de_septiembre(conjuntos):
  historico, retrain = conjuntos

  septiembre = crear_datos(["2026-09-01"])
  datos = pd.concat([retrain, septiembre], ignore_index=True)

  with pytest.raises(ValueError, match="Fechas fuera"):
    preparar_challenger(historico, datos, FEATURES, "2026-08-31")


def test_rechaza_periodo_incorrecto(conjuntos):
  historico, retrain = conjuntos
  retrain.loc[0, "periodo"] = "ab_eval"

  with pytest.raises(ValueError, match="ventana"):
    preparar_challenger(historico, retrain, FEATURES, "2026-08-31")


def test_rechaza_fecha_de_corte_invalida(conjuntos):
  historico, retrain = conjuntos

  with pytest.raises(ValueError, match="train_fin"):
    preparar_challenger(historico, retrain, FEATURES, "2026-09-30")