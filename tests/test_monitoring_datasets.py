"""Conjuntos de monitoreo de P2 (`MiBici/monitoring/datasets.py`).

- Siempre corren (datos sinteticos): construir la tabla con un ŝ₀ externo respeta ese ŝ₀
  y solo completa las estaciones que faltan.
- Corren solo si existen los parquet generados (se generan con `python -m MiBici.monitoring.datasets`):
  mismo esquema en los cuatro conjuntos, fechas dentro de su periodo, ventanas sin traslape 
  y features iguales a la firma del champion.
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from mlflow.models import Model

from MiBici.config import MONITORING_DATA_DIR, TRAIN_FIN
from MiBici.features import completar_s0, construir_dataset_p2, features_de_p2
from MiBici.monitoring.datasets import (
    ACTUAL_DESDE,
    ACTUAL_HASTA,
    AB_DESDE,
    COLUMNAS_ID,
    COLUMNAS_PRED,
    MODELO_EMPACADO,
    RETRAIN_HASTA,
)

CONJUNTOS = ["reference", "current", "retrain", "ab_eval"]
PERIODOS = {  # conjunto -> (desde, hasta, valores de `periodo`)
    "reference": (None, TRAIN_FIN, {"reference"}),
    "current": (ACTUAL_DESDE, ACTUAL_HASTA, {"retrain", "ab_eval"}),
    "retrain": (ACTUAL_DESDE, RETRAIN_HASTA, {"retrain"}),
    "ab_eval": (AB_DESDE, ACTUAL_HASTA, {"ab_eval"}),
}


# --------------------------------------------------------------------------- ŝ₀ externo (sintético)

def test_completar_s0_solo_agrega_las_estaciones_que_faltan():
    s0 = pd.DataFrame({"C": [10, 8], "s0": [3, 6], "metodo": "max_factible"}, index=pd.Index([1, 2], name="estacion"))
    capacidad = pd.Series({1: 10, 2: 8, 3: 9})
    out = completar_s0(s0, [1, 2, 3], capacidad)

    assert out.loc[[1, 2], "s0"].tolist() == [3, 6]
    assert out.loc[3, "s0"] == 9 // 2
    assert out.loc[3, "metodo"] == "fallback_cs2"
    assert completar_s0(s0, [1, 2], capacidad) is s0


def test_tabla_con_s0_externo_igual_a_la_original_en_estaciones_conocidas(p2_csvs, p2_params, p2_dataset):
    """Si el ŝ₀ externo es el mismo, las filas de esas estaciones no cambian; las demás usan ⌊C/2⌋."""
    tabla, s0_est, _ = p2_dataset
    conocidas = [1, 2, 3, 4]
    tabla2, s0_2, info2 = construir_dataset_p2(*p2_csvs, p2_params, s0_est=s0_est.loc[conocidas])

    assert "proxy" not in info2
    assert s0_2.loc[conocidas, "s0"].tolist() == s0_est.loc[conocidas, "s0"].tolist()
    assert (s0_2.loc[[5, 6], "s0"] == s0_2.loc[[5, 6], "C"] // 2).all()

    a = tabla[tabla["estacion"].isin(conocidas)].reset_index(drop=True)
    b = tabla2[tabla2["estacion"].isin(conocidas)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- parquet generados

faltan_parquet = not all((MONITORING_DATA_DIR / f"{c}.parquet").exists() for c in CONJUNTOS)
requiere_parquet = pytest.mark.skipif(faltan_parquet, reason="corre antes `python -m MiBici.monitoring.datasets`")


def leer(conjunto: str, columnas: list[str]) -> pd.DataFrame:
    return pd.read_parquet(MONITORING_DATA_DIR / f"{conjunto}.parquet", columns=columnas)


@requiere_parquet
def test_mismo_esquema_en_los_cuatro_conjuntos():
    esquemas = {c: pq.read_schema(MONITORING_DATA_DIR / f"{c}.parquet").remove_metadata() for c in CONJUNTOS}
    for c in CONJUNTOS[1:]:
        assert esquemas[c].equals(esquemas["reference"]), f"{c} no tiene el esquema de reference"


@requiere_parquet
def test_columnas_y_features():
    columnas = pq.read_schema(MONITORING_DATA_DIR / "reference.parquet").names
    features = features_de_p2(pd.DataFrame(columns=columnas))
    assert columnas[:len(COLUMNAS_ID)] == COLUMNAS_ID
    assert columnas[-len(COLUMNAS_PRED):] == COLUMNAS_PRED
    assert len(features) == 23
    assert json.loads((MONITORING_DATA_DIR / "info.json").read_text())["features"] == features


@requiere_parquet
@pytest.mark.skipif(not (MODELO_EMPACADO / "MLmodel").is_file(), reason="no hay modelo empacado en models/")
def test_features_iguales_a_la_firma_del_champion():

    # Se lee el MLmodel local
    firma = Model.load(MODELO_EMPACADO / "MLmodel").signature.inputs.input_names()
    columnas = pq.read_schema(MONITORING_DATA_DIR / "reference.parquet").names
    assert sorted(firma) == sorted(features_de_p2(pd.DataFrame(columns=columnas)))


@requiere_parquet
@pytest.mark.parametrize("conjunto", CONJUNTOS)
def test_fechas_y_periodo(conjunto):
    df = leer(conjunto, ["dia", "periodo"])
    desde, hasta, periodos = PERIODOS[conjunto]
    if desde is not None:
        assert df["dia"].min() >= pd.Timestamp(desde)
    assert df["dia"].max() <= pd.Timestamp(hasta)
    assert set(df["periodo"].unique()) == periodos


@requiere_parquet
def test_ventanas_sin_traslape():
    retrain, ab, actual = (leer(c, ["estacion", "dia", "hora"]) for c in ["retrain", "ab_eval", "current"])
    assert retrain["dia"].max() < ab["dia"].min()
    assert len(retrain) + len(ab) == len(actual)
    assert not actual.duplicated().any()


@requiere_parquet
@pytest.mark.parametrize("conjunto", ["reference", "ab_eval"])
def test_etiqueta_y_prediccion_validas(conjunto):
    df = leer(conjunto, ["clase", *COLUMNAS_PRED])
    assert df["clase"].isin([0, 1, 2]).all()
    assert df["pred"].isin([0, 1, 2]).all()
    proba = df[COLUMNAS_PRED[1:]].to_numpy()
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(1), 1, atol=1e-4)
