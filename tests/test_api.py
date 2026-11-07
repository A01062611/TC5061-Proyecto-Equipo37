"""Pruebas de la API de P2 con TestClient: predicción válida, errores 422 y modelo no cargado."""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.schemas import P2Prediction

client = TestClient(main.app)

EJEMPLO = P2Prediction.model_config["json_schema_extra"]["examples"][0]

# Casos inventados: estación de 15 anclajes a las 8:00 de un martes
VACIADO = {
    "hora": 8, "ocup_est": 0.1, "estado_est": 1, "tend_1h": -4.0,
    "sal_1h": 6.0, "sal_2h": 5.0, "sal_ayer_2h": 12.0, "sal_sem_2h": 11.0, "sal_media7_2h": 11.5,
    "lle_1h": 0.0, "lle_2h": 1.0, "lle_ayer_2h": 1.0, "lle_sem_2h": 2.0, "lle_media7_2h": 1.5,
    "frac_vacio_7d": 1.0, "frac_lleno_7d": 0.0, "riesgo_ayer": 1,
    "dia_semana": 1, "fin_semana": 0, "festivo": 0,
    "C": 15, "lat": 20.6664, "lon": -103.3488,
}
LLENADO = {
    **VACIADO,
    "ocup_est": 0.9, "estado_est": 2, "tend_1h": 4.0,
    "sal_1h": 0.0, "sal_2h": 1.0, "sal_ayer_2h": 1.0, "sal_sem_2h": 2.0, "sal_media7_2h": 1.5,
    "lle_1h": 6.0, "lle_2h": 5.0, "lle_ayer_2h": 12.0, "lle_sem_2h": 11.0, "lle_media7_2h": 11.5,
    "frac_vacio_7d": 0.0, "frac_lleno_7d": 1.0, "riesgo_ayer": 2,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_prediccion_valida():
    r = client.post("/predict_p2/", json=EJEMPLO)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1, 2)
    assert body["message"]


@pytest.mark.parametrize("entrada, clase", [(VACIADO, 1), (LLENADO, 2)])
def test_prediccion_por_clase(entrada, clase):
    r = client.post("/predict_p2/", json=entrada)
    assert r.status_code == 200
    assert r.json()["prediction"] == clase


def test_campos_opcionales_nulos():
    """Los lags pueden faltar en los primeros días de una estación."""
    r = client.post("/predict_p2/", json={**EJEMPLO, "sal_ayer_2h": None, "frac_vacio_7d": None, "riesgo_ayer": None})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "cambio",
    [
        {"hora": 30},  # fuera de rango
        {"ocup_est": "mucho"},  # tipo incorrecto
        {"lat": 19.4},  # fuera de Guadalajara
        {"estacion_extra": 1},  # campo no permitido
    ],
)
def test_entrada_invalida_422(cambio):
    r = client.post("/predict_p2/", json={**EJEMPLO, **cambio})
    assert r.status_code == 422


def test_campo_faltante_422():
    sin_hora = {k: v for k, v in EJEMPLO.items() if k != "hora"}
    r = client.post("/predict_p2/", json=sin_hora)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "hora"]


def test_modelo_no_cargado_503(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    r = client.post("/predict_p2/", json=EJEMPLO)
    assert r.status_code == 503
    assert client.get("/health").json()["status"] == "error"
