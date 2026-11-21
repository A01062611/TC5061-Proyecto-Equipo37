"""Servicio de inferencia MiBici (API REST)."""
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.model_loader import load_model, model_version, predict
from app.schemas import P2Prediction, P2Response

logger = logging.getLogger(__name__)

app = FastAPI(title="MiBici - riesgo de vaciado/llenado (P2)")

try:
    model = load_model()
    version = model_version()
except Exception:
    logger.exception("Error al cargar el modelo")
    model = None
    version = None

MENSAJES = {
    0: "La estación estará normal por las siguientes 2 hrs, no requiere acción",
    1: "La estación se vaciará en las siguientes 2 hrs, recolocar bicis en la estación",
    2: "La estación se llenará en las siguientes 2 hrs, remover bicis de la estación",
}


# `def` (no `async def`): la inferencia es síncrona y FastAPI la ejecuta en su threadpool
# en vez de bloquear el event loop.
@app.post("/predict_p2/", response_model=P2Response)
def predict_p2(schema: P2Prediction):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    prediction = predict(model, rows=[schema.model_dump()])[0]
    return P2Response(
        prediction=prediction,
        message=MENSAJES.get(prediction, MENSAJES[2]),
        model_version=version,
    )


@app.get("/health")
def health_check():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Modelo no cargado"})
    return {"status": "ok", "model_version": version}
