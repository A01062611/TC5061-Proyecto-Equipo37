from fastapi import FastAPI, HTTPException
from app.schemas import P2Prediction,P2Response
from app.model_loader import predict, load_model, prepare_frame

import pandas as pd

app = FastAPI()
try:
    model = load_model()
except Exception as e:
    model = None
    print(f"Error al cargar el modelo: {e}")

@app.post("/predict_p2/", response_model=P2Response)
async def predict_p2(schema:P2Prediction):

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    prediction = predict(model, rows=pd.DataFrame(schema.model_dump(),index=[0]))
    prediction = prediction[0]

    if prediction == 0:
        res = ("La estacion estara normal por las siguientes 2 hrs, no requiere accion")
    elif prediction == 1:
        res = ("La estacion se vaciara en las siguientes 2 hrs, recolocar bicis en estacion")
    else:
        res =("La estacion se llenara en las siguientes 2 hrs, remover bicies de la estacion")

    return P2Response(prediction=prediction, message=res)

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "error", "message": "Modelo no cargado"}
    return {"status": "ok"}
