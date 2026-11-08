from pydantic import BaseModel, ConfigDict, Field

# Los lags de demanda e historial valen NaN en los primeros días de una estación
# (no hay ayer, semana pasada o 7 días previos); el modelo los acepta como faltantes.


class P2Prediction(BaseModel):
    """Una estación en la hora t: entrada del modelo de riesgo de vaciado/llenado (P2)."""

    # Configuración de Pydantic, se muestra en la documentación de FastAPI
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{
                "hora": 5,
                "ocup_est": 0.6,
                "estado_est": 0,
                "tend_1h": 0.0,
                "sal_1h": 0.0,
                "sal_2h": 0.0,
                "sal_ayer_2h": 6.0,
                "sal_sem_2h": 0.0,
                "sal_media7_2h": 3.14,
                "lle_1h": 0.0,
                "lle_2h": 0.0,
                "lle_ayer_2h": 1.0,
                "lle_sem_2h": 0.0,
                "lle_media7_2h": 2.14,
                "frac_vacio_7d": 0.4,
                "frac_lleno_7d": 0.4,
                "riesgo_ayer": 0,
                "dia_semana": 5,
                "fin_semana": 1,
                "festivo": 0,
                "C": 15,
                "lat": 20.6664,
                "lon": -103.3488,
            }]
        },
    )

    # Estado actual
    hora: int = Field(ge=0, le=23, description="Hora de predicción t")
    ocup_est: float = Field(
        ge=-1.0, le=2.0, description="Bicis estimadas / capacidad; puede salir de [0, 1] en días atípicos"
    )
    estado_est: int = Field(ge=0, le=2, description="0 normal, 1 vacía, 2 llena")
    tend_1h: float = Field(ge=-100, le=100, description="Cambio neto de bicis en la última hora")

    # Demanda reciente: salidas y llegadas hace una y dos horas
    sal_1h: float = Field(ge=0)
    sal_2h: float = Field(ge=0)
    lle_1h: float = Field(ge=0)
    lle_2h: float = Field(ge=0)

    # Demanda estacional: misma ventana de 2 h ayer, hace una semana y promedio de 7 días
    sal_ayer_2h: float | None = Field(default=None, ge=0)
    sal_sem_2h: float | None = Field(default=None, ge=0)
    sal_media7_2h: float | None = Field(default=None, ge=0)
    lle_ayer_2h: float | None = Field(default=None, ge=0)
    lle_sem_2h: float | None = Field(default=None, ge=0)
    lle_media7_2h: float | None = Field(default=None, ge=0)

    # Historial de riesgo
    frac_vacio_7d: float | None = Field(default=None, ge=0, le=1)
    frac_lleno_7d: float | None = Field(default=None, ge=0, le=1)
    riesgo_ayer: int | None = Field(default=None, ge=0, le=2, description="Clase de ayer a la misma hora")

    # Calendario
    dia_semana: int = Field(ge=0, le=6, description="0 lunes ... 6 domingo")
    fin_semana: int = Field(ge=0, le=1)
    festivo: int = Field(ge=0, le=1)

    # Estación
    C: int = Field(gt=0, le=100, description="Capacidad de la estación")
    lat: float = Field(ge=20.5, le=20.8, description="Latitud (zona metropolitana de Guadalajara)")
    lon: float = Field(ge=-103.5, le=-103.2, description="Longitud (zona metropolitana de Guadalajara)")

class P2Response(BaseModel):
    """Respuesta de la predicción de riesgo de vaciado/llenado (P2)."""

    # Configuración de Pydantic, se muestra en la documentación de FastAPI
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{
                "prediction": 0,
                "message": "La estación estará normal por las siguientes 2 hrs, no requiere acción"
            }]
        },
    )

    prediction: int = Field(ge=0, le=2, description="0 normal, 1 vacía, 2 llena")
    message: str = Field(description="Mensaje explicativo del resultado de la predicción")