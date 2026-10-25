"""Feature engineering de MiBici: transformaciones reutilizables para las preguntas del proyecto.

Secciones:
- Común: carga de viajes y estaciones, duración y filtros de viajes, partición temporal y festivos.
- P2: inventario reconstruido, etiqueta de riesgo de vaciado/llenado en las próximas 2 h
  y features por estación × hora (versión en código del notebook `1.0-mg-p2-riesgo-estaciones.ipynb`).
- P3: perfiles horarios de uso por estación para el clustering.

Uso (desde la raíz del repositorio):
    python -m MiBici.features        # construye las features de P2 y de P3
    python -m MiBici.features p2     # solo P2 → data/processed/p2/
    python -m MiBici.features p3     # solo P3 → data/processed/p3_station_profile.csv
"""

import json
from pathlib import Path
import warnings

from loguru import logger
import numpy as np
import pandas as pd
import typer

from MiBici.config import FECHA_INICIO, TRAIN_FIN, VALIDATION_FIN, PROCESSED_DATA_DIR, STATIONS_CSV, TRIPS_CSV

app = typer.Typer(help="Construye las features de P2 y P3. Sin subcomando, corre las dos.")


# ==========================================================================================
# Común
# ==========================================================================================

# Días de descanso obligatorio (art. 74 LFT), más el 1 de octubre de 2024 (transmisión del Poder Ejecutivo)
FESTIVOS = pd.to_datetime([
    "2024-01-01", "2024-02-05", "2024-03-18", "2024-05-01", "2024-09-16", "2024-10-01", "2024-11-18", "2024-12-25",
    "2025-01-01", "2025-02-03", "2025-03-17", "2025-05-01", "2025-09-16", "2025-11-17", "2025-12-25",
])


def load_trips(path: Path = TRIPS_CSV) -> pd.DataFrame:
    """Lee solo fechas y estaciones de los viajes.

    P3 trabaja con agregados por estación, así que no se cargan `Usuario_Id`,
    `Genero` ni `Anio_de_nacimiento` (datos personales).
    """
    return pd.read_csv(
        path,
        usecols=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"],
        dtype={"Origen_Id": "int32", "Destino_Id": "int32"},
        parse_dates=["Inicio_del_viaje", "Fin_del_viaje"],
    )


def add_trip_duration(trips: pd.DataFrame) -> pd.DataFrame:
    """Agrega `duracion_min`, la duración del viaje en minutos.

    Se usa para filtrar viajes inválidos (P1, P2 y P3) y es una variable
    candidata para describir el uso de cada estación.
    """
    dur = (trips["Fin_del_viaje"] - trips["Inicio_del_viaje"]).dt.total_seconds() / 60
    return trips.assign(duracion_min=dur)


def filter_valid_trips(
    trips: pd.DataFrame,
    min_min: float = 1,
    max_min: float = 180,
    drop_round_trips: bool = True,
) -> pd.DataFrame:
    """Quita viajes que no representan un desplazamiento real.

    - Duración <= `min_min`: anclajes fallidos o bicis que se regresan de inmediato.
    - Duración >= `max_min`: bicis no devueltas o errores de registro.
    - Origen = destino (opcional): paseos circulares; para P3 no aportan
      información sobre el flujo entre zonas.

    Los umbrales salen del notebook 1.0 y aplican a cada viaje por separado,
    así que no usan información de otros periodos.
    """
    if "duracion_min" not in trips:
        trips = add_trip_duration(trips)
    mask = trips["duracion_min"].between(min_min, max_min, inclusive="neither")
    if drop_round_trips:
        mask &= trips["Origen_Id"] != trips["Destino_Id"]
    return trips.loc[mask].reset_index(drop=True)


def cargar_estaciones(path: Path = STATIONS_CSV) -> pd.DataFrame:
    """Lee la nomenclatura de estaciones: id, nombre, latitud, longitud y capacidad.

    la capacidad Cₛ (`dpcapacity`) define qué significa "vacía" y "llena" y acota
    el inventario reconstruido; la ubicación entra como feature de la estación.
    """
    return pd.read_csv(path)


def particion(dia: pd.Series, params: dict) -> np.ndarray:
    """Asigna cada día operativo a "train", "val" o "test" según las fechas de corte.

    La partición es temporal y con los mismos cortes para todas las estaciones,
    para que el modelo nunca se evalúe con días anteriores a los que usó para entrenar.
    Cada corte cae entre dos días operativos (de 01:00 a 05:00 no hay servicio), lo que
    deja un hueco de ~4 h, mayor que el horizonte de 2 h.
    """
    return np.select([dia <= params["train_fin"], dia <= params["val_fin"]], ["train", "val"], "test")


# ==========================================================================================
# P2 · riesgo de vaciado o llenado en las próximas 2 horas
# ==========================================================================================

PARAMS_P2 = {
    "fecha_inicio": FECHA_INICIO,
    "fecha_fin": "2025-12-30",  # último día operativo completo
    "train_fin": TRAIN_FIN,
    "val_fin": VALIDATION_FIN,
    "hora_reinicio": 5,  # día operativo = timestamp − 5 h
    "dur_max_h": 24,
    "empates": "neto",  # eventos de una estación en el mismo segundo se suman
    "s0_metodo": "factible_por_estacion",
    "min_dias_s0": 30,  # menos días de entrenamiento → ŝ₀ = ⌊Cₛ/2⌋
    "umbral_vacio": 0.10,
    "umbral_lleno": 0.90,
    "horizonte_h": 2,
    "resolucion_min": 15,
    "n_horas_prediccion": 20,  # 05:00, 06:00, …, 00:00
}
# Los parámetros del modelo y de la evaluación (hiperparámetros, semilla, k) están en
# modeling/train_p2.py: no cambian la tabla, así que se ajustan sin regenerarla.
CLASES_P2 = ["normal", "vaciado", "llenado"]
# `hora` sí es feature
NO_FEATURES_P2 = ["estacion", "dia", "particion", "clase"]

OUT_DIR_P2 = PROCESSED_DATA_DIR / "p2"


def cargar_viajes_p2(path: Path = TRIPS_CSV) -> pd.DataFrame:
    """Lee el CSV consolidado de viajes con solo las columnas que usa P2.

    Carga `Viaje_Id`, las fechas de inicio y fin, y las estaciones de origen y
    destino, convierte las fechas a datetime.

    Para reconstruir el inventario solo importa cuándo y dónde sale o llega cada
    bici. `Usuario_Id`, `Genero` y `Anio_de_nacimiento` no aportan a la pregunta y son
    datos personales, así que no se cargan
    """
    trips = pd.read_csv(
        path,
        usecols=["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"],
        dtype={"Viaje_Id": "int64", "Origen_Id": "int32", "Destino_Id": "int32"},
    )
    for col in ["Inicio_del_viaje", "Fin_del_viaje"]:
        trips[col] = pd.to_datetime(trips[col], format="%Y-%m-%d %H:%M:%S")
    return trips


def limpiar_viajes_p2(trips: pd.DataFrame, stations: pd.DataFrame, dur_max_h: float) -> tuple[pd.DataFrame, dict]:
    """Quita los viajes que no representan un movimiento real de bicis.

    Elimina duraciones negativas o de 0 s, viajes de menos de 1 min entre
    estaciones distintas (físicamente imposibles), viajes de más de `dur_max_h` horas
    (bicis perdidas o recuperadas por el personal) y estaciones que no están en la
    nomenclatura. Devuelve los viajes válidos y cuántos quitó cada regla.

    Cada viaje se convierte en una salida y una llegada que mueven el inventario. 
    Importante no tener datos erroneos que generen cambios de inventario que no ocurrieron.
    """
    dur = (trips["Fin_del_viaje"] - trips["Inicio_del_viaje"]).dt.total_seconds()
    misma = trips["Origen_Id"] == trips["Destino_Id"]
    reglas = {
        "duracion_negativa": dur < 0,
        "duracion_cero": dur == 0,
        "menos_1min_entre_estaciones": (dur > 0) & (dur < 60) & ~misma,
        "mas_dur_max": dur > dur_max_h * 3600,
        "estacion_fuera_catalogo": ~trips["Origen_Id"].isin(stations["id"]) | ~trips["Destino_Id"].isin(stations["id"]),
    }
    quitar = np.logical_or.reduce(list(reglas.values()))
    conteo = {k: int(v.sum()) for k, v in reglas.items()} | {"total_eliminado": int(quitar.sum())}
    return trips.loc[~quitar].reset_index(drop=True), conteo


def construir_eventos(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Convierte los viajes en eventos de inventario: una fila por estación y segundo.

    Cada viaje genera una salida (−1) en su origen a la hora de inicio y una
    llegada (+1) en su destino a la hora de fin. A cada evento se le asigna su propio día
    operativo (`ts − hora_reinicio`), así un viaje que termina después de medianoche cuenta
    en el día de servicio correcto. Los eventos de una estación en el mismo segundo se
    suman en un solo cambio neto. Devuelve, ordenado por estación, día y hora: `estacion`,
    `dia`, `ts`, `salida` y `llegada`, `delta`, `sd` (id de estación-día)
    y `acum` (acumulado del día después del evento).

    El inventario de una estación solo cambia con estos eventos, así que son la
    base de las cotas, la etiqueta y las features de demanda. El neto evita inventar un
    pico que no se puede comprobar. Los conteos de salidas y llegadas se conservan
    por separado para medir la demanda.
    """
    t5 = pd.Timedelta(hours=params["hora_reinicio"])
    n = len(trips)
    ev = pd.DataFrame({
        "estacion": np.concatenate([trips["Origen_Id"].values, trips["Destino_Id"].values]),
        "ts": np.concatenate([trips["Inicio_del_viaje"].values, trips["Fin_del_viaje"].values]),
        "salida": np.concatenate([np.ones(n, "int8"), np.zeros(n, "int8")]),
    })
    ev["llegada"] = (1 - ev["salida"]).astype("int8")
    ev["dia"] = (ev["ts"] - t5).dt.normalize()
    ev = ev[ev["dia"].between(params["fecha_inicio"], params["fecha_fin"])]

    ev = ev.groupby(["estacion", "dia", "ts"], sort=True)[["salida", "llegada"]].sum().reset_index()
    ev["delta"] = (ev["llegada"] - ev["salida"]).astype("int16")
    ev["sd"] = ev.groupby(["estacion", "dia"], sort=True).ngroup()
    ev["acum"] = ev.groupby("sd")["delta"].cumsum()
    return ev


def calcular_cotas(ev: pd.DataFrame, capacidad: pd.Series, params: dict) -> pd.DataFrame:
    """Cotas del inventario inicial por estación-día: L, U y si el día es infactible.

    Con el acumulado del día calcula minimo y maximo. L = −minimo son las bicis que la 
    estación necesitaba al inicio para no quedar en negativo, y U = Cₛ − maximo las que 
    podía tener sin pasarse de su capacidad. Cualquier inventario inicial s con L ≤ s ≤ U 
    hace posible el día. Si L > U el día se marca como infactible. Devuelve una fila 
    por estación-día con eventos

    Ya que los datos no traen inventario, las cotas son lo que los viajes permiten
    deducir de él, y son la base de ŝ₀, del s₀ ajustado y del descarte de días
    infactibles en la etiqueta.
    """
    cotas = ev.groupby("sd").agg(
        estacion=("estacion", "first"),
        dia=("dia", "first"),
        total=("delta", "sum"),
        minimo=("acum", "min"),
        maximo=("acum", "max"),
        eventos=("delta", "size"),
    )
    cotas["minimo"] = cotas["minimo"].clip(upper=0)
    cotas["maximo"] = cotas["maximo"].clip(lower=0)
    cotas["C"] = cotas["estacion"].map(capacidad).astype(int)
    cotas["L"] = -cotas["minimo"]
    cotas["U"] = cotas["C"] - cotas["maximo"]
    cotas["infactible"] = cotas["L"] > cotas["U"]
    cotas["particion"] = particion(cotas["dia"], params)
    return cotas


def calcular_s0(c_train: pd.DataFrame, estaciones, capacidad: pd.Series, min_dias: int) -> pd.DataFrame:
    """Inventario inicial típico de cada estación (ŝ₀), calculado solo con entrenamiento.

    Para cada s ∈ [0, Cₛ] cuenta en cuántos días de entrenamiento es factible
    (L ≤ s ≤ U) y elige el más frecuente; en empate, el más cercano a Cₛ/2 y, a la misma
    distancia, el menor. Con menos de `min_dias` días de entrenamiento, o sin ninguno
    (estaciones nuevas), usa ⌊Cₛ/2⌋. Devuelve también el % de días factibles con ŝ₀ y con
    ⌊Cₛ/2⌋ y el método usado, para el artefacto `s0_por_estacion.csv`.

    Es el punto de partida del inventario estimado que alimenta las features.
    No depende de los viajes del día que se predice, así que se puede usar en t.
    """
    filas = []
    for est, g in c_train.groupby("estacion"):
        C = int(g["C"].iloc[0])
        s = np.arange(C + 1)
        cnt = ((g["L"].values[:, None] <= s) & (s <= g["U"].values[:, None])).sum(0)
        if len(g) >= min_dias:
            cand = s[cnt == cnt.max()]
            s0, metodo = int(cand[np.argmin(np.abs(cand - C / 2))]), "max_factible"
        else:
            s0, metodo = C // 2, "fallback_cs2"
        filas.append((est, C, len(g), s0, metodo, cnt[s0] / len(g), cnt[C // 2] / len(g)))
    out = pd.DataFrame(filas, columns=["estacion", "C", "dias_train", "s0", "metodo", "pct_factible_s0", "pct_factible_cs2"])
    faltan = sorted(set(estaciones) - set(out["estacion"]))
    extra = pd.DataFrame({"estacion": faltan, "C": [int(capacidad[e]) for e in faltan], "dias_train": 0})
    extra["s0"], extra["metodo"] = extra["C"] // 2, "fallback_cs2"
    return pd.concat([out, extra], ignore_index=True).set_index("estacion")


def validar_proxy(cotas: pd.DataFrame, s0_est: pd.DataFrame) -> dict:
    """Mide qué tan bien funciona ŝ₀ fuera de los días con los que se calculó.

    Calcula el % de días en que ŝ₀ es factible, el mismo % con ⌊Cₛ/2⌋ como 
    referencia y el % de días infactibles.

    """
    cv = cotas[cotas["particion"] == "val"].join(s0_est["s0"], on="estacion")
    cs2 = cv["C"] // 2
    return {
        "val_proxy_pct_factible_s0": float(((cv["L"] <= cv["s0"]) & (cv["s0"] <= cv["U"])).mean()),
        "val_proxy_pct_factible_cs2": float(((cv["L"] <= cs2) & (cs2 <= cv["U"])).mean()),
        "val_pct_dias_infactibles": float(cv["infactible"].mean()),
        "train_estaciones_s0_factible_bajo_50pct": int((s0_est["pct_factible_s0"] < 0.5).sum()),
    }


def indices_malla(params: dict) -> dict:
    """Índices de la malla de tiempo según la resolución y el horizonte.

    Devuelve `NI` (intervalos por día operativo), `IPH` (intervalos por hora),
    `KT` (intervalo en el que empieza cada hora de predicción: 05:00, 06:00, …) y `VENT`
    (intervalos que cubre la ventana de la etiqueta). Falla si la resolución no divide a
    60 min o si la última ventana se sale del día operativo.

    Concentra en un solo lugar los números que dependen de `resolucion_min` y
    `horizonte_h`, para que cambiarlos no deje índices fijos desalineados entre la
    etiqueta y las features.
    """
    res = params["resolucion_min"]
    if 60 % res:
        raise ValueError("resolucion_min debe dividir a 60")
    iph = 60 // res
    kt = np.arange(params["n_horas_prediccion"]) * iph
    vent = params["horizonte_h"] * iph
    ni = 24 * iph
    if kt[-1] + vent > ni:
        raise ValueError("la última ventana de predicción se sale del día operativo")
    return {"NI": ni, "IPH": iph, "KT": kt, "VENT": vent}


def malla_y_etiqueta(ev: pd.DataFrame, cotas: pd.DataFrame, s0_est: pd.DataFrame, params: dict) -> dict:
    """Inventario en una malla de 15 min por estación-día y etiqueta `clase_riesgo_2h`.

    - Acomoda los eventos en una matriz (estación-día × intervalo) y calcula `ini`, el
      acumulado al inicio de cada intervalo, junto con el mínimo y el máximo del acumulado
      dentro de cada intervalo.
    - Inventario reconstruido: parte de s₀ = clip(ŝ₀, L, U), que siempre es factible, y
      marca cada intervalo como vacío (ocupación mínima ≤ `umbral_vacio`) o lleno
      (ocupación máxima ≥ `umbral_lleno`).
    - Etiqueta para cada hora t: revisa los intervalos de (t, t + `horizonte_h`]; si alguno
      está vacío → vaciado (1), si alguno está lleno → llenado (2), si aparecen ambos gana
      el que ocurre primero (en el mismo intervalo, vaciado); si ninguno → normal (0).
    Devuelve `ini`, `vacio`, `lleno`, `clase`, `fact` (día factible), `s0_hat` y `C`.

    La pregunta es si la estación quedará sin bicis o sin anclajes *en algún
    momento* de las próximas 2 horas, no exactamente en t + 2 h; por eso se revisa toda la
    ventana y el mínimo y el máximo dentro de cada intervalo. El s₀ ajustado usa los viajes
    de todo el día, así que solo sirve para la etiqueta y nunca entra como feature.
    """
    m = indices_malla(params)
    NI, KT, VENT = m["NI"], m["KT"], m["VENT"]
    n_sd = len(cotas)
    t5 = pd.Timedelta(hours=params["hora_reinicio"])

    k = ((ev["ts"] - ev["dia"] - t5) // pd.Timedelta(minutes=params["resolucion_min"])).to_numpy().astype(int)
    flat = ev["sd"].to_numpy() * NI + k
    net = np.bincount(flat, weights=ev["delta"], minlength=n_sd * NI).reshape(n_sd, NI).astype("int32")
    ini = net.cumsum(axis=1) - net

    # Mínimo y máximo dentro de cada intervalo: nivel al inicio y acumulado después de cada evento
    ext = pd.DataFrame({"f": flat, "a": ev["acum"].to_numpy()}).groupby("f")["a"].agg(["min", "max"])
    pos = ext.index.to_numpy()
    acum_min = ini.ravel().copy()
    acum_min[pos] = np.minimum(acum_min[pos], ext["min"].to_numpy())
    acum_max = ini.ravel().copy()
    acum_max[pos] = np.maximum(acum_max[pos], ext["max"].to_numpy())
    acum_min, acum_max = acum_min.reshape(n_sd, NI), acum_max.reshape(n_sd, NI)
    del net, ext, flat, k

    C = cotas["C"].to_numpy()
    s0_hat = cotas["estacion"].map(s0_est["s0"]).to_numpy()
    s0_adj = np.clip(s0_hat, cotas["L"].to_numpy(), cotas["U"].to_numpy())  # solo tiene sentido en días factibles
    fact = ~cotas["infactible"].to_numpy()

    occ_min = (s0_adj[:, None] + acum_min) / C[:, None]
    occ_max = (s0_adj[:, None] + acum_max) / C[:, None]
    if not ((occ_min[fact] >= 0).all() and (occ_max[fact] <= 1).all()):
        raise AssertionError("el inventario reconstruido se sale de [0, Cₛ] en un día factible")
    vacio = occ_min <= params["umbral_vacio"]
    lleno = occ_max >= params["umbral_lleno"]
    del acum_min, acum_max, occ_min, occ_max

    # Etiqueta: gana el riesgo que ocurre primero en la ventana; empate en el mismo intervalo → vaciado
    win = KT[:, None] + np.arange(VENT)
    V, Ll = vacio[:, win], lleno[:, win]
    primero_v = np.where(V.any(2), V.argmax(2), 99)
    primero_l = np.where(Ll.any(2), Ll.argmax(2), 99)
    clase = np.select([(primero_v < 99) & (primero_v <= primero_l), primero_l < 99], [1, 2], 0)

    return {"ini": ini, "vacio": vacio, "lleno": lleno, "clase": clase, "fact": fact, "s0_hat": s0_hat, "C": C} | m


def construir_tabla_p2(ev: pd.DataFrame, cotas: pd.DataFrame, stations: pd.DataFrame, malla: dict, params: dict) -> pd.DataFrame:
    """Tabla de modelado: una fila por estación-día factible × hora de predicción t.

    - Estado actual (inventario estimado, que parte de ŝ₀ cada día):
      `ocup_est` (bicis estimadas / Cₛ al inicio del intervalo de t; puede salir de [0, 1]
      en días atípicos), `estado_est` (vacío/normal/lleno según los umbrales) y `tend_1h`
      (cambio neto de bicis en la última hora; a las 05:00 vale 0).
    - Demanda reciente: `sal_1h`, `lle_1h` (salidas y llegadas en [t−1h, t)) y `sal_2h`,
      `lle_2h` (en [t−2h, t−1h)).
    - Demanda estacional: `*_ayer_2h` (la misma ventana de 2 h de ayer), `*_sem_2h` (de hace
      una semana) y `*_media7_2h` (promedio de los últimos 7 días).
      Anticipan cuánta demanda habrá en (t, t+2h] con lo que pasó en esa misma
      franja en días anteriores.
    - Historial de riesgo: `frac_vacio_7d`, `frac_lleno_7d` (fracción de los 7 días
      anteriores en que la estación se vació o se llenó) y `riesgo_ayer` (clase de ayer a la
      misma hora).
    - Calendario: `hora`, `dia_semana`, `fin_semana`, `festivo`.
      El uso cambia mucho entre horas pico, fines de semana y festivos.
    - Estación: `C`, `lat`, `lon`.
      Por qué: la capacidad define qué tan rápido se llega a un extremo y la ubicación
      resume el tipo de zona (residencial, laboral, recreativa).

    También incluye `estacion`, `dia`, `particion` y `clase` (la etiqueta), que no son
    features. Los días infactibles se descartan porque no tienen una etiqueta confiable.
    """
    KT, IPH, VENT = malla["KT"], malla["IPH"], malla["VENT"]
    ini, fact, s0_hat, C, clase = malla["ini"], malla["fact"], malla["s0_hat"], malla["C"], malla["clase"]
    t5 = pd.Timedelta(hours=params["hora_reinicio"])
    n_h = len(KT)

    # Estado actual con el inventario estimado (parte de ŝ₀ cada día)
    nivel_t = ini[:, KT]
    ocup_est = (s0_hat[:, None] + nivel_t) / C[:, None]
    ocup_clip = ocup_est.clip(0, 1)
    estado_est = np.select([ocup_clip <= params["umbral_vacio"], ocup_clip >= params["umbral_lleno"]], [1, 2], 0)
    tend_1h = nivel_t - ini[:, np.maximum(KT - IPH, 0)]  # a las 05:00 vale 0: no se lee el final de la matriz

    # Conteos por hora en una línea de tiempo continua (hora 0 = fecha_inicio 05:00)
    est_ids = np.sort(ev["estacion"].unique())
    dia0 = pd.Timestamp(params["fecha_inicio"])
    n_dias = (pd.Timestamp(params["fecha_fin"]) - dia0).days + 1
    st_ev = np.searchsorted(est_ids, ev["estacion"].to_numpy())
    h_ev = (ev["dia"] - dia0).dt.days.to_numpy() * 24 + ((ev["ts"] - ev["dia"] - t5) // pd.Timedelta(hours=1)).to_numpy()
    # una semana de NaN antes del inicio: los lags de los primeros días quedan como "sin dato"
    pad = 168  

    def por_hora(col):
        m = np.bincount(st_ev * n_dias * 24 + h_ev, weights=ev[col], minlength=len(est_ids) * n_dias * 24)
        m = m.reshape(len(est_ids), n_dias * 24).astype("float32")
        return np.concatenate([np.full((len(est_ids), pad), np.nan, "float32"), m], axis=1)

    st_sd = np.searchsorted(est_ids, cotas["estacion"].to_numpy())
    dia_sd = (cotas["dia"] - dia0).dt.days.to_numpy()
    H = pad + dia_sd[:, None] * 24 + KT[None, :] // IPH
    S = st_sd[:, None]

    def ventana2h(M, atras):
        return M[S, H - atras] + M[S, H - atras + 1]

    demanda = {}
    for nombre, col in [("sal", "salida"), ("lle", "llegada")]:
        M = por_hora(col)
        demanda[f"{nombre}_1h"] = M[S, H - 1]
        demanda[f"{nombre}_2h"] = M[S, H - 2]
        demanda[f"{nombre}_ayer_2h"] = ventana2h(M, 24)
        demanda[f"{nombre}_sem_2h"] = ventana2h(M, 168)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            demanda[f"{nombre}_media7_2h"] = np.nanmean([ventana2h(M, 24 * d) for d in range(1, 8)], axis=0)
        del M

    # Historial de riesgo con días cerrados (inventario reconstruido; solo días factibles)
    fin_ventanas = KT[-1] + VENT
    dv = np.where(fact, malla["vacio"][:, :fin_ventanas].any(1), np.nan)
    dl = np.where(fact, malla["lleno"][:, :fin_ventanas].any(1), np.nan)
    DV = np.full((len(est_ids), n_dias + 7), np.nan)
    DV[st_sd, dia_sd + 7] = dv
    DL = np.full((len(est_ids), n_dias + 7), np.nan)
    DL[st_sd, dia_sd + 7] = dl
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        frac_vacio_7d = np.nanmean([DV[st_sd, dia_sd + 7 - d] for d in range(1, 8)], axis=0)
        frac_lleno_7d = np.nanmean([DL[st_sd, dia_sd + 7 - d] for d in range(1, 8)], axis=0)

    CL = np.full((len(est_ids), n_dias + 1, n_h), np.nan)
    CL[st_sd, dia_sd + 1] = np.where(fact[:, None], clase, np.nan)
    # clase de ayer a la misma hora
    riesgo_ayer = CL[st_sd, dia_sd]  

    dias = cotas["dia"]
    geo = stations.set_index("id")[["latitude", "longitude"]]
    F = fact
    rep = lambda a: np.repeat(np.asarray(a)[F], n_h)
    plano = lambda a: np.asarray(a)[F].ravel()
    f32 = lambda a: np.asarray(a, dtype="float32")

    tabla = pd.DataFrame({
        "estacion": rep(cotas["estacion"]),
        "dia": rep(dias),
        "hora": np.tile((params["hora_reinicio"] + KT // IPH) % 24, int(F.sum())).astype("int8"),
        "particion": pd.Categorical(rep(cotas["particion"]), categories=["train", "val", "test"]),
        "clase": plano(clase).astype("int8"),
        "ocup_est": f32(plano(ocup_est)),
        "estado_est": f32(plano(estado_est)),
        "tend_1h": f32(plano(tend_1h)),
        **{c: f32(plano(v)) for c, v in demanda.items()},
        "frac_vacio_7d": f32(rep(frac_vacio_7d)),
        "frac_lleno_7d": f32(rep(frac_lleno_7d)),
        "riesgo_ayer": f32(plano(riesgo_ayer)),
        "dia_semana": f32(rep(dias.dt.dayofweek)),
        "fin_semana": f32(rep(dias.dt.dayofweek >= 5)),
        "festivo": f32(rep(dias.isin(FESTIVOS))),
        "C": f32(rep(C)),
        "lat": f32(rep(cotas["estacion"].map(geo["latitude"]))),
        "lon": f32(rep(cotas["estacion"].map(geo["longitude"]))),
    })
    return tabla


def features_de_p2(tabla: pd.DataFrame) -> list[str]:
    """Columnas de la tabla que entran al modelo: todas menos identificadores, partición y etiqueta.

    Un solo lugar define qué es feature, para que el entrenamiento, la evaluación
    y cualquier predicción usen exactamente las mismas columnas y nunca la etiqueta.
    """
    return [c for c in tabla.columns if c not in NO_FEATURES_P2]


def construir_dataset_p2(trips_path: Path, stations_path: Path, params: dict = PARAMS_P2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Pipeline completo de P2: CSV de viajes → tabla de modelado.

    Carga y limpia los viajes, construye los eventos, calcula las cotas, ŝ₀, la validación 
    del proxy, la malla con la etiqueta y la tabla con las features. Devuelve la tabla,
    ŝ₀ por estación y un resumen (`info`) con los tamaños, lo que quitó cada regla de limpieza, 
    el % de días infactibles, la validación del proxy y los parámetros usados.

    Concentra en una sola llamada todo lo que necesita el entrenamiento, y el
    resumen se guarda junto con la tabla para que los parámetros registrados en MLflow
    sean los que de verdad generaron los datos.
    """
    stations = cargar_estaciones(stations_path)
    capacidad = stations.set_index("id")["dpcapacity"]

    trips = cargar_viajes_p2(trips_path)
    info = {
        "meses": int(trips["Inicio_del_viaje"].dt.to_period("M").nunique()),
        "primer_viaje": str(trips["Inicio_del_viaje"].min()),
        "ultimo_viaje": str(trips["Inicio_del_viaje"].max()),
        "viajes_crudo": len(trips),
        "viaje_id_duplicados": int(trips["Viaje_Id"].duplicated().sum()),
        "estaciones_con_viajes": len(set(trips["Origen_Id"]) | set(trips["Destino_Id"])),
        "estaciones_nomenclatura": len(stations),
    }
    trips, info["limpieza"] = limpiar_viajes_p2(trips, stations, params["dur_max_h"])
    info["viajes_limpios"] = len(trips)

    ev = construir_eventos(trips, params)
    del trips
    info["eventos_netos"] = len(ev)
    cotas = calcular_cotas(ev, capacidad, params)
    info["estacion_dias"] = len(cotas)
    info["pct_infactible"] = cotas.groupby("particion")["infactible"].mean().round(4).to_dict()

    s0_est = calcular_s0(cotas[cotas["particion"] == "train"], cotas["estacion"].unique(), capacidad, params["min_dias_s0"])
    info["s0_metodos"] = s0_est["metodo"].value_counts().to_dict()
    info["proxy"] = validar_proxy(cotas, s0_est)

    malla = malla_y_etiqueta(ev, cotas, s0_est, params)
    tabla = construir_tabla_p2(ev, cotas, stations, malla, params)
    del ev, malla

    info["filas_tabla"] = len(tabla)
    info["n_features"] = len(features_de_p2(tabla))
    info["params"] = params
    return tabla, s0_est, info


# ==========================================================================================
# P3 · perfiles de uso por estación
# ==========================================================================================

DIAS = ["semana", "finde"]
HORAS_PICO_AM = range(6, 10)
HORAS_PICO_PM = range(17, 21)


def _hourly_counts(ts: pd.Series, station: pd.Series, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "estacion_id": station.values,
        "dia": np.where(ts.dt.dayofweek >= 5, "finde", "semana"),
        "hora": ts.dt.hour.values,
    })
    out = df.pivot_table(index="estacion_id", columns=["dia", "hora"], aggfunc="size", fill_value=0)
    # De madrugada no hay viajes; se fuerzan las 24 columnas
    full = pd.MultiIndex.from_product([DIAS, range(24)], names=["dia", "hora"])
    out = out.reindex(columns=full, fill_value=0)
    out.columns = [f"{prefix}_{d}_{h:02d}" for d, h in out.columns]
    return out


def station_hourly_counts(trips: pd.DataFrame) -> pd.DataFrame:
    """Cuenta salidas y llegadas por estación, tipo de día y hora (96 columnas).

    Las salidas se asignan a la hora de inicio en la estación de origen y las
    llegadas a la hora de fin en la estación de destino. Es la base del perfil
    de P3 y también del flujo neto de P2.
    """
    sal = _hourly_counts(trips["Inicio_del_viaje"], trips["Origen_Id"], "sal")
    lle = _hourly_counts(trips["Fin_del_viaje"], trips["Destino_Id"], "lle")
    return sal.join(lle, how="outer").fillna(0)


def _sum_hours(df: pd.DataFrame, hours) -> pd.Series:
    return df[[c for c in df.columns if int(c[-2:]) in hours]].sum(axis=1)


def station_usage_summary(counts: pd.DataFrame, n_dias: int) -> pd.DataFrame:
    """Métricas interpretables por estación a partir de `station_hourly_counts`.

    - `viajes`, `viajes_dia`: volumen total y promedio diario.
    - `balance`: (llegadas - salidas) / total; > 0 la estación recibe más bicis.
    - `pct_finde`: proporción de viajes en fin de semana (uso recreativo).
    - `balance_am`, `balance_pm`: balance en horas pico entre semana; distingue
      estaciones de vivienda (salen en la mañana) de las de trabajo o escuela.
    - `pct_pico`: proporción de viajes entre semana que caen en horas pico.

    Sirven para filtrar estaciones con poco uso y para nombrar los grupos de P3.
    """
    sal, lle = counts.filter(like="sal_"), counts.filter(like="lle_")
    sal_sem, lle_sem = counts.filter(like="sal_semana"), counts.filter(like="lle_semana")
    total = sal.sum(axis=1) + lle.sum(axis=1)
    pico = list(HORAS_PICO_AM) + list(HORAS_PICO_PM)

    def balance(hours):
        a, s = _sum_hours(lle_sem, hours), _sum_hours(sal_sem, hours)
        return (a - s) / (a + s).replace(0, np.nan)

    return pd.DataFrame({
        "viajes": total,
        "viajes_dia": total / n_dias,
        "balance": (lle.sum(axis=1) - sal.sum(axis=1)) / total,
        "pct_finde": counts.filter(like="_finde_").sum(axis=1) / total,
        "balance_am": balance(HORAS_PICO_AM),
        "balance_pm": balance(HORAS_PICO_PM),
        "pct_pico": (_sum_hours(sal_sem, pico) + _sum_hours(lle_sem, pico))
        / (sal_sem.sum(axis=1) + lle_sem.sum(axis=1)),
    })


def station_usage_profile(counts: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Vector de características por estación para el clustering de P3.

    Cada uno de los 4 bloques horarios (salidas/llegadas x semana/fin de semana)
    se normaliza para sumar 1. Así dos estaciones con el mismo ritmo diario
    quedan cerca aunque una tenga mucho más uso. El tamaño se agrega aparte
    como `log_volumen`, junto con `balance` y `pct_finde`.
    """
    blocks = [f"{t}_{d}" for d in DIAS for t in ("sal", "lle")]
    profile = pd.concat(
        [counts.filter(like=b).pipe(lambda d: d.div(d.sum(axis=1).replace(0, 1), axis=0)) for b in blocks],
        axis=1,
    )
    return profile.assign(
        log_volumen=np.log1p(summary["viajes_dia"]),
        balance=summary["balance"],
        pct_finde=summary["pct_finde"],
    )


def n_service_days(trips: pd.DataFrame) -> int:
    """Número de días con al menos un viaje, para pasar totales a promedios diarios."""
    return trips["Inicio_del_viaje"].dt.normalize().nunique()


# ==========================================================================================
# Línea de comandos
# ==========================================================================================

@app.command()
def p2(
    trips_path: Path = TRIPS_CSV,
    stations_path: Path = STATIONS_CSV,
    out_dir: Path = OUT_DIR_P2,
):
    """Construye la tabla de P2 y la guarda en `out_dir` (tabla.parquet, s0_por_estacion.csv e info.json)."""
    logger.info("P2: construyendo eventos, cotas, ŝ₀, etiqueta y features...")
    tabla, s0_est, info = construir_dataset_p2(trips_path, stations_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(out_dir / "tabla.parquet", index=False)
    s0_est.to_csv(out_dir / "s0_por_estacion.csv")
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    logger.success(f"P2: {info['filas_tabla']:,} filas, {info['n_features']} features → {out_dir}")


@app.command()
def p3(
    input_path: Path = TRIPS_CSV,
    output_path: Path = PROCESSED_DATA_DIR / "p3_station_profile.csv",
):
    """Construye el perfil de uso por estación de P3 y lo guarda en `output_path`."""
    logger.info(f"Leyendo viajes de {input_path}")
    trips = filter_valid_trips(load_trips(input_path))
    counts = station_hourly_counts(trips)
    summary = station_usage_summary(counts, n_service_days(trips))
    profile = station_usage_profile(counts, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path)
    logger.success(f"Perfil de {len(profile)} estaciones guardado en {output_path}")


@app.callback(invoke_without_command=True)
def todas(ctx: typer.Context):
    """Sin subcomando, construye las features de P2 y de P3 con las rutas por defecto."""
    if ctx.invoked_subcommand is None:
        p2()
        p3()


if __name__ == "__main__":
    app()
