"""P2.1: inventario reconstruido, etiqueta de riesgo y features por estación × hora.

Pregunta: ¿una estación estará en riesgo de quedarse sin bicicletas o sin anclajes
en las próximas 2 horas? 
Los datos no traen inventario, así que se reconstruye con
los viajes y la capacidad de cada estación:
1. Cada viaje genera una salida (−1) y una llegada (+1); cada evento recibe su propio
   día operativo (ts − 5 h). Los eventos de una estación en el mismo segundo se suman
   como un solo cambio neto.
2. Por estación-día: L = −min(acum, 0), U = Cₛ − max(acum, 0). Si L > U el día es infactible.
3. ŝ₀ por estación (solo entrenamiento): el valor factible en más días; empate → el más
   cercano a Cₛ/2 y, a la misma distancia, el menor.
4. Inventario estimado (ŝ₀) → features. Inventario reconstruido (clip(ŝ₀, L, U)) → etiqueta.

Es la versión en código del notebook `1.0-mg-p2-riesgo-estaciones.ipynb`.

Uso: python -m MiBici.features_p2_1
"""

import json
from pathlib import Path
import warnings

from loguru import logger
import numpy as np
import pandas as pd
import typer

from MiBici.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

PARAMS = {
    "fecha_inicio": "2024-01-01",
    "fecha_fin": "2025-12-30",  # último día operativo completo
    "train_fin": "2025-06-30",
    "val_fin": "2025-09-30",
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
# modeling/train_p2_1.py: no cambian la tabla, así que se ajustan sin regenerarla.
CLASES = ["normal", "vaciado", "llenado"]
NO_FEATURES = ["estacion", "dia", "particion", "clase"]  # `hora` sí es feature

TRIPS_P2_1 = RAW_DATA_DIR / "mibici_2024_2025.csv"
STATIONS_P2_1 = RAW_DATA_DIR / "nomenclatura_2026_09.csv"
OUT_DIR_P2_1 = PROCESSED_DATA_DIR / "p2_1"

# Días de descanso obligatorio (art. 74 LFT), más el 1 de octubre de 2024 (transmisión del Poder Ejecutivo)
FESTIVOS = pd.to_datetime([
    "2024-01-01", "2024-02-05", "2024-03-18", "2024-05-01", "2024-09-16", "2024-10-01", "2024-11-18", "2024-12-25",
    "2025-01-01", "2025-02-03", "2025-03-17", "2025-05-01", "2025-09-16", "2025-11-17", "2025-12-25",
])


# --------------------------------------------------------------------------- carga y limpieza

def cargar_viajes(path: Path = TRIPS_P2_1) -> pd.DataFrame:
    """Lee solo las columnas que usa P2.1; los datos personales no se cargan.

    Se usa el motor C con un formato de fecha explícito: el motor pyarrow retiene ~2 GB
    de memoria después de la lectura.
    """
    trips = pd.read_csv(
        path,
        usecols=["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"],
        dtype={"Viaje_Id": "int64", "Origen_Id": "int32", "Destino_Id": "int32"},
    )
    for col in ["Inicio_del_viaje", "Fin_del_viaje"]:
        trips[col] = pd.to_datetime(trips[col], format="%Y-%m-%d %H:%M:%S")
    return trips


def cargar_estaciones(path: Path = STATIONS_P2_1) -> pd.DataFrame:
    return pd.read_csv(path)


def limpiar_viajes(trips: pd.DataFrame, stations: pd.DataFrame, dur_max_h: float) -> tuple[pd.DataFrame, dict]:
    """Quita duraciones inválidas y estaciones fuera del catálogo.

    Los viajes de menos de 1 min a la misma estación se conservan: su efecto neto es 0
    y no mueven las cotas. Los viajes largos (hasta `dur_max_h`) también, porque su
    salida y su llegada sí ocurrieron.
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


# --------------------------------------------------------------------------- eventos, cotas y ŝ₀

def construir_eventos(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Una fila por estación y segundo, ordenada por estación, día y hora.

    Columnas: estacion, dia, ts, salida, llegada (conteos), delta (neto),
    sd (id de estación-día) y acum (acumulado del día después del evento).
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


def particion(dia: pd.Series, params: dict) -> np.ndarray:
    return np.select([dia <= params["train_fin"], dia <= params["val_fin"]], ["train", "val"], "test")


def calcular_cotas(ev: pd.DataFrame, capacidad: pd.Series, params: dict) -> pd.DataFrame:
    """Una fila por estación-día con eventos; la fila i corresponde a sd = i."""
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
    """ŝ₀ por estación con los días de entrenamiento.

    El valor s ∈ [0, Cₛ] factible (L ≤ s ≤ U) en más días; empate → el más cercano a
    Cₛ/2 y, a la misma distancia, el menor. Con menos de `min_dias` días, o sin días
    de entrenamiento, ŝ₀ = ⌊Cₛ/2⌋.
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
    """% de días de validación en los que ŝ₀ (calculado con entrenamiento) sigue siendo factible."""
    cv = cotas[cotas["particion"] == "val"].join(s0_est["s0"], on="estacion")
    cs2 = cv["C"] // 2
    return {
        "val_proxy_pct_factible_s0": float(((cv["L"] <= cv["s0"]) & (cv["s0"] <= cv["U"])).mean()),
        "val_proxy_pct_factible_cs2": float(((cv["L"] <= cs2) & (cs2 <= cv["U"])).mean()),
        "val_pct_dias_infactibles": float(cv["infactible"].mean()),
        "train_estaciones_s0_factible_bajo_50pct": int((s0_est["pct_factible_s0"] < 0.5).sum()),
    }


# --------------------------------------------------------------------------- malla de 15 min y etiqueta

def indices_malla(params: dict) -> dict:
    """Intervalos por día, por hora, horas de predicción (KT) y largo de la ventana."""
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
    """Inventario sobre la malla (n_sd × NI) y etiqueta `clase` (n_sd × horas de predicción).

    Devuelve `ini` (acumulado al inicio de cada intervalo), `vacio` y `lleno` del
    inventario reconstruido, `clase`, `fact`, `s0_hat` y `C`.
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


# --------------------------------------------------------------------------- features y tabla horaria

def construir_tabla(ev: pd.DataFrame, cotas: pd.DataFrame, stations: pd.DataFrame, malla: dict, params: dict) -> pd.DataFrame:
    """Tabla de modelado: una fila por estación-día factible × hora de predicción.

    Todas las features se calculan con información ≤ t.
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


def features_de(tabla: pd.DataFrame) -> list[str]:
    return [c for c in tabla.columns if c not in NO_FEATURES]


# --------------------------------------------------------------------------- pipeline completo

def construir_dataset(trips_path: Path, stations_path: Path, params: dict = PARAMS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Viajes crudos → (tabla de modelado, ŝ₀ por estación, resumen del proceso)."""
    stations = cargar_estaciones(stations_path)
    capacidad = stations.set_index("id")["dpcapacity"]

    trips = cargar_viajes(trips_path)
    info = {
        "meses": int(trips["Inicio_del_viaje"].dt.to_period("M").nunique()),
        "primer_viaje": str(trips["Inicio_del_viaje"].min()),
        "ultimo_viaje": str(trips["Inicio_del_viaje"].max()),
        "viajes_crudo": len(trips),
        "viaje_id_duplicados": int(trips["Viaje_Id"].duplicated().sum()),
        "estaciones_con_viajes": len(set(trips["Origen_Id"]) | set(trips["Destino_Id"])),
        "estaciones_nomenclatura": len(stations),
    }
    trips, info["limpieza"] = limpiar_viajes(trips, stations, params["dur_max_h"])
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
    tabla = construir_tabla(ev, cotas, stations, malla, params)
    del ev, malla

    info["filas_tabla"] = len(tabla)
    info["n_features"] = len(features_de(tabla))
    info["params"] = params
    return tabla, s0_est, info


@app.command()
def main(
    trips_path: Path = TRIPS_P2_1,
    stations_path: Path = STATIONS_P2_1,
    out_dir: Path = OUT_DIR_P2_1,
):
    logger.info("P2.1: construyendo eventos, cotas, ŝ₀, etiqueta y features...")
    tabla, s0_est, info = construir_dataset(trips_path, stations_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(out_dir / "tabla.parquet", index=False)
    s0_est.to_csv(out_dir / "s0_por_estacion.csv")
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    logger.success(f"P2.1: {info['filas_tabla']:,} filas, {info['n_features']} features → {out_dir}")


if __name__ == "__main__":
    app()
