from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from MiBici.config import PROCESSED_DATA_DIR, TRIPS_CSV

app = typer.Typer()

DIAS = ["semana", "finde"]
HORAS_PICO_AM = range(6, 10)
HORAS_PICO_PM = range(17, 21)


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


@app.command()
def main(
    input_path: Path = TRIPS_CSV,
    output_path: Path = PROCESSED_DATA_DIR / "p3_station_profile.csv",
):
    logger.info(f"Leyendo viajes de {input_path}")
    trips = filter_valid_trips(load_trips(input_path))
    counts = station_hourly_counts(trips)
    summary = station_usage_summary(counts, n_service_days(trips))
    profile = station_usage_profile(counts, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path)
    logger.success(f"Perfil de {len(profile)} estaciones guardado en {output_path}")


if __name__ == "__main__":
    app()
