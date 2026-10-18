"""Ingesta reproducible de los viajes y la nomenclatura de estaciones de MiBici.

Uso (desde la raíz del repositorio):
    python -m MiBici.dataset                    # descarga, normaliza, consolida y escribe el manifiesto
    python -m MiBici.dataset --solo-verificar   # comprueba los datos locales contra el manifiesto

Pasos:
1. Descarga los CSV mensuales y la nomenclatura del portal de datos abiertos a
   `data/raw/mensuales/`. Los archivos que ya existen no se vuelven a descargar.
2. Normaliza cada mes. Los archivos de MiBici no son homogéneos:
   - codificación: unos meses vienen en UTF-8 y otros en latin-1;
   - encabezado: `Año_de_nacimiento` llega con acento, sin él o corrupto
     (`A}�e_nacimiento` en 2024-02 y 2024-05), así que las columnas se nombran por posición;
   - fechas: septiembre y octubre de 2024 vienen como `dd/mm/aaaa h:mm`, sin segundos;
     se convierten a `aaaa-mm-dd hh:mm:ss` (con segundos en 00).
3. Valida cada mes: 8 columnas, fechas legibles, viajes que inician en el mes del archivo
   y `Viaje_Id` sin duplicados.
4. Consolida los meses, en orden, en `TRIPS_CSV`: el CSV que usan P1, P2 y P3.
5. Escribe `references/data_manifest.json` (versionado en Git) con la URL, el SHA-256 y
   el número de registros de cada archivo y del consolidado. La versión de los datos
   (`data_version`) es el SHA-256 del consolidado, el mismo valor que los scripts de
   entrenamiento registran en MLflow como `datos_sha256`.
"""

import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.request

from loguru import logger
import pandas as pd
import typer

from MiBici.config import PROJ_ROOT, RAW_DATA_DIR, STATIONS_CSV, TRIPS_CSV

app = typer.Typer()

FUENTE = "https://www.mibici.net/es/datos-abiertos/"
BASE_URL = "https://www.mibici.net/site/assets/files"
MENSUALES_DIR = RAW_DATA_DIR / "mensuales"
MANIFEST_PATH = PROJ_ROOT / "references" / "data_manifest.json"

# Los enlaces del portal no siguen un patrón (cada archivo tiene su propio id), así que se fijan aquí.
ARCHIVOS_VIAJES = {
    "2024-01": (61450, "datos_abiertos_2024_01.csv"),
    "2024-02": (77251, "datos_abiertos_2024_02.csv"),
    "2024-03": (77252, "datos_abiertos_2024_03.csv"),
    "2024-04": (80028, "datos_abiertos_2024_04.csv"),
    "2024-05": (81663, "datos_abiertos_2024_05.csv"),
    "2024-06": (83901, "datos_abiertos_2024_06.csv"),
    "2024-07": (85500, "datos_abiertos_2024_07.csv"),
    "2024-08": (87665, "datos_abiertos_2024_08.csv"),
    "2024-09": (89736, "datos_abiertos_2024_09.csv"),
    "2024-10": (91758, "datos_abiertos_2024_10.csv"),
    "2024-11": (93516, "datos_abiertos_2024_11.csv"),
    "2024-12": (94962, "datos_abiertos_2024_12.csv"),
    "2025-01": (97653, "datos_abiertos_2025_01.csv"),
    "2025-02": (99879, "datos_abiertos_2025_02.csv"),
    "2025-03": (101986, "datos_abiertos_2025_03.csv"),
    "2025-04": (103813, "datos_abiertos_2025_04.csv"),
    "2025-05": (105513, "datos_abiertos_2025_05.csv"),
    "2025-06": (107088, "datos_abiertos_2025_06.csv"),
    "2025-07": (108898, "datos_abiertos_2025_07.csv"),
    "2025-08": (111021, "datos_abiertos_2025_08.csv"),
    "2025-09": (113216, "datos_abiertos_2025_09.csv"),
    "2025-10": (115104, "datos_abiertos_2025_10.csv"),
    "2025-11": (116773, "datos_abiertos_2025_11.csv"),
    "2025-12": (118153, "datos_abiertos_2025_12.csv"),
}
NOMENCLATURA = (1118, "nomenclatura_2026_09.csv")

COLUMNAS = [
    "Viaje_Id", "Usuario_Id", "Genero", "Anio_de_nacimiento",
    "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id",
]
COLUMNAS_FECHA = ["Inicio_del_viaje", "Fin_del_viaje"]
FORMATO_ISO = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
FORMATO_DMY = "%d/%m/%Y %H:%M"  # sep y oct 2024


# --------------------------------------------------------------------------- descarga

def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def url_de(file_id: int, nombre: str) -> str:
    return f"{BASE_URL}/{file_id}/{nombre}"


def descargar(url: str, destino: Path, forzar: bool = False) -> Path:
    """Descarga `url` a `destino`; si ya existe y no se fuerza, lo deja como está.

    Se escribe primero en un archivo temporal, para que una descarga interrumpida no
    deje un archivo incompleto con el nombre final.
    """
    if destino.exists() and destino.stat().st_size > 0 and not forzar:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "MiBici-Equipo37/ingesta"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(destino)
    logger.info(f"descargado {destino.name}")
    return destino


# --------------------------------------------------------------------------- normalización

def decodificar(raw: bytes) -> tuple[str, str]:
    """Texto del archivo y la codificación con la que se leyó (UTF-8 o latin-1)."""
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def normalizar_fechas(serie: pd.Series) -> tuple[pd.Series, str]:
    """Convierte las fechas a `aaaa-mm-dd hh:mm:ss`.

    Acepta el formato ISO (la mayoría de los meses) y `dd/mm/aaaa h:mm` (sep y oct 2024),
    al que se le ponen los segundos en 00. Falla si alguna fecha no tiene ninguno de los dos.
    """
    iso = serie.str.match(FORMATO_ISO)
    if iso.all():
        return serie, "iso"
    convertidas = pd.to_datetime(serie[~iso], format=FORMATO_DMY).dt.strftime("%Y-%m-%d %H:%M:%S")
    return serie.where(iso, convertidas), "mixto" if iso.any() else "dd/mm/aaaa h:mm"


def leer_mes(path: Path, mes: str) -> tuple[pd.DataFrame, dict]:
    """Lee un CSV mensual y lo deja con columnas, codificación y fechas normalizadas.

    Todo se lee como texto para no alterar los valores (por ejemplo, que un año faltante
    convierta la columna a float y `1984` se escriba como `1984.0`).
    """
    texto, encoding = decodificar(path.read_bytes())
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False)
    if df.shape[1] != len(COLUMNAS):
        raise ValueError(f"{path.name}: se esperaban {len(COLUMNAS)} columnas y hay {df.shape[1]}: {list(df.columns)}")
    df.columns = COLUMNAS

    formatos = set()
    for col in COLUMNAS_FECHA:
        df[col], formato = normalizar_fechas(df[col])
        formatos.add(formato)

    fuera = ~df["Inicio_del_viaje"].str.startswith(mes)
    if fuera.any():
        raise ValueError(f"{path.name}: {int(fuera.sum())} viajes inician fuera de {mes}")
    if df["Viaje_Id"].duplicated().any():
        raise ValueError(f"{path.name}: hay Viaje_Id duplicados")
    for col in ["Viaje_Id", "Origen_Id", "Destino_Id"]:
        if not df[col].str.fullmatch(r"\d+").all():
            raise ValueError(f"{path.name}: {col} tiene valores no numéricos")

    info = {"encoding": encoding, "formato_fecha": "/".join(sorted(formatos)), "registros": len(df)}
    return df, info


def consolidar(archivos: dict[str, Path], destino: Path) -> dict[str, dict]:
    """Normaliza y escribe los meses en orden en un solo CSV; devuelve la info de cada mes.

    Formato (el de la base que usan P1, P2 y P3): encabezado una sola vez, sin comillas,
    faltantes como campo vacío y fin de línea CRLF. Se procesa un mes a la vez para no
    tener los 9 M de viajes en memoria.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    infos = {}
    with tmp.open("wb") as fh:
        for i, (mes, path) in enumerate(archivos.items()):
            df, infos[mes] = leer_mes(path, mes)
            fh.write(df.to_csv(index=False, header=(i == 0), lineterminator="\r\n").encode("ascii"))
    tmp.replace(destino)
    return infos


# --------------------------------------------------------------------------- versionado

def ruta_relativa(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJ_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def data_version(manifest_path: Path = MANIFEST_PATH) -> str:
    """Versión de los datos para registrar en MLflow: el SHA-256 del CSV consolidado."""
    return json.loads(manifest_path.read_text())["data_version"]


def verificar(manifest_path: Path = MANIFEST_PATH) -> bool:
    """Compara los hashes de los archivos locales con los del manifiesto."""
    manifiesto = json.loads(manifest_path.read_text())
    entradas = [*manifiesto["viajes_mensuales"], manifiesto["nomenclatura"], manifiesto["consolidado"]]
    ok = True
    for e in entradas:
        path = PROJ_ROOT / e["archivo"]
        if not path.exists():
            logger.error(f"falta {e['archivo']}")
            ok = False
        elif sha256_archivo(path) != e["sha256"]:
            logger.error(f"{e['archivo']}: el hash no coincide con el manifiesto")
            ok = False
    return ok


# --------------------------------------------------------------------------- punto de entrada

@app.command()
def main(
    desde: str = typer.Option("2024-01", help="Primer mes (aaaa-mm)"),
    hasta: str = typer.Option("2025-12", help="Último mes (aaaa-mm)"),
    salida: Path = TRIPS_CSV,
    estaciones: Path = STATIONS_CSV,
    forzar_descarga: bool = typer.Option(False, help="Volver a descargar aunque los archivos ya existan"),
    solo_verificar: bool = typer.Option(False, help="Solo comparar los archivos locales con el manifiesto"),
):
    if solo_verificar:
        if not verificar():
            raise typer.Exit(code=1)
        logger.success(f"Los datos locales coinciden con el manifiesto (data_version={data_version()[:12]}…)")
        return

    meses = [m for m in ARCHIVOS_VIAJES if desde <= m <= hasta]
    if not meses:
        raise typer.BadParameter(f"no hay meses entre {desde} y {hasta}")

    logger.info(f"Descargando {len(meses)} meses y la nomenclatura en {ruta_relativa(MENSUALES_DIR)}...")
    archivos = {m: descargar(url_de(*ARCHIVOS_VIAJES[m]), MENSUALES_DIR / ARCHIVOS_VIAJES[m][1], forzar_descarga) for m in meses}
    nom = descargar(url_de(*NOMENCLATURA), MENSUALES_DIR / NOMENCLATURA[1], forzar_descarga)

    logger.info(f"Normalizando, validando y consolidando en {ruta_relativa(salida)}...")
    infos = consolidar(archivos, salida)
    mensuales = [{"mes": mes, "archivo": ruta_relativa(path), "url": url_de(*ARCHIVOS_VIAJES[mes]),
                  "sha256": sha256_archivo(path), **infos[mes]} for mes, path in archivos.items()]
    total = sum(m["registros"] for m in mensuales)

    estaciones.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(nom, estaciones)

    version = sha256_archivo(salida)
    manifiesto = {
        "data_version": version,
        "fuente": FUENTE,
        "rango": {"desde": meses[0], "hasta": meses[-1]},
        "viajes_mensuales": mensuales,
        "nomenclatura": {"archivo": ruta_relativa(nom), "url": url_de(*NOMENCLATURA), "sha256": sha256_archivo(nom),
                         "registros": len(pd.read_csv(nom))},
        "consolidado": {"archivo": ruta_relativa(salida), "sha256": version, "registros": total, "columnas": COLUMNAS},
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n")
    logger.success(f"{total:,} viajes de {meses[0]} a {meses[-1]} · data_version={version[:12]}… · "
                   f"manifiesto en {ruta_relativa(MANIFEST_PATH)}")


if __name__ == "__main__":
    app()
