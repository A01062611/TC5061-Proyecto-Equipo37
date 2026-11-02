"""Validación reproducible del CSV consolidado MiBici.

Ejecutar desde la raíz del repositorio:
    python scripts/validate_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

COLUMNAS = [
    "Viaje_Id", "Usuario_Id", "Genero", "Anio_de_nacimiento",
    "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id",
]
IDENTIFICADORES = ("Viaje_Id", "Origen_Id", "Destino_Id")
FECHAS = ("Inicio_del_viaje", "Fin_del_viaje")
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"
NULOS = ("", "na", "n/a", "null", "none", "nan")


def hash_archivo(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(8 * 1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def analizar_bloque(bloque: pd.DataFrame) -> dict:
    """Cuenta problemas locales sin modificar los datos de entrada."""
    if list(bloque.columns) != COLUMNAS:
        raise ValueError("El esquema del CSV no coincide con las ocho columnas esperadas")
    texto = bloque.fillna("").astype(str)
    vacios = {
        col: int(texto[col].str.strip().str.lower().isin(NULOS).sum())
        for col in COLUMNAS
    }
    invalidos = {
        col: int((~texto[col].str.fullmatch(r"[0-9]+", na=False)).sum())
        for col in IDENTIFICADORES
    }
    fechas = {}
    for col in FECHAS:
        validas_formato = texto[col].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", na=False)
        parsed = pd.to_datetime(texto[col], format=FORMATO_FECHA, errors="coerce")
        fechas[col] = parsed
        invalidos[col] = int((~validas_formato | parsed.isna()).sum())
    orden_invalido = int((fechas[FECHAS[0]] > fechas[FECHAS[1]]).sum())
    return {
        "registros": len(bloque),
        "faltantes": vacios,
        "invalidos": invalidos,
        "inicio_posterior_fin": orden_invalido,
    }


def validar_completo(ruta: Path, chunksize: int = 100_000) -> dict:
    """Procesa por bloques; usa SQLite temporal para unicidad global de Viaje_Id."""
    import sqlite3
    import tempfile

    resumen = {
        "registros": 0,
        "faltantes": {col: 0 for col in COLUMNAS},
        "invalidos": {col: 0 for col in (*IDENTIFICADORES, *FECHAS)},
        "inicio_posterior_fin": 0,
        "viaje_id_duplicados": 0,
    }
    with tempfile.TemporaryDirectory(prefix="mibici-calidad-") as carpeta:
        with sqlite3.connect(str(Path(carpeta) / "ids.sqlite")) as conexion:
            conexion.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            for numero, bloque in enumerate(pd.read_csv(ruta, dtype=str, chunksize=chunksize, keep_default_na=False), 1):
                actual = analizar_bloque(bloque)
                resumen["registros"] += actual["registros"]
                for campo in ("faltantes", "invalidos"):
                    for columna, cantidad in actual[campo].items():
                        resumen[campo][columna] += cantidad
                resumen["inicio_posterior_fin"] += actual["inicio_posterior_fin"]
                antes = conexion.total_changes
                conexion.executemany(
                    "INSERT OR IGNORE INTO ids (id) VALUES (?)",
                    ((identificador,) for identificador in bloque["Viaje_Id"].fillna("").astype(str)),
                )
                insertados = conexion.total_changes - antes
                resumen["viaje_id_duplicados"] += len(bloque) - insertados
                conexion.commit()
                print(f"Bloque {numero}: {resumen['registros']:,} registros", flush=True)
    return resumen


def ejecutar_deepchecks(ruta: Path, salida: Path, muestra: int, semilla: int) -> dict:
    from deepchecks.tabular import Dataset, Suite
    from deepchecks.tabular.checks import (
        DataDuplicates, MixedDataTypes, MixedNulls, SpecialCharacters, StringMismatch,
    )

    # Muestreo determinista sobre un archivo de origen fijo, sin cargarlo íntegro.
    # La selección uniforme usa índices de fila y dos pasadas secuenciales.
    total = sum(len(chunk) for chunk in pd.read_csv(ruta, dtype=str, chunksize=100_000, keep_default_na=False))
    if total < muestra:
        raise ValueError(f"El dataset solo tiene {total} filas; se solicitaron {muestra}")
    import numpy as np
    indices = set(np.random.default_rng(semilla).choice(total, size=muestra, replace=False).tolist())
    partes = []
    inicio = 0
    for chunk in pd.read_csv(ruta, dtype=str, chunksize=100_000, keep_default_na=False):
        posiciones = [i - inicio for i in range(inicio, inicio + len(chunk)) if i in indices]
        if posiciones:
            partes.append(chunk.iloc[posiciones])
        inicio += len(chunk)
    df = pd.concat(partes, ignore_index=True)
    # Mantener la semántica de faltantes observada en el diagnóstico inicial.
    df = df.replace(r"^\s*$", float("nan"), regex=True)
    dataset = Dataset(df, cat_features=["Genero"], index_name="Viaje_Id")
    suite = Suite(
        "MiBici - Integridad general",
        MixedNulls(), MixedDataTypes(), DataDuplicates(), StringMismatch(), SpecialCharacters(),
    )
    resultado = suite.run(dataset)
    salida.mkdir(parents=True, exist_ok=True)
    ruta_html = salida / "integridad_general.html"
    if ruta_html.exists():
        ruta_html.unlink()
    resultado.save_as_html(str(ruta_html), as_widget=False)
    detalles = []
    for nombre, check in zip(
        ("MixedNulls", "MixedDataTypes", "DataDuplicates", "StringMismatch", "SpecialCharacters"),
        resultado.results,
    ):
        detalles.append({"chequeo": nombre, "tipo": type(check).__name__, "resultado": str(getattr(check, "value", None))[:5000]})
    return {"muestra": len(df), "semilla": semilla, "chequeos": detalles}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calidad de datos MiBici con pandas y DeepChecks")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/mibici_2024_2025.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("references/data_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/deepchecks"))
    parser.add_argument("--sample", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample <= 0:
        parser.error("--sample debe ser positivo")
    if not args.csv.is_file() or not args.manifest.is_file():
        parser.error("No se encontró el CSV o el manifiesto de datos")
    manifiesto = json.loads(args.manifest.read_text(encoding="utf-8"))
    digest = hash_archivo(args.csv)
    if digest != manifiesto["data_version"]:
        raise ValueError("SHA-256 del CSV no coincide con el manifiesto")
    args.out.mkdir(parents=True, exist_ok=True)
    globales = validar_completo(args.csv)
    deepchecks = ejecutar_deepchecks(args.csv, args.out, args.sample, args.seed)
    reporte = {"archivo": str(args.csv), "sha256": digest, "global": globales, "deepchecks": deepchecks}
    (args.out / "resumen_calidad.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validación completada. Reportes en {args.out}")


if __name__ == "__main__":
    main()
