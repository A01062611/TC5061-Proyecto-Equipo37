# TC5061 · Proyecto MiBici · Equipo 37

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto de la materia de MLOps (TC5061, Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey). Usamos los datos abiertos de **MiBici**, el sistema de bicicleta pública del Área Metropolitana de Guadalajara, para responder tres preguntas sobre la demanda y la operación de sus estaciones, con un flujo reproducible: ingesta automatizada, features en código del repositorio, partición temporal y experimentos registrados en MLflow.

## Contenido

- [Preguntas](#preguntas)
- [Datos](#datos)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Ingesta de datos](#ingesta-de-datos)
- [Experimento base](#experimento-base)
- [Flujo de trabajo con Git](#flujo-de-trabajo-con-git)
- [Integrantes](#integrantes)

## Preguntas

| | Pregunta | Tarea | Unidad | Baseline | Modelo candidato | Métricas |
|---|---|---|---|---|---|---|
| **P1** | ¿Cuántos viajes se iniciarán en una estación el día siguiente? | Regresión | Estación × día | Mismo día de la semana anterior | Random Forest | MAE, RMSE |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las próximas 2 horas? | Clasificación multiclase (`normal` / `vaciado` / `llenado`) | Estación × hora | Persistencia de la ocupación estimada; perfil histórico por estación × día × hora | HistGradientBoosting multiclase con pesos por clase | PR-AUC por clase, recall@k y precision@k por hora, F2, Brier |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Estación | K-Means con k fijo, sin PCA | PCA + K-Means (contraste con Ward) | Silueta, Davies-Bouldin, ARI |

Los datos no traen el inventario de las estaciones, así que P2 lo reconstruye a partir de los viajes y la capacidad de cada estación (ver el docstring de `MiBici/features.py`).

## Datos

**Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/): un CSV de viajes por mes y la nomenclatura de estaciones (id, nombre, ubicación y capacidad).

| | |
|---|---|
| Periodo | Enero 2024 – diciembre 2025 (24 meses) |
| Viajes | 9,177,953 |
| Estaciones | 373 con viajes; 484 en la nomenclatura (`nomenclatura_2026_09.csv`) |
| Variables de viaje | `Viaje_Id`, `Usuario_Id`, `Genero`, `Anio_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id` |

**Por qué este rango:** dos años completos cubren la estacionalidad anual (vacaciones, lluvias, diciembre), y el rango termina antes del cambio de red de 2026 (estaciones eléctricas y 53 estaciones reubicadas).

**Partición temporal** (la misma en P1 y P2): entrenamiento de 2024-01-01 a 2025-06-30, validación de 2025-07-01 a 2025-09-30 y prueba de 2025-10-01 a 2025-12-31. No se usa shuffle ni validación cruzada aleatoria.

**Datos personales:** `Usuario_Id`, `Genero` y `Anio_de_nacimiento` no se usan como variables. Los scripts cargan solo las columnas que necesita cada pregunta. Los datos no se suben al repositorio (`data/` está en `.gitignore`) y los artefactos de MLflow solo contienen agregados por estación.

**Condiciones de uso:** el portal no publica una licencia de datos abiertos explícita, así que los datos se usan solo con fines académicos, citando la fuente y sin redistribuir los archivos.

**Versión de los datos:** [`references/data_manifest.json`](references/data_manifest.json) guarda la URL, el SHA-256 y el número de registros de cada archivo mensual y del CSV consolidado. `data_version` es el SHA-256 del consolidado, el mismo valor que cada run registra en MLflow como `datos_sha256`.

## Estructura del repositorio

```
├── Makefile                <- make data, make lint, make format, make test
├── README.md
├── pyproject.toml          <- metadatos del paquete MiBici y configuración de ruff
├── requirements.txt        <- dependencias (incluye el paquete en modo editable: -e .)
├── data/                   <- no se versiona en Git
│   ├── raw/mensuales/      <- CSV mensuales y nomenclatura, tal como se descargan
│   ├── interim/            <- CSV consolidado y normalizado (lo genera dataset.py)
│   └── processed/          <- tablas de modelado por pregunta
├── models/                 <- modelos entrenados (no se versionan)
├── notebooks/              <- exploración; convención: número-iniciales-descripción
├── references/
│   └── data_manifest.json  <- versión de los datos (hash y registros por archivo)
├── reports/figures/        <- gráficas por pregunta (p2/, p3/) y diagrama de arquitectura
├── tests/                  <- pruebas con pytest
└── MiBici/                 <- código fuente
    ├── config.py           <- rutas, experimento de MLflow y constantes compartidas
    ├── dataset.py          <- ingesta: descarga, normalización, consolidación y manifiesto
    ├── features.py         <- feature engineering por secciones: común, P2 y P3
    ├── plots.py
    └── modeling/
        ├── train_p1.py     <- P1: baseline y Random Forest
        ├── train_p2.py     <- P2: baselines y HistGradientBoosting
        └── train_p3.py     <- P3: K-Means y PCA + K-Means
```

## Instalación

Requisitos: Python 3.10 (ver `requires-python` en `pyproject.toml`) y git.

```bash
git clone https://github.com/A01062611/TC5061-Proyecto-Equipo37.git
cd TC5061-Proyecto-Equipo37

python3.10 -m venv .venv          # o un entorno de pyenv / conda con Python 3.10
source .venv/bin/activate
pip install -r requirements.txt   # instala las dependencias y el paquete MiBici en modo editable
```

Para comprobar la instalación (desde cualquier carpeta):

```bash
python -c "import MiBici.config as c; print(c.TRIPS_CSV)"
```

## Ingesta de datos

Un solo comando descarga y prepara los datos, sin pasos manuales:

```bash
python -m MiBici.dataset          # o: make data
```

1. Descarga los 24 CSV mensuales y la nomenclatura a `data/raw/mensuales/` (si ya existen, no se vuelven a descargar).
2. Normaliza la codificación (los meses vienen en UTF-8 o en latin-1), los nombres de columna y las fechas (septiembre y octubre de 2024 vienen como `dd/mm/aaaa h:mm`).
3. Valida cada mes: columnas, fechas dentro del mes y `Viaje_Id` únicos.
4. Escribe el consolidado en `data/interim/mibici_2024_2025.csv`, copia la nomenclatura a `data/interim/` y actualiza `references/data_manifest.json`.

Tarda alrededor de un minuto con los archivos ya descargados. Para comprobar que los datos locales coinciden con la versión registrada en el manifiesto:

```bash
python -m MiBici.dataset --solo-verificar
```

## Experimento base

Las features se construyen con `MiBici/features.py` y quedan en `data/processed/`. P2 necesita su tabla antes de entrenar; P3 calcula el perfil dentro de `train_p3`, así que `features p3` solo sirve para revisarlo:

```bash
python -m MiBici.features         # P2 y P3
python -m MiBici.features p2      # solo P2 → data/processed/p2/ (tabla.parquet, s0_por_estacion.csv, info.json)
python -m MiBici.features p3      # solo P3 → data/processed/p3_station_profile.csv
```

Después se corren los experimentos. Todos los runs se registran en el experimento **`mibici-equipo37`** de MLflow (configuración en `MiBici/config.py`). Cada run guarda los parámetros para reproducirlo (incluidos el rango de fechas y la versión de los datos), las métricas propuestas, artefactos (gráficas y tablas) y las etiquetas `equipo`, `pregunta` y `git_commit`.

```bash
# P1: baseline semanal y Random Forest
python -m MiBici.modeling.train_p1

# P2: baselines (persistencia y perfil histórico) y HistGradientBoosting; lee data/processed/p2/
python -m MiBici.modeling.train_p2            # --help para ver hiperparámetros, semilla y k

# P3: K-Means base y PCA + K-Means
python -m MiBici.modeling.train_p3
```

Para ver los runs:

```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db"   # desde la raíz; abrir http://127.0.0.1:5000
```

Haz commit antes de correr un experimento: así el `git_commit` registrado corresponde al código que generó el run y `git_worktree_clean` queda en `true`.

## Flujo de trabajo con Git

Seguimos **GitHub Flow**:

1. `main` está protegida: nadie hace push directo y todo cambio entra por Pull Request.
2. Cada cambio se hace en una rama nueva creada desde `main` actualizada (`git switch main && git pull && git switch -c <rama>`).
3. Se hacen commits pequeños y se abre un Pull Request por tema, con una descripción de qué cambia y cómo probarlo.
4. Otra persona del equipo revisa el PR y lo aprueba antes de integrarlo. Cada integrante abre al menos un PR y revisa al menos uno de otro integrante.
5. Después de integrar, se borra la rama.

**Nombres de ramas:** `tipo/descripcion-corta`, en minúsculas y con guiones.

| Tipo | Uso | Ejemplo |
|---|---|---|
| `feature/` | Funcionalidad nueva | `feature/ingesta-datos`, `feature/p2-clasificacion-mg` |
| `fix/` | Corrección de un error | `fix/ruta-csv-p1` |
| `docs/` | Documentación | `docs/readme` |
| `refactor/` | Reorganizar código sin cambiar resultados | `refactor/features-unificado` |

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/), con la pregunta o el área como alcance: `tipo(alcance): descripción en minúsculas`.

- Tipos: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- Alcances: `p1`, `p2`, `p3`, `datos`, `docs`.
- Ejemplos: `feat(datos): ingesta reproducible con manifiesto versionado`, `fix(p1): leer el CSV desde TRIPS_CSV`, `docs: flujo de trabajo con Git en el README`.

## Integrantes

| Integrante | Rol |
|---|---|
| Alan Hernández | P3 · clustering de estaciones; administración del repositorio |
| Isaac González | P1 · pronóstico de viajes diarios |
| Edgar González Paz | P2 · propuesta de desequilibrio horario |
| Mario González | P2 · riesgo de vaciado/llenado; ingesta de datos |
