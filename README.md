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
- [Servidor de MLflow](#servidor-de-mlflow)
- [Ingesta de datos](#ingesta-de-datos)
- [Features](#features)
- [Experimentos con el MLflow Project](#experimentos-con-el-mlflow-project)
- [Reproducir desde un clon limpio](#reproducir-desde-un-clon-limpio)
- [Flujo de trabajo con Git](#flujo-de-trabajo-con-git)
- [Integrantes](#integrantes)

## Preguntas

| | Pregunta | Tarea | Unidad | Baseline | Modelos comparados | Métricas |
|---|---|---|---|---|---|---|
| **P1** | ¿Cuántos viajes se iniciarán en una estación el día siguiente? | Regresión | Estación × día | Mismo día de la semana anterior | Random Forest, HistGradientBoosting, Ridge | MAE, RMSE |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las próximas 2 horas? | Clasificación multiclase (`normal` / `vaciado` / `llenado`) | Estación × hora | Persistencia de la ocupación estimada; perfil histórico por estación × día × hora | HistGradientBoosting multiclase con pesos por clase, regresión logística | PR-AUC por clase, recall@k y precision@k por hora, F2, Brier |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Estación | K-Means con k fijo, sin PCA | PCA + K-Means, clustering jerárquico, mezcla de gaussianas | Silueta, Davies-Bouldin, ARI |

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
├── MLproject               <- MLflow Project: entry points ingest, train, search y main
├── python_env.yaml         <- entorno del MLflow Project (mismas versiones que requirements.txt)
├── .env.example            <- variables de MLflow; copiar a .env
├── Makefile                <- make data, make lint, make format, make test
├── README.md
├── pyproject.toml          <- metadatos del paquete MiBici y configuración de ruff
├── requirements.txt        <- dependencias (incluye el paquete en modo editable: -e .)
├── mlflow.db, mlartifacts/ <- base y artefactos del servidor de MLflow (no se versionan)
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
        ├── train.py        <- punto de entrada del MLproject: despacha a cada pregunta
        ├── helper.py       <- MLflow compartido: servidor, nested runs y artefactos
        ├── train_p1.py     <- P1: baseline, Ridge, HistGradientBoosting y Random Forest
        ├── train_p2.py     <- P2: baselines, HistGradientBoosting y regresión logística
        └── train_p3.py     <- P3: K-Means, PCA + K-Means, jerárquico y mezcla de gaussianas
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

Todo el equipo debe usar la misma versión de MLflow (la de `requirements.txt`). Una base `mlflow.db` creada con una versión más nueva no abre con una más vieja: MLflow responde `out-of-date database schema` y `mlflow db upgrade` no la puede migrar.

## Servidor de MLflow

El Model Registry necesita una base de datos, así que los runs se registran en un servidor local respaldado por SQLite. El servidor se levanta **desde la raíz del repositorio** (`sqlite:///mlflow.db` es relativo a la carpeta actual) y se deja corriendo en su propia terminal:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

La interfaz queda en http://localhost:5000. La primera vez, el servidor crea `mlflow.db` y guarda los artefactos en `mlartifacts/`; ninguno de los dos se versiona.

En cada terminal donde se lancen corridas hay que apuntar a ese servidor y al experimento del equipo:

```bash
cp .env.example .env              # una sola vez
set -a; source .env; set +a       # en cada terminal nueva
```

`.env` define `MLFLOW_TRACKING_URI=http://localhost:5000` y `MLFLOW_EXPERIMENT_NAME=mibici-equipo37`. El equivalente sin `.env` es:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=mibici-equipo37
```

`MLFLOW_TRACKING_URI` tiene que estar en la terminal, no solo en `.env`: `mlflow run` crea el run del proyecto antes de ejecutar nuestro código y solo lee el entorno. Sin ella, `mlflow run` escribe en una base local distinta y la corrida falla al no encontrar su run en el servidor. Si no está definida, el código usa `sqlite:///mlflow.db` en la raíz del repositorio (`MiBici/config.py`).

**Todas las corridas con `mlflow run` llevan `--experiment-name mibici-equipo37`**, como en los comandos de este README. Sin esa opción, `mlflow run` crea el run en el experimento `Default`, el código lo busca en `mibici-equipo37` y la corrida falla con `Cannot start run with ID ... because active experiment ID does not match environment run ID`. `MLFLOW_EXPERIMENT_NAME` en la terminal tiene el mismo efecto que la opción, pero es mejor escribirla en el comando para que quede explícito.

## Ingesta de datos

Un solo comando descarga y prepara los datos, sin pasos manuales:

```bash
mlflow run . -e ingest --experiment-name mibici-equipo37 --env-manager local    # o: python -m MiBici.dataset, make data
```

`ingest` acepta `-P desde=aaaa-mm` y `-P hasta=aaaa-mm` (por defecto, `2024-01` a `2025-12`).

1. Descarga los 24 CSV mensuales y la nomenclatura a `data/raw/mensuales/` (si ya existen, no se vuelven a descargar).
2. Normaliza la codificación (los meses vienen en UTF-8 o en latin-1), los nombres de columna y las fechas (septiembre y octubre de 2024 vienen como `dd/mm/aaaa h:mm`).
3. Valida cada mes: columnas, fechas dentro del mes y `Viaje_Id` únicos.
4. Escribe el consolidado en `data/interim/mibici_2024_2025.csv`, copia la nomenclatura a `data/interim/` y actualiza `references/data_manifest.json`.

Tarda alrededor de un minuto con los archivos ya descargados. Para comprobar que los datos locales coinciden con la versión registrada en el manifiesto:

```bash
python -m MiBici.dataset --solo-verificar
```

## Features

Las features se construyen con `MiBici/features.py` y quedan en `data/processed/`. P2 necesita su tabla antes de entrenar (si no existe, `train_p2` la genera); P3 calcula el perfil dentro de `train_p3`, así que `features p3` solo sirve para revisarlo:

```bash
python -m MiBici.features         # P2 y P3
python -m MiBici.features p2      # solo P2 → data/processed/p2/ (tabla.parquet, s0_por_estacion.csv, info.json)
python -m MiBici.features p3      # solo P3 → data/processed/p3_station_profile.csv
```

## Experimentos con el MLflow Project

Todas las corridas se lanzan con `mlflow run` desde la raíz, con el servidor levantado, `MLFLOW_TRACKING_URI` en la terminal y la opción `--experiment-name mibici-equipo37` (ver [Servidor de MLflow](#servidor-de-mlflow)). Todos los runs quedan en el experimento **`mibici-equipo37`**.

| Entry point | Qué hace | Parámetros |
|---|---|---|
| `ingest` | Descarga y consolida los datos (`dataset.py`) | `desde`, `hasta` |
| `train` | Entrena un modelo de una pregunta, o el modelo por defecto de todas | `question` (`p1`, `p2`, `p3` o `all`), `model` y sus hiperparámetros |
| `search` | Búsqueda de hiperparámetros con nested runs: una corrida padre y una hija por combinación | `question`, `model` |
| `main` | Ingesta, features y `train` con `question=all` | `question` |

Los hiperparámetros se pasan como `-P nombre=valor` y llegan con su tipo al modelo; un hiperparámetro que el modelo no conoce se rechaza o se ignora con un aviso:

```bash
mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300 --experiment-name mibici-equipo37 --env-manager local
mlflow run . -e train -P question=p2 -P model=hist_gradient_boosting -P learning_rate=0.05 --experiment-name mibici-equipo37 --env-manager local
mlflow run . -e train -P question=p3 -P model=pca_kmeans -P n_clusters=6 --experiment-name mibici-equipo37 --env-manager local
mlflow run . -e search -P question=p2 -P model=hist_gradient_boosting --experiment-name mibici-equipo37 --env-manager local
```

`--env-manager local` usa el entorno ya instalado; sin esa opción, MLflow crea uno nuevo a partir de `python_env.yaml`.

### Modelos por pregunta

| Pregunta | `model` | Hiperparámetros por `-P` | Búsqueda (`-e search`) |
|---|---|---|---|
| P1 | `all` (por defecto: baseline y los tres candidatos), `rf`, `hgb`, `ridge` | `n_estimators`, `max_depth`, `min_samples_leaf`, `n_jobs` (Random Forest) | Random Forest, 4 combinaciones alrededor de los valores dados |
| P2 | `baseline` (persistencia y perfil histórico), `hist_gradient_boosting` (por defecto), `logistic_regression` | Los de `DEFAULT_HYPERPARAMS` en `train_p2.py`, por ejemplo `learning_rate`, `max_leaf_nodes`, `C`, `class_weight` | `hist_gradient_boosting` (8 combinaciones, unos 20 minutos) o `logistic_regression` (6) |
| P3 | `baseline`, `pca_kmeans` (por defecto), `agglomerative`, `gaussian_mixture` | `n_clusters`, `n_components`, `linkage`, `covariance_type`; en `pca_kmeans`, `n_clusters=0` elige k por silueta, Davies-Bouldin y concordancia con Ward | `pca_kmeans` (15), `agglomerative` (15), `gaussian_mixture` (20) |

En P1, cada corrida de `train` abre un run padre de comparación con el baseline y los candidatos como runs hijos, y marca al ganador por MAE de validación con la etiqueta `selected_best_p1`.

### Qué registra cada run

- **Etiquetas** para filtrar en la interfaz: `stage_of_work` (`baseline`, `search` o `candidate`), `model_family`, la pregunta (`pregunta` en P2 y P3, `question` en P1), `equipo`, `git_commit` y la versión de los datos.
- **Parámetros** para reproducirlo, incluidos el rango de fechas, la partición y el SHA-256 de los datos.
- **Artefactos**, organizados en subcarpetas:

| Carpeta | Contenido |
|---|---|
| `data/` | Manifiesto de la ingesta (`data_manifest.json`); en P2, también `info.json` y `s0_por_estacion.csv` |
| `config/` | `run_config.json` con la pregunta, el modelo y los hiperparámetros; en P2 y P3, también `MLproject`, `python_env.yaml` y `config.py` |
| `plots/` | P1: real contra predicho y residuales. P2: matriz de confusión, curvas precisión-recall y distribución de clases. P3: mapa y perfiles de los grupos |
| `results/` | P2 y P3: tablas de resultados (métricas de validación, estaciones por grupo, selección de k). P1 guarda su resumen de candidatos en `config/p1_resultados.json` del run de comparación |

- **Modelo** con `log_model`, firma y ejemplo de entrada. En MLflow 3 aparece en la sección *Logged models* del run, no como carpeta de artefactos. El baseline de P1 y el de P2 son reglas, no modelos entrenados, y el clustering jerárquico de P3 no tiene `predict`, así que esos runs no registran modelo.
- **Conjunto de entrenamiento** con `mlflow.log_input` (P2 y P3): siempre la tabla agregada, nunca los viajes individuales.

Haz commit antes de correr un experimento: así el `git_commit` registrado corresponde al código que generó el run y `git_worktree_clean` queda en `true`.

## Reproducir desde un clon limpio

Con la [instalación](#instalación) hecha, el servidor levantado y las variables en la terminal:

```bash
mlflow run . -e ingest --experiment-name mibici-equipo37 --env-manager local      # 1. datos y manifiesto
python -m MiBici.features p2                    # 2. tabla de P2 (opcional: train_p2 la genera si falta)
```

3. Modelo elegido de cada pregunta:

| Pregunta | Modelo elegido | Comando |
|---|---|---|
| P1 | *Pendiente de la comparación en MLflow* | `mlflow run . -e train -P question=p1 -P model=<modelo> [-P hiperparámetros] --experiment-name mibici-equipo37 --env-manager local` |
| P2 | *Pendiente de la comparación en MLflow* | `mlflow run . -e train -P question=p2 -P model=<modelo> [-P hiperparámetros] --experiment-name mibici-equipo37 --env-manager local` |
| P3 | *Pendiente de la comparación en MLflow* | `mlflow run . -e train -P question=p3 -P model=<modelo> [-P hiperparámetros] --experiment-name mibici-equipo37 --env-manager local` |

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
