# Model Card: riesgo de vaciado o llenado de estaciones MiBici (P2)

| | |
|---|---|
| **Modelo registrado** | `mibici-equipo37-p2-riesgo`, versión 1, alias `champion` |
| **Tipo** | Clasificación multiclase: `normal` (0), `vaciado` (1), `llenado` (2) |
| **Algoritmo** | `HistGradientBoostingClassifier` de scikit-learn 1.7.2, con regla de decisión por umbral τ = 0.15 |
| **Autores** | Equipo 37, TC5061 Operaciones de Aprendizaje Automático, Tecnológico de Monterrey |
| **Fecha de esta ficha** | 25 de noviembre de 2026 |
| **Licencia del código** | MIT. Los datos se usan solo con fines académicos (ver [Datos](#datos)) |

## Resumen

El modelo predice, para cada estación de MiBici y cada hora, si en las **próximas 2 horas** la estación se quedará sin bicicletas (vaciado), sin anclajes libres (llenado) o seguirá normal. Su propósito es que el equipo de operación sepa qué estaciones visitar para reubicar bicicletas antes de que un usuario llegue y no encuentre servicio.

En validación detecta **84% de los vaciados y 85% de los llenados**, con F2 macro de 0.736. Un año después, en el periodo de monitoreo de 2026, conserva un F2 macro de 0.726, y la comparación A/B confirmó que sigue siendo mejor que una versión reentrenada.

## Uso previsto

- **Usuarios:** el equipo de operación de MiBici que planea la reubicación de bicicletas.
- **Decisión que apoya:** qué estaciones visitar en la siguiente hora. La métrica de negocio es `recall@20`: la fracción de eventos de riesgo que se cubren si se visitan las 20 estaciones con mayor riesgo cada hora.
- **Forma de uso:** inferencia en tiempo real con `POST /predict_p2/`. La API recibe el estado actual de una estación y responde la clase y un mensaje con la acción sugerida.
- **Horario:** 20 horas de predicción por día. El inventario se reinicia a las 5:00, cuando la operación reacomoda la red.

## Usos fuera de alcance

- Otras ciudades o sistemas de bicicleta pública, o la red de MiBici después de un cambio grande sin reentrenar.
- Horizontes distintos de 2 horas o pronósticos de demanda: para eso está el modelo P1.
- Decisiones sobre personas, como tarifas, sanciones o perfiles de usuarios. El modelo trabaja solo con agregados por estación.
- Usar la clase como inventario real. La ocupación es una estimación reconstruida a partir de los viajes (ver [Limitaciones](#limitaciones)).

## Datos

**Fuente:** [datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/), con un CSV de viajes por mes y la nomenclatura de estaciones. El portal no publica una licencia explícita, así que los datos se usan solo con fines académicos, citando la fuente y sin redistribuir los archivos.

| Partición | Periodo | Filas (estación × hora) |
|---|---|---:|
| Entrenamiento | 2024-01-01 a 2025-06-30 | 3,261,440 |
| Validación | 2025-07-01 a 2025-09-30 | 568,180 |
| Prueba (reservada) | 2025-10-01 a 2025-12-31 | No se usó |
| Monitoreo | 2026-07-01 a 2026-10-30 | 1,014,040 |

- **Versión de los datos:** SHA-256 `2c3c2931…d811314f` del CSV consolidado, registrado en [`references/data_manifest.json`](../references/data_manifest.json) y en el run de MLflow como `datos_sha256`.
- **Partición temporal**, sin mezcla aleatoria, para evaluar como se usaría en producción: con datos posteriores al entrenamiento.
- **Cómo se construye la etiqueta:** los datos no incluyen el inventario de cada estación, así que se reconstruye la ocupación con los viajes de entrada y salida y la capacidad de la estación. Una estación está en riesgo de vaciado si su ocupación baja de 10% de la capacidad en las siguientes 2 horas, y en riesgo de llenado si supera 90%. Los días en que la reconstrucción no es factible se excluyen.
- **Balance de clases en validación:** 86.5% normal, 7.0% vaciado y 6.5% llenado.

### Variables de entrada

Son 23, todas agregadas por estación. La firma del modelo y el esquema de la API las validan.

| Grupo | Variables |
|---|---|
| Estado actual | `hora`, `ocup_est`, `estado_est`, `tend_1h` |
| Demanda reciente | `sal_1h`, `sal_2h`, `lle_1h`, `lle_2h` |
| Demanda estacional | Salidas y llegadas de la misma ventana ayer, hace una semana y en promedio de 7 días |
| Historial de riesgo | `frac_vacio_7d`, `frac_lleno_7d`, `riesgo_ayer` |
| Calendario | `dia_semana`, `fin_semana`, `festivo` |
| Estación | `C` (capacidad), `lat`, `lon` |

`ocup_est`, la ocupación estimada, es por mucho la variable más importante: al permutarla, el F2 macro cae 0.416, 14 veces más que con la siguiente.

## Desempeño

Con la regla de τ = 0.15: si la probabilidad de vaciado o de llenado llega a 0.15, se predice la más probable de las dos; si no, `normal`. El umbral se eligió para maximizar F2, que pondera el recall más que la precisión, porque no visitar una estación en riesgo cuesta más que una visita innecesaria.

La API aplica la misma regla (`MODEL_TAU`, 0.15 por defecto), y en validación sus predicciones coinciden fila por fila con las del entrenamiento. Sin el umbral, eligiendo solo la clase más probable, el recall bajaría a 0.56 en vaciado y 0.59 en llenado.

| Métrica | Validación 2025 | Monitoreo 2026 | Ventana A/B (sep–oct 2026) |
|---|---:|---:|---:|
| F2 macro (vaciado y llenado) | 0.736 | 0.726 | 0.726 |
| Recall vaciado | 0.836 | 0.827 | 0.827 |
| Recall llenado | 0.849 | 0.853 | 0.851 |
| AP vaciado | 0.760 | 0.750 | 0.749 |
| AP llenado | 0.762 | 0.730 | 0.722 |
| Brier vaciado | 0.031 | 0.027 | 0.031 |
| Brier llenado | 0.028 | 0.027 | 0.031 |

**Métrica de negocio:** visitando 20 estaciones por hora se cubre 40.8% de los eventos de riesgo de validación, con 80.2% de precisión. Con 20 visitas por hora, el máximo alcanzable es 53.4%, porque hay horas con más de 20 estaciones en riesgo.

**Por subgrupo, en el monitoreo de 2026:**

| Subgrupo | Estaciones | F2 macro | Comentario |
|---|---:|---:|---|
| Existentes desde el entrenamiento | 369 | 0.724 | Baja 0.012 contra validación; el AP de llenado baja 0.033 |
| Nuevas en 2026 | 101 | 0.755 | No es comparable: tienen pocos eventos de riesgo (2.8% vaciado y 1.7% llenado). Recall de 0.798 en vaciado y 0.882 en llenado |

Las cifras vienen de [`classification_summary.json`](evidently/classification_summary.json), [`02_reporte_drift.md`](monitoring/02_reporte_drift.md) y [`05_ab_test.md`](monitoring/05_ab_test.md).

## Linaje y ciclo de vida

| | |
|---|---|
| **Run de entrenamiento** | `acdae4e1edea4f7bb67d77de0e2ea24e` (`P2 - hist_gradient_boosting`), experimento `mibici-equipo37` |
| **Código** | Commit `6a5c9f8`, con el árbol de trabajo limpio, entrenado con `mlflow run . -e train -P question=p2` |
| **Hiperparámetros** | `learning_rate=0.1`, `max_iter=300`, `max_leaf_nodes=63`, `min_samples_leaf=200`, `l2_regularization=1.0`, sin pesos por clase, `seed=42` |
| **Registro** | Etiquetas `run_origen`, `git_commit`, `version_datos`, `pregunta` y `model_family` en la versión del Registry. Para el monitoreo y el A/B, el champion se reimportó sin reentrenar con [`scripts/import_champion.py`](../scripts/import_champion.py), que verifica que las predicciones sean idénticas |
| **Despliegue** | Exportado con [`scripts/export_model.py`](../scripts/export_model.py) a `models/mibici-equipo37-p2-riesgo/` (9.8 MB, formato `skops`) y empacado en la imagen de la API |
| **Estado** | `champion` en producción. El `challenger` (v2, reentrenado con julio y agosto de 2026) **no se promovió**: su F2 macro fue 0.0038 menor, con IC 95% [−0.0057, −0.0018] |

Se usan aliases en lugar de las etapas `Staging` y `Production`, que MLflow marcó como obsoletas. Promover una versión consiste en mover el alias `champion`, sin cambiar el código que carga `models:/mibici-equipo37-p2-riesgo@champion`.

## Operación

- **Servicio:** FastAPI en contenedor, orquestada con Docker Compose y validada en CI con un smoke test que hace una predicción real.
- **Capacidad:** con 1 CPU y 1 worker atiende 50 usuarios simultáneos (unas 38 peticiones por segundo) con latencia p95 de 210 ms y sin errores. Con 150 usuarios se satura. Detalle en [`load_test/README.md`](load_test/README.md).
- **Costo estimado:** USD 0.264 por millón de inferencias con 1 CPU, según [`finops/analisis_finops.md`](finops/analisis_finops.md).
- **Monitoreo:** cada mes, cuando MiBici publica los viajes, se mide drift con Evidently y desempeño con las etiquetas reales.
- **Reentrenamiento:** la [política de Continuous Training](monitoring/06_ct.md) define disparadores urgentes (F2 macro cae más de 0.02, recall menor a 0.80 o drift en más de 30% de la importancia del modelo) y programados (más de 20% de filas de estaciones nuevas, o datos de más de 12 meses). Hoy están activos los dos programados, así que hay un reentrenamiento planeado, pero no urgente.

## Consideraciones éticas

- **Datos personales:** los viajes incluyen `Usuario_Id`, `Genero` y `Anio_de_nacimiento`, pero no se usan. Los scripts cargan solo las columnas necesarias, y los artefactos de MLflow y el modelo contienen solo agregados por estación.
- **Equidad territorial:** el modelo prioriza estaciones con más eventos de riesgo, que suelen ser las de mayor demanda. Si solo se usa `recall@20`, las estaciones con menos movimiento, como las nuevas o las periféricas, podrían recibir menos visitas. Conviene revisar periódicamente la cobertura por zona.
- **Decisión humana:** el modelo sugiere visitas; la decisión y la planeación de rutas siguen siendo del equipo de operación.

## Limitaciones

- **La etiqueta es una estimación.** El inventario real no está en los datos abiertos, y la ocupación se reconstruye con viajes, capacidad y un reinicio a las 5:00. Los rebalanceos que hace la operación no aparecen en los datos y pueden generar errores en la etiqueta.
- **Estaciones nuevas.** Tienen poca historia, así que las variables de 7 días y `frac_*` se calculan con pocos datos. Concentran el drift más fuerte: 18 de 23 variables.
- **Calibración del llenado.** Con más demanda en 2026, las probabilidades de llenado son menos precisas: el AP bajó de 0.762 a 0.729 en las estaciones existentes.
- **Etiquetas con retraso.** MiBici publica los viajes mensualmente, así que una degradación se detecta con hasta un mes de retraso.
- **Evaluación offline.** El A/B se hizo con datos históricos, no con tráfico real.
- **Partición de prueba.** Octubre a diciembre de 2025 se reservó y no se ha evaluado. El desempeño fuera de la muestra se mide con el periodo de monitoreo de 2026.

## Referencias

- Mitchell, M., et al. (2019). *Model Cards for Model Reporting*. Proceedings of FAT\* 2019.
- Documentación del proyecto: [README](../README.md), [monitoreo y A/B](monitoring/), [pruebas de carga](load_test/README.md), [FinOps](finops/analisis_finops.md) y [seguridad](security/INFORME_SEGURIDAD.md).
