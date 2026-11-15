# Entregable 8 · Plan de trabajo Equipo 37 (Drift + A/B testing)


## Resumen

Cada integrante hace un PR principal y deja su sección del reporte en ese mismo PR. El modelo que vigilamos es el **P2: clasificación de riesgo de vaciado/llenado** (`mibici-equipo37-p2-riesgo@champion`), que ya está registrado y servido en la API.

- Repo: [TC5061-Proyecto-Equipo37](https://github.com/A01062611/TC5061-Proyecto-Equipo37)
- Entregable: `TC5061_Drift_ABTest_Equipo37.pdf`, de 3 a 5 cuartillas, Arial 11, interlineado 1.15
- Lo que ya existe y se conserva: drift con DeepChecks (`scripts/validate_drift.py`), `register.py` con aliases `champion`/`challenger`, la API en contenedor y las pruebas
- La API no registra peticiones, así que el conjunto actual sale de datos abiertos posteriores. MiBici ya publicó de enero a octubre de 2026 ([datos abiertos](https://mibici.net/es/datos-abiertos/))

## Periodos de datos

La referencia es el periodo de entrenamiento del champion y el actual es julio–octubre de 2026. Al actual lo dividimos en dos ventanas para que el A/B no tenga fuga de datos.

| Conjunto | Periodo | Uso |
| --- | --- | --- |
| Referencia | 2024-01 a 2025-06 (partición `train`) | Con lo que se entrenó el champion. Base del reporte de drift |
| Actual (completo) | 2026-07 a 2026-10 | "Producción": un año después y con estaciones de la nomenclatura 2026. Lado actual del reporte de drift |
| Ventana de reentrenamiento | 2026-07 a 2026-08 | Solo se usa para entrenar el challenger |
| Ventana de evaluación A/B | 2026-09 a 2026-10 | Ningún modelo la ve al entrenar. Las dos versiones se evalúan aquí |

Por qué estos periodos: julio–octubre coincide en temporada con meses que sí están en el entrenamiento (jul–dic 2024). Así, el drift que aparezca se puede atribuir al paso del tiempo y a cambios en la red, y no a la estacionalidad. En el reporte hay que documentar el número de filas y de estaciones de cada conjunto, porque la rúbrica pide que los conjuntos estén "definidos con claridad y precisión".

Si el equipo prefiere no descargar datos de 2026, la alternativa es usar como actual la partición `test` (2025-10 a 2025-12), que tampoco se ha usado. Implica menos trabajo, pero muestra menos drift.

## División en PRs

Son cinco PRs para cuatro personas. Quien hace los datos tiene la parte más corta, así que también hace la política de Continuous Training y la integración del PDF.

| Integrante | PR y rama | Código en el repo | Sección del reporte |
| --- | --- | --- | --- |
| **A** | PR1 · `feat/monitoring-data` | Extender `dataset.py` y `data_manifest.json` a 2026-06…10. `MiBici/monitoring/datasets.py` genera `reference.parquet`, `current.parquet` y las dos ventanas, con las 23 features de P2, `clase` y la predicción del champion. Agregar `evidently` con versión fija a `requirements.txt`. Test de esquema: mismas columnas y tipos en ambos conjuntos | 1. Datos y periodos: qué se eligió, por qué, filas y estaciones por conjunto |
| **A** | PR5 · `feat/ct-policy` | `MiBici/monitoring/ct_policy.py` (o un YAML con umbrales) que lee el JSON de drift y el resultado del A/B y responde: reentrenar sí/no y promover sí/no. Targets `make monitor` que encadenan todo. Sección de monitoreo en el README | 6. Recomendación de Continuous Training + integración del PDF |
| **B** | PR2 · `feat/evidently-drift` | `MiBici/monitoring/drift.py` con `DataDriftPreset` + `DataSummaryPreset` + `ClassificationPreset` (este último con las etiquetas reales del periodo actual, para medir si el champion ya se degradó). Salidas en `reports/evidently/`: HTML, JSON resumen y PNG para el PDF | 2–3. Reporte de drift e interpretación por variable |
| **C** | PR3 · `feat/p2-challenger` | Parametrizar `train_p2` (por ejemplo `--train-fin 2026-08-31`) para agregar la ventana de reentrenamiento. Run en MLflow y registro como nueva versión del mismo modelo con alias `challenger`, usando `register.py`. Test de que la v2 carga por `models:/...@challenger` | 4. Cómo se entrenó y registró la v2, y qué cambia respecto a la v1 |
| **D** | PR4 · `feat/ab-test` | `MiBici/monitoring/ab_test.py` carga `@champion` y `@challenger` y los evalúa sobre la ventana A/B con las métricas que ya usa `train_p2` (F-beta, AP por clase, Brier). Bootstrap pareado para el intervalo de confianza de la diferencia y prueba de McNemar. Registra la comparación como run en MLflow. Tabla en `reports/ab_test/` | 5. Tabla A/B, prueba estadística y decisión |

**1. Reporte de drift con Evidently** (B, con datos de A)

- [ ] Periodos con fechas exactas, número de filas y de estaciones en cada conjunto
- [ ] Reporte con drift por variable y calidad de datos (nulos, rangos, duplicados)
- [ ] Gráficas de Evidently exportadas en PNG, con pie de figura, y el HTML adjunto como anexo

**2. Interpretación del drift** (B)

- [ ] Tabla de variables con drift: prueba estadística, score y umbral (Evidently los reporta)
- [ ] Cruzar el drift con la importancia de variables del champion (importancia por permutación). Que se desvíe una variable importante no es lo mismo que se desvíe una irrelevante, y esa distinción es lo que la rúbrica llama "razonamiento profundo"
- [ ] Revisar si cambió la distribución de `clase` (target/prior drift) y si bajó la métrica del champion en el periodo actual (`ClassificationPreset`)
- [ ] Explicar la causa probable de cada drift importante: estaciones nuevas, cambios de tarifa, eventos o temporada
- [ ] Comparar con lo que ya había encontrado DeepChecks

**3. A/B testing** (D, con el modelo de C)

- [ ] Escribir la regla de decisión antes de correr la prueba. Por ejemplo: se promueve si la F-beta mejora, el IC 95% de la diferencia excluye 0 y ninguna clase empeora más de 2 puntos de recall
- [ ] Mismo conjunto de evaluación para las dos versiones, que ninguna haya visto al entrenar
- [ ] Tabla comparativa: métrica por versión, diferencia, IC 95% y p-valor de McNemar
- [ ] Opcional: simular un reparto de tráfico 50/50 por estación para que se parezca más a un A/B en producción

**4. Recomendación de Continuous Training** (A)

- [ ] Decisión explícita: promover o no al challenger, y reentrenar o no
- [ ] Disparadores concretos: porcentaje de variables con drift, caída de la métrica respecto a la validación y periodicidad (por ejemplo, mensual)
- [ ] Riesgos y límites: el challenger solo se evaluó en dos meses, hay etiquetas con retraso, estaciones nuevas, etc.

**5. Claridad y colaboración en Git** (todos)

- [ ] Cada PR tiene al menos un revisor de otro integrante
- [ ] Commits con mensajes convencionales y la cuenta correcta de cada quien
- [ ] El PDF sigue la misma estructura que las secciones de este plan

## Dependencias y calendario

PR1 es el único que bloquea a los demás, así que tiene que estar en `main` el miércoles. PR2 y PR3 corren en paralelo el jueves, y PR4 necesita el challenger ya registrado.

&#91;embedded content: dependencias entre PRs · 11 al 15 de noviembre\]

Si la fecha de entrega cae antes del domingo, se recorre todo un día. Lo que no puede moverse es el orden: datos → challenger → A/B.

## Git y estructura del PDF

- Cada sección va como Markdown en `reports/monitoring/NN_<tema>.md`. A solo junta los archivos y arma el PDF.
- Si los parquet son pesados, no se suben al repo: se documenta el comando que los genera.

Estructura del PDF (3–5 cuartillas sin anexos):

1. Datos y periodos elegidos (A), aproximadamente media cuartilla
2. Reporte de drift con gráficas de Evidently (B), una cuartilla
3. Interpretación del drift (B), media a una cuartilla
4. Versión alternativa del modelo (C), media cuartilla
5. A/B testing: tabla y decisión (D), una cuartilla
6. Recomendación de Continuous Training (A), media cuartilla
7. Anexos: HTML de Evidently, enlace al repo y comandos para reproducir

## Errores comunes que hay que evitar

- **Fuga en el A/B:** si el challenger se entrena con los meses en los que después se evalúa, gana por diseño. La ventana 2026-09…10 no se toca hasta el A/B.
- **API de Evidently:** desde la versión 0.7 se usa `from evidently import Report` y `report.run(current, reference)`, con el actual primero. Hay que fijar la versión y no copiar ejemplos de 0.4.
- **Lags en la ventana nueva:** `riesgo_ayer` y las features con historia necesitan días previos. La construcción tiene que empezar en junio de 2026 para que julio no tenga NaN.
- **Estaciones nuevas:** las que no existen en la referencia pueden inflar el drift. Hay que reportarlas aparte y no mezclarlas sin decirlo.
- **Muestreo:** si se muestrea por rendimiento (DeepChecks usa 10 000 filas), se usa la misma semilla (`SEED = 42`) y se documenta.
- **Decidir antes de ver los resultados:** la regla de promoción queda escrita en el PR4 antes de correr el A/B.
