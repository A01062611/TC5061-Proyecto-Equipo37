## 2. Reporte de drift con Evidently

Se comparó el conjunto de referencia (con el que se entrenó el champion, 2024-01 a 2025-06: 3,261,440 filas y 371 estaciones) contra el periodo actual (2026-07 a 2026-10: 1,014,620 filas y 470 estaciones) usando `DataDriftPreset` y `DataSummaryPreset` de Evidently 0.7.23 sobre las 23 variables de entrada de P2. El reporte completo con las 23 variables está en `reports/evidently/drift_report_*.html` (anexo). Para la calidad de datos, `DataSummaryPreset` incluye nulos, rangos y duplicados.

Evidently usa, por variable, la distancia de Wasserstein normada (numéricas) o Jensen-Shannon (categóricas), con umbral de 0.1. Marca drift del conjunto cuando más de 50% de las variables lo presentan. Por rendimiento, cada lado se muestreó a 10,000 filas con `SEED = 42`.

De las 470 estaciones del periodo actual, 101 no existen en la referencia (220,140 filas, 22%). Si se mezclan con las demás, inflan el drift y ocultan lo que le pasa a la red original. Por eso el reporte se corrió en tres subconjuntos del actual:

| Subconjunto del actual | Filas | Estaciones | Variables con drift | % | ¿Drift del conjunto? |
| --- | ---: | ---: | ---: | ---: | --- |
| Todas | 1,014,620 | 470 | 4 / 23 | 17% | No |
| Solo existentes | 794,480 | 369 | 11 / 23 | 48% | No (casi) |
| Solo nuevas | 220,140 | 101 | 18 / 23 | 78% | Sí |

Con todas las estaciones juntas, el drift parece pequeño (4 variables: `frac_vacio_7d`, `frac_lleno_7d`, `C` y `lon`). Es engañoso: el drift fuerte de las estaciones nuevas se diluye porque 78% de las filas son de estaciones existentes. Al separarlas, aparecen dos fenómenos distintos, uno leve en las estaciones existentes y uno fuerte en las nuevas (figura 1).

![Figura 1. Score de drift por variable, estaciones existentes y nuevas. La línea punteada es el umbral de Evidently (0.1). Fuente: `drift_summary_existentes.json` y `drift_summary_nuevas.json`.](../evidently/figuras/drift_por_variable.png)

**Calidad de datos.** Los conjuntos tienen las mismas columnas y tipos (lo verifica `tests/test_monitoring_datasets.py`). El periodo actual se construyó desde el 2026-06-01 para que los lags de julio no tengan NaN, y se recortó a partir del 2026-07-01. El último día operativo completo es el 2026-10-30. Los HTML contienen el detalle de nulos y rangos por variable.

**Qué cambió en la distribución de `clase`.** Entre las estaciones existentes, la distribución es casi igual a la de entrenamiento (7.0% vaciado y 7.0% llenado, contra 7.0% y 6.9%). En las nuevas, solo 2.8% de las filas son vaciado y 1.7% llenado (figura 2). Es un cambio de prior, no de las variables de entrada, y hace que cualquier métrica global sobre el conjunto actual dependa de cuántas estaciones nuevas incluya.

![Figura 2. Distribución real de `clase` por conjunto. Fuente: `classification_summary.json`.](../evidently/figuras/distribucion_clase.png)

**Desempeño del champion con las etiquetas reales.** Se usa la partición `val` como línea base porque el champion no la vio al entrenar (la partición `train` sale optimista). F2 macro baja de 0.737 en validación a 0.724 en las estaciones existentes (−0.013) y a 0.726 en el actual completo. El AP de llenado cae de 0.762 a 0.728 en existentes (−0.034). El de vaciado casi no cambia (0.760 a 0.748). En las estaciones nuevas el F2 macro es 0.754, pero no es comparable: casi no tienen eventos de riesgo, y con tan pocos positivos la métrica es menos estable (ver figura 3).

![Figura 3. Métricas del champion en validación y en el periodo actual, con las etiquetas reales. Fuente: `classification_summary.json`.](../evidently/figuras/desempeno_champion.png)

Se reproduce con:

```bash
python -m MiBici.monitoring.drift --reference data/processed/monitoring/reference.parquet --current data/processed/monitoring/current.parquet --estaciones todas   # y existentes, nuevas
python -m MiBici.monitoring.clasificacion --tau 0.15
python -m MiBici.monitoring.importancia --tau 0.15
python -m MiBici.monitoring.figuras
```
