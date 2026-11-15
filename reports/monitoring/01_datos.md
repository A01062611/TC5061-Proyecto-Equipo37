## 1. Datos y periodos

Se monitorea el modelo P2 (`mibici-equipo37-p2-riesgo@champion`), que clasifica el riesgo de vaciado o llenado de cada estación en las próximas dos horas. El periodo de producción se reconstruyó con los datos abiertos de MiBici de junio a octubre de 2026. Se procesaron con el mismo código y el mismo inventario inicial por estación que la tabla de entrenamiento, para no introducir drift artificial.

| Conjunto | Periodo (día operativo) | Filas | Estaciones (nuevas) | Uso |
|---|---|---:|---:|---|
| Referencia | 2024-01-01 a 2025-06-30 | 3,244,300 | 369 (0) | Entrenamiento del champion; base del drift |
| Actual | 2026-07-01 a 2026-10-30 | 1,014,040 | 470 (101) | Producción; lado actual del drift |
| Reentrenamiento | 2026-07-01 a 2026-08-31 | 511,780 | 468 (100) | Solo para entrenar el challenger |
| Evaluación A/B | 2026-09-01 a 2026-10-30 | 502,260 | 469 (101) | Solo para comparar las dos versiones |

Cada fila es una estación en una hora de predicción de un día con inventario factible.

**Por qué estos periodos.** Julio a octubre coincide en temporada con meses de la referencia (julio a diciembre de 2024), así que el drift se puede atribuir al paso del tiempo y a los cambios en la red, no a la estacionalidad. El periodo actual se divide en dos ventanas sin traslape para que el A/B no tenga fuga de datos: el challenger se entrena con julio y agosto, y las dos versiones se evalúan con septiembre y octubre.

**Cambios visibles antes del análisis de drift.**

- **Volumen:** 15,501 viajes diarios de julio a octubre de 2026, 21.6% más que en los mismos meses de 2024.
- **Estaciones nuevas:** 101 estaciones no aparecen en la referencia y son el 21.7% de las filas actuales. Se marcan con `estacion_nueva` para reportarlas aparte.
- **Etiqueta:** las horas en riesgo bajaron de 7.0% a 6.1% en vaciado y de 7.0% a 5.9% en llenado.

La construcción, la limpieza y las diferencias con los datos de entrenamiento se detallan en el Anexo A.
