## 3. Interpretación del drift

Que una variable se desvíe no pesa igual si el modelo la usa mucho o poco. Por eso se midió la importancia por permutación del champion (caída de F2 macro al permutar cada variable en la partición `val`, 100,000 filas, 3 repeticiones, `SEED = 42`) y se cruzó con el drift (figura 4; tabla completa en `reports/evidently/drift_vs_importancia.csv`).

![Figura 4. Importancia por permutación contra score de drift. Arriba y a la derecha están las variables cuyo drift más debería preocupar. Fuente: `drift_vs_importancia.csv`.](../evidently/figuras/drift_vs_importancia.png)

| Variable | Importancia (caída de F2) | Score existentes | Score nuevas |
| --- | ---: | ---: | ---: |
| `ocup_est` | 0.416 | 0.057 (no) | 0.368 (sí) |
| `sal_media7_2h` | 0.028 | 0.158 (sí) | 0.513 (sí) |
| `lle_media7_2h` | 0.028 | 0.140 (sí) | 0.494 (sí) |
| `lon` | 0.020 | 0.060 (no) | 0.576 (sí) |
| `C` (capacidad) | 0.019 | 0.016 (no) | 0.487 (sí) |
| `sal_2h` / `lle_2h` | 0.010 / 0.007 | 0.102 / 0.104 (sí) | 0.372 / 0.384 (sí) |
| `frac_vacio_7d` / `frac_lleno_7d` | 0.001 / 0.001 | 0.105 / 0.099 | 0.635 / 0.725 |

**Lectura 1: el drift fuerte está en variables poco importantes, salvo `ocup_est`.** `frac_vacio_7d` y `frac_lleno_7d` tienen los scores más altos en estaciones nuevas (0.64 y 0.73) pero casi no mueven el F2 (importancia 0.001). `ocup_est` es la excepción: es con mucho la variable más importante (0.416, 14 veces la siguiente) y se desvía en las estaciones nuevas (0.368). Es el principal riesgo para el modelo, pero está acotado a 101 estaciones.

**Lectura 2: en la red original el drift es leve y está en variables de segundo orden.** 11 de 23 variables superan el umbral, casi todas de salidas y llegadas (`sal_*`, `lle_*`). Sus scores van de 0.10 a 0.16, apenas sobre el umbral. Las de mayor importancia entre ellas son `sal_media7_2h` y `lle_media7_2h` (0.028 cada una). El desplazamiento tiene dirección: la media de `sal_2h` pasa de 1.34 a 1.61 salidas en 2 h (+20%) y la de `lle_2h` de 1.33 a 1.59. Sube la demanda, no cambia la forma del patrón (`hora`, `dia_semana` y `festivo` no tienen drift). Esto es coherente con el desempeño: el F2 macro baja solo 0.013 y la recall de vaciado y llenado se mantiene arriba de 83% (0.832 y 0.850, contra 0.838 y 0.849 en validación).

**Lectura 3: no hay degradación importante, pero el champion calibra peor el llenado.** El AP de llenado cae 0.034 en existentes y el Brier sube de 0.028 a 0.033. Coincide con que la demanda subió: las probabilidades que el modelo aprendió con 1.3 llegadas por 2 h ya no son exactas con 1.6.

**Estaciones nuevas.** Tienen menos movimiento (`sal_2h` promedio 0.58 contra 1.34; mediana 0 contra 1), más capacidad (`C` 17.6 contra 17.0; mediana 19 contra 15) y menos eventos de riesgo (figura 2). Es lo esperable de estaciones recientes y sin demanda estable. Su ubicación también difiere (`lon` con drift), aunque la diferencia de medianas es pequeña (−103.355 contra −103.360). Su historia es corta, y las variables de 7 días y `frac_*` se calculan con pocos datos. Aun así, el champion mantiene recall de 0.798 en vaciado y 0.881 en llenado. No tiene sentido promediarlas con el resto: conviene reportarlas aparte y vigilar si maduran.

**Causa probable.** Son hipótesis; los datos no permiten comprobarlas:
- *Estaciones nuevas:* explican el drift de `lon`, `C` y `ocup_est` y la caída del prior. El número de estaciones pasó de 371 a 470 (+27%).
- *Aumento de demanda en existentes:* crecimiento de uso en un año, o un efecto de temporada. Se descarta en parte lo segundo: el periodo (jul–oct) coincide con meses presentes en el entrenamiento (jul–dic 2024).
- *Cambios de tarifa o eventos puntuales:* no los pudimos verificar con los datos disponibles.

**Comparación con DeepChecks.** El análisis previo (`reports/deepchecks/INFORME_DRIFT_P2.md`) comparó train contra val y test, ambos de 2025, con KS y Cramér's V sobre 10,000 filas. No encontró drift relevante (el máximo fue 0.050, `frac_vacio_7d` en test) y recomendó monitorear sin reentrenar. Evidently, con el periodo de 2026, sí detecta drift. No es una contradicción: los dos periodos de DeepChecks eran inmediatos al entrenamiento y con la misma red de 371 estaciones, y el actual está un año después y con 101 estaciones nuevas. Los scores de las dos herramientas no son comparables entre sí (KS y Wasserstein normada tienen escalas distintas), pero `frac_vacio_7d` y `lon` aparecen en ambas como las más desviadas, lo que refuerza que la dirección del cambio es consistente.

**Implicación para el modelo.** El champion sigue funcionando en la red original (F2 macro 0.724 contra 0.737 en validación), y el drift fue leve donde hay historia. Por eso no justificaba un reentrenamiento urgente, pero sí probar una versión reentrenada con datos de 2026, que incorpora las estaciones nuevas y la demanda más alta. El A/B (sección 5) la comparó en una ventana que ninguna versión vio y el resultado fue consistente con esta lectura: el challenger no mejora el F2 macro (0.723 contra 0.726; diferencia −0.004, IC 95% [−0.006, −0.002]) y calibra peor (Brier 0.07 contra 0.03), aunque sube el recall de vaciado y llenado en 3.4 y 2.8 puntos. Es decir, el drift observado no bastó para que el reentrenamiento superara al champion, y se conserva el champion. La principal vigilancia pendiente son las estaciones nuevas, donde está el drift fuerte en `ocup_est`.
