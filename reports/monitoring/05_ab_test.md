## 5. Comparación A/B entre versiones del modelo P2

Se realizó una evaluación comparativa offline de dos versiones del
modelo P2 de MiBici, registradas en MLflow Model Registry como
`champion` (v1) y `challenger` (v2). Ambos modelos fueron evaluados
sobre las mismas 502,260 observaciones de 469 estaciones,
correspondientes al periodo del 1 de septiembre al 30 de octubre
de 2026. Esta ventana no se utilizó durante el entrenamiento ni
la calibración del challenger.

Para generar las predicciones se aplicó la regla de clasificación
de P2, utilizando el umbral propio de cada versión:
0.150 para champion y 0.575 para challenger. La métrica principal
fue F2 macro de las clases de riesgo de vaciado y llenado.

| Métrica | Champion v1 | Challenger v2 |
|---|---:|---:|
| F2 macro | **0.726443** | 0.722661 |
| AP vaciado | **0.748555** | 0.745967 |
| AP llenado | 0.722362 | **0.725426** |
| Brier vaciado | **0.030996** | 0.070873 |
| Brier llenado | **0.030549** | 0.066186 |
| Recall vaciado | 0.826859 | **0.860663** |
| Recall llenado | 0.850525 | **0.878374** |

El challenger presentó mayor recall en ambas clases de riesgo:
3.38 puntos porcentuales adicionales para vaciado y 2.78 para
llenado. Sin embargo, obtuvo un F2 macro ligeramente inferior
y mayores valores de Brier Score, lo que indica un mayor error
en sus probabilidades estimadas.

Para cuantificar la incertidumbre se aplicó bootstrap pareado
por estación con 2,000 réplicas y semilla 42. La diferencia
de F2 macro (challenger menos champion) fue de -0.003782,
con intervalo de confianza del 95% [-0.005659, -0.001815].
El intervalo se mantuvo por debajo de cero, favoreciendo
al champion bajo el método de remuestreo utilizado.

Complementariamente, la prueba de McNemar identificó
18,470 observaciones acertadas exclusivamente por el champion
y 5,964 acertadas exclusivamente por el challenger.
El estadístico fue 6399.895 (p < 0.001), lo que evidencia
diferencias en los errores de clasificación. No obstante,
su interpretación debe considerar la dependencia entre
observaciones repetidas por estación y periodo.

De acuerdo con la política de promoción definida antes
de ejecutar la comparación, **no se recomienda promover el
challenger**, ya que no mejora el F2 macro y el intervalo de
confianza de la diferencia no es positivo. Se recomienda
conservar el champion como versión vigente y continuar
monitoreando el comportamiento del modelo.

La evaluación fue offline, no un experimento con distribución
de tráfico real. Los resultados y la metodología quedaron
registrados en MLflow bajo el Run ID
`623e25d125c641d8927c27b4dc0489d3`.
