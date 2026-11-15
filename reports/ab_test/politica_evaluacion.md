# Política de evaluación A/B — P2 MiBici

## Objetivo
Comparar champion y challenger sobre el mismo conjunto
de evaluación de septiembre-octubre de 2026.

## Modelos
- Champion: versión 1, tau = 0.150.
- Challenger: versión 2, tau = 0.575.
- Modelo registrado: mibici-equipo37-p2-riesgo.

## Datos
- Archivo: data/processed/monitoring/ab_eval.parquet
- Registros: 502260.
- Periodo: 2026-09-01 a 2026-10-30.
- Ninguna versión utilizó esta ventana para entrenar.

## Predicciones
Se utilizará predict_proba y la función pred_umbral
del entrenamiento P2, con el tau de cada versión.

## Métrica principal
F2 macro de las clases 1 (vaciado) y 2 (llenado).

## Regla de promoción
Se recomendará promover el challenger solo si:

1. Su F2 macro es superior al champion.
2. El límite inferior del IC 95% de la diferencia
   de F2 macro es mayor que cero.
3. El recall de ninguna clase de riesgo disminuye
   más de 0.02 en términos absolutos.

## Pruebas y métricas complementarias
- Bootstrap pareado por estación, semilla 42.
- Prueba de McNemar sobre aciertos y errores.
- Average Precision por clase de riesgo.
- Brier Score por clase de riesgo.
- Matrices de confusión y recall por clase.

McNemar es evidencia complementaria y no sustituye
el intervalo de confianza de F2.

## Limitaciones
La comparación es offline sobre el mismo conjunto,
no un experimento con tráfico real en producción.
Se documentarán la dependencia temporal, estaciones
nuevas y diferencias de entorno entre modelos.
