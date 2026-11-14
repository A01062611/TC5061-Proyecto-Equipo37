## 4. Versión alternativa del modelo P2

Para evaluar una posible mejora del modelo P2 de clasificación de riesgo de vaciado y llenado de estaciones, se entrenó una segunda versión denominada *challenger*. Se utilizó el algoritmo HistGradientBoostingClassifier y las mismas 23 características del modelo original (*champion*).

El entrenamiento incorporó los registros históricos de enero de 2024 a junio de 2025 y los datos de julio y agosto de 2026. Primero se entrenó con el histórico y julio, utilizando agosto para validar y calibrar el umbral de clasificación. Posteriormente, el modelo definitivo se ajustó con ambos meses, utilizando un total de **3,756,080 registros**.

El nuevo modelo obtuvo un umbral de clasificación de 0.575, mientras que el umbral recalculado del champion fue de 0.15. Ambos modelos se registraron en MLflow bajo el nombre `mibici-equipo37-p2-riesgo`, utilizando los alias `challenger` y `champion`, respectivamente. Se verificó que ambos podían cargarse y generar predicciones.

Los datos de septiembre y octubre de 2026 quedaron reservados para la evaluación A/B, evitando su utilización durante el entrenamiento. Las métricas de agosto corresponden a la validación previa al ajuste definitivo, por lo que no sustituyen la evaluación independiente. También se utilizó la nomenclatura de octubre de 2026, generando pequeñas diferencias respecto a los conjuntos inicialmente reportados.

Finalmente, se ejecutaron **107 pruebas automatizadas**, todas satisfactorias, incluyendo validaciones de separación temporal, compatibilidad de datos y funcionamiento general del proyecto.