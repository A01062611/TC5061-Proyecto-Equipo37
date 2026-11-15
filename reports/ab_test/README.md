# A/B Testing — P2 MiBici, Equipo37

## Objetivo

Comparar offline dos versiones registradas en MLflow del modelo
P2 de riesgo de vaciado y llenado:

- Champion: versión 1.
- Challenger: versión 2.

La evaluación utiliza el mismo conjunto de septiembre-octubre
de 2026, sin incorporar esos registros al entrenamiento.

## Requisitos

- Python y dependencias del repositorio.
- Datos históricos procesados de P2.
- Datos abiertos de MiBici de junio a octubre de 2026.
- MLflow Model Registry con los aliases `champion` y `challenger`.

Los datos grandes, modelos registrados y bases MLflow locales
no se distribuyen mediante este directorio.

## 1. Preparar los datos

Desde la raíz del repositorio:

```bash
python -m MiBici.dataset \
  --desde 2026-06 \
  --hasta 2026-10 \
  --salida data/interim/mibici_2026_06_10.csv \
  --estaciones data/interim/nomenclatura_2026_10.csv \
  --manifiesto-path data/interim/data_manifest_ab_test.json
```

Se requieren también los archivos históricos de P2:

- `data/processed/p2/tabla.parquet`
- `data/processed/p2/s0_por_estacion.csv`
- `data/processed/p2/info.json`

Generar los conjuntos de monitoreo:

```bash
python -m MiBici.monitoring.datasets \
  --trips-path data/interim/mibici_2026_06_10.csv \
  --stations-path data/interim/nomenclatura_2026_10.csv \
  --out-dir data/processed/monitoring \
  --tau 0.15
```

El conjunto de evaluación queda en:

`data/processed/monitoring/ab_eval.parquet`

## 2. Preparar las versiones en MLflow

El modelo champion se puede importar desde el artefacto
exportado en `models/mibici-equipo37-p2-riesgo/`:

```bash
python -m scripts.import_champion
```

Para reproducir el challenger:

```bash
python -m MiBici.modeling.challenger_p2 \
  --train-fin 2026-08-31
```

Recuperar el Run ID generado y registrarlo:

```bash
python -m MiBici.modeling.register \
  --question p2 \
  --run-id RUN_ID_GENERADO \
  --alias challenger
```

Si las versiones ya existen en el registry utilizado,
no es necesario volver a entrenar ni registrar modelos.

## 3. Ejecutar A/B Testing

Prueba diagnóstica:

```bash
python -m MiBici.monitoring.ab_test --sample 1000
```

Evaluación completa:

```bash
python -m MiBici.monitoring.ab_test --bootstrap 2000
```

El script recupera los umbrales desde los parámetros de
MLflow y aplica la regla de clasificación de P2.

## 4. Ejecutar pruebas automatizadas

```bash
python -m pytest tests/test_ab_statistics.py -v
```

## 5. Registrar resultados en MLflow

Después de ejecutar la evaluación completa:

```bash
python -m MiBici.monitoring.log_ab_mlflow
```

El registro crea una nueva corrida con métricas y artefactos.
La evaluación no modifica los aliases de los modelos.

## 6. Archivos de resultados

- `comparacion_metricas.csv`: tabla de métricas.
- `resultado_ab.json`: resultados estadísticos y decisión.
- `politica_evaluacion.md`: criterios de promoción.
- `../monitoring/05_ab_test.md`: texto para el reporte final.

## 7. Metodología

- Evaluación offline pareada sobre 502,260 registros.
- Periodo: 2026-09-01 a 2026-10-30.
- Métrica principal: F2 macro de las clases 1 y 2.
- Bootstrap pareado por estación, 2,000 réplicas.
- Semilla: 42.
- Prueba complementaria de McNemar.
- Regla de promoción establecida antes de evaluar.

La prueba no implementa un reparto real de tráfico.
McNemar convencional tampoco corrige la dependencia
entre observaciones de una misma estación.

## 8. Resultado obtenido

Champion v1: F2 macro = 0.726443.

Challenger v2: F2 macro = 0.722661.

Diferencia: -0.003782.

IC 95%: [-0.005659, -0.001815].

Recomendación: mantener el champion.

Los identificadores de corrida pueden cambiar al reproducir
el experimento en otra instalación de MLflow.
