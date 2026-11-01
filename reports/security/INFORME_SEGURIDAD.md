# Informe de auditoría básica de seguridad — MiBici, Equipo 37

**Actividad 7 · 31 de octubre de 2026**  
**Alcance:** revisión estática del código `MiBici/` y `scripts/`, auditoría de dependencias declaradas en `requirements.txt`, nombres de archivos rastreados por Git, reglas de exclusión y búsqueda básica de patrones de secretos. No constituye una auditoría integral de seguridad ni un análisis del historial completo de Git.

## Resultados

| Control | Resultado observado | Evaluación |
|---|---|---|
| Bandit 1.9.4 | 13 hallazgos: 1 medio y 12 bajos | Requiere revisión contextual |
| pip-audit 2.10.1 | 42 avisos asociados a 2 paquetes | Riesgo pendiente; sin corrección aplicada |
| Git / `.gitignore` | 61 archivos rastreados; `data/` ignorado | Control de exclusión confirmado |
| Búsqueda por expresiones regulares | 0 coincidencias potenciales en los archivos de texto revisados | No descarta secretos con otros formatos |
| `.env.example` | Variables `MLFLOW_TRACKING_URI` y `MLFLOW_EXPERIMENT_NAME` | Valores no inspeccionados en esta revisión |

## Bandit

- **B310, severidad MEDIUM, confianza HIGH:** `MiBici/dataset.py`, línea 108. Uso de apertura de URL que debe revisarse para restringir esquemas y fuentes permitidas. **No se ha demostrado explotación ni se ha aplicado corrección.**
- **12 alertas LOW, confianza HIGH:** `B404`, `B603`, `B607` en `train_p1.py`, `train_p2.py` y `train_p3.py`; asociadas al uso de `subprocess` y rutas de ejecutables. Revisar si los argumentos proceden de entradas no confiables y si se resuelven rutas ejecutables seguras. No asumir automáticamente que son falsos positivos.

## pip-audit

- **Pillow 12.3.0:** 39 avisos; versión corregida indicada por el reporte: **13.0.0**.
- **Tornado 6.5.10:** 3 avisos; versión corregida indicada por el reporte: **6.6**.

Las cifras son las devueltas por la auditoría local; no equivalen necesariamente a 42 fallas independientes explotables en este proyecto. No se actualizaron paquetes para evitar alterar el entorno de experimentación sin comprobar compatibilidad. La recomendación es validar actualizaciones en un entorno aislado y repetir pruebas y auditoría antes de aprobar cambios.

## Protección de datos y secretos

El archivo `.gitignore` excluye `data/`, `.env`, artefactos de MLflow y modelos serializados. Una inspección de nombres encontró solamente `.env.example` entre los archivos rastreados con patrones sensibles; la búsqueda básica de asignaciones de contraseñas y tokens no produjo coincidencias. **Limitaciones:** no se inspeccionaron los valores de `.env.example`, todos los formatos binarios, patrones de secretos no contemplados ni el historial completo de Git.

## Acciones pendientes antes de cerrar el riesgo

1. Revisar el contexto de `B310` y de las llamadas a `subprocess`, documentando si hay controles suficientes o si procede corregirlas.
2. Evaluar en un entorno de pruebas la compatibilidad de Pillow 13.0.0 y Tornado 6.6; no cambiar los requisitos de entrenamiento sin aprobación del equipo.
3. Confirmar que `.env.example` contiene valores de ejemplo y no credenciales.
4. Revisar el contenido de los reportes y capturas antes de publicarlos; conservar evidencia de los comandos ejecutados.

## Comandos reproducibles

```bash
python -m bandit -r MiBici scripts -f json -o reports/security/bandit.json
python -m pip_audit -r requirements.txt -f json -o reports/security/pip_audit.json
make test
```

**Conclusión:** se ejecutaron los controles básicos solicitados y se identificaron riesgos pendientes. No se afirma ausencia de vulnerabilidades ni remediación completa.
