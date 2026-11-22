# Análisis FinOps: costo por inferencia y optimización del servicio P2

**Proyecto:** MiBici — TC5061, Operaciones de Aprendizaje Automático  
**Equipo:** 37  
**Actividad:** Entregable 7 — Microservicios, CI/CD y FinOps  
**Fecha de consulta de precios:** 22 de noviembre de 2026  
**Estado del análisis:** estimación económica basada en pruebas locales; no corresponde a una factura ni a un benchmark ejecutado en la nube.

## 1. Objetivo y alcance

Estimar el costo de procesar una inferencia de la API de clasificación P2 de MiBici e identificar una configuración de infraestructura razonable, combinando los resultados de las pruebas de carga con precios públicos de cómputo. El servicio analizado es `POST /predict_p2/`, que devuelve una predicción de riesgo de vaciado o llenado para una estación. El análisis se concentra en el **costo de infraestructura de cómputo por predicción atendida**; no pretende representar el costo total de propiedad del sistema.

Para dar seguimiento a la eficiencia económica se utiliza una métrica de *unit economics*: USD por inferencia, y su equivalente por 1,000 y por 1,000,000 de inferencias (FinOps Foundation, s. f.). La comparación incorpora latencia, capacidad y utilización de CPU/memoria, porque una configuración barata que no cumpla el objetivo de servicio no es una alternativa operativamente suficiente.

## 2. Evidencia experimental utilizada

Los datos corresponden a las pruebas documentadas por el equipo en [`reports/load_test/README.md`](../load_test/README.md), con detalle adicional en [`reports/load_test/resumen.csv`](../load_test/resumen.csv). Locust simuló 10, 50 y 150 usuarios con peticiones al endpoint P2; las corridas duraron cinco minutos, con rampa de diez segundos. El consumo se observó con `docker stats` cada dos segundos. Se compararon una API limitada a 1 CPU y 1 worker, y otra a 2 CPUs y 2 workers. Ambas configuraciones contaron con 1 GiB de límite de memoria para la API.

El criterio definido por el equipo fue **p95 menor a 500 ms y tasa de error inferior a 1%**.

| Configuración local de la API | Usuarios | Rendimiento (req/s) | p95 (ms) | Errores | CPU media | Memoria máxima API (MiB) | ¿Cumple objetivo? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 CPU / 1 worker | 10 | 7.8 | 32 | 0% | 11% | 206 | Sí |
| 1 CPU / 1 worker | 50 | 37.6 | 210 | 0% | 57% | 213 | Sí |
| 1 CPU / 1 worker | 150 | 35.3 | 4,300 | 0% | 98% | 217 | No |
| 2 CPUs / 2 workers | 10 | 7.8 | 29 | 0% | 11% | 414 | Sí |
| 2 CPUs / 2 workers | 50 | 38.9 | 59 | 0% | 54% | 413 | Sí |
| 2 CPUs / 2 workers | 150 | 74.6 | 1,700 | 0% | 188% | 428 | No |

**Interpretación de la evidencia.** Con 50 usuarios las dos configuraciones satisfacen el criterio del equipo. La variante de 2 CPUs reduce el p95 de 210 a 59 ms (aproximadamente 72%), aunque el rendimiento cambia poco porque la demanda todavía no satura a la versión de 1 CPU. Con 150 usuarios, la variante de 1 CPU alcanza 98% de un núcleo y la de 2 CPUs llega a 188% de CPU (donde 100% equivale a un núcleo). Se observa saturación de cómputo y fuerte degradación de p95. Ninguna es apta para esa carga conforme al objetivo establecido. La memoria de la **API** permanece por debajo de 428 MiB en las corridas, pero esto no representa el consumo total del servidor: también deben considerarse MLflow, el sistema operativo y Docker.

Las pruebas no representan una medición en DigitalOcean: Locust y la API se ejecutaron en la misma máquina local, se hizo una sola corrida por combinación y cada una duró cinco minutos. Por tanto, no puede suponerse que una VM de nube entregue exactamente el mismo rendimiento.

## 3. Supuestos y precios de infraestructura

Como referencia se utiliza **DigitalOcean Basic Droplets, CPU Regular**, con los precios publicados por el proveedor al 22/11/2026. No se contrató ningún servicio para este análisis.

| Máquina virtual de referencia | vCPU | RAM | Tarifa publicada (USD/h) | Tope publicado (USD/mes) | Uso planteado |
|---|---:|---:|---:|---:|---|
| Escenario A | 2 | 4 GiB | 0.03571 | 24 | API limitada a 1 CPU / 1 worker, más servicios de apoyo |
| Escenario B | 4 | 8 GiB | 0.07143 | 48 | API limitada a 2 CPUs / 2 workers, más servicios de apoyo |
| Alternativa A optimizada* | 2 | 2 GiB | 0.02679 | 18 | Igual que A, con menos RAM aprovisionada |

\* La alternativa optimizada es una **hipótesis de dimensionamiento**, no una configuración desplegada o validada por el equipo. Es necesario medir el uso total del host, incluido MLflow, antes de adoptarla.

**Fuente oficial:** [DigitalOcean. *Droplet Pricing*](https://www.digitalocean.com/pricing/droplets), sección *Basic Droplets*, opción *Regular*. Los precios están expresados en dólares estadounidenses. El proveedor publica tarifas por hora y topes mensuales; la facturación real puede depender del tiempo facturable y de los términos vigentes.

**Hipótesis para poder comparar:** (a) todos los servicios caben en una sola VM de cada escenario; (b) no hay cuellos de botella nuevos por red, disco o virtualización; (c) la tasa de solicitudes medida localmente se puede utilizar solo como aproximación de capacidad en la nube; (d) cada solicitud exitosa a `POST /predict_p2/` equivale a una inferencia; (e) las máquinas están dedicadas a este sistema. Si alguno de estos supuestos cambia, el resultado debe recalcularse.

**Costos excluidos:** transferencia que exceda la incluida, respaldos, almacenamiento adicional, balanceador, dominios, monitoreo gestionado, impuestos, tiempo de desarrollo y cualquier otro servicio cobrado por separado. El objetivo es comparar el costo **directo de cómputo**, no estimar una factura completa.

## 4. Método de cálculo

Dado un costo horario de la VM `C_h` y un rendimiento sostenido `R` en inferencias por segundo, se calcula:

**Inferencias por hora = R × 3,600**

**Costo por inferencia (USD) = C_h / (R × 3,600)**

**Costo por 1,000 inferencias (USD) = 1,000 × C_h / (R × 3,600)**

**Costo por 1,000,000 de inferencias (USD) = 1,000,000 × C_h / (R × 3,600)**

Estas fórmulas aplican a **una hora de servicio bajo el rendimiento sostenido supuesto**. No deben interpretarse como el costo de una predicción aislada: si la VM permanece aprovisionada sin recibir peticiones, sigue generando cargos y el costo efectivo por predicción aumenta. Tampoco se deben multiplicar las tarifas horarias por un mes completo sin tener en cuenta el tope mensual del proveedor.

### 4.1. Ejemplo reproducible: 50 usuarios

**Escenario A (1 CPU / 1 worker):**

- Rendimiento observado: `37.6 req/s`.
- Inferencias por hora: `37.6 × 3,600 = 135,360`.
- Costo por inferencia: `0.03571 / 135,360 = 0.000000263815 USD`.
- Costo por 1,000 inferencias: `0.000264 USD` (redondeado).
- Costo por 1,000,000 de inferencias: **`0.264 USD`** (redondeado).

**Escenario B (2 CPUs / 2 workers):**

- Rendimiento observado: `38.9 req/s`.
- Inferencias por hora: `38.9 × 3,600 = 140,040`.
- Costo por inferencia: `0.07143 / 140,040 = 0.000000510069 USD`.
- Costo por 1,000 inferencias: `0.000510 USD` (redondeado).
- Costo por 1,000,000 de inferencias: **`0.510 USD`** (redondeado).

### 4.2. Comparación por nivel de demanda

| Escenario | Usuarios simulados | RPS observadas | p95 (ms) | USD/1,000 inferencias* | USD/1,000,000 inferencias* | ¿Cumple el objetivo? |
|---|---:|---:|---:|---:|---:|:---:|
| A: 2 vCPU, 4 GiB; API 1 CPU | 10 | 7.8 | 32 | 0.001272 | 1.272 | Sí |
| A: 2 vCPU, 4 GiB; API 1 CPU | 50 | 37.6 | 210 | 0.000264 | 0.264 | Sí |
| A: 2 vCPU, 4 GiB; API 1 CPU | 150 | 35.3 | 4,300 | 0.000281 | 0.281 | **No** |
| B: 4 vCPU, 8 GiB; API 2 CPUs | 10 | 7.8 | 29 | 0.002544 | 2.544 | Sí |
| B: 4 vCPU, 8 GiB; API 2 CPUs | 50 | 38.9 | 59 | 0.000510 | 0.510 | Sí |
| B: 4 vCPU, 8 GiB; API 2 CPUs | 150 | 74.6 | 1,700 | 0.000266 | 0.266 | **No** |

\* Costo **teórico de cómputo** calculado con los RPS locales y las tarifas públicas por hora. Los importes del escenario de 150 usuarios se muestran para transparentar la fórmula, **no** como una opción de operación viable mientras se incumpla el p95.

Para la carga de 50 usuarios, la variante B tiene un costo teórico por inferencia aproximadamente **93% mayor** que A (`0.510 / 0.264 ≈ 1.93`), aunque reduce el p95 de 210 a 59 ms. Con baja carga (10 usuarios), la capacidad adicional permanece poco utilizada, por lo que el costo unitario se duplica aproximadamente. Los resultados muestran que sobredimensionar una máquina sin aumentar las inferencias realizadas eleva el costo por unidad.

## 5. Optimización de costo y capacidad

### 5.1. Dimensionar los recursos según demanda observada

**Propuesta:** para una demanda comparable a las pruebas de 50 usuarios, iniciar la evaluación operativa con una configuración de 1 CPU / 1 worker y la VM A, porque el p95 de 210 ms y la tasa de errores de 0% cumplen el criterio. Elegir 2 CPUs / 2 workers cuando el margen de latencia o los picos justifiquen el mayor costo. Esta recomendación no sustituye una prueba de producción ni garantiza el comportamiento futuro.

**Efecto estimado:** mantener A frente a B a la carga de 50 usuarios reduce el costo directo por millón de inferencias de `0.510` a `0.264 USD` (**aproximadamente 48% menos**), a cambio de un p95 mayor (210 ms frente a 59 ms). La decisión se basa en el objetivo medido, no solamente en el precio.

### 5.2. Evaluar una VM con menos memoria

**Propuesta:** probar la VM Basic de 2 vCPU y 2 GiB (0.02679 USD/h) conservando, inicialmente, la limitación de 1 CPU / 1 worker. La memoria máxima observada en el contenedor de API fue 213 MiB con 50 usuarios y 217 MiB con 150; sin embargo, debe verificarse el uso conjunto del host, MLflow, contenedores y sistema operativo antes de reducir RAM.

**Efecto teórico, si mantiene 37.6 RPS y el p95 objetivo:**

- Costo por millón de inferencias: `0.02679 / (37.6 × 3,600) × 1,000,000 = 0.198 USD`.
- Comparación contra A (`0.264 USD`): **reducción aproximada de 25%**.
- Tope mensual publicado: `18 USD` frente a `24 USD`, con **6 USD de diferencia** para un mes en que se alcance el tope en ambas opciones.

Estos ahorros son **proyectados**, no observados; se perderían si la máquina más pequeña degradara latencia o throughput.

### 5.3. Escalamiento sujeto al cumplimiento de latencia

Con 150 usuarios, subir de 1 a 2 CPUs aumenta el rendimiento de `35.3` a `74.6 req/s` (aproximadamente 111% más), pero el p95 sigue siendo 1,700 ms, por encima del máximo de 500 ms. Por lo tanto, **no se recomienda declarar que B soporta 150 usuarios** solo porque duplica la tasa de procesamiento.

**Siguiente experimento propuesto:** probar 4 CPUs / 4 workers con memoria suficiente y repetir Locust con 100, 150 y 200 usuarios, separando el generador de carga del host de la API. Un aumento aproximadamente lineal del throughput es únicamente una hipótesis. La selección de tamaño y número de réplicas debe depender de que las mediciones nuevas cumplan p95 y error, además del costo unitario.

### 5.4. Evitar capacidad ociosa y vigilar la factura real

**Propuesta:** registrar periódicamente `costo de cómputo / predicciones exitosas`, RPS, p95, errores y CPU por entorno; revisar si la capacidad reservada corresponde al tráfico efectivo. Automatizar la destrucción y recreación de **ambientes temporales** cuando no se necesiten puede evitar pagar por infraestructura sin uso, siempre que los artefactos necesarios se conserven de manera segura.

**Precaución:** en DigitalOcean, **apagar** una VM (*power off*) **no elimina** los cargos; la VM sigue reservando recursos y se factura. El ahorro requiere dejar de aprovisionar los recursos facturables, no simplemente apagar el sistema. Esta acción solo es apropiada para ambientes temporales reproducibles y debe contemplar respaldos y costos de almacenamiento persistente.

### 5.5. Optimizar inferencia antes de añadir servicios nuevos

La saturación observada es principalmente de CPU, mientras que MLflow registró menos de 1% de CPU promedio durante estas pruebas. Por ello, antes de introducir Redis/RQ, más servicios o un despliegue distribuido, conviene medir tiempo de inferencia por modelo, probar la configuración de workers y evaluar el costo de una versión más ligera siempre que conserve los requisitos de calidad predictiva. El procesamiento asíncrono puede amortiguar picos de solicitudes, pero **no reduce automáticamente el costo ni la latencia de una predicción individual**. No se le atribuye un porcentaje de ahorro sin un experimento adicional.

## 6. Limitaciones y validaciones pendientes

1. El benchmark fue local; los RPS y la latencia **no fueron medidos en las VMs** cuyos precios se emplearon.
2. La comparación A/B cambia a la vez tamaño de VM, límite de CPU y número de workers. No permite atribuir causalmente toda la diferencia a uno solo de esos factores.
3. Se utilizó la tarifa horaria del proveedor para comparar una hora bajo carga sostenida. Una proyección mensual requiere un patrón de tráfico real y aplicar las reglas de facturación y el tope mensual correspondientes.
4. No se comprobó el consumo total de la VM, especialmente MLflow, sistema operativo y tareas complementarias; las cifras de memoria citadas son de la API.
5. Los escenarios de 150 usuarios incumplen el objetivo de latencia, aunque no presenten errores HTTP en la prueba.
6. Para una estimación productiva completa se deben medir en nube CPU, memoria, latencia, solicitudes exitosas, tiempo ocioso, red, almacenamiento, respaldos y otros servicios, además de verificar que las VMs soporten la carga real.

## 7. Conclusión

Las mediciones del Equipo 37 permiten estimar un costo directo teórico de cómputo de **0.264 USD por millón de inferencias** con 1 CPU / 1 worker y **0.510 USD por millón** con 2 CPUs / 2 workers, usando las tarifas de las VMs A y B y la demanda medida de 50 usuarios. Ambas configuraciones cumplieron el objetivo de servicio a esa carga, pero la opción de más recursos mejora la latencia a cambio de mayor costo. En cambio, ninguna configuración cumplió el objetivo con 150 usuarios. Una posible reducción de RAM podría bajar el costo teórico de A a **0.198 USD por millón**, pero requiere validación del consumo total del sistema. En términos FinOps, la recomendación es elegir recursos a partir de métricas de uso, rendimiento y costo unitario, y validar cualquier ahorro estimado antes de tomarlo como resultado real.

## 8. Fuentes y referencias

- DigitalOcean. (s. f.). *Droplet pricing*. Consultado el 22 de noviembre de 2026. https://www.digitalocean.com/pricing/droplets
- DigitalOcean. (s. f.). *Droplet pricing* [Documentación de facturación de Droplets]. Consultado el 22 de noviembre de 2026. https://docs.digitalocean.com/products/droplets/details/pricing/
- Equipo 37. (2026). *Pruebas de carga y monitoreo de recursos* [Documentación técnica del proyecto]. `reports/load_test/README.md`. https://github.com/A01062611/TC5061-Proyecto-Equipo37/blob/main/reports/load_test/README.md
- FinOps Foundation. (s. f.). *FinOps Framework*. https://www.finops.org/framework/
- FinOps Foundation. (s. f.). *Capability: Unit economics*. https://www.finops.org/framework/capabilities/unit-economics/
- Gift, N., & Deza, A. (2021). *Practical MLOps: Operationalizing machine learning models*. O’Reilly Media.
- Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O’Reilly Media.
- Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.

> **Nota de reproducibilidad:** los datos de carga se transcribieron de `reports/load_test/README.md` publicado en `main` el 22/11/2026. Si el equipo actualiza los CSV o repite las pruebas, se deberán recalcular las tablas y actualizar la fecha antes de cerrar el reporte PDF.