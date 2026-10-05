# 2. Comprensión de los datos

> **Estado:** completar con resultados de la exploración. No reemplazar los marcadores hasta contar con evidencia reproducible.

## 2.1 Recopilar datos iniciales

Los datos proceden de la competición Home Credit – Credit Risk Model Stability y se almacenan localmente en formato Parquet.

| Recurso | Ubicación | Uso |
|---|---|---|
| Definiciones de variables | `data/feature_definitions.csv` | Interpretación de campos. |
| Formato de predicción | `data/sample_submission.csv` | Validación de la salida final. |
| Entrenamiento | `data/parquet_files/train/` | Preparación y modelado. |
| Prueba | `data/parquet_files/test/` | Predicciones finales. |

### Entorno y herramientas comunes (Etapa A, paso 1)

Entorno Python 3.11 con `polars`, `pyarrow`, `numpy` y `scikit-learn`; instalación y comandos en el [README](../README.md#entorno-y-herramientas). Lectura única mediante `src/dm_credit/io.py`; perfilador en `src/dm_credit/profiling.py`; esqueleto del baseline en `src/dm_credit/baseline.py`. Verificación: 29 pruebas automáticas (`pytest`).

Observaciones verificadas al ejecutar el perfilador sobre `train_base.parquet` (`results/profile_train_base/`):

- 1 526 659 filas y 5 columnas: `case_id`, `date_decision`, `MONTH`, `WEEK_NUM`, `target`; 61 MB en memoria.
- **La tabla base no contiene predictores propios**: solo clave, fechas/semana y objetivo. Un baseline entrenado únicamente con ella es la referencia trivial (prevalencia, Gini 0 por construcción); la señal predictiva llegará al integrar las demás fuentes.
- `test_base.parquet` local tiene solo 10 filas (`WEEK_NUM` 100) y no incluye `target`; sirve para comprobar esquema, no para evaluar.

Contrato técnico del baseline: ver README. Este contrato no fija el corte OOT; se completa con el protocolo del paso 4.

### Inventario de tablas

<!-- Insertar tabla generada: archivo, filas, columnas, clave, granularidad, WEEK_NUM disponible y observaciones. -->

## 2.2 Describir datos

### Tabla base y unidad de análisis

<!-- Confirmar: tabla base, clave del caso, variable objetivo, número de casos y definición de una observación. -->

### Variable objetivo

<!-- Insertar: conteo y porcentaje de cada clase, distribución por WEEK_NUM y comentario sobre desbalance. -->

### Tipos, cobertura y variables relevantes

<!-- Insertar: resumen de tipos, nulos, cardinalidad y grupos de variables según feature_definitions.csv. -->

## 2.3 Explorar datos

### Análisis univariado y bivariado

<!-- Insertar figuras y hallazgos: distribuciones, valores extremos, categorías, asociaciones iniciales con target. -->

### Análisis temporal

<!-- Insertar: cobertura de semanas, variación de target y cambios de distribución por WEEK_NUM. -->

### Hipótesis iniciales

<!-- Registrar hipótesis verificables sobre fuentes o variables prometedoras. -->

## 2.4 Verificar la calidad de los datos

| Hallazgo | Fuente/variable | Impacto potencial | Tratamiento propuesto |
|---|---|---|---|
| <!-- Completar --> |  |  |  |

### Riesgos de integración y fuga de información

<!-- Describir cardinalidades, posibles duplicaciones y campos cuya disponibilidad temporal deba validarse. -->

### Conclusión de la fase

<!-- Resumir qué fuentes se usarán primero y qué riesgos condicionan la preparación. -->
