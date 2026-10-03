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
