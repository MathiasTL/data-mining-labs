# 2. Comprensión de los datos

> **Estado:** completar con resultados de la exploración. No reemplazar los marcadores hasta contar con evidencia reproducible.

## 2.1 Recopilar datos iniciales

Los datos proceden de la competición Home Credit – Credit Risk Model Stability y se almacenan localmente en formato Parquet.

| Recurso                   | Ubicación                      | Uso                            |
| ------------------------- | ------------------------------ | ------------------------------ |
| Definiciones de variables | `data/feature_definitions.csv` | Interpretación de campos.      |
| Formato de predicción     | `data/sample_submission.csv`   | Validación de la salida final. |
| Entrenamiento             | `data/parquet_files/train/`    | Preparación y modelado.        |
| Prueba                    | `data/parquet_files/test/`     | Predicciones finales.          |

### Entorno y herramientas comunes (Etapa A, paso 1)

Entorno Python 3.11 con `polars`, `pyarrow`, `numpy` y `scikit-learn`; instalación y comandos en el [README](../README.md#entorno-y-herramientas). Lectura única mediante `src/dm_credit/io.py`; perfilador en `src/dm_credit/profiling.py`; esqueleto del baseline en `src/dm_credit/baseline.py`. Verificación: 29 pruebas automáticas (`pytest`).

Observaciones verificadas con los reportes del perfilador (`results/profile_train_base/` y `results/profile_test_base/`):

- `train_base.parquet` contiene 1 526 659 filas y 5 columnas: `case_id`, `date_decision`, `MONTH`, `WEEK_NUM` y `target`. El tamaño estimado en memoria es 61,1 MB.
- **La tabla base no contiene predictores propios**: solo clave del caso, campos temporales y objetivo. La señal predictiva deberá provenir de las demás fuentes cuando se integren en etapas posteriores.
- `test_base.parquet` disponible contiene solo 10 filas y 4 columnas; no incluye `target`. Su perfil sirve para revisar el esquema de ese archivo, no para evaluar rendimiento ni representar la distribución completa del test.

#### Perfil descriptivo y compatibilidad train/test (Etapa A, paso 5)

Los CSV de `results/profile_train_base/` y `results/profile_test_base/` muestran lo siguiente:

| Control  | Entrenamiento (`train_base`)                                                 | Prueba (`test_base`)                                               |
| -------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Filas    | 1 526 659                                                                    | 10                                                                 |
| Columnas | 5                                                                            | 4                                                                  |
| Nulos    | 0 en todas las columnas                                                      | 0 en todas las columnas                                            |
| Columnas | `case_id`, `date_decision`, `MONTH`, `WEEK_NUM`, `target`                    | `case_id`, `date_decision`, `MONTH`, `WEEK_NUM`                    |
| Tipos    | `case_id`, `MONTH`, `WEEK_NUM`, `target`: `Int64`; `date_decision`: `String` | `case_id`, `MONTH`, `WEEK_NUM`: `Int64`; `date_decision`: `String` |

En entrenamiento, `case_id` tiene 1 526 659 valores únicos para 1 526 659 filas y no presenta nulos, consistente con una fila por caso. El perfil reporta cero filas exactamente duplicadas; este control no sustituye la comprobación específica de claves repetidas, aunque la cardinalidad observada de `case_id` sí coincide con el número de filas. `target` contiene las clases 0 y 1: 47 994 positivos (3,14 %) y 1 478 665 negativos (96,86 %), por lo que existe un desbalance marcado. `WEEK_NUM` va de 0 a 91 y `MONTH` de 201901 a 202010. `date_decision` es texto y presenta 644 valores distintos; se recomienda tratarla como fecha si un análisis posterior requiere operaciones temporales.

La comparación de esquema indica que las cuatro columnas de `test_base` existen en entrenamiento con tipos compatibles; `target` está únicamente en entrenamiento, como se espera. En las 10 filas disponibles de test, `WEEK_NUM` es 100 y `MONTH` es 202201; por el tamaño reducido de ese archivo, estos valores no deben interpretarse como una distribución representativa del conjunto de prueba.

El perfilador no detecta nulos ni filas idénticas. En las variables numéricas no objetivo, `WEEK_NUM` contiene 16 735 ceros y `case_id` un cero; se registran como observaciones descriptivas, no como errores. No hay predictores en esta tabla base cuya cardinalidad o distribución requiera interpretación funcional por sí sola.

Contrato técnico del baseline: ver README. Este contrato no fija el corte OOT; se completa con el protocolo del paso 4.

### Inventario de tablas

<!-- Insertar tabla generada: archivo, filas, columnas, clave, granularidad, WEEK_NUM disponible y observaciones. -->

## 2.2 Describir datos

### Tabla base y unidad de análisis

<!-- Confirmar: tabla base, clave del caso, variable objetivo, número de casos y definición de una observación. -->

Integridad estructural de `train_base.parquet` (Etapa A, paso 2). Fuente: `src/dm_credit/integrity.py`, resultados en `results/integrity_train_base/`, ejecución del 2026-10-10.

- **Tamaño y estructura:** 1 526 659 filas y 5 columnas (61,15 MB en memoria). Esquema: `case_id` Int64, `date_decision` String, `MONTH` Int64, `WEEK_NUM` Int64, `target` Int64. Están presentes la clave (`case_id`), el objetivo (`target`) y la semana (`WEEK_NUM`).
- **Clave `case_id`:** 0 claves nulas, 0 claves repetidas (0 filas afectadas) y 0 filas duplicadas exactas. `case_id` identifica de forma única cada fila, por lo que una observación es un caso de crédito.
- **Consistencia temporal:** `date_decision` se almacena como String pero el 100 % de sus valores se interpreta con el formato `%Y-%m-%d` (rango 2019-01-01 a 2020-10-05). `MONTH` difiere de `yyyymm(date_decision)` en 0 filas y `WEEK_NUM` difiere de `floor((date_decision − mín(date_decision)) / 7 días)` en 0 filas.
- **Objetivo:** `target` solo toma valores en {0, 1} (0 valores fuera de ese conjunto).
- **Veredicto:** ningún hallazgo bloquea el baseline.

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

| Hallazgo           | Fuente/variable | Impacto potencial | Tratamiento propuesto |
| ------------------ | --------------- | ----------------- | --------------------- |
| <!-- Completar --> |                 |                   |                       |
| `date_decision` almacenada como String en lugar de fecha (observación no bloqueante; 1 columna con incompatibilidad tipo/contenido) | `train_base.parquet` / `date_decision` | Las comparaciones y los cortes por fecha requieren conversión explícita; sin ella se ordenan como texto. | Convertir a `Date` con `%Y-%m-%d` al preparar los datos. Fuente: `src/dm_credit/integrity.py`, `results/integrity_train_base/` (2026-10-10). |

### Riesgos de integración y fuga de información

<!-- Describir cardinalidades, posibles duplicaciones y campos cuya disponibilidad temporal deba validarse. -->

### Conclusión de la fase

<!-- Resumir qué fuentes se usarán primero y qué riesgos condicionan la preparación. -->
