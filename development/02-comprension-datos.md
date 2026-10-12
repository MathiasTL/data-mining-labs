# 2. Comprensión de los datos

> **Estado:** en desarrollo. Inventario y catálogo de variables completados; continúan pendientes los análisis de calidad, integración y validación temporal indicados al final del capítulo.

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

Contrato técnico del baseline: ver README. El protocolo temporal del paso 4 fija `WEEK_NUM = 74` como corte OOT inicial: entrenamiento en semanas `< 74` y validación en semanas `>= 74`.

### Inventario de tablas Parquet

Se inspeccionaron los metadatos Parquet de los archivos locales, sin ejecutar el perfilamiento descriptivo completo. El inventario detallado por archivo (filas, columnas, tipos y columnas candidatas a clave) está en `results/parquet_inventory.csv`.

| Familia de archivos    | Archivos train/test | Filas por archivo en train | Columnas por archivo | Claves observadas / granularidad inferida                                                              |
| ---------------------- | ------------------: | -------------------------: | -------------------: | ------------------------------------------------------------------------------------------------------ |
| `base`                 |               1 / 1 |                  1 526 659 |                5 / 4 | `case_id`; una fila por caso, verificado para train.                                                   |
| `applprev_1`           |               2 / 3 |        2 638 295–3 887 684 |                   41 | `case_id`, `num_group1`; historial de solicitudes, probablemente varias filas por caso.                |
| `applprev_2`           |               1 / 1 |                 14 075 487 |                    6 | `case_id`, `num_group1`, `num_group2`; registros anidados por caso.                                    |
| `credit_bureau_a_1`    |               4 / 5 |        2 079 323–6 009 192 |                   79 | `case_id`, `num_group1`; registros de bureau agrupados por caso.                                       |
| `credit_bureau_a_2`    |             11 / 12 |       4 386 062–33 053 760 |                   19 | `case_id`, `num_group1`, `num_group2`; registros anidados de bureau.                                   |
| `credit_bureau_b_1`    |               1 / 1 |                     85 791 |                   45 | `case_id`, `num_group1`; registros de bureau agrupados por caso.                                       |
| `credit_bureau_b_2`    |               1 / 1 |                  1 286 755 |                    6 | `case_id`, `num_group1`, `num_group2`; registros anidados de pagos de bureau.                          |
| `person`               |               2 / 2 |        1 643 410–2 973 991 |                11–37 | `case_id`, `num_group1` y, en `person_2`, `num_group2`; personas/direcciones relacionadas con el caso. |
| `static_0`             |               2 / 3 |          522 902–1 003 757 |                  168 | `case_id`; información estática por caso (granularidad por validar para cada archivo).                 |
| `static_cb_0`          |               1 / 1 |                  1 500 476 |                   53 | `case_id`; información estática de bureau asociada al caso.                                            |
| `debitcard_1`          |               1 / 1 |                    157 302 |                    6 | `case_id`, `num_group1`; registros de tarjeta agrupados por caso.                                      |
| `deposit_1`            |               1 / 1 |                    145 086 |                    5 | `case_id`, `num_group1`; registros de depósitos agrupados por caso.                                    |
| `other_1`              |               1 / 1 |                     51 109 |                    7 | `case_id`, `num_group1`; registros de otros productos agrupados por caso.                              |
| `tax_registry_[a-c]_1` |               3 / 3 |        1 107 933–3 343 800 |                    5 | `case_id`, `num_group1`; registros fiscales agrupados por caso.                                        |

En los 68 archivos inventariados aparece la columna `case_id`. En tablas con múltiples registros por caso también aparecen `num_group1` y/o `num_group2`, que son candidatos a índices de registros dentro del caso; no se deben tratar como claves foráneas independientes sin validar su significado. Las combinaciones candidatas serían `case_id + num_group1` o `case_id + num_group1 + num_group2`, según el archivo; su unicidad debe verificarse antes de integrar tablas. La granularidad de las fuentes relacionadas se infiere aquí por sus columnas y nombres, no se considera confirmada hasta revisar las claves y duplicados.

Hay 32 archivos de train y 36 de test. Los Parquet de test disponibles contienen 10 filas cada uno, salvo `test_tax_registry_c_1.parquet`, que tiene 0; por ello, sirven para una inspección inicial de compatibilidad de esquema, pero no representan el volumen ni la distribución completa del test. En tablas con columnas tipadas como `null` en test, la muestra pequeña puede no contener valores observados; no se debe concluir que el tipo difiere sin revisar el esquema lógico esperado.

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

En `train_base`, `target` es binario y contiene 1 478 665 casos de clase 0 (96,86 %) y 47 994 casos de clase 1 (3,14 %). La distribución muestra un desbalance marcado. El detalle semanal y la selección del corte OOT se documentan en la sección de análisis temporal y en `results/target_analysis/`.

### Tipos, cobertura y variables relevantes

El perfil de `train_base` no detecta valores nulos en sus cinco columnas. `case_id` tiene un valor único por fila; `date_decision` está almacenada como texto, aunque sus valores son interpretables como fechas; `MONTH` y `WEEK_NUM` son campos temporales enteros. La tabla base no contiene predictores propios.

#### Catálogo funcional de columnas de `train_base` (Etapa A, paso 3)

| Columna         | Tipo observado | Función                         | Acción inicial                                                                                                | Justificación                                                                                          |
| --------------- | -------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `case_id`       | `Int64`        | Identificador técnico           | Excluir como predictor; conservar para trazabilidad y unión de tablas.                                        | Identifica el caso; no es una característica disponible para medir riesgo.                             |
| `date_decision` | `String`       | Variable temporal               | Conservar para trazabilidad y análisis temporal; convertir a fecha si se requieren operaciones de calendario. | El contenido se interpreta con formato `YYYY-MM-DD`; no usar como predictor en el baseline inicial.    |
| `MONTH`         | `Int64`        | Variable temporal derivada      | Excluir inicialmente como predictor; conservar para análisis temporal y control.                              | Representa el mes de decisión y duplica información temporal disponible en `date_decision`/`WEEK_NUM`. |
| `WEEK_NUM`      | `Int64`        | Variable temporal de validación | Usar para partición OOT y análisis temporal; excluir inicialmente como predictor.                             | Define el eje temporal de evaluación y su uso como predictor podría introducir atajos temporales.      |
| `target`        | `Int64`        | Variable objetivo               | Usar solo como etiqueta de entrenamiento; no incluir como predictor.                                          | Etiqueta binaria de incumplimiento; no está presente en test.                                          |

Esta clasificación cubre las columnas de `train_base`. El catálogo reproducible que extiende el análisis a las demás tablas se presenta a continuación.

#### Catálogo de variables de todas las fuentes

Se cruzaron los esquemas de las 17 familias de entrenamiento con las 465 entradas de `data/feature_definitions.csv`. El catálogo detallado está en `results/variable_catalog.csv`, su resumen por familia en `results/variable_catalog_summary.csv` y puede regenerarse con `uv run -m dm_credit.variable_catalog` mediante `src/dm_credit/variable_catalog.py`.

El resultado contiene 503 apariciones de columnas por familia y 471 nombres de variable distintos. De estos, 464 nombres tienen descripción oficial. Los únicos siete nombres sin entrada en `feature_definitions.csv` son los campos estructurales o propios de la tabla base: `case_id`, `num_group1`, `num_group2`, `date_decision`, `MONTH`, `WEEK_NUM` y `target`; por tanto, no se detectaron predictores de negocio sin definición oficial.

| Grupo de familias              | Familias incluidas                                                                                | Apariciones de columnas | Con definición oficial |
| ------------------------------ | ------------------------------------------------------------------------------------------------- | ----------------------: | ---------------------: |
| Base y estáticas               | `base`, `static_0`, `static_cb_0`                                                                 |                     226 |                    219 |
| Solicitudes previas            | `applprev_1`, `applprev_2`                                                                        |                      47 |                     42 |
| Bureau de crédito              | `credit_bureau_a_1`, `credit_bureau_a_2`, `credit_bureau_b_1`, `credit_bureau_b_2`                |                     149 |                    139 |
| Personas                       | `person_1`, `person_2`                                                                            |                      48 |                     43 |
| Productos y registros fiscales | `debitcard_1`, `deposit_1`, `other_1`, `tax_registry_a_1`, `tax_registry_b_1`, `tax_registry_c_1` |                      33 |                     21 |

La clasificación funcional, aplicada de forma reproducible a partir del nombre, tipo y descripción oficial, agrupa las 503 apariciones principalmente en producto/contrato (118), importe o capacidad financiera (115), mora o incumplimiento histórico (65), variables temporales (60), persona/demografía (39), claves o índices estructurales (35), otras medidas numéricas (31), empleo/ingresos (24), atributos categóricos (15) y objetivo (1). Es una taxonomía de navegación: no reemplaza la interpretación de dominio ni autoriza automáticamente el uso de una variable como predictor.

El catálogo también registra:

- `temporal_relevance`: distingue campos de decisión/partición, fechas o referencias temporales explícitas, historia implícita y variables sin referencia temporal evidente;
- `identifier_risk`: excluye las claves técnicas y señala nombres, direcciones, teléfonos, códigos postales, distritos e instituciones como posibles cuasi-identificadores o campos de alta cardinalidad;
- `leakage_review`: marca fechas para compararlas con `date_decision` y estados, mora, deuda, pagos, aprobaciones o rechazos para comprobar si estaban disponibles al decidir. Estas marcas son candidatos de revisión, no evidencia de fuga;
- `pct_null` y `n_unique`: incorpora los resultados de los perfiles existentes cuando están disponibles.

Las 484 apariciones de 16 familias proceden de `results/profile_train_*/columns.csv`. Las 19 columnas de `credit_bureau_a_2`, cuya materialización completa es especialmente costosa, se incorporan desde `results/parquet_inventory.csv`; quedan identificadas mediante `profile_source=inventory_schema` y no se les atribuyen estadísticas de nulos o cardinalidad. De este modo se completa el catálogo semántico sin presentar como calculadas métricas que todavía no se han obtenido para esa familia.

**Decisión inicial:** `case_id`, `num_group1` y `num_group2` se conservan para trazabilidad, orden y agregación, pero no se usarán directamente como predictores. `target` queda reservado como etiqueta, y `date_decision`, `MONTH` y `WEEK_NUM` se reservan inicialmente para construcción temporal y validación. Las demás variables requieren comprobar disponibilidad temporal y definir una agregación acorde con la granularidad de su familia antes de entrar al modelo.

## 2.3 Explorar datos

### Análisis univariado y bivariado

En la tabla base, `target` presenta un desbalance de 96,86 % de clase 0 frente a 3,14 % de clase 1. `WEEK_NUM` toma valores de 0 a 91; `MONTH`, de 201901 a 202010; y `date_decision` tiene 644 valores distintos. `case_id` es un identificador y sus valores no se interpretan como una distribución de predictor. Como la tabla base no contiene predictores, no corresponde extraer asociaciones entre predictores y `target` en esta etapa.

### Análisis temporal

El análisis reproducible se ejecuta con `uv run -m dm_credit.target_analysis --out results/target_analysis`. `weekly.csv` contiene, para cada una de las 92 semanas observadas (`WEEK_NUM` 0–91), casos, positivos, negativos, porcentajes de ambas clases, tasa móvil ponderada de cuatro semanas, cambio semanal en puntos porcentuales y banderas de soporte bajo. No hay semanas ausentes, etiquetas nulas ni valores de `target` fuera de `{0, 1}`.

La cobertura semanal no es uniforme: la mediana es 17 088,5 casos, con un mínimo de 825 en la semana 68 y un máximo de 35 920 en la semana 51. Las semanas 65–70 tienen menos de 5 000 casos y menos de 100 positivos; sus tasas individuales deben interpretarse con cautela. Aun así, todas las semanas contienen ambas clases: el mínimo semanal es 21 positivos.

La tasa de incumplimiento tampoco es estable. Su mediana semanal es 2,79 %, el mínimo es 1,77 % en la semana 86 y el máximo 5,21 % en la semana 62. Por bloques de diez semanas, aumenta de 2,68 % en 0–9 a 4,07 % en 50–59 y 4,28 % en 60–69, y luego desciende a 2,33 % en 70–79 y 2,03 % en 80–89. Los mayores cambios consecutivos se observan alrededor de las semanas 47–48 y 63–65. Esto justifica una validación temporal: una partición aleatoria mezclaría regímenes con tasas base diferentes.

**Corte OOT propuesto: `WEEK_NUM = 74`.** La regla reproducible reserva aproximadamente el último 20 % de las semanas observadas: entrenamiento con semanas 0–73 y validación con 74–91. El entrenamiento contiene 1 332 001 casos y 43 734 positivos (3,2833 %); el OOT contiene 194 658 casos y 4 260 positivos (2,1885 %), equivalentes a 18 semanas, 12,75 % de los casos y 8,88 % de los positivos. Aunque el OOT tiene una tasa menor, su tamaño y cantidad de eventos permiten calcular métricas globales y semanales, y el cambio de prevalencia constituye precisamente una prueba de estabilidad. El corte y sus cifras quedan registrados en `results/target_analysis/oot_recommendation.csv`; los 91 cortes posibles están en `cutoff_candidates.csv`.

Este será el corte primario del baseline. `WEEK_NUM` no se usará como predictor y toda transformación se ajustará únicamente con semanas `< 74`. Como análisis de sensibilidad se podrán comparar cortes cercanos (por ejemplo, 72 y 76), pero no se escogerá retrospectivamente el que produzca la mejor métrica.

### Hipótesis iniciales

<!-- Registrar hipótesis verificables sobre fuentes o variables prometedoras. -->

## 2.4 Verificar la calidad de los datos

| Hallazgo                                                                                                                                                           | Fuente/variable                        | Impacto potencial                                                                            | Tratamiento propuesto                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `date_decision` está almacenada como `String`, aunque el 100 % de los valores se interpreta como fecha con formato `%Y-%m-%d`; no se detectaron valores inválidos. | `train_base.parquet` / `date_decision` | Las operaciones de calendario requieren convertir explícitamente el campo a fecha.           | Convertir a tipo fecha cuando se realicen operaciones de calendario en preparación de datos. Fuente: `src/dm_credit/integrity.py`, `results/integrity_train_base/` (2026-10-10). |
| Desbalance de clases: 3,14 % de casos positivos.                                                                                                                   | `train_base.parquet` / `target`        | La exactitud puede ser engañosa y el entrenamiento/evaluación debe considerar el desbalance. | Reportar métricas adecuadas al problema y respetar el protocolo OOT; no modificar la distribución antes de acordar el método.                                                    |

### Compatibilidad inicial de esquemas train/test

La comparación de metadatos indica que los nombres de columnas coinciden entre train y test para las familias comparables, excepto `target`, presente solo en `train_base` como corresponde. Se observaron diferencias de tipo físico en algunos campos de fecha, pero los valores disponibles en los archivos de test son nulos:

| Familia       | Campo(s)                                                      | Tipo train | Tipo test | Observación                                                                                               |
| ------------- | ------------------------------------------------------------- | ---------- | --------- | --------------------------------------------------------------------------------------------------------- |
| `static_0`    | `validfrom_1069D`                                             | `string`   | `double`  | Los 10 valores observados en test son nulos; no se puede concluir que exista incompatibilidad semántica.  |
| `static_cb_0` | `assignmentdate_238D`, `birthdate_574D`, `responsedate_1012D` | `string`   | `double`  | Los 10 valores observados en test son nulos; la diferencia física no demuestra una incompatibilidad real. |

En algunas columnas de test el tipo Parquet aparece como `null`; esto ocurre cuando no hay valores observados para esa columna, no significa que la columna falte. En `test_tax_registry_c_1.parquet` hay 0 filas, por lo que sus campos no tienen valores observados. Los fragmentos de train que parecen pertenecer a una misma tabla mantienen esquemas consistentes; algunos fragmentos de test muestran tipos distintos en campos sin valores. Por tanto, estas diferencias deberán resolverse contra el esquema lógico de train y las definiciones antes de preparar el pipeline, sin tratarlas por ahora como errores confirmados.

Fuente: comparación de metadatos Parquet registrada en `results/parquet_inventory.csv` y lectura de los valores de las columnas citadas en los archivos `static_0` y `static_cb_0` disponibles. Este control no incluye todavía una comparación de distribuciones o categorías entre particiones.

### Perfil de calidad de las fuentes relacionadas

Se perfilan las 17 familias de entrenamiento sin combinar tablas ni imputar o eliminar valores. El detalle por columna está en `results/quality_catalog.csv`, el resumen por familia en `results/quality_catalog_summary.csv` y los reportes de origen en `results/profile_train_*/`. Las tablas menores se procesaron en memoria; las mayores se escanearon en streaming. `credit_bureau_a_2`, con 188 298 452 filas, se procesó en 11 checkpoints monohilo y se agregó sin volver a materializar los Parquet.

| Fuente              |     Filas | Cobertura y hallazgos descriptivos                                                                                                                                                                                                                                                                               |
| ------------------- | --------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `debitcard_1`       |   157 302 | Ausencia alta en saldos/turnover de 180 días (92,23 % y 92,96 %), menor en fecha de apertura (8,08 %). Se observan valores negativos en las tres métricas numéricas de tarjeta.                                                                                                                                  |
| `deposit_1`         |   145 086 | `contractenddate_991D` tiene 54,92 % de nulos. `amount_416A` va de −40 000 a 12 213 286, con 10 valores negativos.                                                                                                                                                                                               |
| `other_1`           |    51 109 | No se detectan nulos. `amtdepositbalance_4809441A` tiene un valor negativo (mínimo −335 718). `num_group1` es 0 en todas las filas, consistente con un índice estructural constante en este archivo.                                                                                                             |
| `credit_bureau_b_1` |    85 791 | Faltantes importantes: `periodicityofpmts_997L` 97,64 %, `interesteffectiverate_369L` 88,92 %, y varios campos de límites/residuales entre 49 % y 81 %. Se observan valores extremos en variables de mora (por ejemplo, máximos superiores a 200 millones) y una tasa negativa; requieren validación de dominio. |
| `credit_bureau_b_2` | 1 286 755 | `pmts_dpdvalue_108P` y `pmts_pmtsoverdue_635A` tienen 0,42 % de nulos. El máximo de `pmts_dpdvalue_108P` es 185 124 192; revisar si corresponde a una codificación o valor extremo.                                                                                                                              |
| `tax_registry_a_1`  | 3 275 770 | No se detectan nulos ni importes negativos. `name_4527232M` tiene 147 037 valores distintos; revisar su papel antes de usarlo directamente como predictor.                                                                                                                                                       |
| `tax_registry_b_1`  | 1 107 933 | No se detectan nulos ni importes negativos. `name_4917606M` tiene 55 857 valores distintos.                                                                                                                                                                                                                      |
| `tax_registry_c_1`  | 3 343 800 | No se detectan nulos ni importes negativos. `employername_160M` tiene 152 835 valores distintos.                                                                                                                                                                                                                 |

La ausencia es estructuralmente heterogénea. `credit_bureau_a_1` tiene 65 de 79 columnas con al menos 50 % de nulos; `static_cb_0`, 36 de 53; `static_0`, 49 de 168; y `person_1`, 15 de 37. Los máximos observados alcanzan 99,97 % en `clientscnt_136L` (`static_0`), 99,76 % en `prolongationcount_599L` (`credit_bureau_a_1`) y 99,67 % en `role_993L` (`person_1`). En `credit_bureau_a_2`, 8 de 19 columnas superan 50 % de nulos: destacan `collater_valueofguarantee_1124L` (98,54 %) y `collater_valueofguarantee_876L` (96,34 %). La imputación deberá preservar indicadores de ausencia y evaluarse por fuente; no se eliminarán columnas solo por este perfil.

Se obtuvieron categorías frecuentes exactas para las 170 columnas de texto o booleanas de las 17 familias mediante conteos por lotes. Hay 32 columnas cuya categoría dominante representa al menos 95 % de los valores no nulos. Algunas son constantes entre los valores observados, como `contaddr_matchlist_1032L=False`, `remitter_829L=False` y `role_993L=FULL` en `person_1`, o `bankacctype_710L=CA` e `isdebitcard_729L=False` en `static_0`. En `credit_bureau_a_2`, el código enmascarado `a55475b1` domina sus seis categóricas entre 96,34 % y 98,58 %. Estas variables pueden aportar poca separación o representar una categoría por defecto; se medirán en entrenamiento antes de excluirlas.

Solo se detectan dos columnas constantes globales según los perfiles: el índice estructural `num_group1` de `other_1` y `deferredmnthsnum_166L` de `static_0`. También aparecen rangos que requieren interpretación, no corrección automática: garantías de hasta 98 780 millones y años de pago hasta 2028 en `credit_bureau_a_2`; máximos de mora superiores a 185 millones en `credit_bureau_b_2`; importes negativos en depósitos, tarjetas y saldos; y valores negativos frecuentes en métricas DPD de `static_0`. Un valor negativo puede representar días antes de vencimiento o movimientos contables, por lo que se conserva hasta validar su semántica.

La integridad de claves se verificó exactamente para las 17 familias y quedó consolidada en `results/key_integrity_all_families.csv`. `case_id` no tiene nulos y las claves candidatas —`case_id`, `case_id + num_group1` o `case_id + num_group1 + num_group2`, según la granularidad— tienen cero grupos y cero filas duplicadas. En consecuencia, tampoco existen filas exactamente duplicadas dentro de cada familia. Para las fuentes grandes y fragmentadas se utilizó una agregación externa DuckDB limitada a un hilo y 1 GB de memoria; para la tanda menor se conservaron los controles Polars previos. La unicidad de la clave compuesta no implica que `case_id` sea único: las fuentes de profundidad 1 y 2 contienen múltiples observaciones por caso y deberán agregarse antes del join.

### Riesgos de integración y fuga de información

`case_id` es único en `train_base` y las claves candidatas de las 17 familias son íntegras, pero las tablas de profundidad 1 y 2 tienen varias filas por caso. Por ello, un join directo sobre `case_id` multiplicaría casos aunque no existan duplicados de clave. Cada fuente deberá agregarse a una fila por caso —respetando `num_group1` y `num_group2`— antes de integrarse. La disponibilidad temporal y el riesgo de fuga de los campos marcados en `results/variable_catalog.csv` siguen requiriendo revisión de dominio.

### Conclusión de la fase

La revisión confirma una unidad de análisis identificable, un catálogo funcional completo y perfiles de calidad e integridad de clave para las 17 familias de entrenamiento. No se detectaron claves candidatas nulas o duplicadas, pero sí ausencia extrema, categorías casi constantes, alta cardinalidad y rangos que requieren interpretación. Antes de preparar la tabla de modelado aún corresponde definir agregaciones por granularidad, revisar disponibilidad temporal y fugas, comparar las fuentes adicionales con test y cerrar el análisis temporal y el protocolo OOT.
