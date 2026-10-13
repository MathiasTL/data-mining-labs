# 2. Comprensión de los datos

> **Estado:** capítulo 02 cerrado. Los puntos 1–8 están completados y trazados a código y reportes reproducibles. Este cierre fija decisiones y prerrequisitos para la fase siguiente, pero no implementa preparación de datos, variables ni modelos.

## 2.1 Recopilar datos iniciales

Los datos proceden de la competición Home Credit – Credit Risk Model Stability y se almacenan localmente en formato Parquet.

| Recurso                   | Ubicación                      | Uso                            |
| ------------------------- | ------------------------------ | ------------------------------ |
| Definiciones de variables | `data/feature_definitions.csv` | Interpretación de campos.      |
| Formato de predicción     | `data/sample_submission.csv`   | Validación de la salida final. |
| Entrenamiento             | `data/parquet_files/train/`    | Preparación y modelado.        |
| Prueba                    | `data/parquet_files/test/`     | Predicciones finales.          |

### Entorno y herramientas comunes (Etapa A, paso 1)

Entorno Python 3.11 con `polars`, `pyarrow`, `numpy` y `scikit-learn`; instalación y comandos en el [README](../README.md#entorno-y-herramientas). Lectura única mediante `src/dm_credit/io.py`; perfilador en `src/dm_credit/profiling.py`; baseline y métrica en `src/dm_credit/baseline.py`. Verificación al cerrar el capítulo: 81 pruebas automáticas con `uv run --extra dev pytest -q`.

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

La tabla base, su clave, el objetivo y la unidad de observación quedaron confirmados mediante la siguiente comprobación reproducible.

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

Las 503 apariciones de las 17 familias proceden de `results/profile_train_*/columns.csv`. Las 19 columnas de `credit_bureau_a_2` se incorporan desde su perfil agregado en `results/profile_train_credit_bureau_a_2/` y quedan identificadas mediante `profile_source=column_profile`. El catálogo incluye su porcentaje de nulos; `n_unique` permanece vacío para esta familia porque el perfil agregado conserva cotas inferior y superior de cardinalidad, no un conteo único exacto.

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

### Revisión de la métrica oficial

La métrica de selección será **Gini Stability**, según la página oficial [Evaluation de Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/evaluation), consultada el **2026-10-12**. El endpoint oficial de la competición identifica el algoritmo como `Home Credit 2023 - Gini Stability` (ID 1126, kernel de métrica 47057943, versión 16032). No es un AUC calculado sobre todas las semanas juntas: primero se calcula un Gini independiente para cada `WEEK_NUM` observado y luego se agregan esos valores sin ponderarlos por cantidad de casos.

Para cada semana `t`, con ambas clases presentes:

```text
G_t = 2 × ROC_AUC(y_t, p_t) − 1
Gini Stability = mean(G_t) + 88 × min(0, a) − 0,5 × std_pop(r_t)
r_t = G_t − (a × WEEK_NUM_t + b)
```

`a` y `b` son respectivamente la pendiente y el intercepto de una regresión lineal ordinaria de los Gini semanales contra los valores numéricos observados de `WEEK_NUM`. La media es aritmética y semanal, no ponderada. La desviación de los residuos es **poblacional** (`numpy.std`, `ddof=0`). El término `88 × min(0, a)` penaliza únicamente una pendiente negativa; una pendiente positiva no recibe premio. El término `0,5 × std_pop(r_t)` penaliza variación alrededor de la tendencia. Las predicciones constantes producen AUC 0,5 y Gini 0 si existen ambas clases.

El código oficial no define una sustitución para entradas inválidas: una semana monoclase hace que ROC AUC no sea calculable; nulos o valores no finitos tampoco reciben un valor especial. Para evitar resultados silenciosamente distintos de la competición, el proyecto **rechaza** el cálculo completo si cualquier semana observada contiene una sola clase, si hay nulos/no finitos, si el objetivo no pertenece a `{0, 1}` o si las longitudes no coinciden. Las semanas ausentes no se interpolan: solo participan los `WEEK_NUM` observados. Se exigen al menos dos semanas evaluables para estimar la pendiente. Estas convenciones están implementadas en `src/dm_credit/baseline.py` y probadas con datos sintéticos en `tests/test_baseline.py`.

La auditoría corrigió la implementación anterior, que descartaba silenciosamente Gini semanales no finitos. La API separa ahora Gini global, tabla de Gini semanal y conteos, regresión temporal, desviación residual y resultado final mediante `gini`, `weekly_gini` y `gini_stability_components`. La evidencia reproducible está en `results/metric_review/definition.csv`, `example_components.csv` y `source.csv`.

#### Métricas internas por experimento

| Métrica                                                  | Partición              | Papel                                                              | Requiere probabilidades                      |
| -------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------ | -------------------------------------------- |
| Gini Stability                                           | OOT (`WEEK_NUM >= 74`) | **Métrica oficial y criterio primario de selección**               | No; depende del orden de los scores          |
| ROC AUC y Gini global                                    | Entrenamiento y OOT    | Diagnóstico de discriminación global; no sustituyen Gini Stability | No                                           |
| Gini por semana; media, desviación y pendiente semanales | OOT                    | Diagnóstico temporal y componentes auditables                      | No                                           |
| Casos y positivos por semana                             | OOT                    | Soporte de cada estimación semanal                                 | No                                           |
| Log loss y Brier score                                   | OOT                    | Diagnóstico de calidad probabilística/calibración                  | **Sí**                                       |
| PR AUC                                                   | OOT                    | Diagnóstico complementario por desbalance                          | No, aunque se calcula sobre scores continuos |

Cada comparación conservará entrenamiento `WEEK_NUM < 74` y OOT `WEEK_NUM >= 74`. El modelo se seleccionará por mayor Gini Stability OOT; las demás métricas explicarán discriminación global, estabilidad, soporte y calidad probabilística, pero no reemplazarán la métrica oficial.

### Hipótesis iniciales

La tabla base no permite formular hipótesis bivariadas sobre predictores porque carece de ellos. Los perfiles de fuentes relacionadas sugieren revisar en fases posteriores —sin asumir capacidad predictiva— la ausencia estructural, historiales de mora/pagos, importes, categorías casi constantes y variables de alta cardinalidad. Estas son líneas de revisión, no hallazgos causales ni autorización para usar campos cuya disponibilidad temporal no esté demostrada.

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

Fuente: comparación de metadatos Parquet registrada en `results/parquet_inventory.csv` y lectura de los valores de las columnas citadas en los archivos `static_0` y `static_cb_0` disponibles. La comparación completa de cobertura, tipos, categorías y rangos quedó posteriormente consolidada por `src/dm_credit/schema_compatibility.py` en `results/schema_compatibility/`; por tanto, esta subsección se conserva solo como control inicial.

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

### Riesgos de integración y fuga de información (Etapa A, punto 6)

El análisis reproducible está en `src/dm_credit/integration_analysis.py` y sus resultados en `results/integration_analysis/`. Se leyeron únicamente las claves, una familia y un lote a la vez, con un hilo; no se materializaron joins entre fuentes. Los fragmentos físicos se consolidaron como una familia lógica y todas las métricas se calcularon contra la unidad final de modelado: una fila de `train_base` por `case_id`.

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.integration_analysis
```

#### Cardinalidad y cobertura observadas

Las 17 familias tienen cero filas huérfanas respecto de `train_base`. Las claves candidatas no tienen nulos ni repeticiones según `results/key_integrity_all_families.csv`. En profundidad 2, `case_id + num_group1` se repite, como corresponde a un padre con varios hijos, mientras que `case_id + num_group1 + num_group2` sí es única; `num_group1` y `num_group2` nunca se consideran claves globales. Los percentiles se calculan entre casos presentes en cada fuente.

| Familia             | Prof. | Clave candidata                     | Archivos |       Filas |     Casos | Cobertura | Filas/caso mediana / p99 / máx. | Cardinalidad |
| ------------------- | ----: | ----------------------------------- | -------: | ----------: | --------: | --------: | ------------------------------: | ------------ |
| `base`              |     0 | `case_id`                           |        1 |   1 526 659 | 1 526 659 |  100,00 % |                       1 / 1 / 1 | `1:1`        |
| `static_0`          |     0 | `case_id`                           |        2 |   1 526 659 | 1 526 659 |  100,00 % |                       1 / 1 / 1 | `1:1`        |
| `static_cb_0`       |     0 | `case_id`                           |        1 |   1 500 476 | 1 500 476 |   98,28 % |                       1 / 1 / 1 | `1:0..1`     |
| `applprev_1`        |     1 | `case_id + num_group1`              |        2 |   6 525 979 | 1 221 522 |   80,01 % |                     4 / 20 / 20 | `1:N`        |
| `credit_bureau_a_1` |     1 | `case_id + num_group1`              |        4 |  15 940 537 | 1 386 273 |   90,80 % |                   10 / 31 / 529 | `1:N`        |
| `credit_bureau_b_1` |     1 | `case_id + num_group1`              |        1 |      85 791 |    36 500 |    2,39 % |                      2 / 7 / 21 | `1:N`        |
| `debitcard_1`       |     1 | `case_id + num_group1`              |        1 |     157 302 |   111 772 |    7,32 % |                      1 / 5 / 66 | `1:N`        |
| `deposit_1`         |     1 | `case_id + num_group1`              |        1 |     145 086 |   105 111 |    6,89 % |                      1 / 5 / 65 | `1:N`        |
| `other_1`           |     1 | `case_id + num_group1`              |        1 |      51 109 |    51 109 |    3,35 % |                       1 / 1 / 1 | `1:0..1`     |
| `person_1`          |     1 | `case_id + num_group1`              |        1 |   2 973 991 | 1 526 659 |  100,00 % |                      1 / 5 / 10 | `1:N`        |
| `tax_registry_a_1`  |     1 | `case_id + num_group1`              |        1 |   3 275 770 |   457 934 |   30,00 % |                     6 / 20 / 99 | `1:N`        |
| `tax_registry_b_1`  |     1 | `case_id + num_group1`              |        1 |   1 107 933 |   150 732 |    9,87 % |                    6 / 21 / 101 | `1:N`        |
| `tax_registry_c_1`  |     1 | `case_id + num_group1`              |        1 |   3 343 800 |   482 265 |   31,59 % |                    6 / 19 / 121 | `1:N`        |
| `applprev_2`        |     2 | `case_id + num_group1 + num_group2` |        1 |  14 075 487 | 1 221 522 |   80,01 % |                     8 / 48 / 79 | `1:N`        |
| `credit_bureau_a_2` |     2 | `case_id + num_group1 + num_group2` |       11 | 188 298 452 | 1 385 288 |   90,74 % |                96 / 588 / 9 404 | `1:N`        |
| `credit_bureau_b_2` |     2 | `case_id + num_group1 + num_group2` |        1 |   1 286 755 |    36 447 |    2,39 % |                  36 / 108 / 301 | `1:N`        |
| `person_2`          |     2 | `case_id + num_group1 + num_group2` |        1 |   1 643 410 | 1 435 105 |   94,00 % |                      1 / 4 / 34 | `1:N`        |

El detalle reproducible incluye mínimo, máximo, media, mediana, p90, p95 y p99 en `family_cardinality.csv`. Que `other_1` tenga profundidad 1 no obliga a que sea `1:N`: en los datos observados `num_group1=0` y hay como máximo una fila por caso.

#### Riesgo de multiplicación

Un `LEFT JOIN` directo de una sola fuente conserva una fila para casos sin cobertura y replica los cubiertos. Los riesgos máximos son:

| Familia             | Filas tras join directo | Factor medio sobre toda la base | Media por caso cubierto | Máximo | Casos con ≥100 filas |
| ------------------- | ----------------------: | ------------------------------: | ----------------------: | -----: | -------------------: |
| `credit_bureau_a_2` |             188 439 823 |                         123,43× |                  135,93 |  9 404 |              630 603 |
| `credit_bureau_a_1` |              16 080 923 |                          10,53× |                   11,50 |    529 |                  170 |
| `applprev_2`        |              14 380 624 |                           9,42× |                   11,52 |     79 |                    0 |
| `applprev_1`        |               6 831 116 |                           4,47× |                    5,34 |     20 |                    0 |
| `tax_registry_c_1`  |               4 388 194 |                           2,87× |                    6,93 |    121 |                    3 |
| `tax_registry_a_1`  |               4 344 495 |                           2,85× |                    7,15 |     99 |                    0 |
| `credit_bureau_b_2` |               2 776 967 |                           1,82× |                   35,30 |    301 |                  597 |

Las cifras completas están en `join_risk.csv`. Dos fuentes `1:N` no deben unirse entre sí por `case_id`: para un caso con 2 filas en A y 3 en B el resultado tiene `2 × 3 = 6` filas, no 5; el ejemplo determinista de dos casos está en `cartesian_example.csv`. En el peor caso teórico, combinar los máximos de `credit_bureau_a_2` (9 404) y `credit_bureau_a_1` (529) podría producir 4 974 716 combinaciones para un solo `case_id` si coincidieran. Por ello, cada familia se filtra temporalmente y se agrega por separado antes de cualquier join con la base.

#### Disponibilidad temporal y fuga

`temporal_leakage_review.csv` registra por familia las columnas temporales, estados/resultados, identificadores y cuasi-identificadores, regla previa a agregación y acción. La evidencia usada es la definición oficial más el nombre de columna; esto confirma la semántica, **no** el instante real de extracción. No se declara seguro ningún predictor solo por su nombre.

- `base`: `date_decision` es la referencia _as-of_; `WEEK_NUM` crea la partición y `target` es el resultado. `case_id`, `date_decision`, `MONTH`, `WEEK_NUM` y `target` quedan fuera de predictores.
- Con fechas o periodos explícitos (`applprev_1`, ambas familias `credit_bureau_a`, ambas `credit_bureau_b`, `debitcard_1`, `deposit_1`, `person_1`, `person_2`, `static_0`, `static_cb_0` y los tres registros fiscales), se exige que la fecha aplicable del evento o _snapshot_ sea `<= date_decision` y que la ventana retrospectiva esté definida **antes** de agregar. Fechas de fin futuras pueden describir calendario contractual conocido, pero no prueban disponibilidad del resto del estado; requieren una regla por campo.
- `applprev_2` y `other_1` no tienen referencia temporal de fila suficiente. Deben vincularse primero con un padre cuya disponibilidad esté validada o excluirse preventivamente; `credacc_cards_status_52L` no se admite sin esa trazabilidad.
- Estados, aprobaciones/rechazos, cancelaciones, deuda “actual”, pagos, DPD/mora, cierres y respuestas pueden haberse actualizado después de la decisión. Se excluyen hasta demostrar una instantánea válida a `date_decision`; tener una fecha en la misma familia no demuestra automáticamente que esa fecha gobierne cada medida.
- Claves técnicas y cuasi-identificadores —nombres, empleadores/instituciones, direcciones, distritos, códigos postales y teléfonos— no entran directamente como predictores. Cualquier uso posterior exige justificación, tratamiento de cardinalidad y revisión de privacidad/estabilidad.
- Variables sin fecha, linaje o evidencia _as-of_ quedan en “revisión de dominio”; no pasan por defecto al modelado.

#### Contrato seguro de integración

El contrato auditable completo y su condición explícita de fallo están en `integration_contract.csv`:

1. Partir de `train_base`, con una fila y clave no nula por `case_id`.
2. Crear primero entrenamiento (`WEEK_NUM < 74`) y OOT (`WEEK_NUM >= 74`), sin solapamiento ni omisiones.
3. Aplicar el corte _as-of_ y la ventana retrospectiva antes de cualquier agregación.
4. Agregar cada familia lógica independientemente hasta una fila por `case_id`; en profundidad 2 respetar primero el padre `case_id + num_group1` y luego sus hijos `num_group2` cuando la característica lo requiera.
5. Preservar `source_present`, conteo de registros y conteos/indicadores de ausencia relevantes.
6. Fallar ante `case_id` nulo o repetido después de cada agregación.
7. Unir a base únicamente agregados cuya relación sea `1:1` por `case_id`.
8. Fallar si cualquier join cambia el número de filas, la unicidad o el conjunto de `case_id` de la base.
9. Ajustar imputación, codificación, selección y toda estadística aprendida exclusivamente con `WEEK_NUM < 74`.
10. Excluir `case_id`, `num_group1`, `num_group2`, `date_decision`, `MONTH` y `WEEK_NUM` de predictores salvo decisión posterior explícita; `target` permanece exclusivamente como etiqueta.

`safe_left_join` implementa las invariantes mínimas de unicidad y número de filas, y `aggregate_one_row_per_case` ofrece una referencia pequeña para conteos auditables. Las agregaciones de negocio concretas (por ejemplo, último valor válido, suma en ventana, máximo histórico o número de contratos) se definirán por semántica y nunca mezclando primero dos fuentes multirregistro.

### Compatibilidad train/test de las fuentes adicionales (Etapa A, punto 7)

El análisis reproducible está en `src/dm_credit/schema_compatibility.py` y sus resultados en `results/schema_compatibility/`. Consolida los shards por familia lógica a partir de `parquet_inventory.csv`, reutiliza los perfiles de train para tipos y rangos, y lee secuencialmente la muestra local de test. Para las categorías de test que no aparecían entre las cinco más frecuentes guardadas, realiza únicamente una búsqueda dirigida de pertenencia en train, en lotes monohilo; no calcula frecuencias nuevas ni aprende vocabularios desde test.

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.schema_compatibility
```

#### Cobertura y columnas

Las 17 familias lógicas están presentes en ambos splits. Train tiene 32 archivos físicos y test 36 porque difiere el número de shards en `static_0`, `applprev_1`, `credit_bureau_a_1` y `credit_bureau_a_2`; esto no constituye una diferencia de esquema lógico. Las 503 apariciones de columnas por familia coinciden por nombre salvo `target`, correctamente presente solo en `train_base`. No hay predictores exclusivos de train ni columnas adicionales de test. `test_tax_registry_c_1.parquet` está presente pero vacío; los demás archivos test tienen solo 10 filas. `family_coverage.csv` conserva archivos, filas, columnas compartidas y diferencias por familia.

#### Tipos y coerción

De las 503 columnas, 428 tienen tipo físico idéntico, 53 son convertibles explícitamente al tipo lógico, 17 aparecen físicamente como `null` en al menos un shard test, cuatro fechas requieren validación con un test real porque se almacenan como `Float64` pero sus 10 valores son nulos, y una es `target`. No se confirmó ningún valor no convertible ni incompatibilidad de tipo en la muestra. Las cuatro diferencias potenciales son `static_0.validfrom_1069D` y `static_cb_0.assignmentdate_238D`, `birthdate_574D` y `responsedate_1012D`: son `String` en train, `Float64` sin valores observados en test y su tipo lógico es fecha. Las 17 columnas `null` se consideran sin observaciones, no de tipo lógico `Null`; incluyen fechas, categóricas, booleanas, numéricas y todas las columnas del archivo vacío `tax_registry_c_1`.

El esquema lógico se deriva de train, definiciones oficiales y sufijos Home Credit: `D` se interpreta como fecha `%Y-%m-%d`, `M` como categórica textual, los tipos numéricos `A`/`P`/`L`/`T` conservan la representación compatible definida por train, y claves/partición mantienen tipos enteros. Las diferencias entero/flotante solo se aceptan para enteros si todo valor finito tiene parte fraccionaria cero. Texto booleano solo admite `true`, `false`, `1` o `0`. Toda coerción es explícita y cualquier valor no convertible produce error; nunca se convierte silenciosamente en nulo.

#### Categorías

Se revisaron 112 columnas categóricas o booleanas: 101 tienen todos los valores observados de test confirmados en train (71 ya estaban en el `top-k` y 30 se comprobaron mediante búsqueda dirigida), ocho no tienen categorías observables y tres contienen cuatro categorías no observadas en train:

| Familia             | Columna                     | Categorías desconocidas observadas |
| ------------------- | --------------------------- | ---------------------------------- |
| `applprev_1`        | `cancelreason_3545846M`     | `f2052b23`                         |
| `credit_bureau_a_1` | `financialinstitution_382M` | `da981185`                         |
| `tax_registry_a_1`  | `name_4527232M`             | `cc68a90a`, `ff9eb829`             |

Esto es una comprobación técnica de la muestra, no una estimación de frecuencia o drift. Los vocabularios se ajustan solo con `WEEK_NUM < 74`; OOT/test nunca los amplían. Todo valor no visto se asigna al bucket reservado `__UNKNOWN__` o se entrega a un codificador con soporte equivalente. Las categorías de train ausentes en estas diez filas no se interpretan por ser una muestra no representativa.

#### Rangos y convertibilidad

No se observaron infinitos, valores numéricos no convertibles ni fechas no parseables. Hay 80 observaciones fuera del rango de train distribuidas en nueve columnas: `base.MONTH` y `base.WEEK_NUM`; `applprev_1.revolvingaccount_394A`; cuatro años `T` de `credit_bureau_a_1` (`dpdmaxdateyear_596T`, `dpdmaxdateyear_896T`, `overdueamountmaxdateyear_2T`, `overdueamountmaxdateyear_994T`); `other_1.num_group1`; y `credit_bureau_a_2.pmts_year_1139T`. Por ejemplo, test contiene `MONTH=202201`, `WEEK_NUM=100` y años hasta 2022, posteriores a los máximos de train. Son alertas técnicas esperables en un corte temporal futuro, no errores ni prueba de drift. Los valores numéricos nuevos se admiten salvo restricción de dominio explícita; infinitos, conversiones fallidas o violaciones de dominio sí bloquean el pipeline. El detalle de mínimos, máximos y conteos está en `range_compatibility.csv`.

#### Contrato de compatibilidad

`compatibility_contract.csv` registra once reglas auditables:

1. Derivar el esquema lógico de train y definiciones, nunca de test.
2. Permitir `target` únicamente en `train_base`.
3. Exigir todas las variables seleccionadas al transformar OOT/test.
4. Alinear y reordenar por nombre, no por posición.
5. Coercionar explícitamente y fallar ante cualquier valor no convertible.
6. Tratar columnas físicas `Null` o enteramente nulas como no observadas y aplicarles el tipo lógico de train.
7. Ajustar imputación, escalado, selección y vocabularios solo con `WEEK_NUM < 74`.
8. Reservar `__UNKNOWN__` —o soporte equivalente— para categorías desconocidas sin reajustar el codificador.
9. Admitir rangos numéricos nuevos salvo una regla de dominio explícita y registrarlos como advertencia.
10. Verificar tras cada transformación las mismas features, orden y tipos, además de ausencia de infinitos.
11. No usar estadísticas de test para elegir columnas, umbrales o tratamientos.

Producen **error**: predictor requerido ausente, `target` en test, columna no convertible, fecha inválida, fracción al coercionar a entero, infinito, violación explícita de dominio, vocabulario sin manejo de desconocidos o cambio de features/orden/tipos tras transformar. Producen **advertencia**: columna extra no seleccionada, columna física `Null`, categoría desconocida manejable y extensión del rango de train. La diferencia física de las cuatro fechas completamente nulas queda como incompatibilidad potencial que exige validación con el test real, no como error confirmado.

Los reportes son `family_coverage.csv`, `column_schema.csv`, `categorical_compatibility.csv`, `range_compatibility.csv`, `compatibility_contract.csv` y `run_metadata.csv`. Los metadatos dejan explícitos el corte 74, formato de fecha, bucket desconocido, ejecución monohilo y limitación de la muestra. Test se usa exclusivamente para compatibilidad técnica: no participa en imputación, selección, umbrales, escalado, vocabularios ni decisiones de modelado.

## 2.5 Trazabilidad y cierre de los ocho puntos

La siguiente matriz identifica el origen reproducible de cada resultado. Los comandos que leen Parquet se muestran para reproducibilidad, no como instrucción de reejecutarlos durante este cierre documental.

|                   Punto | Estado  | Implementación o comando principal                                                                                                                                                                           | Evidencia                                                                                            | Decisión principal                                                                                                                    |
| ----------------------: | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
|           1. Inventario | Cerrado | Lectura común en `src/dm_credit/io.py`; inspección de metadatos que produjo el inventario consolidado                                                                                                        | `results/parquet_inventory.csv`                                                                      | Trabajar con 17 familias lógicas y no confundir shards físicos con fuentes distintas.                                                 |
|   2. Catálogo funcional | Cerrado | `uv run -m dm_credit.variable_catalog`                                                                                                                                                                       | `results/variable_catalog.csv`, `results/variable_catalog_summary.csv`                               | Reservar claves y campos temporales; revisar semántica y disponibilidad antes de admitir predictores.                                 |
| 3. Calidad e integridad | Cerrado | `uv run -m dm_credit.profiling`; `uv run -m dm_credit.streaming_profiling`; `uv run -m dm_credit.streaming_categories`; `uv run -m dm_credit.streaming_key_integrity`; `uv run -m dm_credit.quality_catalog` | `results/profile_train_*/`, `results/quality_catalog*.csv`, `results/key_integrity_all_families.csv` | No corregir extremos ni eliminar columnas por ausencia sin interpretación; las claves candidatas validadas gobiernan la granularidad. |
|    4. Objetivo y tiempo | Cerrado | `uv run -m dm_credit.target_analysis --out results/target_analysis`                                                                                                                                          | `results/target_analysis/`                                                                           | Entrenar con `WEEK_NUM < 74`, evaluar OOT con `WEEK_NUM >= 74` y excluir `WEEK_NUM` de predictores.                                   |
|      5. Métrica oficial | Cerrado | Implementación en `src/dm_credit/baseline.py`; pruebas en `tests/test_baseline.py`                                                                                                                           | `results/metric_review/`                                                                             | Seleccionar modelos por Gini Stability OOT; usar AUC/Gini global, PR AUC, log loss y Brier solo como diagnósticos acordados.          |
|          6. Integración | Cerrado | `POLARS_MAX_THREADS=1 uv run -m dm_credit.integration_analysis`                                                                                                                                              | `results/integration_analysis/`                                                                      | Filtrar temporalmente y agregar cada familia por separado antes del join; fallar si se rompe cualquier invariante de caso.            |
|       7. Compatibilidad | Cerrado | `POLARS_MAX_THREADS=1 uv run -m dm_credit.schema_compatibility`                                                                                                                                              | `results/schema_compatibility/`                                                                      | Derivar el esquema y transformaciones de train temporal; coercionar explícitamente y reservar tratamiento para desconocidos.          |
|        8. Documentación | Cerrado | Auditoría integral de este capítulo y del `README.md`; validación con `git diff --check` y suite completa                                                                                                    | `development/02-comprension-datos.md`, `README.md`                                                   | Transferir un contrato explícito a la fase siguiente sin implementar el capítulo 03.                                                  |

Los módulos auxiliares `src/dm_credit/shard_profile_aggregation.py`, `streaming_profiling.py`, `streaming_categories.py` y `streaming_key_integrity.py` explican cómo se construyeron perfiles agregados, categorías e integridad sin materializar todas las fuentes. Sus interfaces concretas se consultan con `uv run -m dm_credit.<módulo> --help`; algunos requieren rutas de shards y destinos específicos, por lo que la matriz evita presentar argumentos inventados.

### Hechos, inferencias, decisiones y limitaciones

- **Hechos observados:** existen 68 Parquet locales —32 train y 36 test— organizados en 17 familias lógicas. `train_base` contiene 1 526 659 casos únicos y 47 994 positivos (3,14 %). El catálogo contiene 503 apariciones, 471 nombres distintos y 464 nombres con definición oficial. Los perfiles muestran ausencia heterogénea, 32 categóricas casi constantes entre valores observados, alta cardinalidad y extremos que requieren interpretación. Las claves candidatas son únicas, no nulas y no hay filas huérfanas respecto de la base.
- **Hechos de integración:** los joins directos de fuentes `1:N` multiplican filas; `credit_bureau_a_2` alcanzaría 188 439 823 filas y un factor medio de 123,43× al unirse sin agregar. La unidad final debe seguir siendo una fila por `case_id`.
- **Inferencias conservadoras:** nombres, sufijos y definiciones oficiales ayudan a interpretar semántica, tipo lógico y posibles riesgos, pero no prueban por sí solos que un campo estuviera disponible en `date_decision`. Campos sin fecha, linaje o evidencia _as-of_, incluidos los marcados en revisión de dominio, no se consideran seguros automáticamente.
- **Decisiones del equipo:** corte OOT en 74; `WEEK_NUM` fuera de predictores; Gini Stability OOT como criterio primario; cada familia se filtra y agrega independientemente; identificadores, índices de grupo y campos temporales reservados quedan fuera de predictores; test se usa solo para compatibilidad técnica.
- **Limitaciones de perfilado:** las fuentes grandes se procesaron mediante escaneo streaming, lotes, checkpoints y agregaciones externas. En particular, la cardinalidad de `credit_bureau_a_2` se conserva como cotas en el perfil agregado, no como un `n_unique` exacto por columna. Los `top-k` categóricos son exactos para las categorías reportadas, pero no constituyen vocabularios completos en el catálogo de calidad.
- **Limitaciones de test:** cada archivo local tiene 10 filas, salvo `test_tax_registry_c_1.parquet`, que está vacío. Cuatro fechas físicamente `Float64` están completamente nulas y no permiten validar la coerción con valores reales. Las cuatro categorías desconocidas y los rangos nuevos observados prueban que el pipeline debe tolerar esos casos, pero la muestra no permite estimar frecuencias ni interpretar extensiones de rango como drift.
- **Reglas de fallo:** deben bloquear la integración las claves nulas o duplicadas, la alteración del número o conjunto de casos y cualquier join no `1:1` tras agregar. Deben bloquear la compatibilidad los predictores requeridos ausentes, `target` en test, coerciones inválidas, fechas inválidas, fracciones al convertir a entero, infinitos, violaciones de dominio, falta de manejo de desconocidos o cambios de features, orden o tipos. Columnas extra no seleccionadas, tipos físicos `Null`, categorías desconocidas manejables y rangos nuevos son advertencias registrables, no aprendizaje desde test.

### Contrato documental de entrada al capítulo 03

La fase posterior deberá respetar estos prerrequisitos; aquí solo se documentan y **no se implementan**:

1. Conservar exactamente una fila por `case_id` y usar `train_base` como universo de casos.
2. Crear la partición temporal antes de ajustar cualquier transformación: train con `WEEK_NUM < 74` y OOT con `WEEK_NUM >= 74`, sin solapamiento ni omisiones.
3. Filtrar cada evento o _snapshot_ por disponibilidad `<= date_decision` y por una ventana retrospectiva definida antes de agregar; excluir preventivamente lo que no tenga evidencia temporal suficiente.
4. Agregar cada familia lógica de forma independiente a una fila por `case_id`, respetando la jerarquía `num_group1`/`num_group2`, antes de unirla a la base.
5. Validar después de cada agregación y join la no nulidad y unicidad de `case_id`, el número de filas y el conjunto exacto de casos; fallar explícitamente ante una violación.
6. Ajustar imputación, codificación, escalado, vocabularios, umbrales y selección exclusivamente con las semanas `< 74`; OOT y test solo reciben transformaciones ya ajustadas.
7. Reservar `__UNKNOWN__` o soporte equivalente para categorías no vistas, sin ampliar vocabularios con OOT/test.
8. Alinear columnas por nombre y orden, aplicar el esquema lógico derivado de train y realizar coerciones estrictas; fallar ante valores no convertibles, infinitos o cambios de contrato.
9. Excluir de predictores `case_id`, `num_group1`, `num_group2`, `date_decision`, `MONTH`, `WEEK_NUM` y `target`; cualquier excepción futura requerirá una decisión explícita y evidencia temporal.
10. Evaluar finalmente en OOT con Gini Stability como criterio de selección y conservar Gini/AUC global, Gini semanal y su soporte, PR AUC, log loss y Brier como diagnósticos acordados.

### Conclusión de la fase

El capítulo 02 queda formalmente cerrado con los ocho puntos completados. La evidencia confirma 68 archivos, 17 familias, una base desbalanceada de 1 526 659 casos, cobertura semántica amplia pero no equivalente a disponibilidad temporal, calidad heterogénea, claves candidatas íntegras, ausencia de huérfanos y riesgo severo de multiplicación en joins directos. El protocolo queda fijado en `WEEK_NUM = 74`, con Gini Stability OOT como métrica primaria y reglas conservadoras de integración y fuga.

La compatibilidad lógica train/test es completa salvo `target`, esperado solo en train; las coerciones, categorías desconocidas y extensiones de rango tienen tratamiento explícito. La muestra local de test limita cualquier conclusión distributiva. Los asuntos transferidos son la validación de dominio y disponibilidad _as-of_, la definición semántica de agregaciones, el tratamiento de ausencia/extremos/cardinalidad y la implementación de transformaciones bajo el contrato anterior. No se prepararon datos, no se crearon features y no se entrenaron modelos como parte de este cierre.
