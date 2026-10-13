# A5 — Estrategia de streaming y checkpoints para bureau de crédito

> **Tarea:** A5 del backlog del capítulo 03 (`context/03-preparacion-datos-backlog.md`).
>
> **Responsable:** Mathias (P4).
>
> **Estado:** propuesta para la compuerta G1. Nada de lo descrito aquí está congelado.
>
> **Documentos relacionados:** contratos A1/A2 (`context/03-preparation-A1-A2-contracts.md`) y propuesta A4 (`context/03-preparacion-A4-solicitudes-personas.md`).
>
> **Alcance:** `credit_bureau_a_1`, `credit_bureau_a_2`, `credit_bureau_b_1` y `credit_bureau_b_2`. Cubre cómo leer, filtrar, agregar y recuperar; no fija todavía la lista de columnas (B5) ni la interfaz del adaptador (G1).

## 1. Problema

Las cuatro fuentes de bureau son `1:N` respecto de `case_id` y suman más de 205 millones de filas en train. `credit_bureau_a_2` concentra 188 298 452. El objetivo es producir un agregado con una fila por `case_id` para cada fuente sin materializar ninguna de ellas completa y sin ejecutar joins entre fuentes `1:N`.

## 2. Evidencia medida

Las cifras de filas y cobertura proceden de `results/integration_analysis/family_cardinality.csv`. Las de grupos de filas y rangos de `case_id` se obtuvieron el 12/10/2026 leyendo únicamente el pie de cada Parquet (metadatos y estadísticas), sin leer datos.

| Fuente              | Shards train |       Filas | Casos distintos | Cobertura base | Filas por caso (mediana / p99 / máx.) | Columnas |
| ------------------- | -----------: | ----------: | --------------: | -------------: | ------------------------------------- | -------: |
| `credit_bureau_a_1` |            4 |  15 940 537 |       1 386 273 |        90,80 % | 10 / 31 / 529                         |       79 |
| `credit_bureau_a_2` |           11 | 188 298 452 |       1 385 288 |        90,74 % | 96 / 588 / 9 404                      |       19 |
| `credit_bureau_b_1` |            1 |      85 791 |          36 500 |         2,39 % | 2 / 7 / 21                            |       45 |
| `credit_bureau_b_2` |            1 |   1 286 755 |          36 447 |         2,39 % | 36 / 108 / 301                        |        6 |

Hechos que condicionan el diseño:

1. **Los grupos de filas son uniformes.** Todos los shards usan grupos de 1 048 576 filas como máximo. El shard más grande, `train_credit_bureau_a_2_5.parquet`, tiene 33 053 760 filas en 32 grupos.
2. **Cada shard está ordenado por `case_id`.** Las estadísticas mínimo/máximo de los grupos de filas son monótonas dentro de cada uno de los 15 shards de `a_1` y `a_2`.
3. **Los shards no están particionados por rango.** Cada shard recorre casi todo el dominio de `case_id` (por ejemplo, `a_2_0` va de 388 a 2 548 729 y `a_2_10` de 56 408 a 2 703 454). Esto explica las 10 violaciones de orden registradas en `results/case_order_train_credit_bureau_a_2.csv`: son los 10 bordes entre los 11 shards.
4. **No se puede descartar que un `case_id` aparezca en más de un shard.** Los rangos se solapan por completo. No se ha medido si ocurre realmente (ver verificación V1).
5. **Los shards de `a_1` y `a_2` no se corresponden.** Hay 4 y 11 respectivamente, con rangos solapados, de modo que el padre de una fila de `a_2` puede estar en cualquier shard de `a_1`.
6. **`credit_bureau_a_2` no tiene columnas de fecha `D`.** Su única referencia temporal son los pares año/mes `pmts_year_1139T` + `pmts_month_706T` (contrato activo) y `pmts_year_507T` + `pmts_month_158T` (contrato cerrado). La granularidad es mensual. El emparejamiento activo/cerrado se deduce de las definiciones oficiales, que presentan incoherencias de redacción entre sí, y debe confirmarse en B5.
7. **`dm_credit.io` solo ofrece lectura completa en memoria** (`load_parquet` usa `pl.read_parquet`). Los módulos de streaming del capítulo 02 leen por su cuenta con `pl.scan_parquet` y `pyarrow.parquet.ParquetFile.iter_batches`.

## 3. Decisiones propuestas

### 3.1 Unidad de lectura: el grupo de filas

Se lee un grupo de filas por iteración, con proyección de columnas. Un lote de `a_2` con las 19 columnas ocupa del orden de 160 MB decodificado (1 048 576 filas × 19 columnas × 8 bytes, estimación; las columnas de texto pueden ocupar más) y menos al proyectar solo las columnas aprobadas en B5.

El orden interno por `case_id` garantiza que un caso solo puede quedar partido entre dos lotes consecutivos del mismo shard, lo que el diseño de 3.2 resuelve sin lógica especial.

### 3.2 Agregados parciales combinables

Como un `case_id` puede aparecer en varios lotes y, potencialmente, en varios shards, cada lote no produce el agregado final sino un **parcial** por `case_id` con estadísticos que se pueden combinar sin pérdida:

| Estadístico parcial           | Regla de combinación | Característica final derivada |
| ----------------------------- | -------------------- | ----------------------------- |
| `n` (filas válidas)           | suma                 | conteo                        |
| `n_null`                      | suma                 | nulos                         |
| `sum`                         | suma                 | suma; media = `sum / n`       |
| `sum_sq`                      | suma                 | desviación estándar           |
| `min`                         | mínimo               | mínimo                        |
| `max`                         | máximo               | máximo                        |
| `n_true`                      | suma                 | proporción verdadera          |
| conteo por categoría aprobada | suma                 | frecuencia y proporción       |

Este conjunto cubre la estrategia mínima de características de la sección 8 del backlog. Quedan **fuera** de la primera versión los estadísticos no combinables de forma exacta: mediana, percentiles y conteo distinto exacto. Si el equipo los necesita, requieren una decisión explícita en G1 (aproximación o segunda pasada).

El agregado final se obtiene combinando todos los parciales de la fuente y derivando las características. La combinación es asociativa y conmutativa, por lo que el resultado no depende del orden de los shards.

### 3.3 Una sola pasada para train y OOT

El contrato A2 define `split` como `train`, `oot` o `test`. Train y OOT proceden de los mismos archivos físicos `train_*` y solo se distinguen por el `WEEK_NUM` del caso. Leer bureau una vez por split duplicaría la lectura de `a_2` (376 millones de filas en lugar de 188).

Propuesta: los parciales por `case_id` se calculan **una sola vez** sobre los archivos `train_*` y son independientes del split. La separación ocurre al finalizar, repartiendo los casos según la partición de la tabla base (`WEEK_NUM < 74` o `>= 74`). `build_family_features(family, "train", ...)` y `build_family_features(family, "oot", ...)` reutilizan los mismos parciales si su versión de reglas y la huella de los shards coinciden. Test se procesa aparte desde los archivos `test_*`.

Esto es válido porque ningún parcial aprende parámetros entre casos: cada fila solo contribuye a su propio `case_id`.

### 3.4 Filtro temporal antes de agregar

El filtro se aplica a cada lote, antes de calcular el parcial, uniendo el lote con `date_decision` de la tabla base por `case_id` (relación `N:1`, no multiplica filas). Las filas descartadas se cuentan por regla para el informe de fuga temporal (F7).

Propuesta por fuente, a confirmar en B5 y G2:

| Fuente | Referencia temporal disponible                                                                     | Regla propuesta                                                                              |
| ------ | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `a_1`  | Fechas `D` por contrato (`dateofcredstart_*`, `lastupdate_*`, `refreshdate_3813885D`, entre otras) | Fecha aplicable `<= date_decision`; regla por campo en B5.                                   |
| `a_2`  | Solo año y mes de pago                                                                             | Conservadora: el mes del pago debe ser **estrictamente anterior** al mes de `date_decision`. |
| `b_1`  | `contractdate_551D`, `lastupdate_260D`                                                             | Fecha aplicable `<= date_decision`.                                                          |
| `b_2`  | `pmts_date_1107D`                                                                                  | `pmts_date_1107D <= date_decision`.                                                          |

Criterios comunes, alineados con la propuesta A4:

- **Fecha nula:** la fila se descarta, porque no se puede demostrar su disponibilidad.
- **Igualdad:** una fecha `D` igual a `date_decision` se conserva.
- **Fechas posteriores:** se cuentan únicamente en el manifiesto de la familia, nunca como característica por caso.

La regla de `a_2` descarta los pagos del mismo mes de la decisión porque con granularidad mensual no se puede demostrar que sean anteriores al día de la decisión. Es la única excepción al criterio de igualdad y queda como punto abierto (D1).

### 3.5 Relación padre-hijo sin join masivo

Para `a_2` y `b_2`, una fila hija solo se usa si su contrato padre `(case_id, num_group1)` es temporalmente válido en `a_1` o `b_1`.

1. Al procesar la fuente padre se escribe una tabla de **claves de padres válidos** con solo `case_id` y `num_group1`.
2. Cada lote de la fuente hija se filtra con un semi-join contra esa tabla. Un semi-join no añade columnas ni multiplica filas.

Para `a_1` la tabla tiene como máximo 15 940 537 claves, unos 255 MB como dos columnas `Int64`, y puede mantenerse en memoria durante toda la pasada de `a_2`. Para `b_1` son como máximo 85 791 claves.

Consecuencia de orden: `a_1` se procesa por completo antes que `a_2`, y `b_1` antes que `b_2`.

Esto exige que el adaptador de la fuente hija conozca el resultado de la fuente padre, mientras que el contrato A2 indica que `build_family_features` no recibe tablas de otras familias. La propuesta A4 (sección 8) plantea la misma necesidad para solicitudes y personas y propone definir "familia" como el grupo padre-hijo, citando bureau A y B. Esta propuesta la respalda: familia `credit_bureau_a` = `a_1` + `a_2` y familia `credit_bureau_b` = `b_1` + `b_2` (punto abierto D5).

### 3.6 Checkpoints y recuperación

- **Unidad de checkpoint:** el shard. Al terminar un shard se escribe su parcial y una entrada de manifiesto.
- **Publicación atómica:** el parcial se escribe en un directorio temporal de la familia y se renombra al completar sus validaciones. Un parcial visible es siempre un parcial válido.
- **Entrada de manifiesto por shard:** nombre del archivo, tamaño y fecha de modificación, filas leídas, filas descartadas por regla temporal, filas descartadas por padre inválido, casos distintos, versión de reglas, duración y memoria máxima.
- **Reanudación:** al arrancar se omiten los shards con entrada válida cuya versión de reglas y huella del archivo coinciden. Un cambio de reglas invalida todos los parciales de la fuente.
- **Cierre:** la combinación final se ejecuta solo cuando todos los shards descubiertos tienen parcial válido.

### 3.7 Descubrimiento de shards

Los shards se descubren por patrón de nombre o por `results/parquet_inventory.csv`, nunca con una lista fija, y se ordenan por su índice numérico (no alfabéticamente, para que `_10` no preceda a `_2`). Train y test difieren en el número de shards de `a_1` y `a_2`.

## 4. Presupuesto de memoria

Estimaciones para la ejecución real con un solo hilo (`POLARS_MAX_THREADS=1`). Son cotas de diseño que el prototipo y la ejecución E4 deben confirmar con medición.

| Componente                                                                | Estimación              |
| ------------------------------------------------------------------------- | ----------------------- |
| Lote de `a_2` en memoria (un grupo de filas, 19 columnas)                 | ~160 MB                 |
| Claves de padres válidos de `a_1`                                         | ≤ 255 MB                |
| `date_decision` de la base para el filtro (1 526 659 casos, dos columnas) | ~25 MB                  |
| Parcial acumulado de un shard (casos del shard × estadísticos)            | depende de B5; ver nota |
| **Objetivo de pico por proceso**                                          | **≤ 2 GB**              |

Nota: el parcial crece con el número de columnas aprobadas. Con 1 385 288 casos y, por ejemplo, 60 estadísticos `Float64`, el parcial completo de `a_2` ocupa unos 665 MB. Si B5 aprueba un conjunto mayor, el parcial del shard se vuelca a disco por tramos de `case_id` en lugar de mantenerse en memoria.

## 5. Orden de ejecución

Coherente con la sección 7 del backlog (una sola ejecución pesada activa):

1. `credit_bureau_b_1` → claves de padres válidos de B.
2. `credit_bureau_b_2`.
3. `credit_bureau_a_1` (4 shards) → claves de padres válidos de A.
4. `credit_bureau_a_2` (11 shards), del shard más pequeño al más grande, para detectar fallos pronto y dejar `a_2_5` para el final.

Bureau B sirve como ensayo de extremo a extremo del mismo código antes de lanzar A.

## 6. Verificaciones previstas

| ID  | Verificación                                                                                                                                                                                                                                                                                                          | Cuándo                  |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| V1  | Medir si algún `case_id` aparece en más de un shard de `a_1` o `a_2` (lectura de la columna `case_id` únicamente). No cambia el diseño; documenta si la combinación entre shards es necesaria o solo defensiva.                                                                                                       | Antes de E4             |
| V2  | Pruebas sintéticas: caso partido entre dos lotes y entre dos shards; fecha anterior, igual, posterior y nula respecto de `date_decision`; hijo con padre inválido; train y OOT derivados de los mismos parciales; caso sin cobertura; reanudación tras interrupción; resultado independiente del orden de los shards. | Prototipo (antes de G1) |
| V3  | Casos distintos del agregado real iguales a los de `family_cardinality.csv` cuando no hay filtro temporal; con filtro, la diferencia queda explicada por los conteos de exclusión.                                                                                                                                    | E4                      |
| V4  | Pico de memoria y duración por shard registrados en el manifiesto.                                                                                                                                                                                                                                                    | E4                      |

## 7. Puntos abiertos para G1 y G2

| ID  | Decisión                                                                                                                                                                               | Propuesta                                                                                                                                                     | Compuerta                |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| D1  | Regla temporal de `a_2` con granularidad mensual.                                                                                                                                      | Mes estrictamente anterior al de `date_decision`.                                                                                                             | G2                       |
| D2  | Lector por lotes en `dm_credit.io`. El seam actual solo lee en memoria; la regla 2.2.4 del backlog prohíbe lectores propios y el contrato A2 exige usar la capa de lectura compartida. | Añadir una función de iteración por grupo de filas con proyección de columnas; definir quién la implementa, porque `io.py` es compartido.                     | G1                       |
| D3  | Estadísticos no combinables (mediana, percentiles, conteo distinto exacto).                                                                                                            | Excluirlos del primer corte.                                                                                                                                  | G1                       |
| D4  | Ventanas retrospectivas (A7). Cada ventana multiplica el número de estadísticos del parcial.                                                                                           | Acotar el número de ventanas para bureau; impacta el presupuesto de memoria.                                                                                  | G1                       |
| D5  | Definición de "familia" y exclusión de hijos con padre temporalmente inválido.                                                                                                         | Familia = grupo padre-hijo (`credit_bureau_a`, `credit_bureau_b`), como propone A4; excluir hijos de padres inválidos, igual que en solicitudes previas (C2). | G1 (familia), G2 (regla) |
| D6  | Emparejamiento activo/cerrado de las columnas año/mes de `a_2`.                                                                                                                        | Confirmar con las definiciones oficiales y con los datos.                                                                                                     | G2 (B5)                  |
| D7  | Ubicación de parciales y manifiestos, fuera de git.                                                                                                                                    | Un directorio por familia bajo la ruta de artefactos grandes que fije G1.                                                                                     | G1                       |
| D8  | Reutilización de parciales entre los splits `train` y `oot` (sección 3.3).                                                                                                             | Una sola pasada sobre los archivos `train_*`; el split se aplica al finalizar.                                                                                | G1                       |
