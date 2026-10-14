# 3. Preparación de datos

> **Estado (13/10/2026): capítulo cerrado (G4).** F5 consolidada con las evidencias finales de E4–E7 y F3/F4/F7; F6 registra la revisión técnica y sus límites en [validación F](../context/03-preparacion-F-validacion.md). Los riesgos de 3.3 y 3.5 se resolvieron en el cierre ([G4](../context/03-preparacion-G4-cierre.md)): la cobertura de productos y registros ahora se filtra por fecha y las columnas constantes se descartan. Todas las cifras salen de una reconstrucción desde cero posterior a esos cambios. Criterios: [backlog](../context/03-preparacion-datos-backlog.md).

## Resumen

La ejecución registrada produjo **una matriz de modelado con una fila por caso** para train, OOT y test. Los controles verifican paridad de características y tipos (salvo `target` en test); todas las fuentes con fecha se filtran contra `date_decision`, aunque eso no demuestra ausencia total de fuga (véase 3.3).

| Resultado                                     | Valor                                                                     | Evidencia                                                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Casos                                         | train 1 332 001 · OOT 194 658 · test 10                                   | `results/preparation/F3_final_checks.csv` (`base_cases`)                                                               |
| Características                               | 124: 94 predictores y 30 de cobertura                                     | `results/preparation/E5_master_tables.csv`, `C8_feature_catalog.csv`                                                   |
| Familias integradas                           | 12 (incluidas las tablas hijas donde procede)                             | `results/preparation/family_results.csv` (36 filas), `F3_final_checks.csv` (`family_set`)                              |
| Validación final F3 (E7 desde cero)           | 29/29 correctos, incluidos tres cotejos `published_matrix`                | `results/preparation/integration_checks.csv` = `F3_final_checks.csv`; `E7_manifest.json` = `F3_final_run_manifest.json` |
| Vistas para el modelado                       | lineal (234 predictores, sin nulos) y de árboles (114, con nulos nativos); constantes en train descartadas | `results/preparation/E6_views.csv`, `matrix_views_spec.csv`                                          |

Las cifras citadas proceden de los informes de `results/preparation/` (rutas relativas a la raíz del repositorio), generados por la reconstrucción desde cero del cierre (G4) con el comando de 3.5. Los Parquet grandes de train de E5/E6 quedan fuera de git; se regeneran con ese comando y sus SHA-256 están en `E7_manifest.json`.

## 3.1 Seleccionar datos

### Criterio de selección

La base de cada split define su universo de casos. Las fuentes con varias filas por caso se agregan a una fila por `case_id` antes de unirlas; toda familia con fecha de disponibilidad aplica filtro temporal (véase F7). Una columna entra como predictor solo si la decisión **G2** la aprueba:

1. No es clave, cuasi-identificador ni atributo sensible (sexo y género quedan fuera).
2. Su disponibilidad al decidir está justificada por una regla temporal, no por su nombre.
3. No es una fecha con aspecto de calendario o proyección.
4. Tiene al menos 5 % de cobertura en su tabla fuente.
5. Pasa un filtro empírico de fuga calculado **solo con train** para las características numéricas o booleanas: Gini univariado y del indicador de ausencia no mayores a 0,5 en valor absoluto, sin cambio de signo entre tres bloques de semanas. Las ocho de texto se aprobaron por regla, sin medición.

Detalle: [decisión G2](../context/03-preparacion-G2-decision.md).

### Fuentes y columnas incluidas

En este primer corte, **solo solicitudes previas y personas aportan predictores**; el resto de familias aporta indicadores de cobertura (`n_records` y `source_present`), porque G2 no encontró evidencia temporal suficiente para sus columnas. Esos indicadores también cumplen la regla temporal: en productos y registros solo cuentan filas con fecha de disponibilidad anterior o igual a la decisión (`products-v2`).

| Familia                               | Tablas                                   | Aporta                                                                                                 | Predictores | Regla temporal                                                                  | Cobertura train / OOT |
| ------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------: | ------------------------------------------------------------------------------- | --------------------: |
| `applprev`                            | `applprev_1`, `applprev_2`               | importes, plazos, categorías, indicadores y recencias                                                  |          61 | fila válida si `creationdate_885D <= date_decision`; hijo solo con padre válido |       78,2 % / 85,2 % |
| `person`                              | `person_1`, `person_2`                   | atributos de la persona principal (`num_group1 = 0`), edad, antigüedad laboral y relación de contactos |          33 | atributos de la solicitud actual; fechas internas `<= date_decision`            |         100 % / 100 % |
| `static_0`                            | `static_0`                               | cobertura                                                                                              |           0 | —                                                                               |         100 % / 100 % |
| `static_cb_0`                         | `static_cb_0`                            | cobertura                                                                                              |           0 | —                                                                               |       98,1 % / 99,9 % |
| `credit_bureau_a`                     | `credit_bureau_a_1`                      | cobertura                                                                                              |           0 | ancla `<= date_decision`; `credit_bureau_a_2` se omite sin ancla aprobada       |       71,0 % / 87,6 % |
| `credit_bureau_b`                     | `credit_bureau_b_1`, `credit_bureau_b_2` | cobertura                                                                                              |           0 | ancla `<= date_decision`; hijo con padre válido                                 |         0,4 % / 0,9 % |
| `debitcard_1`, `deposit_1`            | una tabla cada una                       | cobertura                                                                                              |           0 | `openingdate <= date_decision`; fecha nula descartada                           |         6,7 % / 8,1 % |
| `tax_registry_b_1`, `_c_1`            | una tabla cada una                       | cobertura                                                                                              |           0 | `deductiondate` / `processingdate <= date_decision`                             | b 1,8/64,7 · c 30,3/0 % |
| `tax_registry_a_1`                    | `tax_registry_a_1`                       | nada: constante en train, se descarta                                                                  |           0 | `recorddate_4527225D <= date_decision`; el 100 % de las filas es posterior      |                 0 % / 0 % |
| `other_1`                             | `other_1`                                | nada: constante en train, se descarta                                                                  |           0 | sin fecha: no se puede comprobar, todas las filas se descartan                  |                 0 % / 0 % |

Las 94 columnas predictoras salen de 35 columnas fuente aprobadas: 19 de `applprev_1` y 16 de `person_1` (`results/preparation/G2_applications_people_columns.csv`, `family_results.csv`). Las coberturas de esta tabla proceden de `results/preparation/family_results.csv`; desde `products-v2`, en productos y registros representan filas disponibles a la decisión (`context/03-preparacion-F7-informe-fuga.md`).

### Fuentes y columnas excluidas

| Selección                   | Columnas | Excluidas | Bloqueadas | Aplazadas | Aprobadas |
| --------------------------- | -------: | --------: | ---------: | --------: | --------: |
| B3 · profundidad 0          |      226 |        31 |        195 |         0 |         0 |
| B4 · solicitudes y personas |       95 |        34 |         21 |         5 |        35 |
| B5 · bureau                 |      149 |        23 |        126 |         0 |         0 |
| B6 · productos y registros  |       33 |        15 |         18 |         0 |         0 |

**Excluida** es definitiva en el primer corte; **bloqueada** es una exclusión preventiva que se puede revisar con evidencia de disponibilidad al decidir. Casos destacados:

| Elemento                                                                | Motivo                                                                         | Evidencia                         |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------- |
| Claves, índices y fechas de partición                                   | Control, nunca predictores                                                     | `columns.py`, `baseline.RESERVED` |
| Direcciones, distritos, códigos postales, empleadores, nombres fiscales | Cuasi-identificadores                                                          | Selecciones B3 a B6               |
| `gender_992L`, `sex_738L`                                               | Atributos sensibles                                                            | G2 §2                             |
| `riskassesment_302T`                                                    | Describe una probabilidad de incumplimiento futura: fuga del objetivo          | G2 §3                             |
| `dtlastpmtallstes_3545839D`                                             | 5,4 % de filas válidas con fecha posterior a la decisión: calendario, no hecho | G2 §2                             |
| `firstnonzeroinstldate_307D`                                            | Fecha de primera cuota: calendario contractual                                 | G2 §2                             |
| Estados, mora, pagos, deuda y motivos de rechazo                        | Pudieron actualizarse después de la decisión                                   | B3 a B6                           |
| `credit_bureau_a_2`                                                     | Su única referencia es año y mes de pago; no hay ancla aprobada                | C4-C5                             |

### Partición temporal

| Split | Semanas                 |     Casos | Positivos |   Tasa |
| ----- | ----------------------- | --------: | --------: | -----: |
| train | `WEEK_NUM < 74`         | 1 332 001 |    43 734 | 3,28 % |
| OOT   | `WEEK_NUM >= 74`        |   194 658 |     4 260 | 2,19 % |
| test  | muestra local de Kaggle |        10 |         — |      — |

Train y OOT suman 1 526 659 casos, sin solapamiento (`results/preparation/F3_final_checks.csv`, `train_oot_partition`; positivos y corte en `F3_final_run_manifest.json`, `partitions`). Test no tiene `target` y se usa aquí para comprobar compatibilidad de esquema.

## 3.2 Datos limpios

| Problema                          | Regla aplicada                                                                                                                                                | Variables afectadas                   | Justificación                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Valores faltantes                 | Las numéricas se imputan con la mediana de train y llevan un indicador `<feat>__missing`; en las categóricas, el nulo pasa a `__missing__`.                   | Todas las características             | La ausencia puede ser señal; los parámetros solo se aprenden con train.                                                              |
| Valores atípicos                  | Un valor fuera del rango de train genera un aviso, sin recorte. Ninguna columna tiene recorte aprobado.                                                       | Numéricas                             | Un rango nuevo en un periodo futuro es esperable y no es un error.                                                                   |
| Tipos y codificación              | Coerción estricta al tipo lógico (fecha `%Y-%m-%d`, entero, decimal, booleano, texto). Un valor no convertible o infinito es un error, y `NaN` es ausencia.   | Todas las columnas fuente             | Nunca se convierte un valor inválido en nulo en silencio.                                                                            |
| Categorías raras o nuevas         | Las de frecuencia menor a 1 % en train, o no vistas, pasan a `__UNKNOWN__`, con el vocabulario congelado en train.                                            | Categóricas                           | OOT y test no amplían el vocabulario.                                                                                                |
| Eventos posteriores a la decisión | La fila se excluye (solicitudes previas, anclas de bureau y fechas de productos y registros) o el campo se anula (fechas internas); también se descartan filas sin ancla o con padre inválido. | Fuentes con fecha                     | Los conteos figuran en `results/preparation/temporal_leakage_report.csv` y `F7_temporal_leakage_by_family.csv`, no como predictores. |
| Claves inválidas                  | Una clave nula o repetida es un error en solicitudes, personas, profundidad 0, productos y bureau (C4).                                                       | `case_id`, `num_group1`, `num_group2` | Garantiza una fila por caso.                                                                                                         |

Módulos: `coercion.py` (D4), `outliers.py` (D5), `preprocessing.py` (D1 a D3).

## 3.3 Datos de construcción

### Variables agregadas y derivadas

Los nombres siguen la gramática congelada en G1: `<fuente>__<variable>__<agregación>__<ventana>`, en minúsculas y con ventana `all` (todo el historial hasta la decisión).

| Tipo                                 | Ejemplo                                               | Familias             |
| ------------------------------------ | ----------------------------------------------------- | -------------------- |
| Conteo, mínimo, máximo, media y suma | `applprev_1__credamount_590a__mean__all`              | `applprev`           |
| Proporción de una categoría aprobada | `applprev_1__credtype_587l__share_cal__all`           | `applprev`, `person` |
| Número de categorías distintas       | `applprev_1__profession_152m__n_distinct__all`        | `applprev`           |
| Recencia o antigüedad en días        | `applprev_1__creationdate_885d__days_since_last__all` | `applprev`, `person` |
| Atributo de la persona principal     | `person_1__incometype_1044t__main__all`               | `person`             |
| Edad en años cumplidos a la decisión | `person_1__birth_259d__age_years__all`                | `person`             |
| Cobertura                            | `<fuente>__n_records`, `<fuente>__source_present`     | todas                |

El catálogo completo (nombre, rol, tipo lógico, columnas fuente, agregación, ventana, regla temporal y descripción) está en `results/preparation/C8_feature_catalog.csv`, con 124 filas.

### Prevención de fuga de información

1. **Filtro temporal antes de agregar.** Cada fila o valor se compara con `date_decision` de su caso; un hijo solo se usa si su padre es válido.
2. **Sin aprendizaje fuera de train.** El preprocesador y las vistas exigen `WEEK_NUM < 74` al ajustar. Transformar OOT y test no cambia el estado (control `views_fit_on_train_only`).
3. **Filtro empírico de fuga (G2).** Evaluó 102 características de train de solicitudes y personas: las 94 numéricas o booleanas tienen un Gini univariado máximo de 0,184 en valor absoluto, sin ninguna bandera; las 8 de texto se aprobaron por regla, sin medición (`results/preparation/G2_applications_people_screen.csv`; `context/03-preparacion-G2-decision.md`, §2). No garantiza ausencia de fugas moderadas.
4. **Exclusiones registradas.** Los manifiestos E4 y `results/preparation/temporal_leakage_report.csv` registran descartes y anulaciones; F7 consolida reglas y decisiones G2 por las 12 familias (`results/preparation/F7_temporal_leakage_by_family.csv`). Algunas filas carecen de ancla o de padre válido, no son exclusivamente fechas futuras:

| Regla                                   | Familia           |     Train |     OOT |
| --------------------------------------- | ----------------- | --------: | ------: |
| Solicitud creada después de la decisión | `applprev`        |   117 977 |  22 063 |
| Solicitud sin fecha de creación         | `applprev`        |        65 |       1 |
| Hijo de solicitud con padre inválido    | `applprev`        |   178 626 |  34 586 |
| Fecha de último pago posterior, anulada | `applprev`        |    35 089 |   6 595 |
| Inicio de empleo posterior, anulado     | `person`          |       312 |       4 |
| Fila de bureau A con ancla posterior    | `credit_bureau_a` | 1 583 190 | 249 695 |
| Fila de bureau A sin ancla              | `credit_bureau_a` | 7 528 249 | 832 131 |
| Hijo de bureau B con padre inválido     | `credit_bureau_b` |   913 190 | 245 050 |
| Registro fiscal A con fecha posterior   | `tax_registry_a_1` | 3 260 956 |  14 814 |
| Registro fiscal C con fecha posterior   | `tax_registry_c_1` |   726 355 |       0 |
| Registro fiscal B con fecha posterior   | `tax_registry_b_1` |    14 236 |  66 436 |
| Fila sin fecha de disponibilidad        | `other_1`         |    25 859 |  25 250 |
| Tarjeta de débito con fecha nula        | `debitcard_1`     |    10 823 |   1 888 |

Totales F7 de filas descartadas (incluidas anclas nulas y padres inválidos) y valores anulados, train / OOT: `applprev` 296 668 / 56 650 y 45 669 / 8 108; `person` 0 / 0 y 312 / 4; bureau A 9 111 439 / 1 081 826 y 0 / 0; bureau B 980 770 / 262 433 y 0 / 0. En test se descartaron 16 filas: 13 de bureau A, 2 de `other_1` y 1 de `tax_registry_b_1` (`F7_temporal_leakage_by_family.csv`). Los indicadores de productos y registros, que en la primera versión no tenían filtro as-of, ahora cumplen su regla (ver 3.4).

## 3.4 Integrar datos

### Proceso de integración

Cada familia se agrega por separado y luego se une a la base con un **left join 1:1 por `case_id`**. Nunca se unen dos fuentes 1:N entre sí.

| Familia                   | Relación con la base antes de agregar | Filas fuente (train, capítulo 02) | Después de agregar |
| ------------------------- | ------------------------------------- | --------------------------------: | ------------------ |
| `applprev`                | 1:N (profundidades 1 y 2)             |            6 525 979 + 14 075 487 | 1 fila por caso    |
| `person`                  | 1:N (profundidades 1 y 2)             |             2 973 991 + 1 643 410 | 1 fila por caso    |
| `static_0`, `static_cb_0` | 1:1 y 1:0..1                          |             1 526 659 y 1 500 476 | 1 fila por caso    |
| `credit_bureau_a`         | 1:N                                   |                  15 940 537 (a_1) | 1 fila por caso    |
| `credit_bureau_b`         | 1:N (profundidades 1 y 2)             |                85 791 + 1 286 755 | 1 fila por caso    |
| productos y registros     | 1:N y 1:0..1                          |          entre 51 109 y 3 343 800 | 1 fila por caso    |

### Validación de cardinalidad

Controles de la ejecución F3 archivada (`results/preparation/F3_final_checks.csv`), no repetidos ahora sobre los Parquet train ausentes:

- Las 12 familias de cada split usan **la misma base** (una única huella por split).
- Cada familia trae exactamente una fila por caso de la base.
- La matriz ensamblada conserva el número, el orden y la unicidad de los casos: 1 332 001 × 129 en train, 194 658 × 129 en OOT y 10 × 128 en test (sin `target`).
- Ninguna columna reservada es característica y todas están en el catálogo; F3 añade tres cotejos `published_matrix` de E5 (train, OOT y test).

### Hallazgo: la cobertura sin filtro de productos y registros tenía fuga

En la primera versión (`products-v1`), los indicadores de productos y registros contaban **todas** las filas de la fuente, sin mirar su fecha. Al medirlo, el 100 % de las filas de `tax_registry_a_1` tenía `recorddate` posterior a la decisión (mediana +14 días), y también el 21,7 % de `tax_registry_c_1`. Desde `products-v2`, la cobertura solo cuenta filas disponibles a `date_decision`.

Una ablación con el mismo modelo, con y sin estos indicadores, midió Gini Stability en OOT (`results/preparation/F7_coverage_ablation.csv` para las vistas finales y `F7_coverage_ablation_history.csv` para las corridas previas):

| Indicadores de productos y registros               | Gradient boosting (árboles) | Regresión logística (lineal) |
| -------------------------------------------------- | --------------------------: | ---------------------------: |
| Sin filtro (`products-v1`, con fuga)               |                      0,4914 |                       0,4503 |
| Solo `debitcard_1` y `deposit_1` (`products-v1`)   |                      0,4851 |                       0,4417 |
| Ninguno                                            |                      0,4861 |                       0,4423 |
| Filtrados por fecha (`products-v2`, final)         |                  **0,4913** |                   **0,4523** |

La ganancia se conserva con el filtro: la aportan `tax_registry_b_1` y `_c_1` ya filtrados, y es legítima. Lo que cambió fue el ajuste en train (Gini de 0,595 a 0,590), que antes aprovechaba la fuga.

| Fuente (cobertura filtrada) | Cobertura train | Cobertura OOT |
| --------------------------- | --------------: | ------------: |
| `tax_registry_b_1`          |           1,8 % |        64,7 % |
| `tax_registry_c_1`          |          30,3 % |         0,0 % |
| `credit_bureau_a`           |          71,0 % |        87,6 % |

Aun filtrados, los registros fiscales se reemplazan unos a otros con el tiempo: la deriva no es fuga, sino disponibilidad de la fuente. El capítulo 04 debe vigilar su estabilidad semanal.

## 3.5 Formatear datos

La matriz ensamblada pasa por dos vistas, ajustadas solo con train y con las mismas filas y el mismo orden de casos:

| Vista   | Numéricas                                                                                         | Categóricas                                                               | Nulos   | Predictores |
| ------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------- | ----------: |
| Lineal  | Imputadas con la mediana de train, escaladas con su media y desviación, con indicador de ausencia | One-hot sobre el vocabulario congelado, más `__missing__` y `__UNKNOWN__` | Ninguno |         234 |
| Árboles | Decimales con sus nulos                                                                           | `pl.Enum` congelado, con `__UNKNOWN__`                                    | Nativos |         114 |

**Columnas constantes (pedido de G2):** una característica con un solo valor en train sale de las dos vistas (10, entre ellas `tax_registry_a_1`, `other_1` y `persontype`), y en la vista lineal sale además toda salida constante en train (43 indicadores `__missing` o niveles one-hot que train nunca vio). Quedan registradas con `role = dropped` en `matrix_views_spec.csv`. Una categoría nueva en OOT o test queda en la vista lineal con todos sus niveles conocidos en cero, sin ampliar el vocabulario.

Las columnas de control (`case_id`, `WEEK_NUM`, `date_decision`, `MONTH` y `target`, este último solo en train y OOT) preceden a las características en ambas vistas. Los tipos y el orden de las características son iguales entre splits, con la excepción esperada de `target` en test (`results/preparation/F3_final_checks.csv`, `schema_parity`); en esa ejecución la matriz entró sin cambios a `dm_credit.baseline` (`baseline_contract`). Filas y nulos por vista: `results/preparation/E6_views.csv`.

**Hallazgos resueltos en el cierre (G4):**

| Hallazgo                                                       | Resolución                                                                        |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Columnas constantes en train (D1)                               | Se descartan y se registran (ver arriba).                                         |
| `pl.concat(how="horizontal")` obsoleto en `views.py` (D7)       | Reemplazado por `hstack`, que además exige la misma altura.                       |
| Cobertura de productos y registros sin filtro as-of (F7)        | `products-v2` filtra por fecha; la ablación confirma que la ganancia es legítima. |
| E5 publicaba las tablas antes de la validación final de esquemas (F6) | Los tres splits se publican como unidad: un fallo tardío borra lo escrito.  |

### Resultado de la fase

El comando E7 para reproducir toda la preparación desde los Parquet originales es (ejecución desde cero del cierre: 101,7 s según `results/preparation/E7_manifest.json`, `steps`):

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json --from-scratch
```

Desde cero reconstruyó las 36 piezas E4, las auditó, ensambló E5, ajustó E6 con train y pasó **29 de 29 controles**, incluidos los tres cotejos `published_matrix` de E5 (`results/preparation/integration_checks.csv`, archivado también como `F3_final_checks.csv`; manifiesto `E7_manifest.json`, archivado como `F3_final_run_manifest.json`). `run_metadata.csv` y `final_schema.csv` conservan metadatos y esquema. Los manifiestos e informes guardan rutas absolutas solo en campos informativos; la carga usa rutas relativas y SHA-256, así que para regenerar hay que usar el comando, no esas rutas. La integración sintética se prueba en `tests/test_preparation_g3_integration.py`. El cierre del capítulo está en [G4](../context/03-preparacion-G4-cierre.md).
