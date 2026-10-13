# 3. Preparación de datos

> **Estado (13/10/2026):** F5 consolidada con las evidencias finales de E4–E7 y F3/F4/F7. El cierre del capítulo (G4) sigue pendiente de F6 y de los riesgos indicados en 3.3 y 3.5. Criterios: [backlog](../context/03-preparacion-datos-backlog.md).

## Resumen

La ejecución registrada produjo **una matriz de modelado con una fila por caso** para train, OOT y test. Los controles verifican paridad de características y tipos (salvo `target` en test); el filtro temporal cubre las fuentes con ancla, pero no demuestra ausencia total de fuga (véase 3.3).

| Resultado                                     | Valor                                                                     | Evidencia                                                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Casos                                         | train 1 332 001 · OOT 194 658 · test 10                                   | `results/preparation/F3_final_checks.csv` (`base_cases`)                                                               |
| Características                               | 124: 94 predictores y 30 de cobertura                                     | `results/preparation/E5_master_tables.csv`, `C8_feature_catalog.csv`                                                   |
| Familias integradas                           | 12 (incluidas las tablas hijas donde procede)                             | `results/preparation/family_results.csv` (36 filas), `F3_final_checks.csv` (`family_set`)                              |
| Validación histórica E7 / validación final F3 | 26/26 / 29/29 correctos (F3 añade tres cotejos `published_matrix`)        | `results/preparation/integration_checks.csv`, `E7_manifest.json` / `F3_final_checks.csv`, `F3_final_run_manifest.json` |
| Vistas para el modelado                       | lineal (303 predictores, sin nulos) y de árboles (124, con nulos nativos) | `results/preparation/E6_views.csv`                                                                                     |

Las cifras citadas proceden de los informes archivados en `results/preparation/` (rutas relativas a la raíz del repositorio); el comando de 3.5 permite reproducirlas, pero **no se ha vuelto a ejecutar** para este texto. Los Parquet grandes de train de E5/E6 no están presentes actualmente; sus cifras y SHA-256 son evidencia de ejecuciones pasadas, no una comprobación de archivos locales actuales.

## 3.1 Seleccionar datos

### Criterio de selección

La base de cada split define su universo de casos. Las fuentes con varias filas por caso se agregan a una fila por `case_id` antes de unirlas; solo las familias con ancla aprobada aplican filtro temporal (véase F7). Una columna entra como predictor solo si la decisión **G2** la aprueba:

1. No es clave, cuasi-identificador ni atributo sensible (sexo y género quedan fuera).
2. Su disponibilidad al decidir está justificada por una regla temporal, no por su nombre.
3. No es una fecha con aspecto de calendario o proyección.
4. Tiene al menos 5 % de cobertura en su tabla fuente.
5. Pasa un filtro empírico de fuga calculado **solo con train** para las características numéricas o booleanas: Gini univariado y del indicador de ausencia no mayores a 0,5 en valor absoluto, sin cambio de signo entre tres bloques de semanas. Las ocho de texto se aprobaron por regla, sin medición.

Detalle: [decisión G2](../context/03-preparacion-G2-decision.md).

### Fuentes y columnas incluidas

En este primer corte, **solo solicitudes previas y personas aportan predictores**; el resto de familias aporta indicadores de cobertura (`n_records` y `source_present`), porque G2 no encontró evidencia temporal suficiente para sus columnas.

| Familia                               | Tablas                                   | Aporta                                                                                                 | Predictores | Regla temporal                                                                  | Cobertura train / OOT |
| ------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------: | ------------------------------------------------------------------------------- | --------------------: |
| `applprev`                            | `applprev_1`, `applprev_2`               | importes, plazos, categorías, indicadores y recencias                                                  |          61 | fila válida si `creationdate_885D <= date_decision`; hijo solo con padre válido |       78,2 % / 85,2 % |
| `person`                              | `person_1`, `person_2`                   | atributos de la persona principal (`num_group1 = 0`), edad, antigüedad laboral y relación de contactos |          33 | atributos de la solicitud actual; fechas internas `<= date_decision`            |         100 % / 100 % |
| `static_0`                            | `static_0`                               | cobertura                                                                                              |           0 | —                                                                               |         100 % / 100 % |
| `static_cb_0`                         | `static_cb_0`                            | cobertura                                                                                              |           0 | —                                                                               |       98,1 % / 99,9 % |
| `credit_bureau_a`                     | `credit_bureau_a_1`                      | cobertura                                                                                              |           0 | ancla `<= date_decision`; `credit_bureau_a_2` se omite sin ancla aprobada       |       71,0 % / 87,6 % |
| `credit_bureau_b`                     | `credit_bureau_b_1`, `credit_bureau_b_2` | cobertura                                                                                              |           0 | ancla `<= date_decision`; hijo con padre válido                                 |         0,4 % / 0,9 % |
| `debitcard_1`, `deposit_1`, `other_1` | una tabla cada una                       | cobertura                                                                                              |           0 | presencia física **sin filtro as-of**                                           |               ver 3.4 |
| `tax_registry_a_1`, `_b_1`, `_c_1`    | una tabla cada una                       | cobertura                                                                                              |           0 | presencia física **sin filtro as-of**                                           |               ver 3.4 |

Las 94 columnas predictoras salen de 35 columnas fuente aprobadas: 19 de `applprev_1` y 16 de `person_1` (`results/preparation/G2_applications_people_columns.csv`, `family_results.csv`). Las coberturas de esta tabla proceden de `results/preparation/family_results.csv`; para productos y registros representan presencia física, **no disponibilidad probada a la decisión**. La decisión sobre esos indicadores antes del modelado sigue abierta (`context/03-preparacion-F7-informe-fuga.md`).

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
| Eventos posteriores a la decisión | La fila se excluye (solicitudes previas y anclas de bureau) o el campo se anula (fechas internas); también se descartan filas sin ancla o con padre inválido. | Fuentes con fecha                     | Los conteos figuran en `results/preparation/temporal_leakage_report.csv` y `F7_temporal_leakage_by_family.csv`, no como predictores. |
| Claves inválidas                  | Una clave nula o repetida es un error en solicitudes, personas, profundidad 0 y productos.                                                                    | `case_id`, `num_group1`, `num_group2` | Garantiza una fila por caso. Bureau aún no las valida (ver 3.5).                                                                     |

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

Totales F7 de filas descartadas (incluidas anclas nulas y padres inválidos) y valores anulados, train / OOT: `applprev` 296 668 / 56 650 y 45 669 / 8 108; `person` 0 / 0 y 312 / 4; bureau A 9 111 439 / 1 081 826 y 0 / 0; bureau B 980 770 / 262 433 y 0 / 0. En test solo se descartaron 13 filas de bureau A (`F7_temporal_leakage_by_family.csv`). **Límite:** los 12 indicadores de cobertura de productos y registros no tienen filtro as-of; deben excluirse o justificarse temporalmente antes de usarse en el modelo (`context/03-preparacion-F7-informe-fuga.md`, §Riesgo abierto).

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

### Hallazgo: la cobertura de algunas fuentes cambia entre periodos

| Fuente             | Cobertura train | Cobertura OOT |
| ------------------ | --------------: | ------------: |
| `tax_registry_a_1` |          34,2 % |         1,1 % |
| `tax_registry_b_1` |           1,8 % |        64,9 % |
| `tax_registry_c_1` |          36,2 % |         0,0 % |
| `other_1`          |           1,9 % |        13,0 % |
| `credit_bureau_a`  |          71,0 % |        87,6 % |

Las coberturas fiscales muestran una deriva marcada entre train y OOT (`results/preparation/family_results.csv`); no se ha demostrado aquí su causa ni su efecto cuantitativo en Gini Stability. Además de esa deriva, falta acreditar su disponibilidad temporal: antes del modelado hay que justificar o retirar los indicadores sin filtro as-of (F7).

## 3.5 Formatear datos

La matriz ensamblada pasa por dos vistas, ajustadas solo con train y con las mismas filas y el mismo orden de casos:

| Vista   | Numéricas                                                                                         | Categóricas                                                               | Nulos   | Predictores |
| ------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------- | ----------: |
| Lineal  | Imputadas con la mediana de train, escaladas con su media y desviación, con indicador de ausencia | One-hot sobre el vocabulario congelado, más `__missing__` y `__UNKNOWN__` | Ninguno |         303 |
| Árboles | Decimales con sus nulos                                                                           | `pl.Enum` congelado, con `__UNKNOWN__`                                    | Nativos |         124 |

Las columnas de control (`case_id`, `WEEK_NUM`, `date_decision`, `MONTH` y `target`, este último solo en train y OOT) preceden a las características en ambas vistas. Los tipos y el orden de las características son iguales entre splits, con la excepción esperada de `target` en test (`results/preparation/F3_final_checks.csv`, `schema_parity`); en esa ejecución la matriz entró sin cambios a `dm_credit.baseline` (`baseline_contract`). Filas y nulos por vista: `results/preparation/E6_views.csv`.

**Pendientes conocidos:**

| Hallazgo                                                                                                                                                         | Responsable    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| En la salida limpia de solicitudes y personas, 23 de 204 columnas son constantes en train; el preprocesador no las descarta todavía.                             | Gabriel (D1)   |
| `views.py` usa una forma de `pl.concat` que polars va a cambiar en una versión mayor.                                                                            | Giancarlo (D7) |

### Resultado de la fase

El comando E7 para reproducir toda la preparación desde los Parquet originales es (ejecución histórica desde cero: 98,6 s según `results/preparation/E7_manifest.json`, `steps`):

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json --from-scratch
```

Desde cero reconstruyó las 36 piezas E4, auditó, ensambló E5, ajustó E6 con train y pasó **26 controles** (`results/preparation/integration_checks.csv`, `E7_manifest.json`). La reproducción byte a byte consta en `context/03-preparacion-E7-reproduccion.md`, §Reproducibilidad; `run_metadata.csv` y `final_schema.csv` conservan metadatos y esquema. Posteriormente F3 ejecutó E7 **sin** `--from-scratch` y archivó **29/29 controles**, incluidos los tres cotejos E5, en `F3_final_checks.csv` y `F3_final_run_manifest.json`. Este último documenta una ejecución pasada: sus rutas y hashes no prueban que las tres salidas grandes de train sigan presentes (no están). Las rutas absolutas de los informes históricos E5/E6 corresponden a la máquina de origen; usar el comando, no esas rutas, para regenerar. La integración sintética se prueba en `tests/test_preparation_g3_integration.py`. F5 documenta los resultados, pero **no cierra G4**: faltan las revisiones F6 y resolver o aceptar explícitamente el riesgo temporal de los indicadores de cobertura sin as-of.
