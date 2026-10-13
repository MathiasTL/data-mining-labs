# 3. Preparación de datos

> **Estado (13/10/2026):** pipeline implementado y validado. La ejecución real (E4), las tablas maestras (E5) y el ajuste del preprocesador y las vistas (E6) están cerrados; queda el comando único de reproducción (E7). El reparto del trabajo está en el [backlog del capítulo 03](../context/03-preparacion-datos-backlog.md).

## Resumen

La fase produce una **matriz de modelado con una fila por caso** para train, OOT y test, sin fuga temporal y con el mismo esquema en los tres splits.

| Resultado | Valor | Evidencia |
| --- | --- | --- |
| Casos | train 1 332 001 · OOT 194 658 · test 10 | `results/preparation/integration_checks.csv` |
| Características | 124: 94 predictores y 30 de cobertura | `results/preparation/C8_feature_catalog.csv` |
| Familias integradas | 12 (con las tablas padre e hija de cada una) | `results/preparation/family_results.csv` |
| Controles de validación sobre datos reales | 26 de 26 correctos | `results/preparation/integration_checks.csv` |
| Vistas para el modelado | lineal (303 predictores, sin nulos) y de árboles (124, con nulos nativos) | `src/dm_credit/preparation/views.py` |

Todas las cifras de este capítulo se regeneran con los comandos de la sección 3.5.

## 3.1 Seleccionar datos

### Criterio de selección

La tabla base (`train_base`) define el universo de casos. Cada fuente con varias filas por caso se filtra temporalmente y se agrega a una fila por `case_id` antes de unirla. Una columna entra como predictor solo si la decisión **G2** la aprueba:

1. No es clave, cuasi-identificador ni atributo sensible (sexo y género quedan fuera).
2. Su disponibilidad al decidir está justificada por una regla temporal, no por su nombre.
3. No es una fecha con aspecto de calendario o proyección.
4. Tiene al menos 5 % de cobertura en su tabla fuente.
5. Pasa un filtro empírico de fuga calculado **solo con train**: Gini univariado y Gini del indicador de ausencia no mayores a 0,5 en valor absoluto, sin cambio de signo entre tres bloques de semanas.

Detalle: [decisión G2](../context/03-preparacion-G2-decision.md).

### Fuentes y columnas incluidas

En este primer corte, **solo solicitudes previas y personas aportan predictores**; el resto de familias aporta indicadores de cobertura (`n_records` y `source_present`), porque G2 no encontró evidencia temporal suficiente para sus columnas.

| Familia | Tablas | Aporta | Predictores | Regla temporal | Cobertura train / OOT |
| --- | --- | --- | ---: | --- | ---: |
| `applprev` | `applprev_1`, `applprev_2` | importes, plazos, categorías, indicadores y recencias | 61 | fila válida si `creationdate_885D <= date_decision`; hijo solo con padre válido | 78,2 % / 85,2 % |
| `person` | `person_1`, `person_2` | atributos de la persona principal (`num_group1 = 0`), edad, antigüedad laboral y relación de contactos | 33 | atributos de la solicitud actual; fechas internas `<= date_decision` | 100 % / 100 % |
| `static_0` | `static_0` | cobertura | 0 | — | 100 % / 100 % |
| `static_cb_0` | `static_cb_0` | cobertura | 0 | — | 98,1 % / 99,9 % |
| `credit_bureau_a` | `credit_bureau_a_1` | cobertura | 0 | ancla `<= date_decision`; `credit_bureau_a_2` se omite sin ancla aprobada | 71,0 % / 87,6 % |
| `credit_bureau_b` | `credit_bureau_b_1`, `credit_bureau_b_2` | cobertura | 0 | ancla `<= date_decision`; hijo con padre válido | 0,4 % / 0,9 % |
| `debitcard_1`, `deposit_1`, `other_1` | una tabla cada una | cobertura | 0 | presencia física | ver 3.4 |
| `tax_registry_a_1`, `_b_1`, `_c_1` | una tabla cada una | cobertura | 0 | presencia física | ver 3.4 |

Las 94 columnas predictoras salen de 35 columnas fuente aprobadas: 19 de `applprev_1` y 16 de `person_1`. Fuente: `results/preparation/G2_applications_people_columns.csv`.

### Fuentes y columnas excluidas

| Selección | Columnas | Excluidas | Bloqueadas | Aplazadas | Aprobadas |
| --- | ---: | ---: | ---: | ---: | ---: |
| B3 · profundidad 0 | 226 | 31 | 195 | 0 | 0 |
| B4 · solicitudes y personas | 95 | 34 | 21 | 5 | 35 |
| B5 · bureau | 149 | 23 | 126 | 0 | 0 |
| B6 · productos y registros | 33 | 15 | 18 | 0 | 0 |

**Excluida** es definitiva en el primer corte; **bloqueada** es una exclusión preventiva que se puede revisar con evidencia de disponibilidad al decidir. Casos destacados:

| Elemento | Motivo | Evidencia |
| --- | --- | --- |
| Claves, índices y fechas de partición | Control, nunca predictores | `columns.py`, `baseline.RESERVED` |
| Direcciones, distritos, códigos postales, empleadores, nombres fiscales | Cuasi-identificadores | Selecciones B3 a B6 |
| `gender_992L`, `sex_738L` | Atributos sensibles | G2 §2 |
| `riskassesment_302T` | Describe una probabilidad de incumplimiento futura: fuga del objetivo | G2 §3 |
| `dtlastpmtallstes_3545839D` | 5,4 % de filas válidas con fecha posterior a la decisión: calendario, no hecho | G2 §2 |
| `firstnonzeroinstldate_307D` | Fecha de primera cuota: calendario contractual | G2 §2 |
| Estados, mora, pagos, deuda y motivos de rechazo | Pudieron actualizarse después de la decisión | B3 a B6 |
| `credit_bureau_a_2` | Su única referencia es año y mes de pago; no hay ancla aprobada | C4-C5 |

### Partición temporal

| Split | Semanas | Casos | Positivos | Tasa |
| --- | --- | ---: | ---: | ---: |
| train | `WEEK_NUM < 74` | 1 332 001 | 43 734 | 3,28 % |
| OOT | `WEEK_NUM >= 74` | 194 658 | 4 260 | 2,19 % |
| test | muestra local de Kaggle | 10 | — | — |

Train y OOT suman 1 526 659 casos, sin solapamiento (control `train_oot_partition`). Test no tiene `target` y solo se usa para comprobar compatibilidad de esquema.

## 3.2 Datos limpios

| Problema | Regla aplicada | Variables afectadas | Justificación |
| --- | --- | --- | --- |
| Valores faltantes | Las numéricas se imputan con la mediana de train y llevan un indicador `<feat>__missing`; en las categóricas, el nulo pasa a `__missing__`. | Todas las características | La ausencia puede ser señal; los parámetros solo se aprenden con train. |
| Valores atípicos | Un valor fuera del rango de train genera un aviso, sin recorte. Ninguna columna tiene recorte aprobado. | Numéricas | Un rango nuevo en un periodo futuro es esperable y no es un error. |
| Tipos y codificación | Coerción estricta al tipo lógico (fecha `%Y-%m-%d`, entero, decimal, booleano, texto). Un valor no convertible o infinito es un error, y `NaN` es ausencia. | Todas las columnas fuente | Nunca se convierte un valor inválido en nulo en silencio. |
| Categorías raras o nuevas | Las de frecuencia menor a 1 % en train, o no vistas, pasan a `__UNKNOWN__`, con el vocabulario congelado en train. | Categóricas | OOT y test no amplían el vocabulario. |
| Eventos posteriores a la decisión | La fila se excluye (solicitudes previas y anclas de bureau) o el campo se anula (fechas internas). El total va solo al manifiesto. | Fuentes con fecha | Contar fechas futuras por caso sería fuga. |
| Claves inválidas | Una clave nula o repetida es un error en solicitudes, personas, profundidad 0 y productos. | `case_id`, `num_group1`, `num_group2` | Garantiza una fila por caso. Bureau aún no las valida (ver 3.5). |

Módulos: `coercion.py` (D4), `outliers.py` (D5), `preprocessing.py` (D1 a D3).

## 3.3 Datos de construcción

### Variables agregadas y derivadas

Los nombres siguen la gramática congelada en G1: `<fuente>__<variable>__<agregación>__<ventana>`, en minúsculas y con ventana `all` (todo el historial hasta la decisión).

| Tipo | Ejemplo | Familias |
| --- | --- | --- |
| Conteo, mínimo, máximo, media y suma | `applprev_1__credamount_590a__mean__all` | `applprev` |
| Proporción de una categoría aprobada | `applprev_1__credtype_587l__share_cal__all` | `applprev`, `person` |
| Número de categorías distintas | `applprev_1__profession_152m__n_distinct__all` | `applprev` |
| Recencia o antigüedad en días | `applprev_1__creationdate_885d__days_since_last__all` | `applprev`, `person` |
| Atributo de la persona principal | `person_1__incometype_1044t__main__all` | `person` |
| Edad en años cumplidos a la decisión | `person_1__birth_259d__age_years__all` | `person` |
| Cobertura | `<fuente>__n_records`, `<fuente>__source_present` | todas |

El catálogo completo (nombre, rol, tipo lógico, columnas fuente, agregación, ventana, regla temporal y descripción) está en `results/preparation/C8_feature_catalog.csv`, con 124 filas.

### Prevención de fuga de información

1. **Filtro temporal antes de agregar.** Cada fila o valor se compara con `date_decision` de su caso; un hijo solo se usa si su padre es válido.
2. **Sin aprendizaje fuera de train.** El preprocesador y las vistas exigen `WEEK_NUM < 74` al ajustar. Transformar OOT y test no cambia el estado (control `views_fit_on_train_only`).
3. **Filtro empírico de fuga (G2).** Evaluó 102 características de train de solicitudes y personas: las 94 numéricas o booleanas tienen un Gini univariado máximo de 0,184 en valor absoluto, sin ninguna bandera; las 8 de texto se aprobaron por regla, sin medición.
4. **Exclusiones registradas.** Cada familia guarda en su manifiesto lo que descartó por temporalidad (`results/preparation/temporal_leakage_report.csv`):

| Regla | Familia | Train | OOT |
| --- | --- | ---: | ---: |
| Solicitud creada después de la decisión | `applprev` | 117 977 | 22 063 |
| Hijo de solicitud con padre inválido | `applprev` | 178 626 | 34 586 |
| Fecha de último pago posterior, anulada | `applprev` | 35 089 | 6 595 |
| Inicio de empleo posterior, anulado | `person` | 312 | 4 |
| Fila de bureau A con ancla posterior | `credit_bureau_a` | 1 583 190 | 249 695 |
| Fila de bureau A sin ancla | `credit_bureau_a` | 7 528 249 | 832 131 |
| Hijo de bureau B con padre inválido | `credit_bureau_b` | 913 190 | 245 050 |

## 3.4 Integrar datos

### Proceso de integración

Cada familia se agrega por separado y luego se une a la base con un **left join 1:1 por `case_id`**. Nunca se unen dos fuentes 1:N entre sí.

| Familia | Relación con la base antes de agregar | Filas fuente (train, capítulo 02) | Después de agregar |
| --- | --- | ---: | --- |
| `applprev` | 1:N (profundidades 1 y 2) | 6 525 979 + 14 075 487 | 1 fila por caso |
| `person` | 1:N (profundidades 1 y 2) | 2 973 991 + 1 643 410 | 1 fila por caso |
| `static_0`, `static_cb_0` | 1:1 y 1:0..1 | 1 526 659 y 1 500 476 | 1 fila por caso |
| `credit_bureau_a` | 1:N | 15 940 537 (a_1) | 1 fila por caso |
| `credit_bureau_b` | 1:N (profundidades 1 y 2) | 85 791 + 1 286 755 | 1 fila por caso |
| productos y registros | 1:N y 1:0..1 | entre 51 109 y 3 343 800 | 1 fila por caso |

### Validación de cardinalidad

Controles sobre los artefactos reales (`uv run -m dm_credit.preparation.validation`):

- Las 12 familias de cada split usan **la misma base** (una única huella por split).
- Cada familia trae exactamente una fila por caso de la base.
- La matriz ensamblada conserva el número, el orden y la unicidad de los casos: 1 332 001 × 129 en train, 194 658 × 129 en OOT y 10 × 128 en test (sin `target`).
- Ninguna columna reservada es característica y todas están en el catálogo.

### Hallazgo: la cobertura de algunas fuentes cambia entre periodos

| Fuente | Cobertura train | Cobertura OOT |
| --- | ---: | ---: |
| `tax_registry_a_1` | 34,2 % | 1,1 % |
| `tax_registry_b_1` | 1,8 % | 64,9 % |
| `tax_registry_c_1` | 36,2 % | 0,0 % |
| `other_1` | 1,9 % | 13,0 % |
| `credit_bureau_a` | 71,0 % | 87,6 % |

Los registros fiscales se reemplazan unos a otros con el tiempo. Sus indicadores de cobertura serán **inestables** entre train y OOT, que es justo lo que penaliza Gini Stability. El capítulo 04 debe evaluarlos con cuidado o excluirlos.

## 3.5 Formatear datos

La matriz ensamblada pasa por dos vistas, ajustadas solo con train y con las mismas filas y el mismo orden de casos:

| Vista | Numéricas | Categóricas | Nulos | Predictores |
| --- | --- | --- | --- | ---: |
| Lineal | Imputadas con la mediana de train, escaladas con su media y desviación, con indicador de ausencia | One-hot sobre el vocabulario congelado, más `__missing__` y `__UNKNOWN__` | Ninguno | 303 |
| Árboles | Decimales con sus nulos | `pl.Enum` congelado, con `__UNKNOWN__` | Nativos | 124 |

Las columnas de control (`case_id`, `WEEK_NUM`, `date_decision`, `MONTH` y `target`, este último solo en train y OOT) preceden a los predictores en las dos vistas. El esquema es idéntico en train, OOT y test (control `schema_parity`), y la matriz entra sin cambios a `dm_credit.baseline` (control `baseline_contract`).

**Pendientes conocidos:**

| Hallazgo | Responsable |
| --- | --- |
| En la salida limpia de solicitudes y personas, 23 de 204 columnas son constantes en train; el preprocesador no las descarta todavía. | Gabriel (D1) |
| `views.py` usa una forma de `pl.concat` que polars va a cambiar en una versión mayor. | Giancarlo (D7) |
| Bureau no rechaza claves `(case_id, num_group1)` repetidas: las cuenta dos veces. No afecta a los datos reales, que tienen claves únicas, pero falta la defensa. | Mathias (C4) |

### Resultado de la fase

Hasta que exista el comando único (E7), la matriz se regenera por familia y se valida así, con `POLARS_MAX_THREADS=1`:

```bash
uv run -m dm_credit.preparation.families.depth0 --splits train oot test
uv run -m dm_credit.preparation.families.applications_people_run
uv run -m dm_credit.preparation.families.bureau_run
uv run -m dm_credit.preparation.families.products_registry --splits train oot test
uv run -m dm_credit.preparation.c8_catalog
uv run -m dm_credit.preparation.validation
uv run -m dm_credit.preparation.e5   # tablas maestras (E5)
uv run -m dm_credit.preparation.e6   # preprocesador y vistas (E6)
```

Los artefactos quedan en `data/artifacts/preparation/` (`C1-v1`, `C2-C3-v1`, `C4-C5-v2` y `C6-v1`), cada uno con su manifiesto y su suma de verificación. La integración completa sobre datos sintéticos se prueba en `tests/test_preparation_g3_integration.py`.
