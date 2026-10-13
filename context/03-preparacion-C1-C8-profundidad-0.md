# C1/C8 — Profundidad 0

## Implementación

`src/dm_credit/preparation/families/depth0.py` implementa adaptadores para
`static_0` y `static_cb_0`. La base se valida y se particiona con el corte G1 de
`WEEK_NUM = 74`; los shards train se reutilizan para train/OOT y los shards test para
test. El código descubre archivos mediante `dm_credit.io`, no mediante una lista fija.

B3 registró 40 candidatas provisionales de `static_0` y 17 de `static_cb_0`, pero no
son predictores aprobados. G2 marca las 195 columnas de ambas familias como `bloqueada`
(144 y 51, respectivamente); las restantes están `excluida`. Por ello, la ejecución
actual conserva únicamente las dos características de cobertura por familia y no
publica predictores. Esta aplicación de G2 concuerda con la decisión de construcción
documentada en `03-preparacion-G2-decision.md`.

Las fuentes tienen como máximo una fila por `case_id`. Por eso, la agregación `first`
es la identidad de esa única fila, no una elección entre observaciones repetidas.
Se añade a cada familia `n_records` y `source_present`. Para `static_cb_0`, los casos
sin fila conservan cobertura cero. Como no hay predictores autorizados en G2, la salida
actual no contiene columnas predictoras. Los tipos se coercionan al tipo lógico inferido
desde train; así, columnas físicas `Null` en test adoptan el tipo de train, y valores
no convertibles o infinitos fallan explícitamente.

La ventana declarada es `all`, como establece G1. Estas fuentes son snapshots sin una
fecha individual para cada columna seleccionada; el artefacto no permite probar la
fecha de cálculo de cada valor. La disponibilidad temporal de las candidatas es, por
tanto, la regla de trabajo aceptada a partir de B3, no una verificación independiente
del instante de extracción.

La base de cada partición pasa por validación, y `case_id`/controles no se publican
como predictores de estos artefactos. El despacho compartido `build.py` no se modifica:
su propiedad corresponde a P1 según G1.

## Ejecución reproducible

Desde la raíz del repositorio, con los Parquet locales disponibles:

```powershell
.\.venv\Scripts\python.exe -m dm_credit.preparation.families.depth0 --splits train oot
```

Los Parquet de entrada y artefactos Parquet generados viven en `data/`, excluido de
Git. La ejecución escribe el catálogo de características en
`results/preparation/depth0_feature_catalog.csv` y el resumen medido en
`results/preparation/C1_depth0_run.csv`. Se puede pedir `--splits test` cuando existan
ambas fuentes para esa partición.

## Resultado ejecutado con Parquet reales

El reporte local `results/preparation/C1_depth0_run.csv` registra la ejecución para
train, OOT y test. En los tres splits se conserva una fila por caso; la cobertura parcial
de `static_cb_0` queda explícita:

| Partición | Familia       | Casos base | Filas fuente enlazadas | Casos con fuente | Casos sin fuente | Predictores | Cobertura |
| --------- | ------------- | ---------: | ---------------------: | ---------------: | ---------------: | ----------: | --------: |
| train     | `static_0`    |  1,332,001 |              1,332,001 |        1,332,001 |                0 |           0 |         2 |
| train     | `static_cb_0` |  1,332,001 |              1,306,106 |        1,306,106 |           25,895 |           0 |         2 |
| OOT       | `static_0`    |    194,658 |                194,658 |          194,658 |                0 |           0 |         2 |
| OOT       | `static_cb_0` |    194,658 |                194,370 |          194,370 |              288 |           0 |         2 |
| test      | `static_0`    |         10 |                     10 |               10 |                0 |           0 |         2 |
| test      | `static_cb_0` |         10 |                     10 |               10 |                0 |           0 |         2 |

Los manifiestos registran esquemas, conteos y huellas de entrada. G2 bloquea todas las
columnas candidatas de estas familias; por tanto, los artefactos no contienen las 40 y
17 columnas provisionales de B3 como predictores.

La ejecución registrada también incluye test: 10 casos por familia, con 2 columnas de
cobertura en cada artefacto. El reporte `C1_depth0_run.csv` registra train, OOT y test;
no respalda la afirmación anterior de que faltaba `test_static_cb_0.parquet` ni la de
que train/test contienen 42 y 19 características.

## C8 y validaciones

El catálogo C1 actual, `results/preparation/depth0_feature_catalog.csv`, contiene 4
definiciones: `n_records` y `source_present` para cada una de las dos familias. No
incluye predictores porque G2 mantiene bloqueadas las columnas candidatas. Los seis
manifiestos locales de train, OOT y test confirman dos características de cobertura y
cero predictores por familia. Las cifras previas de 61 características (40 + 17
predictores y cuatro de cobertura) describen una selección anterior a la decisión G2 y
no representan el catálogo vigente.

El catálogo C8 completo se generó en `results/preparation/C8_feature_catalog.csv` tras
las ejecuciones reales de C4–C7. Consolida 278 definiciones únicas: 248 predictores y
30 características de cobertura. El desglose por catálogos de frente es C1: 4, C2–C3:
102, C4–C5: 160 y C6–C7: 12. El archivo previo
`C8_feature_catalog_partial.csv` conserva el corte parcial de 106 filas; ya no es el
catálogo vigente. Para reproducir la consolidación:

```bash
uv run -m dm_credit.preparation.c8_catalog
```

El comando requiere los catálogos individuales en `results/preparation/`: `depth0_feature_catalog.csv`,
`applications_people_feature_catalog.csv`, `bureau_feature_catalog.csv` y
`products_feature_catalog.csv`.

C4–C5 se ejecutaron en train, OOT y test con corte 74; sus artefactos están en
`data/artifacts/preparation/C4-C5-v1/`. Se obtuvieron 82 características por split para
`credit_bureau_a` y 78 para `credit_bureau_b`; el catálogo consolidado del frente tiene
160 definiciones. C6–C7 se ejecutaron en los mismos splits y generaron 12 definiciones
de cobertura en `products_feature_catalog.csv`, sin predictores. `C6_products_run.csv`
registra cobertura y conteos por familia y partición.

Pruebas sintéticas de los casos de borde:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_preparation_depth0.py tests\test_preparation_build_family.py -q
```

El reporte local de C1 contiene resultados para train, OOT y test. La suite citada
arriba valida casos sintéticos; el catálogo y los manifiestos son evidencia de esquema
real. El cierre de C8 queda pendiente de incorporar los catálogos C4–C7.
