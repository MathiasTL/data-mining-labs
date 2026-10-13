# C1/C8 — Profundidad 0

## Implementación

`src/dm_credit/preparation/families/depth0.py` implementa adaptadores para
`static_0` y `static_cb_0`. La base se valida y se particiona con el corte G1 de
`WEEK_NUM = 74`; los shards train se reutilizan para train/OOT y los shards test para
test. El código descubre archivos mediante `dm_credit.io`, no mediante una lista fija.

La selección operativa toma las columnas `provisional_candidate` de
`results/preparation/B3_depth0_columns.csv`: 40 de `static_0` y 17 de `static_cb_0`.
Esta selección se adopta para avanzar C1 según la decisión de Giancarlo. Las columnas
`manual_review` y `exclude_predictor` no se usan como predictores. No se convierten las
57 selecciones en recomendaciones independientes: se usan exactamente las decisiones
que registra el manifiesto B3.

Las fuentes tienen como máximo una fila por `case_id`. Por eso, la agregación `first`
es la identidad de esa única fila, no una elección entre observaciones repetidas.
Se añade a cada familia `n_records` y `source_present`; un caso sin fila en
`static_cb_0` conserva predictores nulos y cobertura cero. Los tipos se coercionan al
tipo lógico inferido desde train; así, columnas físicas `Null` en test adoptan el tipo
de train, y valores no convertibles o infinitos fallan explícitamente.

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

El 13/10/2026 se ejecutó el comando anterior con los archivos locales de train. Train
y OOT cubren todos los casos de sus bases. `static_0` tiene una fila por caso; la
cobertura parcial de `static_cb_0` queda explícita:

| Partición | Familia | Casos base | Filas fuente enlazadas | Casos con fuente | Casos sin fuente | Predictores |
|---|---|---:|---:|---:|---:|---:|
| train | `static_0` | 1,332,001 | 1,332,001 | 1,332,001 | 0 | 40 |
| train | `static_cb_0` | 1,332,001 | 1,306,106 | 1,306,106 | 25,895 | 17 |
| OOT | `static_0` | 194,658 | 194,658 | 194,658 | 0 | 40 |
| OOT | `static_cb_0` | 194,658 | 194,370 | 194,370 | 288 | 17 |

Cada artefacto contiene además dos columnas de cobertura, por lo que hay 42
características en `static_0` y 19 en `static_cb_0`. Las particiones conservan una
fila por caso. Los manifiestos verifican y registran los esquemas, conteos, selección
B3 y huellas de entrada.

La prueba real de `static_0` sobre la base local de test produjo 10 filas y el mismo
esquema de 42 características que train. **No se pudo completar test para C1:** falta
`test_static_cb_0.parquet` en `data/parquet_files/test/`. El runner debe fallar
explícitamente mientras falte ese shard; no se debe sustituir por datos sintéticos.

## C8 y validaciones

El catálogo C8 de profundidad 0 contiene 61 definiciones únicas: 40 predictores y dos
características de cobertura por `static_0`, y 17 predictores y dos de cobertura por
`static_cb_0`. Su contenido procede de los esquemas de los artefactos reales train/OOT;
no contiene valores de casos. El catálogo global de la épica C seguirá creciendo con
las familias de C2–C7.

Pruebas sintéticas de los casos de borde:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_preparation_depth0.py tests\test_preparation_build_family.py -q
```

Resultado en esta revisión: 29 pruebas pasaron. También se procesaron los Parquet
reales train/OOT como se detalla arriba. La ejecución no incorpora las otras familias
de la épica C, que corresponden a otros frentes.
