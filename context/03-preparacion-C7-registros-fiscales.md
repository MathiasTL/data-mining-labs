# C7 — Adaptadores de registros fiscales con cobertura y cero predictores

> **Responsable:** Gabriel (P5). **Estado:** implementado y ejecutado con Parquet reales en train, OOT y test (13/10/2026).
> **Alcance:** `tax_registry_a_1`, `tax_registry_b_1`, `tax_registry_c_1` (tarea C7 del [backlog](03-preparacion-datos-backlog.md)).
> **Insumos:** B6, G2 (`G2_products_registry_columns.csv`, 0 aprobadas), contratos G1, lector `dm_credit.io`.
> **Código:** `src/dm_credit/preparation/families/products_registry.py` (módulo compartido C6/C7). **Pruebas:** `tests/test_preparation_c7.py`.

## 1. Qué entrega

Tres constructores (`build_tax_registry_{a,b,c}_1_features`) con la firma G1. Reutilizan `_build_source_features` y `_selected_columns` de C6: cobertura física (`n_records` int64, `source_present` bool), una fila por `case_id`, manifiesto con B6/G2 como procedencia y cero predictores (un G2 futuro que apruebe columnas falla explícito hasta definir su agregación).

## 2. Cardinalidad y lectura

Las tres son 1:N (media ~7 filas/caso, máx. 121; 1–3,3 M de filas en train con 5 columnas): lectura directa con proyección por el seam, sin streaming pesado. Se valida unicidad de `(case_id, num_group1)`. El `[R]` del backlog aplica a la ejecución real en cola, no al código.

## 3. Prueba reina: test vacío con columnas Null

`test_tax_registry_c_1.parquet` está vacío con todo en tipo físico `Null`. Dos garantías implementadas y probadas con un fixture idéntico (0 filas, 5 columnas `Null`):

1. Las claves `Null` se convierten a `Int64` **antes** del semi-join perezoso (el join `Null` vs `Int64` falla en el motor; la conversión es exacta porque no hay valores). Otro tipo no entero en la clave es error explícito.
2. El agregado sale con el esquema correcto (`n_records` int64 en ceros, `source_present` en falso), una fila por caso de la base, manifiesto recargable.

## 4. Alta cardinalidad

Los nombres fiscales (`name_4527232M`, `name_4917606M`, `employername_160M`) están excluidos en B6/G2 como cuasi-identificadores: nunca entran como categorías libres (criterio de aceptación C7).

## 5. Reproducir y ejecutar

```bash
uv run --extra dev pytest -q tests/test_preparation_c7.py tests/test_preparation_c6.py
uv run -m dm_credit.preparation.families.products_registry --splits train oot test  # ejecución real E4 en cola pesada ([R])
```

Ejecución real completada junto con C6 mediante `POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.products_registry --splits train oot test`. Los tres adaptadores fiscales aparecen en `results/preparation/C6_products_run.csv`; cada uno genera dos columnas de cobertura y cero predictores en cada split. `tax_registry_c_1` está vacío en test y produjo el agregado con el esquema correcto. Las seis familias C6/C7 aportan 12 filas al catálogo `results/preparation/products_feature_catalog.csv`; la consolidación global está en `results/preparation/C8_feature_catalog.csv`.
