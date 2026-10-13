# C6 — Adaptadores de productos con cobertura y cero predictores

> **Responsable:** Gabriel (P5). **Estado:** implementado y ejecutado con Parquet reales en train, OOT y test (13/10/2026).
> **Alcance:** `debitcard_1`, `deposit_1`, `other_1` (tarea C6 del [backlog](03-preparacion-datos-backlog.md)).
> **Insumos:** B6 (`results/preparation/B6_products_registry_columns.csv`), G2 (`results/preparation/G2_products_registry_columns.csv`, 0 aprobadas), contratos G1 (`contracts.py`), lector `dm_credit.io`.
> **Código:** `src/dm_credit/preparation/families/products_registry.py`. **Pruebas:** `tests/test_preparation_c6.py` (13 casos, resultados a mano).

## 1. Qué entrega

Tres constructores (`build_debitcard_1_features`, `build_deposit_1_features`, `build_other_1_features`) con la firma G1 `(base, split, *, cutoff_week=74, output_dir, selection_path, g2_path)`. Cada uno publica un agregado con una fila por `case_id` de la base y su manifiesto JSON (vía `contracts.write_family_features`, publicación atómica):

- `<fuente>__n_records` (int): filas físicas de la fuente para el caso.
- `<fuente>__source_present` (bool): `n_records > 0`.

Cero predictores: G2 no aprueba ninguna columna de estas fuentes en el primer corte. Si un G2 futuro aprueba alguna, el constructor falla explícitamente (`follow-up`) en vez de inventar una agregación sobre datos 1:N.

## 2. Semántica de cobertura (decisión deliberada)

`n_records` cuenta presencia física **sin filtro temporal**. Ninguna regla `as-of` está aprobada para estas fuentes, así que filtrar sería presumir disponibilidad (invariante A1.5). El `temporal_rule` del manifiesto lo declara. Cardinalidades reales (`family_cardinality.csv`): `debitcard_1` y `deposit_1` son 1:N (máx. 66/65 filas por caso; se valida unicidad de `(case_id, num_group1)`); `other_1` es 1:0..1 (51 109 filas, ~3,3 %; se valida unicidad de `case_id`). Filas de casos fuera de la base se ignoran (semi-join). Fuente vacía → ceros, sin error (caso `tax_registry_c_1` de C7, misma vía).

## 3. other_1

Bloqueado en B6/G2 por falta de linaje temporal (`insufficient_temporal_evidence`). Este módulo solo aporta su presencia física para auditarla; si G2 nunca confirma el linaje, C6 exige excluirlo con evidencia y no usarlo.

## 4. Registro en el despacho

`BUILDERS` propio expuesto en el módulo (`debitcard_1`, `deposit_1`, `other_1`), listo para que P1 lo fusione en `build.build_family_features` (su propiedad, no se toca aquí). Ningún test existente fija las claves del despacho salvo `applprev`/`person`/`nope`, que siguen verdes.

## 5. Reproducir y ejecutar

```bash
uv run --extra dev pytest -q tests/test_preparation_c6.py
uv run -m dm_credit.preparation.families.products_registry --splits train oot test  # ejecución real E4 en cola pesada ([R], POLARS_MAX_THREADS=1)
```

Ejecución real completada con `POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.products_registry --splits train oot test`. El reporte `results/preparation/C6_products_run.csv` registra los seis orígenes en los tres splits, una fila por caso y dos características de cobertura por familia; no se generaron predictores. El catálogo de C6/C7 se escribió en `results/preparation/products_feature_catalog.csv` (12 filas, rol `cobertura`). Los artefactos y manifiestos están bajo `data/artifacts/preparation/C6-v1/` (ruta por defecto del módulo). La ejecución también verificó `tax_registry_c_1` vacío en test y produjo cobertura cero sin fallo.

## 6. Aporte F7

`results/preparation/F7_products_registry_exclusions.csv` (`split,family,metric,value`, mismo esquema que el de applprev/person): 108 filas desde los 18 manifiestos reales vía `collect_f7_exclusions`/`--f7-report`. El detalle por columna excluida vive en B6/G2; aquí van los conteos auditables (`g2_approved` = 0 en todo, `source_rows_outside_base` = 0).
