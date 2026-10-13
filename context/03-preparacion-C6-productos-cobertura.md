# C6 — Adaptadores de productos con cobertura y cero predictores

> **Responsable:** Gabriel (P5). **Estado:** implementado con sintéticos; ejecución real pendiente de la cola pesada (E4).
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

La ejecución real E4 va en la cola pesada compartida tras G3; el catálogo C8 sale de `contracts.write_feature_catalog` sobre los artefactos (probado en sintético: 6 filas, rol `cobertura`).
