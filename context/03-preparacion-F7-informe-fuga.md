# F7 — Informe de fuga temporal por familia

**Resultado:** las 12 familias tienen regla de disponibilidad declarada y evidencia trazable, y **todas las que usan filas con fecha las filtran contra `date_decision`**. Al consolidar F7 apareció una fuga: la cobertura de productos y registros contaba filas posteriores a la decisión (el 100 % en `tax_registry_a_1`). Se corrigió con `products-v2`, y una ablación confirmó que la ganancia que queda es legítima.

## Ruta rápida

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.f7            # informe por familia
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.f7_ablation   # ablación de los indicadores (unos 5 min)
```

1. `f7` lee los manifiestos de `data/artifacts/preparation/E4-v1/` (36 piezas) y las cuatro hojas G2, y escribe `results/preparation/F7_temporal_leakage_by_family.csv`.
2. `f7_ablation` entrena el mismo modelo con y sin los indicadores de productos y registros sobre las vistas de E6, y escribe `results/preparation/F7_coverage_ablation.csv`.
3. Pruebas: `tests/test_preparation_f7.py`, `tests/test_preparation_f7_ablation.py` y `tests/test_preparation_products_as_of.py`.

## Qué contiene cada columna del informe

| Columna | Origen |
|---|---|
| `temporal_rule` | `features[].temporal_rule` del manifiesto de train |
| `as_of_row_filter` | `true` si la familia descartó filas por fecha en algún split |
| `rows_excluded_<split>` | suma de métricas de filas (`*_rows_after_decision`, `*_rows_null_*`, `*_rows_no_anchor`, `*_rows_invalid_parent`); no cuenta `source_rows_outside_base`, que no es temporal |
| `values_nulled_<split>` | suma de métricas `*_values_after_decision` (fechas internas anuladas) |
| `columns_approved/blocked/excluded/deferred` | columnas G2 por decisión; las tablas hijas cuentan en su familia y `base` en `static_0` |
| `evidence` | manifiestos E4 y hojas G2 de donde sale cada número |

## Resultados reales (E4-v1, `products-v2`)

| Familia | Regla | Predictores | Filas excluidas train / OOT | Valores anulados train / OOT | G2 aprob. / bloq. / excl. / aplaz. |
|---|---|---:|---:|---:|---|
| applprev | `creationdate_885D <= date_decision`; hijos con padre válido | 61 | 296 668 / 56 650 | 45 669 / 8 108 | 19 / 17 / 11 / 0 |
| person | solicitud actual; fechas internas `<=` decisión | 33 | 0 / 0 | 312 / 4 | 16 / 4 / 23 / 5 |
| credit_bureau_a | `lastupdate <= date_decision`; ancla nula descartada | 0 | 9 111 439 / 1 081 826 | 0 / 0 | 0 / 84 / 14 / 0 |
| credit_bureau_b | ídem, hijo con padre válido | 0 | 980 770 / 262 433 | 0 / 0 | 0 / 42 / 9 / 0 |
| tax_registry_a_1 | `recorddate_4527225D <= date_decision` | 0 | 3 260 956 / 14 814 | — | 0 aprobadas |
| tax_registry_b_1 | `deductiondate_4917603D <= date_decision` | 0 | 14 236 / 66 436 | — | 0 aprobadas |
| tax_registry_c_1 | `processingdate_168D <= date_decision` | 0 | 726 355 / 0 | — | 0 aprobadas |
| debitcard_1 | `openingdate_857D <= date_decision`; fecha nula descartada | 0 | 10 823 / 1 888 | — | 0 aprobadas |
| deposit_1 | `openingdate_313D <= date_decision` | 0 | 0 / 0 | — | 0 aprobadas |
| other_1 | sin fecha: todas las filas se descartan | 0 | 25 859 / 25 250 | — | 0 aprobadas |
| static_0, static_cb_0 | fila única estática por caso | 0 | — | — | 0 aprobadas |

Totales: 94 predictores y 30 indicadores de cobertura, igual que la tabla maestra de E5. En test solo se excluyen 16 filas.

## La fuga encontrada y cómo se resolvió

**Qué pasaba:** en `products-v1`, `n_records` y `source_present` de productos y registros contaban todas las filas de la fuente, sin mirar su fecha. Medido contra `date_decision`:

| Fuente | Filas posteriores (train) | Mediana | Casos de train que pierden cobertura |
|---|---:|---:|---:|
| `tax_registry_a_1` | 100 % | +14 días | 100 % |
| `tax_registry_c_1` | 21,7 % | +14 días | 16,4 % |
| `tax_registry_b_1` | 7,8 % | +7 días | 0,8 % |
| `debitcard_1`, `deposit_1` | 0 % | — | — (`debitcard_1` tiene 8 % de fechas nulas) |
| `other_1` | sin fecha | — | no se puede comprobar |

**Ablación** (Gini Stability en OOT, mismo modelo y misma vista; historial en `results/preparation/F7_coverage_ablation_history.csv`):

| Variante | Gradient boosting (árboles) | Regresión logística (lineal) |
|---|---:|---:|
| `products-v1`, sin filtro (con fuga) | 0,4914 | 0,4503 |
| `products-v1`, solo `debitcard_1` y `deposit_1` | 0,4851 | 0,4417 |
| Sin ningún indicador de productos y registros | 0,4861 | 0,4423 |
| **`products-v2`, filtrados por fecha (final)** | **0,4913** | **0,4523** |

Las dos últimas filas salen de `F7_coverage_ablation.csv`, sobre las vistas finales (con las columnas constantes ya descartadas).

**Decisión:** filtrar por fecha (`products-v2`) en lugar de excluir los indicadores. La ganancia sin filtro venía de la fuga, pero con el filtro se conserva casi entera gracias a `tax_registry_b_1` y `_c_1`, que ya cumplen la regla. `tax_registry_a_1` y `other_1` quedan en cero en todos los casos y el preprocesador los descarta por constantes.

## Riesgo que queda para el capítulo 04

Aun filtrados, `tax_registry_b_1` (1,8 % → 64,7 %) y `tax_registry_c_1` (30,3 % → 0 %) cambian de cobertura entre train y OOT. Las fuentes se reemplazan en el tiempo: es deriva de disponibilidad, no fuga. Hay que vigilar su efecto en el Gini semanal.

## Checklist

- [x] Regla de disponibilidad por familia
- [x] Filas excluidas y valores anulados por split
- [x] Columnas por decisión G2 y motivo (detalle por columna en las hojas G2)
- [x] Evidencia trazable a manifiestos y hojas G2
- [x] Cobertura de productos y registros filtrada por fecha, con ablación como evidencia
