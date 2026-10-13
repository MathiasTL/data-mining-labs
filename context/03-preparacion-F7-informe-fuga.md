# F7 — Informe de fuga temporal por familia

**Resultado:** las 12 familias tienen regla de disponibilidad declarada y evidencia trazable. Solo 4 familias (applprev, person, bureau A y bureau B) filtran filas o anulan valores por fecha; las otras 8 aportan únicamente indicadores de cobertura física o nada al modelo. Ningún predictor entra sin regla temporal aprobada en G2.

## Ruta rápida

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.f7
```

1. Lee los manifiestos de `data/artifacts/preparation/E4-v1/` (36 piezas) y las cuatro hojas G2 de `results/preparation/`.
2. Escribe `results/preparation/F7_temporal_leakage_by_family.csv` (una fila por familia).
3. Pruebas: `tests/test_preparation_f7.py` (5 pruebas sintéticas).

## Qué contiene cada columna

| Columna | Origen |
|---|---|
| `temporal_rule` | `features[].temporal_rule` del manifiesto de train |
| `as_of_row_filter` | `true` si la familia descartó filas por fecha en algún split |
| `rows_excluded_<split>` | suma de métricas de filas (`*_rows_after_decision`, `*_rows_null_*`, `*_rows_invalid_parent`); no cuenta `source_rows_outside_base`, que no es temporal |
| `values_nulled_<split>` | suma de métricas `*_values_after_decision` (fechas internas anuladas) |
| `columns_approved/blocked/excluded/deferred` | conteo de columnas G2 por decisión; las tablas hijas (`applprev_2`, `person_2`, `credit_bureau_a_2`, `b_2`) cuentan en su familia y `base` en `static_0` |
| `evidence` | manifiestos E4 y hojas G2 de donde sale cada número |

## Resultados reales (E4-v1)

| Familia | Filtro as-of | Predictores | Filas excluidas train / OOT | Valores anulados train / OOT | G2 aprob. / bloq. / excl. / aplaz. |
|---|---|---|---|---|---|
| applprev | sí | 61 | 296 668 / 56 650 | 45 669 / 8 108 | 19 / 17 / 11 / 0 |
| person | sí | 33 | 0 / 0 | 312 / 4 | 16 / 4 / 23 / 5 |
| credit_bureau_a | sí | 0 | 9 111 439 / 1 081 826 | 0 / 0 | 0 / 84 / 14 / 0 |
| credit_bureau_b | sí | 0 | 980 770 / 262 433 | 0 / 0 | 0 / 42 / 9 / 0 |
| static_0 | no | 0 | — | — | 0 / 144 / 29 / 0 |
| static_cb_0 | no | 0 | — | — | 0 / 51 / 2 / 0 |
| debitcard_1, deposit_1, other_1, tax_registry_a/b/c_1 | no | 0 | — | — | 0 aprobadas |

Totales: 94 predictores y 30 indicadores de cobertura, igual que la tabla maestra de E5. Test solo excluye 13 filas (bureau A), por lo que no hay diferencia de trato entre splits.

## Riesgo abierto

Los 12 indicadores de cobertura de productos y registros (`n_records`, `source_present`) se calculan **sin filtro as-of** (lo declara su propia regla). Muestran deriva fuerte entre train y OOT (p. ej. `tax_registry_c_1` 36,2 % → 0 %, `tax_registry_b_1` 1,8 % → 64,9 %). Opciones: excluirlos del modelo o filtrarlos por su fecha de registro. Decisión pendiente antes de modelado.

## Checklist

- [x] Regla de disponibilidad por familia
- [x] Filas excluidas y valores anulados por split
- [x] Columnas por decisión G2 y motivo (detalle por columna en las hojas G2)
- [x] Evidencia trazable a manifiestos y hojas G2
- [ ] Decidir el tratamiento de los indicadores de cobertura sin filtro as-of
