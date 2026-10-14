# G4 — Cierre del capítulo 03

**Resultado:** los 15 puntos de la definición de terminado se cumplen sobre los datos reales. Toda la preparación se reconstruye desde cero con un solo comando, la suite completa pasa sin fallos esperados y `git diff --check` está limpio. Solo queda un paso humano: las revisiones cruzadas de F6 de Nick, Giancarlo, Mathias y Gabriel.

## Ruta rápida

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json --from-scratch
uv run --extra dev pytest -q
git diff --check
```

1. La reconstrucción corre 7 pasos: E4, sus checkpoints, las dos auditorías, E5, E6 y la validación de F3.
2. La suite completa pasa: 695 pruebas, sin fallos esperados (13/10/2026).
3. Las sumas SHA-256 de cada salida quedan en `results/preparation/E7_manifest.json`.

## Definición de terminado

| # | Criterio | Evidencia |
| ---: | --- | --- |
| 1 | Un único comando reconstruye las matrices desde los Parquet | `build.py --from-scratch`: 7 de 7 pasos ok; `E7_manifest.json` |
| 2 | Train `WEEK_NUM < 74`, OOT `>= 74` | control `train_oot_partition` (`integration_checks.csv`) |
| 3 | Ninguna transformación aprende de OOT o test | control `views_fit_on_train_only`; `fit_matrix_views` exige `WEEK_NUM < 74` |
| 4 | Una fila por `case_id` en cada agregado y matriz | controles `artifacts_cover_base` y `assembled_matrix` |
| 5 | Todo evento usado cumple su regla temporal | `F7_temporal_leakage_by_family.csv`: las 12 familias declaran su regla; desde `products-v2` también la cobertura de productos y registros |
| 6 | Los joins conservan número, unicidad y conjunto de casos | ensamblador E5 (controles tras cada join) y `assembled_matrix` |
| 7 | Train, OOT y test comparten características, orden y tipos | control `schema_parity`; `final_schema.csv` |
| 8 | Categorías desconocidas sin ampliar vocabularios | vocabulario congelado; `__UNKNOWN__` en árboles y fila en cero en la vista lineal (prueba G3) |
| 9 | Faltantes, extremos y coerciones con reglas probadas | `coercion.py` (D4), `outliers.py` (D5), `preprocessing.py` (D1-D3) y sus pruebas |
| 10 | Catálogo con origen, transformación y regla temporal | `C8_feature_catalog.csv` (124 filas) |
| 11 | Suite completa y documentación enlazada | suite sin fallos; [capítulo F5](../development/03-preparacion-datos.md) |
| 12 | Dos vistas, mismas filas y trazabilidad de columnas | lineal 234 y árboles 114 predictores; `matrix_views_spec.csv` |
| 13 | Controles fuera del conjunto predictor | `case_id`, `WEEK_NUM`, `date_decision`, `MONTH` y `target` como controles en ambas vistas |
| 14 | La matriz entra sin cambios a `dm_credit.baseline` | control `baseline_contract` |
| 15 | Informe de fuga por familia con filas excluidas por regla | `F7_temporal_leakage_by_family.csv` y `temporal_leakage_report.csv` |

## Cambios del cierre

| Cambio | Motivo | Efecto |
| --- | --- | --- |
| Productos y registros filtran por fecha (`products-v2`) | Punto 5: el 100 % de `tax_registry_a_1` era posterior a la decisión | `tax_registry_a_1` y `other_1` quedan en cero; la ablación confirma que la ganancia restante es legítima ([F7](03-preparacion-F7-informe-fuga.md)) |
| Columnas constantes en train se descartan | Pedido de G2 (D1) | 10 características y 43 salidas lineales registradas con `role = dropped` |
| `hstack` en lugar de `pl.concat(how="horizontal")` | Comportamiento obsoleto en polars (D7) | Exige además la misma altura |
| Bureau rechaza claves `(case_id, num_group1)` repetidas | Hallazgo de F1 (C4) | Ya no quedan pruebas con fallo esperado |
| E5 publica los tres splits como unidad | Hallazgo de F6 | Un fallo tardío no deja tablas que parezcan válidas |

## Entregables del backlog (§10) y sus equivalentes

| Backlog | Archivo real |
| --- | --- |
| `family_manifests/` | `data/artifacts/preparation/E4-v1/<familia>/<split>/*.manifest.json` (36) |
| `feature_catalog.csv` | `results/preparation/C8_feature_catalog.csv` |
| `integration_checks.csv` | `results/preparation/integration_checks.csv` |
| `preprocessing_spec.csv` | `data/artifacts/preparation/E6-v1/matrix_views_spec.csv` |
| `matrix_views.csv` | `results/preparation/E6_views.csv` |
| `temporal_leakage_report.csv` | `results/preparation/temporal_leakage_report.csv` y `F7_temporal_leakage_by_family.csv` |
| `run_metadata.csv` | `results/preparation/run_metadata.csv` |
| `final_schema.csv` | `results/preparation/final_schema.csv` |

Las matrices grandes de train (E5 y E6) quedan fuera de git; su ruta, suma y comando de regeneración están en `run_metadata.csv` y `E7_manifest.json`.

## Limitaciones documentadas

- **Rutas absolutas informativas:** los campos `inputs` de los manifiestos y algunos informes de E4/E7 guardan rutas de la computadora donde se generaron. La carga usa la ruta relativa al manifiesto y la suma SHA-256, así que los artefactos funcionan en otra computadora.
- **Validación de claves en bureau:** se comprueba dentro de cada lote leído. En los datos reales, `credit_bureau_a_1` y `credit_bureau_b_1` no tienen claves repetidas (verificado el 13/10/2026 sobre train y test).
- **Deriva de registros fiscales:** aun filtrados, `tax_registry_b_1` (1,8 % → 64,7 %) y `tax_registry_c_1` (30,3 % → 0 %) cambian de cobertura entre train y OOT, porque las fuentes se reemplazan en el tiempo. No es fuga; el capítulo 04 debe vigilar su estabilidad semanal.
- **Test local:** solo 10 casos; la validación sobre test es estructural.

## Checklist

- [x] Definición de terminado: 15 de 15
- [x] Reconstrucción desde cero con el comando único
- [x] Suite completa: 695 pruebas, sin fallos ni fallos esperados
- [x] `git diff --check` limpio
- [x] F7 consolidado y riesgo de productos resuelto
- [x] Hallazgos de F1, G3 y F6 (Andre) resueltos o documentados
- [ ] F6: revisión cruzada de Nick, Giancarlo, Mathias y Gabriel (paso humano)
