# C4/C5 — Ejecución real de bureau

**Estado:** la ejecución real del 13/10/2026 (train, OOT y test) usó la versión pre-G2 de los adaptadores (`C4-C5-v1`, guiada por B5) y queda **reemplazada**. La versión vigente (`C4-C5-v2`) lee la decisión G2 y aún no se ha reejecutado con datos reales. Los adaptadores y el plan de streaming están en `src/dm_credit/preparation/families/bureau.py`; el runner reproducible es `src/dm_credit/preparation/families/bureau_run.py`.

## Ejecución

Desde la raíz del repositorio:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.bureau_run
```

El runner particiona train con corte `WEEK_NUM = 74`, procesa las dos familias secuencialmente y reusa checkpoints válidos por shard. Escribe artefactos y checkpoints bajo `data/artifacts/preparation/C4-C5-v2/` (carpeta derivada de `bureau.RULES_VERSION`) y el catálogo del frente en `results/preparation/bureau_feature_catalog.csv`.

## Resultado histórico (C4-C5-v1, reemplazado)

| Familia | Split | Casos base | Características por caso |
|---|---|---:|---:|
| `credit_bureau_a` | train | 1,332,001 | 82 |
| `credit_bureau_a` | OOT | 194,658 | 82 |
| `credit_bureau_a` | test | 10 | 82 |
| `credit_bureau_b` | train | 1,332,001 | 78 |
| `credit_bureau_b` | OOT | 194,658 | 78 |
| `credit_bureau_b` | test | 10 | 78 |

Ese catálogo tenía 160 definiciones (154 predictores y 6 de cobertura). Los predictores salían de la selección B5 sin aprobación G2, por lo que **no** deben usarse; los artefactos `C4-C5-v1` dejan de versionarse en git.

## Versión vigente (C4-C5-v2)

- **Modo `approved` (por defecto):** solo las columnas con `g2_decision = aprobada` en `results/preparation/G2_bureau_columns.csv` generan predictores. Con la matriz G2 actual no hay ninguna, de modo que cada tabla procesada aporta solo `n_records` y `source_present`. `credit_bureau_a_2` se omite porque B5 no autoriza su ancla temporal.
- **Catálogo:** `bureau_feature_catalog.csv` tiene 6 definiciones de cobertura y cero predictores (`credit_bureau_a_1`, `credit_bureau_b_1`, `credit_bureau_b_2`). Se regeneró sin datos reales ejecutando los mismos constructores con B5/G2 reales sobre datos sintéticos: los nombres, tipos y reglas dependen solo del plan B5+G2, no de los valores.
- **Modo `screening` (explícito, solo train):** construye el artefacto de evaluación aislado para el filtro de fuga de G2 §6 con las columnas `pendiente_filtro`. Se publica como `<familia>__screening` bajo `_screening/`, con versión `C4-C5-v2-screening`; el ensamblador rechaza estos artefactos, por lo que nunca entran en una matriz. Con la matriz G2 actual ninguna columna está `pendiente_filtro` y el modo falla en lugar de escribir un artefacto vacío.
- **Pendiente:** reejecutar el runner con los Parquet reales para obtener los artefactos `C4-C5-v2` y sus conteos de cobertura.

La consolidación global está en `results/preparation/C8_feature_catalog.csv`; contiene C1–C7 y no reemplaza los catálogos por frente.
