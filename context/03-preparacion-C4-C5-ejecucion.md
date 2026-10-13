# C4/C5 — Ejecución real de bureau

**Estado:** ejecutado con Parquet locales el 13/10/2026, en train, OOT y test. Los adaptadores y el plan de streaming están en `src/dm_credit/preparation/families/bureau.py`; el runner reproducible es `src/dm_credit/preparation/families/bureau_run.py`.

## Ejecución

Desde la raíz del repositorio:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.bureau_run
```

El runner particiona train con corte `WEEK_NUM = 74`, procesa las dos familias secuencialmente y reusa checkpoints válidos por shard. Escribe artefactos y checkpoints bajo `data/artifacts/preparation/C4-C5-v1/`, y el catálogo del frente en `results/preparation/bureau_feature_catalog.csv`.

## Resultado observado

| Familia | Split | Casos base | Características por caso |
|---|---|---:|---:|
| `credit_bureau_a` | train | 1,332,001 | 82 |
| `credit_bureau_a` | OOT | 194,658 | 82 |
| `credit_bureau_a` | test | 10 | 82 |
| `credit_bureau_b` | train | 1,332,001 | 78 |
| `credit_bureau_b` | OOT | 194,658 | 78 |
| `credit_bureau_b` | test | 10 | 78 |

El catálogo del frente tiene 160 definiciones únicas: 154 predictores y 6 características de cobertura. La tabla hija `credit_bureau_a_2` se omite según la selección temporal vigente de B5, que no autoriza su ancla; por ello, no aporta columnas al esquema de `credit_bureau_a`. Cada artefacto mantiene una fila por caso de la partición y su manifiesto registra el esquema y las entradas.

La consolidación global posterior está en `results/preparation/C8_feature_catalog.csv`; contiene C1–C7 y no reemplaza los catálogos por frente.
