# E6 — Preprocesador y vistas sobre las tablas maestras

> **Estado: cerrada el 13/10/2026** (Andre, P3, en lugar de Gabriel). El preprocesador (D1 a D3, con D4 y D5) y las vistas (D7) se ajustaron **solo con train** y se aplicaron a las tablas maestras de E5.

## Resultado

| Vista | Split | Filas | Predictores | Nulos en predictores | Avisos de rango | Segundos |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lineal | train | 1 332 001 | 303 | 0 | 0 | 3,86 |
| lineal | OOT | 194 658 | 303 | 0 | 1 | 0,62 |
| lineal | test | 10 | 303 | 0 | 1 | 0,04 |
| árboles | train | 1 332 001 | 124 | 39 734 211 | 0 | 2,61 |
| árboles | OOT | 194 658 | 124 | 5 069 844 | 0 | 0,38 |
| árboles | test | 10 | 124 | 676 | 0 | 0,01 |

Ejecución con un hilo: 11,2 s y un pico de unos 3,8 GB.

- La **vista lineal** no tiene nulos: imputa con la mediana de train, escala con su media y desviación, y codifica categorías en one-hot con `__missing__` y `__UNKNOWN__`.
- La **vista de árboles** conserva los nulos nativos (casos sin historial, por ejemplo) y las categorías como `pl.Enum` congelado.
- Los avisos de rango en OOT y test son la política de D5: valores fuera del rango de train, sin recorte.

## Qué se verifica

- El ajuste rechaza cualquier fila con `WEEK_NUM >= 74`.
- Transformar OOT y test no cambia el estado ajustado.
- Cada vista tiene el mismo esquema en los tres splits (salvo `target`, ausente en test).
- Las columnas de control preceden a los predictores y conservan el orden de los casos.

## Dónde están

- `data/artifacts/preparation/E6-v1/{linear,tree}__{train,oot,test}.parquet`.
- Estado ajustado: `matrix_views_state.json`. Especificación de cada columna de salida: `matrix_views_spec.csv`.
- Informe con la suma de verificación de cada salida: `results/preparation/E6_views.csv`.
- Las **vistas de train (132 y 103 MiB) no se versionan en git**: superan el límite de GitHub. Se regeneran con el comando de abajo y su SHA-256 está en el informe.

## Reproducir

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e5   # si falta la tabla maestra de train
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e6
uv run --extra dev pytest -q tests/test_preparation_e6.py
```

**Actualizado en G4:** las columnas constantes en train ya se descartan (pedido de G2): 10 características salen de las dos vistas y 43 salidas lineales constantes salen de la lineal, registradas con `role = dropped` en `matrix_views_spec.csv`. Con eso y el filtro por fecha de productos (`products-v2`), la vista lineal queda en **234** predictores y la de árboles en **114** (`results/preparation/E6_views.csv`). Las cifras de la tabla de arriba son las de la primera ejecución.
