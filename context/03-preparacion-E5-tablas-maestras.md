# E5 — Tablas maestras

> **Estado: cerrada el 13/10/2026** (Andre, P3, en lugar de Nick). Se ensamblaron las tablas maestras de train, OOT y test a partir de las 36 piezas validadas de E4.

## Resultado

| Split | Filas | Columnas | Características | Predictores | Cobertura | Familias | Segundos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1 332 001 | 129 | 124 | 94 | 30 | 12 | 4,31 |
| OOT | 194 658 | 129 | 124 | 94 | 30 | 12 | 0,56 |
| test | 10 | 128 (sin `target`) | 124 | 94 | 30 | 12 | 0,01 |

Ejecución con un hilo, un split por vez: 6,0 s en total y un pico de unos 2,0 GB.

## Qué se verifica

- **Antes de escribir:** cada split tiene exactamente las 12 familias de E4, que se unen en el orden fijo de E4.
- **En cada join** (`assembly.assemble_feature_matrix`): la pieza coincide con su suma de verificación, y no cambian el número, el orden ni la unicidad de los casos de la base.
- **Al final:** las tres tablas tienen los mismos controles (salvo `target`) y las mismas características, en el mismo orden y con los mismos tipos (`schema.validate_matrix_schemas`).

## Dónde están

- `data/artifacts/preparation/E5-v1/feature_matrix__{train,oot,test}.parquet` y sus manifiestos.
- Informe: `results/preparation/E5_master_tables.csv`, con la suma de verificación de cada tabla.
- La tabla de **train (92,6 MiB) no se versiona en git**, por su tamaño, como prevé G1 para las matrices grandes. Se regenera en 6 s; su SHA-256 está en el informe y en el manifiesto, así que se puede comprobar que la copia regenerada es idéntica.

## Reproducir

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e5
uv run --extra dev pytest -q tests/test_preparation_e5.py
```

Siguiente paso: E6, ajustar el preprocesador con train y transformar OOT y test (Gabriel).
