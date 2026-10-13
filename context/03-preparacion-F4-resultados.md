# F4 — Resultados por fuente con comando, informe y fecha

## Qué pide F4

Cada cifra por fuente debe ir acompañada del comando que la produjo, del informe donde vive y de la fecha (backlog, tabla de la épica F).

## La tabla de evidencia

`results/preparation/family_results.csv` es la tabla de evidencia: una fila por familia y split (12 familias × train, OOT y test = 36 filas). La escribe `dm_credit.preparation.validation` (`family_results` y `write_reports`).

| Columna | Significado |
| --- | --- |
| `split` | `train`, `oot` o `test` |
| `family` | Familia de fuente |
| `rules_version` | Versión de las reglas con que se construyó la pieza, tomada de su manifiesto |
| `cases` | Casos de la pieza (uno por caso de la base del split) |
| `cases_with_source` | Casos con al menos un registro válido de la fuente |
| `coverage_pct` | `cases_with_source` sobre los casos de la base, en porcentaje |
| `predictor_features` | Columnas con rol predictor |
| `coverage_features` | Columnas con rol de cobertura (`n_records`, `source_present`) |
| `report` | Informe de la fuente: manifiesto de la familia (reglas, exclusiones, entradas y validaciones), con ruta relativa a la raíz del repositorio y separadores POSIX; nunca una ruta absoluta |
| `command` | Comando único de reproducción (E7) |
| `date` | Fecha ISO (AAAA-MM-DD) de la ejecución de la validación que calculó las cifras |

## Cómo regenerarla

Reconstruye todo, incluida la validación:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json --from-scratch
```

Solo la validación, sobre las piezas ya publicadas:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.validation
```

## Limitación

`date` es la fecha de la ejecución de la validación que calculó las cifras, no la fecha en que se construyó cada pieza. Por eso el archivo cambia cada vez que se vuelve a ejecutar el pipeline otro día, aunque las demás columnas no cambien.

## Resumen de las 12 familias

Cobertura en train y OOT (`coverage_pct`) y columnas por rol, según el archivo actual. Las columnas por rol son iguales en los tres splits.

| Familia | Reglas | Train % | OOT % | Predictores | Cobertura |
| --- | --- | ---: | ---: | ---: | ---: |
| `applprev` | A4-v1 | 78,2386 | 85,2264 | 61 | 4 |
| `credit_bureau_a` | C4-C5-v2 | 71,0328 | 87,646 | 0 | 2 |
| `credit_bureau_b` | C4-C5-v2 | 0,4242 | 0,8903 | 0 | 4 |
| `debitcard_1` | products-v1 | 7,1321 | 8,6167 | 0 | 2 |
| `deposit_1` | products-v1 | 6,7071 | 8,1024 | 0 | 2 |
| `other_1` | products-v1 | 1,9414 | 12,9715 | 0 | 2 |
| `person` | A4-v1 | 100,0 | 100,0 | 33 | 4 |
| `static_0` | depth0-v3 | 100,0 | 100,0 | 0 | 2 |
| `static_cb_0` | depth0-v3 | 98,0559 | 99,852 | 0 | 2 |
| `tax_registry_a_1` | products-v1 | 34,2114 | 1,1497 | 0 | 2 |
| `tax_registry_b_1` | products-v1 | 1,8336 | 64,8871 | 0 | 2 |
| `tax_registry_c_1` | products-v1 | 36,2061 | 0,0 | 0 | 2 |
