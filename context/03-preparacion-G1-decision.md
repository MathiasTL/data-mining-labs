# G1 — Decisión propuesta para congelar interfaces

> **Estado:** decisión **vigente** de Andre (P3), tomada el 12/10/2026 con apoyo de Claude, y aplicada desde ya a sus propios módulos. Los demás integrantes pueden objetar por escrito; una objeción se resuelve en la reunión del **vie 16/10** y, si prospera, se corrige este documento. G1 la había dejado abierta el acta de Nick; este documento cierra los puntos para que nadie quede bloqueado.
> **Insumos:** [acta de Nick](03-preparacion-G1-acta.md) y su [revisión](03-preparacion-G1-acta.md#revisión-de-andre-p3-12102026), [A1/A2](03-preparation-A1-A2-contracts.md), [A3/A7](<politica-agregaciones-y-ventanas(A3-A7).md>), [A4](03-preparacion-A4-solicitudes-personas.md), [A5](03-preparacion-A5-bureau-streaming.md) y [A6](03-preparacion-A6-faltantes-categorias.md).
> **Qué no decide:** qué columnas entran (B3–B6, G2) ni la aprobación de ningún predictor.

## 1. Los seis puntos que quedaban abiertos

| # | Punto | Decisión | Responsable |
| -: | --- | --- | --- |
| 1 | Firma de `build_family_features` | Con `base` explícito, como en el acta §1: `build_family_features(family, base, split, *, cutoff_week=74, output_dir)`. `base` trae `case_id`, `WEEK_NUM` y `date_decision`, sin `target`, y ya está particionada: para `"oot"` solo casos con `WEEK_NUM >= 74`. | P1 congela las dataclasses |
| 2 | Familias padre-hijo | Una familia es un grupo con relación padre-hijo: `applprev`, `person`, `credit_bureau_a`, `credit_bureau_b`. Las demás fuentes son familias de una tabla (`base`, `static_0`, `static_cb_0`, `debitcard_1`, `deposit_1`, `other_1`, `tax_registry_a_1`, `tax_registry_b_1`, `tax_registry_c_1`). Una familia nunca cruza con otra. | Todos |
| 3 | Lector por lotes en `io` | Lo implementa P1 y lo revisa P4, según el acta. Interfaz fijada aquí (sección 2) para no bloquear a nadie. Hasta que exista, los módulos reciben tablas ya leídas, sin lector propio. | P1 (revisa P4) |
| 4 | Gramática de nombres | La del acta §2 (sección 3 de este documento), con una expresión regular de hasta cuatro segmentos tras la fuente. | Todos; P5 ajusta A6 |
| 5 | Caché de parciales | Solo para bureau (A5 §3.3). Clave: huella de shards + versión de reglas + huella de `base` + corte. El resto de familias no usa caché. | P4 |
| 6 | Personas y snapshots | Se aprueba la **regla estructural** "atributos de la solicitud actual, persona principal `num_group1 = 0`" para poder agregar. **Ningún atributo de persona es predictor** hasta B4/G2. Sexo y género se excluyen por defecto, pendiente de decisión de equidad. | P3 (B4), equipo (G2) |

## 2. Interfaz del lector compartido (`dm_credit.io`)

```python
def scan_parquet_shards(split: Literal["train", "test"], source: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Escaneo perezoso de todos los shards de una fuente física, con proyección de columnas."""

def iter_row_groups(path: Path, columns: list[str] | None = None) -> Iterator[pl.DataFrame]:
    """Un grupo de filas por iteración, con proyección (para bureau)."""
```

- **Descubrimiento de shards:** `{split}_{source}.parquet` o `{split}_{source}_{n}.parquet`, ordenados por el índice **numérico** (`_2` antes que `_10`). Nunca una lista fija: train tiene 32 archivos y test 36.
- **`split` es el de los archivos** (`train` o `test`); `"oot"` lee archivos de train y se separa con `base`.
- Un shard requerido ausente es un error; un archivo vacío con el esquema esperado es válido (`test_tax_registry_c_1`).
- `load_parquet` se mantiene para tablas pequeñas.

## 3. Gramática de nombres

```text
<fuente_fisica>__<variable_o_metrica>__<agregacion>__<ventana>
```

Minúsculas, sin tildes. Ventana en la primera versión: **solo `all`**; `3m`, `6m`, `12m` y `source_defined` se admiten únicamente con justificación aprobada en G2.

| Tipo | Forma | Ejemplo |
| --- | --- | --- |
| Agregación numérica | `<fuente>__<col>__<count\|min\|max\|mean\|sum\|std>__all` | `applprev_1__credamount_590a__mean__all` |
| Recencia o antigüedad | `<fuente>__<campo>__days_since_last\|days_since_first__all` | `applprev_1__creationdate_885d__days_since_last__all` |
| Proporción de categoría | `<fuente>__<col>__share_<cat>__all` | `applprev_1__credtype_587l__share_cal__all` |
| Conteo de distintos | `<fuente>__<col>__n_distinct__all` | `applprev_1__credtype_587l__n_distinct__all` |
| Cobertura | `<fuente>__n_records`, `<fuente>__source_present` | `applprev_2__source_present` |
| Ausencia (solo la crea D2) | `<característica>__missing` | `applprev_1__credamount_590a__mean__all__missing` |

Expresión de validación: `^[a-z][a-z0-9_]*(__[a-z0-9_]+){1,4}$`. La `<cat>` se normaliza (minúsculas, caracteres no alfanuméricos a `_`); dos categorías que colisionen son un error, no se renombran en silencio. Prohibido el uso de `__unknown__` y `__missing__` como categoría natural.

## 4. Catálogo y manifiesto

```python
@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    role: Literal["predictor", "cobertura"]
    logical_type: Literal["date", "int", "float", "bool", "string"]  # coercion.LOGICAL_TYPES
    source_table: str
    source_columns: tuple[str, ...]
    aggregation: str
    window: str
    temporal_rule: str
    description: str
```

- **Catálogo:** `results/preparation/feature_catalog.csv`, una fila por `FeatureDefinition`.
- **Manifiesto por agregado:** JSON con versión de formato y de reglas, familia, split, ruta relativa, filas, casos, esquema ordenado, exclusiones por motivo (totales, nunca por caso), validaciones, huellas de entrada y métricas de ejecución. Sin muestras de datos.
- **`FeatureMatrixArtifact`:** `split`, `data_path`, `manifest_path`, `row_count`, `control_columns`, `feature_schema` y los manifiestos de entrada.
- Publicación atómica: se escribe en un directorio temporal y se renombra al terminar las validaciones.

## 5. Las doce decisiones del acta

| # | Decisión del acta | Resolución |
| -: | --- | --- |
| 1 | Interfaz de familia | Sección 1, puntos 1 y 2. |
| 2 | Nombres | Sección 3. |
| 3 | Catálogo y manifiesto | Sección 4. |
| 4 | Agregaciones iniciales | Conteos y cobertura en toda fuente; numéricas con `count`, `min`, `max`, `mean`, `sum`, `std`; booleanas con proporción; categóricas solo con categorías aprobadas. `n_distinct` se admite en fuentes pequeñas (`applprev`, `person`, productos) y **no** en bureau ni registros fiscales en la primera versión. Sin mediana ni percentiles. |
| 5 | Tipos, nulos y categorías | `coercion.LOGICAL_TYPES`; `NaN` es nulo; infinito y no convertible son error. Fechas futuras: se excluye la fila o se anula el campo, y el total va solo al manifiesto. `__missing__` es ausencia y `__UNKNOWN__` es categoría nueva o rara. `min_frequency = 0.01` es un valor por defecto configurable de la vista lineal. |
| 6 | Errores y advertencias | Como el acta. Sin cambiar la jerarquía de excepciones; P1 las envuelve conservando la causa. D1 debe exigir `WEEK_NUM` antes de `fit`. |
| 7 | Rutas | `io.data_dir() / "artifacts" / "preparation" / <versión> / <familia> / <split>`; no un `data/` literal. |
| 8 | Comando reproducible | `uv run -m dm_credit.preparation.build --config configs/preparation.json`, JSON, a cargo de P1. |
| 9 | Orden y presupuesto | Cola única, `POLARS_MAX_THREADS=1`, medición de tiempo y pico. El orden del backlog §7. El objetivo de 2 GB de A5 es una estimación a confirmar en E4. |
| 10 | Propiedad | La del acta. P3: núcleo de `applprev`/`person` (`families/applications_people.py`), coerción y extremos. |
| 11 | Ventana común | `all` por defecto hasta la decisión inclusiva. |
| 12 | Controles y vistas | `case_id`, `WEEK_NUM`, `date_decision`, `target` (solo train/OOT) y `MONTH` si está presente son controles; `num_group1` y `num_group2` nunca llegan a la matriz. D6 comprueba `date_decision` y la ausencia de `target` en test. |

## 6. Qué hace cada persona después de ratificar G1

| Quién | Tarea |
| --- | --- |
| Nick (P1) | Congelar las dataclasses en el contrato canónico; implementar `scan_parquet_shards` e `iter_row_groups`; CLI y configuración. |
| Giancarlo (P2) | Adaptador de profundidad 0, vistas D7 e informe de fuga F7. |
| Andre (P3) | Envolver el núcleo en `build_family_features`; B4 final tras G2. |
| Mathias (P4) | Revisar el lector por lotes; prototipo de bureau con caché de parciales. |
| Gabriel (P5) | Ajustar la expresión regular de A6 a `{1,4}`; preprocesador D1–D3. |

## 7. Lo que ya está implementado bajo esta decisión

| Pieza | Dónde | Estado |
| --- | --- | --- |
| Lector compartido (`shard_paths`, `scan_parquet_shards`, `iter_row_groups`) | `src/dm_credit/io.py` | Hecho, con pruebas. Con los datos reales encuentra 32 shards de train y 36 de test, en orden numérico. |
| `FeatureDefinition`, artefactos, manifiesto JSON y publicación atómica | `src/dm_credit/preparation/contracts.py` | Hecho, con pruebas. Si una validación falla no se escribe nada y no se sobrescribe una salida válida. |
| Validación de la base de cada familia (`validate_base`) | `contracts.py` | Hecho: rechaza claves, semanas o fechas nulas, duplicados, train con semanas `>= 74`, OOT con semanas `< 74` y test con `target`. |
| Despacho `build_family_features` | `src/dm_credit/preparation/build.py` | Hecho para `applprev` y `person`; las demás familias se registran en `BUILDERS`. |
| Adaptadores `applprev` y `person` | `src/dm_credit/preparation/families/applications_people.py` | Hecho, con pruebas. |

| Catálogo C8, informe de ejecución y exclusiones F7 de `applprev` y `person` | `families/applications_people_run.py`, `contracts.feature_catalog` y `results/preparation/` | Hecho. Ejecuta train, OOT y test, y comprueba que las características sean idénticas en los tres splits. |
| Datos y artefactos pequeños para el ensamblador (E3) | `tests/preparation_artifacts.py` | Hecho: `write_sample_data` y `build_small_artifacts` generan los datos y artefactos sintéticos de train, OOT y test. |

**Verificación con datos reales** (un hilo, ruta `data/artifacts/preparation/A4-v1/`): train (`WEEK_NUM < 74`) y OOT (`>= 74`) de las dos familias.

| Familia | Casos train | Casos OOT | Suma | Solapamiento |
| --- | ---: | ---: | ---: | ---: |
| `applprev` | 1 042 139 | 165 900 | 1 208 039 | 0 |
| `person` | 1 332 001 | 194 658 | 1 526 659 | 0 |

La suma coincide con la cobertura documentada en A4, y el OOT de `person` coincide con los 194 658 casos del backlog. El conjunto completo tardó 13 s, con un pico de memoria de unos 4,8 GB.

## 8. Límites

Esta decisión no aprueba columnas, no autoriza ningún predictor sensible y no sustituye la reunión: si alguien objeta, vale la discusión conjunta del vie 16/10.
