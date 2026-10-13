# A6 — Política de faltantes, categorías, coerción, extremos y nombres

> **Responsable:** Gabriel (P5). **Estado:** propuesta para G1; nada de esto está aprobado todavía.
> **Alcance:** transversal a todas las familias, según el [backlog del capítulo 03](03-preparacion-datos-backlog.md) (tarea A6) y la estrategia mínima de la sección 8.
> **Complementos:** fixtures `tests/preparation_a6_fixtures.py`, pruebas `tests/test_preparation_a6.py`, coerción D4 (`src/dm_credit/preparation/coercion.py`, Andre) y extremos D5 (`src/dm_credit/preparation/outliers.py`, Andre).
> **Documentos relacionados:** contratos A1/A2 (`03-preparation-contracts.md`), propuestas A4 (`03-preparacion-A4-solicitudes-personas.md`) y A5 (`03-preparacion-A5-bureau-streaming.md`).
>
> **Qué no es:** no implementa `preprocessing.py` (D1–D3, después de G1) ni fija la lista de columnas (B6/G2). Es la especificación que G1 debe congelar.

## 1. Principios

1. **Nada se aprende de OOT/test.** `fit` recibe exclusivamente `WEEK_NUM < 74` validado; `transform` no recalcula medianas, vocabularios, rangos ni columnas. Test solo valida contrato.
2. **Nada implícito.** Sin imputación silenciosa, sin conversión silenciosa a nulo, sin ampliación de vocabulario, sin recorte automático.
3. **Error frente a advertencia.** Error = contrato o dato irrecuperable (columna faltante, no convertible, infinito, dominio violado, `case_id` nulo/duplicado). Advertencia = señal informativa (valor fuera del rango de train).
4. **Trazabilidad.** Cada característica se describe con `familia + columnas fuente + filtro temporal + agregación + significado`. Cobertura y ausencia son características de rol `cobertura`, no predictores silenciosos.
5. **Reutilización.** Coerción y extremos ya existen (D4/D5 de Andre). A6 no los duplica: fija cómo usarlos y qué queda para D1–D3 (Gabriel).

## 2. Faltantes (D2)

| Tipo | Regla |
| --- | --- |
| Numérica agregada | Indicador `<feat>__missing` (1 si el agregado no tuvo observaciones válidas) + `source_present` por familia. Imputación solo con estadístico de train (mediana por defecto); el valor imputado y la mediana quedan registrados en `preprocessing_spec`. |
| Categórica agregada | Ausencia = categoría `__missing__` antes de codificar, distinta de `__UNKNOWN__`. Proporciones se calculan sobre observaciones válidas; el denominador cero da proporciones nulas + `__missing=1`. |
| Booleana agregada | Proporción verdadera sobre válidos + conteo de válidos. Sin válidos → nulo + `__missing=1`. |
| Fecha interna | Valor posterior a `date_decision` = nulo (misma regla que A4 §2 y A5 §3.4). El conteo de anulados va solo al manifiesto, nunca como característica por caso. |
| Sin cobertura de familia | El ensamblador completa `n_records=0`, `source_present=False`; el resto queda nulo para que D2 lo trate (propuesta pendiente de confirmar en G1, coherente con A4 §8.3). |

`NaN` se trata como ausente (ya lo hace `coercion.py`). Infinito es error, nunca ausente.

Casos físicos especiales (verificados en `test_preparation_coercion.py` y cubiertos por los fixtures A6):

- Columna física `Null` o 100 % nula adopta el tipo lógico de train. Cubre `test_tax_registry_c_1.parquet` vacío y sus columnas `Null`.
- Las cuatro fechas de test que llegan como `Float64` totalmente nulo se convierten a `Date` sin error; con valores serían rechazadas.

## 3. Categóricas (D3)

1. **Vocabulario congelado en `fit`.** Lista ordenada por familia, aprendida solo de train `< 74`. `transform` no añade categorías.
2. **Desconocida en OOT/test → `__UNKNOWN__`.** Token reservado, nunca aparece como categoría natural (los adaptadores deben rechazarlo en train).
3. **Raras en train → `__UNKNOWN__`.** Umbral por defecto `min_frequency=0.01` (el mismo que usa `baseline.py` para el baseline logístico); G1 puede ajustarlo. El mapeo de raras queda registrado en la especificación.
4. **Nulo → `__missing__`,** distinto de `__UNKNOWN__`. La vista lineal los distingue; la vista de árboles puede dejar nativo según D7.
5. **Orden estable.** El orden del vocabulario (incluidos `__UNKNOWN__` y `__missing__`) forma parte del esquema final ordenado (D6).
6. **Alta cardinalidad** (`tax_registry_*`, `profession_152M` con 11 558 valores): solo conteo de distintos y frecuencias de categorías aprobadas; prohibidas las categorías libres con identificadores o nombres directos. Coherente con C7.
7. **Identificadores y cuasi-identificadores** (direcciones, distritos, postales, empleador, claves `case_id/num_group1/num_group2`) excluidos de predictores directos (B6/G2).

Constante de referencia (propuesta, a congelar en G1):

```python
UNKNOWN = "__UNKNOWN__"
MISSING = "__missing__"  # coincide con dm_credit.baseline.MISSING
```

## 4. Coerción (D4, Andre — A6 fija su uso)

Delegado a `src/dm_credit/preparation/coercion.py` (`LOGICAL_TYPES = date/int/float/bool/string`, `CoercionError`). A6 no cambia ese módulo; establece:

- Todo agregado y matriz final pasa por `coerce_to_logical` con el esquema lógico declarado en el catálogo.
- Error explícito ante: valor no convertible (texto no numérico, fecha fuera de `%Y-%m-%d`, booleano fuera de `true/false/1/0`), infinito (incluido texto `"inf"`), columna requerida ausente, entero con parte fraccionaria o fuera del rango exacto (`> 2**53`), `datetime` con hora no medianoche.
- `FeatureDefinition` reutiliza `coercion.LOGICAL_TYPES`. Sin tipos nuevos hasta G1.

## 5. Extremos (D5, Andre — A6 fija su uso)

Delegado a `src/dm_credit/preparation/outliers.py` (`RangePolicy`, `RangeWarning`, `DomainViolationError`, `fit_range_policy` solo con train). A6 establece:

- Fuera del rango de train → `RangeWarning`, sin modificación ni recorte automático.
- Recorte (winsorización) solo para columnas aprobadas por el equipo, con límites aprendidos en train (`clip_quantiles` por defecto `(0.001, 0.999)`); queda registrado por columna.
- Regla de dominio explícita violada (p. ej. importe `< 0` donde aplique) → `DomainViolationError` en `fit` y en `apply`. Infinito → error siempre.
- `fit` rechaza tablas con `WEEK_NUM >= 74` (guarda OOT); quien llama pasa la partición train ya validada (D1).

## 6. Nombres de características

Convención (a congelar en G1, coherente con A4 §3):

```text
<familia>__<columna>__<agreg>__<ventana>
```

- Minúsculas, `__` como separador, sin espacios ni tildes. `<ventana>` es `all` hasta que A7/G1 fije ventanas (`3m/6m/12m` multiplican características y parciales de bureau: ver A5 D4).
- Cobertura: `<familia>__n_records`, `<familia>__source_present`. Ausencia: `<feat>__missing`.
- Recencia/antigüedad: `<familia>__days_since_last_<campo>`, `<familia>__days_since_first_<campo>`.
- Proporciones categóricas: `<familia>__<col>__share_<cat>` con `<cat>` normalizada (minúsculas, no alfanumérico → `_`).
- Prohibido colisionar con `dm_credit.baseline.RESERVED` (`case_id`, `WEEK_NUM`, `target`, `date_decision`, `MONTH`, `num_group1`, `num_group2`) y duplicar nombres entre familias (lo valida el ensamblador A2).
- Expresión de validación: `^[a-z][a-z0-9_]*(__[a-z0-9_]+){1,3}$` más minúsculas obligatorias y prohibición de `__unknown__`/`__missing__` como segmento natural en train.

## 7. Seam `fit`/`transform` (D1, propuesta para G1)

```python
spec = PreprocessingSpec(...)  # medianas, vocabularios, rangos, columnas, orden
fitted = fit_preprocessor(train_features, spec)   # solo WEEK_NUM < 74; estado serializable
X_train = fitted.transform(train_features)
X_oot = fitted.transform(oot_features)            # no muta vocabularios ni imputadores
X_test = fitted.transform(test_features)          # sin target; ausencia de target no se imputa
```

Invariantes (las mismas del backlog §3.2):

1. `fit` rechaza OOT/test o recibe partición validada; `WEEK_NUM` nula o `>= 74` en `fit` = error.
2. `transform` con categoría desconocida → `__UNKNOWN__`; con columna requerida ausente, valor no coercible o infinito → error.
3. Numérico fuera del rango de train → advertencia (`RangeWarning`), no recorte salvo columna aprobada.
4. Esquema final único, ordenado y tipado; `target` ausente en test; controles separados de predictores (D6).
5. Las dos vistas de D7 (lineal imputada/escalada/codificada y árboles con nulos/categóricas nativos) comparten filas y trazabilidad de columnas; ambas se ajustan solo con train.

## 8. Fixtures y pruebas

Fixtures sintéticos en `tests/preparation_a6_fixtures.py` (pequeños, sin Parquet reales; resultados calculados a mano):

- Numérica con nulos, `NaN`, infinito, texto no numérico y texto numérico.
- Categórica train con rara (`min_frequency`), OOT con desconocida y nulo.
- Booleana textual válida/inválida; fechas válida/inválida/`Float64` todo-nulo/`Null`.
- Train con `WEEK_NUM` 70–73 y OOT con 74 para la guarda temporal; columna con rango train conocido y valores OOT por debajo/encima.
- Nombres válidos e inválidos (reservada, mayúsculas, segmento `__unknown__` natural).

Pruebas en `tests/test_preparation_a6.py`: cada fila de las secciones 2–6 tiene al menos un caso (faltantes con indicador, desconocida → `__UNKNOWN__`, rara → `__UNKNOWN__`, `Null` adopta tipo, `Float64` nulo → `Date`, infinito/no-convertible → error, `NaN` → nulo, OOT en `fit` → error, fuera de rango → aviso sin modificar, recorte solo aprobado, dominio → error, nombres válidos/inválidos, vocabulario/mediana congelados ante OOT).

## 9. Compatibilidad con A1/A2 y decisiones para G1

Coincide con `03-preparation-contracts.md`: partición 74, `RESERVED = baseline.RESERVED`, `target` binario solo train/OOT, una fila por `case_id`, esquema idéntico salvo `target`, `CoercionError`/`DomainViolationError` como errores de entrada/datos, manifiestos sin muestras sensibles.

Pide a G1 (orden del día §13 del backlog, puntos 2/5/6/12):

1. Congelar `UNKNOWN`/`MISSING`, umbral de raras y orden del vocabulario.
2. Confirmar que el ensamblador completa `n_records=0`/`source_present=False` en casos sin cobertura.
3. Confirmar cuantiles de recorte por defecto y que el recorte va solo en la vista lineal.
4. Congelar la convención de nombres y su regex.
5. Confirmar que `FeatureDefinition` reutiliza `coercion.LOGICAL_TYPES` sin tipos nuevos.
6. Confirmar la firma `fit_preprocessor`/`transform` y su serialización (JSON + Parquet, publicación atómica como en A2/A5).

## 10. Reproducir

```bash
uv run --extra dev pytest -q tests/test_preparation_a6.py tests/test_preparation_coercion.py tests/test_preparation_outliers.py
```
