# D1–D3 — Preprocesador ajustado solo con train

> **Responsable:** Gabriel (P5). **Estado:** implementado con sintéticos + esquema C8 real como fixture; ejecución E6 pendiente de E5.
> **Alcance:** D1 (ajuste solo train), D2 (faltantes), D3 (categorías) del [backlog](03-preparacion-datos-backlog.md).
> **Spec:** [A6](03-preparacion-A6-faltantes-categorias.md). **Reúsos:** D4 (`coercion.py`) y D5 (`outliers.py`) de Andre, sin duplicarlos.
> **Código:** `src/dm_credit/preparation/preprocessing.py`. **Pruebas:** `tests/test_preparation_preprocessing.py` (11 casos).

## 1. Qué entrega

- `PreprocessingSpec`: columnas numéricas/categóricas, `min_frequency` (default 0.01), `clip_columns`, `clip_quantiles`, `domain`. `build_spec_from_catalog` la deriva de un catálogo C8 (`int`/`float` → numéricas, `string`/`bool` → categóricas; `date` directa es error).
- `fit_preprocessor(train, spec, cutoff_week=74)`: exige `WEEK_NUM` presente, sin nulos y `< corte` (falla ante OOT); aprende medianas, vocabularios ordenados y rangos; devuelve `FittedPreprocessor` frozen serializable (`save/load_preprocessor` JSON).
- `FittedPreprocessor.transform(df)`: coerción estricta, indicadores `<feat>__missing`, imputación con medianas de train, categóricas a texto con nulo→`__missing__` y rara/desconocida→`__UNKNOWN__`, aviso `RangeWarning` sin recorte salvo aprobadas. Mismo esquema y orden en train/OOT/test; nunca muta el estado.

## 2. Límites con D7

Aquí sale la tabla limpia con tokens. El one-hot/escalado lineal y los nativos de árboles son de las vistas (Giancarlo): este módulo no codifica ni escala.

## 3. Reproducir

```bash
uv run --extra dev pytest -q tests/test_preparation_preprocessing.py
```
