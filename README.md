# Predicción del incumplimiento crediticio y estabilidad temporal

Proyecto integrador del curso de Minería de Datos basado en el dataset de Kaggle [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).

El objetivo es construir un prototipo reproducible que estime la probabilidad de incumplimiento crediticio y evaluar que su desempeño se mantenga estable a través del tiempo mediante `WEEK_NUM`.

## Documentación de trabajo

El proyecto separa las instrucciones de desarrollo de los borradores del informe:

| Ubicación                      | Propósito                                                                                                                                                                                                        |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`context/`](context/)         | Guías por fase CRISP-DM: qué hacer, qué validar y cuándo avanzar. Comenzar por [`context/README.md`](context/README.md).                                                                                         |
| [`development/`](development/) | Borradores estructurados según el índice del profesor. Aquí se incorporan resultados, tablas, gráficos y decisiones que pasarán al informe final. Comenzar por [`development/README.md`](development/README.md). |
| `docs/us/`                     | Caso de negocio y _Project Charter_ elaborados por el equipo.                                                                                                                                                    |
| `docs/` fuera de `docs/us/`    | Índice y materiales entregados por el profesor. No deben modificarse.                                                                                                                                            |

La fase de **Comprensión del negocio** corresponde a [`context/01-entendimiento-empresarial.md`](context/01-entendimiento-empresarial.md) y su borrador para el informe es [`development/01-entendimiento-empresarial.md`](development/01-entendimiento-empresarial.md). El capítulo 02 de **Comprensión de los datos** está cerrado en [`development/02-comprension-datos.md`](development/02-comprension-datos.md), con trazabilidad de sus ocho puntos, limitaciones y contrato documental para la fase siguiente.

## Datos

| Ubicación                      | Contenido                                |
| ------------------------------ | ---------------------------------------- |
| `data/parquet_files/train/`    | Datos de entrenamiento.                  |
| `data/parquet_files/test/`     | Datos de prueba.                         |
| `data/feature_definitions.csv` | Diccionario de variables.                |
| `data/sample_submission.csv`   | Formato de referencia para predicciones. |

## Entorno y herramientas

Requiere Python 3.11+ y [uv](https://docs.astral.sh/uv/) (`brew install uv` o `pip install uv`). Los datos viven fuera de git (`data/` está ignorado); colócalos en `data/` o apunta `DM_DATA_DIR` a tu copia local.

```bash
uv sync --extra dev                   # crea .venv con las versiones fijadas en uv.lock
uv run --extra dev pytest -q          # suite completa: 81 pruebas al cerrar el capítulo 02
```

| Comando                                                                     | Qué hace                                                                                                                                                                                                       |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv run -m dm_credit.profiling [archivo.parquet] --out results/profile_x`   | Perfilador común: filas/columnas, tipos, nulos, únicos, duplicados, estadísticos numéricos y categorías principales. Sin argumento perfila `train_base`.                                                       |
| `uv run -m dm_credit.integrity [archivo.parquet] --out results/integrity_x` | Integridad estructural de la tabla base: tamaño, clave, tipos, coherencia temporal y objetivo. Sin argumento revisa `train_base`.                                                                              |
| `uv run -m dm_credit.variable_catalog`                                      | Consolida el catálogo funcional de las 17 familias en `results/variable_catalog*.csv` a partir de perfiles existentes y definiciones oficiales.                                                                |
| `uv run -m dm_credit.quality_catalog`                                       | Consolida perfiles existentes, sin releer Parquet, en `results/quality_catalog*.csv`.                                                                                                                          |
| `uv run -m dm_credit.target_analysis --out results/target_analysis`         | Analiza objetivo, semanas y cortes OOT; documenta el corte primario `WEEK_NUM = 74`.                                                                                                                           |
| `uv run -m dm_credit.baseline --cutoff-week 74 --model prevalence`          | Valida el contrato del baseline y calcula métricas globales, probabilísticas y componentes auditables de Gini Stability. No es necesario ejecutarlo para reproducir los reportes descriptivos del capítulo 02. |
| `POLARS_MAX_THREADS=1 uv run -m dm_credit.integration_analysis`             | Audita secuencialmente las 17 familias de train: cardinalidad, cobertura, riesgo de joins, disponibilidad temporal y contrato seguro. Escribe `results/integration_analysis/`.                                 |
| `POLARS_MAX_THREADS=1 uv run -m dm_credit.schema_compatibility`             | Compara las 17 familias lógicas train/test, valida coerciones, categorías desconocidas y rangos, y escribe `results/schema_compatibility/`. Test se usa solo como comprobación técnica.                        |

En código, **toda** lectura pasa por `dm_credit.io` (`load_base("train")`, `load_parquet(ruta)`, `load_feature_definitions()`); no se implementan lectores propios en notebooks.

### Contrato del baseline

```text
case_id      clave única, una fila por caso
WEEK_NUM     eje temporal para el corte OOT (nunca predictor)
target       0/1, solo en entrenamiento
predictores  el resto, salvo reservadas: case_id, WEEK_NUM, target, date_decision, MONTH
```

Partición OOT: entrenamiento `WEEK_NUM < corte`, validación `WEEK_NUM >= corte`. Las transformaciones se ajustan solo con entrenamiento. El protocolo temporal inicial fija `--cutoff-week 74`; su evidencia y análisis de sensibilidad están en `results/target_analysis/` y `development/02-comprension-datos.md`.

La selección de modelos usa la métrica oficial **Gini Stability** en OOT: media del Gini semanal (`2 × AUC − 1`), más `88 × min(0, pendiente)` y menos `0,5 × desviación poblacional de los residuos` de la regresión contra `WEEK_NUM`. El AUC/Gini global es diagnóstico y no la sustituye. Semanas monoclase, objetivos inválidos y valores no finitos producen un error explícito; las semanas ausentes no se interpolan. La definición, fuente oficial consultada el 2026-10-12 y ejemplos reproducibles están en `results/metric_review/`.

### Contrato de integración

La tabla de modelado conserva exactamente una fila por `case_id`. La partición temporal se crea antes de ajustar transformaciones; cada fuente se filtra con información disponible a `date_decision`, se agrega independientemente a una fila por caso y solo entonces se une a `train_base`. Cada join debe preservar número de filas, unicidad y conjunto de casos. Imputación, codificación, selección y estadísticas se ajustan únicamente con `WEEK_NUM < 74`. `case_id`, `num_group1`, `num_group2`, `date_decision`, `MONTH` y `WEEK_NUM` permanecen fuera de los predictores; `target` se usa solo como etiqueta. Las reglas y condiciones explícitas de fallo están en `results/integration_analysis/integration_contract.csv` y el análisis completo en `development/02-comprension-datos.md`.

### Contrato de compatibilidad train/test

El esquema lógico se deriva de train y las definiciones, nunca de test. `target` solo se admite en entrenamiento; OOT/test deben contener las variables seleccionadas, que se alinean por nombre y se coercionan explícitamente con error ante valores no convertibles. Una columna física `Null` se considera sin observaciones y adopta el tipo lógico de train. Imputación, escalado, selección y vocabularios se ajustan exclusivamente con `WEEK_NUM < 74`; las categorías nuevas se asignan a `__UNKNOWN__` (o a un mecanismo equivalente) sin ampliar el vocabulario. Los rangos numéricos nuevos generan advertencia salvo restricción de dominio, mientras que infinitos y fallos de coerción generan error. Tras transformar se verifican features, orden, tipos y ausencia de infinitos. Test no decide columnas, umbrales ni tratamientos. El contrato completo y los reportes están en `results/schema_compatibility/`.

## Alcance

El proyecto integrará las fuentes del dataset, construirá y comparará modelos de clasificación binaria, aplicará validación fuera de tiempo con `WEEK_NUM`, analizará estabilidad temporal y generará un informe final, código reproducible y un prototipo demostrativo viable. No incluye un despliegue productivo ni decisiones crediticias reales.
