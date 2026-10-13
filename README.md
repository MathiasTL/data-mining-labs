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

La fase de **Comprensión del negocio** corresponde a [`context/01-entendimiento-empresarial.md`](context/01-entendimiento-empresarial.md) y su borrador para el informe es [`development/01-entendimiento-empresarial.md`](development/01-entendimiento-empresarial.md).

## Datos

| Ubicación                      | Contenido                                |
| ------------------------------ | ---------------------------------------- |
| `data/parquet_files/train/`    | Datos de entrenamiento.                  |
| `data/parquet_files/test/`     | Datos de prueba.                         |
| `data/feature_definitions.csv` | Diccionario de variables.                |
| `data/sample_submission.csv`   | Formato de referencia para predicciones. |

## Entorno y herramientas

Requiere Python 3.11+. Los datos viven fuera de git (`data/` está ignorado); colócalos en `data/` o apunta `DM_DATA_DIR` a tu copia local.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # polars, pyarrow, numpy, scikit-learn, pytest
pytest                       # verifica el entorno
```

| Comando                                                                     | Qué hace                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python -m dm_credit.profiling [archivo.parquet] --out results/profile_x`   | Perfilador común: filas/columnas, tipos, nulos, únicos, duplicados, estadísticos numéricos y categorías principales. Sin argumento perfila `train_base`.                                                                                                                                                                            |
| `python -m dm_credit.integrity [archivo.parquet] --out results/integrity_x` | Integridad estructural de la tabla base: tamaño y estructura, clave (`case_id` nula, repetida, filas duplicadas), tipos incompatibles con el contenido, columnas vacías, coherencia `MONTH`/`WEEK_NUM` con `date_decision` y valores de `target`. Sin argumento revisa `train_base` e indica si algún hallazgo bloquea el baseline. |
| `python -m dm_credit.baseline --cutoff-week N --model prevalence\|logistic` | Baseline sobre `train_base`: valida el contrato, parte en OOT por `WEEK_NUM`, calcula métricas globales, probabilísticas y los componentes auditables de Gini Stability, y registra el experimento en `results/experiments.jsonl`.                                                                                                  |
| `POLARS_MAX_THREADS=1 python -m dm_credit.integration_analysis`             | Audita secuencialmente las 17 familias de train: cardinalidad y cobertura por `case_id`, expansión de joins, profundidad 2, disponibilidad temporal y contrato seguro. Escribe evidencia en `results/integration_analysis/` sin materializar joins multirregistro.                                                                  |

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

## Alcance

El proyecto integrará las fuentes del dataset, construirá y comparará modelos de clasificación binaria, aplicará validación fuera de tiempo con `WEEK_NUM`, analizará estabilidad temporal y generará un informe final, código reproducible y un prototipo demostrativo viable. No incluye un despliegue productivo ni decisiones crediticias reales.
