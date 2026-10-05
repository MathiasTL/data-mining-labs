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

| Comando | Qué hace |
| --- | --- |
| `python -m dm_credit.profiling [archivo.parquet] --out results/profile_x` | Perfilador común: filas/columnas, tipos, nulos, únicos, duplicados, estadísticos numéricos y categorías principales. Sin argumento perfila `train_base`. |
| `python -m dm_credit.baseline --cutoff-week N --model prevalence\|logistic` | Baseline sobre `train_base`: valida el contrato, parte en OOT por `WEEK_NUM`, calcula Gini, Gini Stability y Gini semanal, y registra el experimento en `results/experiments.jsonl`. |

En código, **toda** lectura pasa por `dm_credit.io` (`load_base("train")`, `load_parquet(ruta)`, `load_feature_definitions()`); no se implementan lectores propios en notebooks.

### Contrato del baseline

```text
case_id      clave única, una fila por caso
WEEK_NUM     eje temporal para el corte OOT (nunca predictor)
target       0/1, solo en entrenamiento
predictores  el resto, salvo reservadas: case_id, WEEK_NUM, target, date_decision, MONTH
```

Partición OOT: entrenamiento `WEEK_NUM < corte`, validación `WEEK_NUM >= corte`. Las transformaciones se ajustan solo con entrenamiento. El corte es un parámetro obligatorio: lo define el protocolo de validación (paso 4 de la etapa A).

## Alcance

El proyecto integrará las fuentes del dataset, construirá y comparará modelos de clasificación binaria, aplicará validación fuera de tiempo con `WEEK_NUM`, analizará estabilidad temporal y generará un informe final, código reproducible y un prototipo demostrativo viable. No incluye un despliegue productivo ni decisiones crediticias reales.
