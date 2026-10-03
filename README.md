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

## Alcance

El proyecto integrará las fuentes del dataset, construirá y comparará modelos de clasificación binaria, aplicará validación fuera de tiempo con `WEEK_NUM`, analizará estabilidad temporal y generará un informe final, código reproducible y un prototipo demostrativo viable. No incluye un despliegue productivo ni decisiones crediticias reales.
