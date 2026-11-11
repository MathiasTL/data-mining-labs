# 1. Entendimiento empresarial

> En CRISP-DM, esta fase también se denomina **Comprensión del negocio**.

## 1.1 Determinar los objetivos comerciales

### Contexto del problema

En el crédito de consumo dirigido a personas con historial crediticio limitado o inexistente, un modelo de riesgo puede perder capacidad predictiva con el tiempo si cambian el perfil de los solicitantes, su comportamiento de pago o el contexto. Esa pérdida de estabilidad puede contribuir tanto a subestimar el riesgo de incumplimiento como a sobreestimar el riesgo de clientes solventes.

El proyecto estudia este problema en un entorno académico con el dataset **Home Credit – Credit Risk Model Stability** de Kaggle, que reúne más de 30 tablas de distintas granularidades y permite analizar el desempeño por semana de observación. El proyecto no representa una evaluación ni una solución desplegada en una entidad financiera real.

### Objetivo general de negocio

Contar con una evaluación del riesgo crediticio para la originación que sea precisa, explicable y que avise cuándo deja de ser confiable ante cambios en la población de solicitantes.

### Objetivos de negocio

| ON  | Objetivo de negocio                                                                                                                                                  | Criterio de éxito de negocio                                                                                                  |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| ON1 | Reducir las pérdidas asociadas al incumplimiento, mejorando la identificación de solicitudes con mayor riesgo.                                                       | El modelo ordena el riesgo mejor que una referencia simple en un periodo posterior al entrenamiento.                          |
| ON2 | Favorecer el acceso al crédito de solicitantes solventes, incluidos quienes tienen historial limitado, reduciendo el riesgo de rechazarlos indebidamente.              | Menos solventes rechazados que con la referencia, a igual detección de incumplimientos.                                       |
| ON3 | Mantener confiable la evaluación de riesgo a lo largo del tiempo y saber cuándo es necesario reentrenarla.                                                            | Las pérdidas de confiabilidad se detectan ubicadas en el tiempo y se explican por los factores que cambiaron.                 |

Los objetivos de negocio expresan lo que una entidad querría conseguir; los objetivos de minería de datos (sección 1.3) describen los resultados analíticos que aportan evidencia para esos fines. No necesitan corresponder uno a uno: un objetivo de minería puede servir a varios objetivos de negocio.

### Beneficio esperado

En un escenario hipotético de adopción financiera, una estimación de riesgo estable podría aportar evidencia para priorizar la revisión de solicitudes y monitorear la pérdida de validez de un modelo. En este proyecto académico, el beneficio directo es experimental: estudiar el problema, documentar resultados reproducibles y construir un prototipo demostrativo; no se tomarán decisiones crediticias reales.

### Criterios de finalización del proyecto

Los criterios de éxito de cada objetivo de negocio están en la tabla anterior. Como proyecto académico, además se considera terminado al:

- Completar las seis fases de CRISP-DM y documentar el proceso de forma reproducible.
- Obtener al menos un modelo funcional y comparar modelos de clasificación.
- Evaluar el desempeño predictivo y su variación temporal por `WEEK_NUM`, incluyendo validación fuera de tiempo.
- Entregar el conjunto de datos preparado para modelado, la evaluación comparativa, el análisis de estabilidad, el informe y la presentación final.
- Construir un prototipo funcional de demostración en un entorno controlado, sin conexión a sistemas reales.

## 1.2 Evaluar la situación

### Alcance

El proyecto contempla integrar las tablas disponibles del dataset respetando sus distintas granularidades, preparar datos y características, entrenar y comparar modelos de clasificación y evaluar su desempeño predictivo y estabilidad temporal. La incorporación de fuentes se realizará de forma controlada durante las fases de comprensión y preparación de datos.

### Fuera de alcance

No se integrará el modelo con sistemas financieros reales, no se automatizarán decisiones de crédito y no se usarán datos externos a los distribuidos para la competición.

### Recursos y responsables

El proyecto utilizará el tiempo y la capacidad de cómputo disponibles del equipo, sin presupuesto monetario. El entorno de trabajo contempla Python 3.11 o superior y las bibliotecas `polars`, `pyarrow`, `numpy`, `scikit-learn` y TensorFlow. La selección de modelos se basará en su desempeño y estabilidad temporal.

| Integrante                        | Responsabilidad                                                                                          |
| --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Pedro Lezama Gonzales             | Patrocinador; valida entregables y resuelve conflictos escalados.                                        |
| Nick Emanuel Salcedo Alfaro       | Líder del proyecto; coordina cronograma y tareas, contacto con patrocinador y consolidación del informe. |
| Mathias James Torres Lezama       | Comprensión y preparación de datos; tablas de solicitud actual y bureau.                                 |
| Andre Alexander Cuenca Echevarria | Modelado y prototipo demostrativo (API o dashboard).                                                     |
| Gabriel Mathias Davalos Velazco   | Comprensión y preparación de datos; solicitudes previas, depósitos y tarjetas.                           |
| Giancarlo Villavicencio Davila    | Evaluación comparativa y estabilidad temporal con validación OOT por `WEEK_NUM`.                         |

### Riesgos y contingencias

| Riesgo                              | Impacto                                 | Mitigación acordada                                                     | Estado    |
| ----------------------------------- | --------------------------------------- | ----------------------------------------------------------------------- | --------- |
| Alto volumen y complejidad de datos | Procesamiento lento o fallos de memoria | Procesamiento incremental, agregaciones por fuente y control de memoria | Pendiente |
| Fuga de información                 | Métricas artificialmente altas          | Validación OOT y revisión de disponibilidad temporal                    | Pendiente |
| Sobreajuste                         | Bajo rendimiento futuro                 | Regularización, _early stopping_ y evaluación por semana                | Pendiente |
| Inestabilidad temporal              | Degradación en periodos posteriores     | Seguimiento de Gini/AUC por `WEEK_NUM`                                  | Pendiente |

## 1.3 Determinar los objetivos de la minería de datos

### Objetivos de minería de datos

| OM  | Objetivo de minería de datos                                                                                                                                                                 | Sirve a  | Criterio de éxito de minería                                                                       |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------- |
| OM1 | Clasificación binaria supervisada que estime la probabilidad de incumplimiento de cada solicitud, validada fuera de tiempo en periodos posteriores al entrenamiento.                            | ON1, ON3 | Gini Stability fuera de tiempo superior a la del modelo de referencia; probabilidades calibradas.  |
| OM2 | Identificar las variables que más contribuyen a las predicciones, global y por caso, sin asumir que esa influencia implica causalidad, comparando el periodo de entrenamiento con el posterior. | ON3, ON1 | Ranking de factores reproducible y cambios entre periodos identificados.                           |
| OM3 | Evaluar, para distintos umbrales, la relación entre detectar incumplimientos y clasificar como riesgosos a clientes que sí pagan.                                                             | ON2, ON1 | Menos falsos positivos que el modelo de referencia a igual detección de incumplimientos.           |
| OM4 | Detectar los momentos en que cambian el desempeño del modelo o la población de solicitantes, para definir ventanas de reentrenamiento.                                                        | ON3      | Alertas ubicadas en el tiempo y contrastadas con el desempeño de cada periodo.                     |

Las técnicas concretas para cada objetivo se eligen y justifican en la fase de modelado (sección 4.1).

### Métricas

La métrica principal será la **Gini Stability Metric** de la competición. Para el análisis temporal se calculará el Gini por `WEEK_NUM`:

```text
Gini semanal = 2 × ROC-AUC semanal − 1
```

Se ajustará una regresión lineal sobre los Gini semanales (`gini = a × WEEK_NUM + b`). La métrica final penaliza tanto el deterioro temporal como la variabilidad de los resultados:

```text
Gini Stability = media(Gini semanal) + 88 × min(0, a) − 0.5 × desviación estándar(residuos)
```

Donde `a` representa la tendencia temporal del Gini y los residuos corresponden a las diferencias entre los Gini observados y los estimados por la regresión. Un valor más alto indica mejor poder predictivo, mayor estabilidad en el tiempo y menor variabilidad.

### Criterio de selección

El modelo final se elegirá por mayor Gini Stability fuera de tiempo. Interpretabilidad, coste computacional y reproducibilidad se usan como criterios secundarios.

## 1.4 Producir el plan del proyecto

| Fase                                | Fecha límite | Producto esperado                               |
| ----------------------------------- | -----------: | ----------------------------------------------- |
| Comprensión del negocio             |   05/12/2026 | Objetivos, alcance y plan definidos.            |
| Comprensión de datos                |   05/12/2026 | Inventario, exploración y calidad de datos.     |
| Preparación de datos                |   05/12/2026 | Conjunto de datos preparado para modelado.      |
| Modelado                            |   05/12/2026 | Modelos comparados con validación temporal.     |
| Evaluación                          |   05/12/2026 | Desempeño comparativo y estabilidad analizados. |
| Implementación y presentación final |   05/12/2026 | Prototipo demostrativo, informe y presentación. |

### Cambios respecto al plan inicial

El cronograma se ajustó para establecer el 05/12/2026 como fecha límite común para todas las fases del proyecto.
