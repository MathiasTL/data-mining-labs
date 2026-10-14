# 1. Entendimiento empresarial

> En CRISP-DM, esta fase también se denomina **Comprensión del negocio**.

## 1.1 Determinar los objetivos comerciales

### Contexto del problema

En el crédito de consumo dirigido a personas con historial crediticio limitado o inexistente, un modelo de riesgo puede perder capacidad predictiva con el tiempo si cambian el perfil de los solicitantes, su comportamiento de pago o el contexto. Esa pérdida de estabilidad puede contribuir tanto a subestimar el riesgo de incumplimiento como a sobreestimar el riesgo de clientes solventes.

El proyecto estudia este problema en un entorno académico con el dataset **Home Credit – Credit Risk Model Stability** de Kaggle, que reúne más de 30 tablas de distintas granularidades y permite analizar el desempeño por semana de observación. El proyecto no representa una evaluación ni una solución desplegada en una entidad financiera real.

### Objetivos de negocio

| ID | Objetivo de negocio |
|---|---|
| ON1 | Estimar el riesgo de incumplimiento de cada solicitante para respaldar la decisión de crédito. |
| ON2 | Lograr que el criterio de riesgo no pierda validez con el paso del tiempo. |
| ON3 | Comprender qué factores explican el riesgo. |
| ON4 | No rechazar por error a solicitantes solventes. |
| ON5 | Saber a tiempo cuándo el criterio de riesgo deja de ser confiable. |

El ON5 no figura entre los cuatro objetivos oficiales del caso de negocio. Se justifica por la sección «Eficiencia» (*Efficiency*) del *Business Case* y por la Resolución SBS N° 00053-2023.

### Beneficio esperado

En un escenario hipotético de adopción financiera, una estimación de riesgo estable podría aportar evidencia para priorizar la revisión de solicitudes y monitorear la pérdida de validez de un modelo. En este proyecto académico, el beneficio directo es experimental: estudiar el problema, documentar resultados reproducibles y construir un prototipo demostrativo; no se tomarán decisiones crediticias reales.

### Criterios de éxito

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

### Objetivos de minería de datos y trazabilidad

Cada objetivo de negocio (ON) se traduce en exactamente un objetivo de minería de datos (OM) con su medida de éxito.

| ON | OM | Objetivo de minería de datos | Medida de éxito |
|---|---|---|---|
| ON1 | OM1 | Clasificación binaria: score de probabilidad de incumplimiento entre 0 y 1 para cada `case_id`. | Gini/AUC global en validación OOT. |
| ON2 | OM2 | Validación fuera de tiempo (OOT) por `WEEK_NUM` para verificar que el desempeño se mantiene. | **Gini Stability** (métrica de la competición). |
| ON3 | OM3 | Identificar las variables o grupos de variables más influyentes. | Ranking de importancia reproducible. |
| ON4 | OM4 | Optimizar el umbral de decisión y evaluar falsos positivos, precisión, recall y calibración. | Tasa de falsos positivos (FPR) a recall fijo. Punto de operación (por ejemplo, TPR = 0,80): *supuesto por acordar*. |
| ON5 | OM5 | Regla de alerta sobre el Gini semanal y su tendencia. | Detecta la degradación en las semanas OOT. Tamaño de la caída y número de semanas k: *supuesto por acordar*. |

Nota: el OM2 evalúa en tiempo de diseño, mientras que el OM5 decide en operación. La Gini Stability sigue siendo la métrica de la competición; el Gini/AUC global (OM1) no la reemplaza.

La variable `WEEK_NUM` se utilizará para evaluar la estabilidad temporal de las predicciones y del desempeño del modelo.

### Métricas

La métrica de la competición, y la que determina la estabilidad temporal (OM2), es la **Gini Stability Metric**. El Gini/AUC global corresponde al OM1. Para el análisis temporal se calculará el Gini por `WEEK_NUM`:

```text
Gini semanal = 2 × ROC-AUC semanal − 1
```

Se ajustará una regresión lineal sobre los Gini semanales (`gini = a × WEEK_NUM + b`). La métrica final penaliza tanto el deterioro temporal como la variabilidad de los resultados:

```text
Gini Stability = media(Gini semanal) + 88 × min(0, a) − 0.5 × desviación estándar(residuos)
```

Donde `a` representa la tendencia temporal del Gini y los residuos corresponden a las diferencias entre los Gini observados y los estimados por la regresión. Un valor más alto indica mejor poder predictivo, mayor estabilidad en el tiempo y menor variabilidad.

### Criterio de selección

El modelo final se elegirá por el equilibrio entre desempeño global, estabilidad temporal, interpretabilidad, coste computacional y reproducibilidad.

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
