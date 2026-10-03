# 1. Entendimiento empresarial

> En CRISP-DM, esta fase también se denomina **Comprensión del negocio**.

## 1.1 Determinar los objetivos comerciales

### Contexto del problema

En el crédito de consumo, un modelo de riesgo puede perder capacidad predictiva con el tiempo debido a cambios en el perfil de los solicitantes, su comportamiento de pago o el contexto. Esta pérdida de estabilidad puede llevar a subestimar el riesgo de clientes que incumplirán o a sobreestimar el riesgo de clientes solventes.

El proyecto utiliza el dataset **Home Credit – Credit Risk Model Stability** de Kaggle para estudiar este problema en un entorno académico.

### Objetivo de negocio

Desarrollar una solución predictiva que permita identificar el riesgo de incumplimiento crediticio y analizar la estabilidad de su desempeño a través del tiempo.

### Beneficio esperado

El modelo busca ordenar a los solicitantes según su riesgo estimado, aportando evidencia para decisiones de crédito y para el monitoreo de la pérdida de validez de un modelo. El prototipo es demostrativo y no toma decisiones reales de aprobación o rechazo.

### Criterios de éxito

- Aplicar las seis fases de CRISP-DM.
- Obtener y comparar modelos de clasificación funcionales.
- Evaluar desempeño y estabilidad temporal por `WEEK_NUM`.
- Entregar un informe, código reproducible y prototipo demostrativo.

## 1.2 Evaluar la situación

### Alcance

El proyecto integra las tablas disponibles del dataset, prepara datos, construye características, entrena modelos de clasificación y evalúa su desempeño predictivo y estabilidad temporal.

### Fuera de alcance

No se integrará el modelo con sistemas financieros reales, no se automatizarán decisiones de crédito y no se usarán datos externos a los distribuidos para la competición.

### Recursos y responsables

<!-- Completar o ajustar desde el Project Charter con recursos de cómputo, herramientas y responsables confirmados. -->

### Riesgos y contingencias

| Riesgo | Impacto | Mitigación acordada | Estado |
|---|---|---|---|
| Alto volumen y complejidad de datos | Procesamiento lento o fallos de memoria | Procesamiento incremental, agregaciones por fuente y control de memoria | Pendiente |
| Fuga de información | Métricas artificialmente altas | Validación OOT y revisión de disponibilidad temporal | Pendiente |
| Sobreajuste | Bajo rendimiento futuro | Regularización, *early stopping* y evaluación por semana | Pendiente |
| Inestabilidad temporal | Degradación en periodos posteriores | Seguimiento de Gini/AUC por `WEEK_NUM` | Pendiente |

## 1.3 Determinar los objetivos de la minería de datos

### Definición técnica

Se construirá un modelo de clasificación binaria que estime la probabilidad de incumplimiento para cada caso de crédito. La clave, el nombre final de la variable objetivo y la distribución de clases se confirmarán durante la comprensión de datos.

### Métricas

- Métrica principal: Gini y/o ROC-AUC.
- Métrica obligatoria de estabilidad: Gini/AUC por semana o bloque temporal mediante `WEEK_NUM`.
- Métricas complementarias: PR-AUC, precisión, recall y calibración, si son aplicables.

### Criterio de selección

El modelo final se elegirá por el equilibrio entre desempeño global, estabilidad temporal, interpretabilidad, coste computacional y reproducibilidad.

## 1.4 Producir el plan del proyecto

| Fase | Fecha límite | Producto esperado |
|---|---:|---|
| Comprensión del negocio | 03/10/2026 | Objetivos, alcance y plan validados. |
| Comprensión de datos | 24/10/2026 | Inventario, EDA y calidad. |
| Preparación | 14/11/2026 | Tabla maestra y pipeline. |
| Modelado | 28/11/2026 | Modelos comparados con validación OOT. |
| Evaluación | 05/12/2026 | Modelo seleccionado y estabilidad analizada. |
| Implementación y entrega | 19/12/2026 | Prototipo, informe y presentación. |

### Cambios respecto al plan inicial

<!-- Registrar aquí decisiones, cambios de alcance o desviaciones del cronograma. -->
