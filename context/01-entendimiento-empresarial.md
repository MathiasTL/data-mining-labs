# 1. Entendimiento empresarial

## Punto de partida

La fuente vigente de esta fase es `../docs/us/IT Business Case Template G3 MD.docx.pdf` y `../docs/us/Project Charter G3 MD-3.pdf`. Esta guía transforma esos acuerdos en tareas ejecutables; no los sustituye.

## Objetivo de la fase

Alinear al equipo sobre el problema de riesgo de crédito, el valor que se busca generar, las restricciones académicas y la forma en que se juzgará el éxito técnico.

## Problema definido

Un modelo de riesgo crediticio puede perder capacidad predictiva con el tiempo debido a cambios en los solicitantes, su comportamiento de pago o el contexto. El proyecto debe predecir la probabilidad de incumplimiento y comprobar que el desempeño no se degrade significativamente entre semanas de observación.

## Objetivos de negocio y de minería de datos

Cada objetivo de negocio (ON) se traduce en exactamente un objetivo de minería de datos (OM). La tabla de trazabilidad completa está en [`../development/01-entendimiento-empresarial.md`](../development/01-entendimiento-empresarial.md), sección 1.3.

| ON | Objetivo de negocio | OM | Objetivo de minería de datos | Medida de éxito |
|---|---|---|---|---|
| ON1 | Estimar el riesgo de incumplimiento para respaldar la decisión de crédito | OM1 | Clasificación binaria: score 0-1 por `case_id` | Gini/AUC global en OOT |
| ON2 | Que el criterio de riesgo no pierda validez con el tiempo | OM2 | Validación OOT por `WEEK_NUM` | Gini Stability (métrica de la competición) |
| ON3 | Comprender qué factores explican el riesgo | OM3 | Identificar variables o grupos más influyentes | Ranking de importancia reproducible |
| ON4 | No rechazar por error a solicitantes solventes | OM4 | Optimizar umbral; evaluar FP, precisión, recall y calibración | FPR a recall fijo (punto de operación: *supuesto por acordar*) |
| ON5 | Saber a tiempo cuándo el criterio deja de ser confiable | OM5 | Regla de alerta sobre el Gini semanal y su tendencia | Detecta degradación en semanas OOT (caída y k semanas: *supuesto por acordar*) |

El ON5 se justifica por la sección «Eficiencia» (*Efficiency*) del *Business Case* y por la Resolución SBS N° 00053-2023. El OM2 evalúa en tiempo de diseño y el OM5 decide en operación.

## Tareas

1. Confirmar el alcance: análisis experimental con datos de Kaggle, sin integración financiera ni automatización de decisiones reales.
2. Definir el coste conceptual de errores:
   - falso negativo: cliente riesgoso tratado como solvente;
   - falso positivo: cliente solvente tratado como riesgoso.
3. Acordar que la salida del modelo será una **probabilidad de riesgo**, no solo una etiqueta binaria.
4. Precisar métricas:
   - principal de la competición: Gini Stability (OM2);
   - desempeño global: Gini/AUC (OM1);
   - obligatoria: desempeño por `WEEK_NUM` y variación temporal;
   - complementarias: PR-AUC, recall, precisión y calibración cuando sean aplicables.
5. Confirmar recursos, responsables, riesgos, contingencias y fechas del *Project Charter*.
6. Crear un glosario inicial: `target`, `case_id`, `WEEK_NUM`, AUC, Gini, OOT, *drift*, fuga de información y calibración.

## Decisiones que deben quedar registradas

- Definición exacta de la variable objetivo tras comprobar la tabla base.
- Unidad de análisis y clave primaria de la tabla maestra.
- Criterio que se usará para decidir entre dos modelos con AUC similar pero estabilidad diferente.
- Recursos de cómputo y límites de tiempo para los experimentos.
- Riesgos priorizados y responsable de cada mitigación.

## Evidencias y entregable

- Declaración del problema y objetivos de negocio/técnicos.
- Matriz de interesados y responsabilidades.
- Matriz de riesgos y contingencias.
- Criterios de éxito medibles.
- Plan de trabajo actualizado.

## Criterio para avanzar

El equipo puede pasar a comprensión de datos cuando todos compartan la misma definición de problema, métricas, alcance y condición de estabilidad temporal.
