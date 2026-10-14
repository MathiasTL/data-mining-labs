# 7. Cierre del informe

## Conclusiones

Estructurar las conclusiones por par ON/OM (ON1-ON5 / OM1-OM5), indicando para cada uno la medida de éxito y el resultado obtenido.

<!-- Responder con evidencia:
1. ¿El modelo logró diferenciar solicitantes con mayor y menor riesgo?
2. ¿Cómo se comportó en periodos posteriores según WEEK_NUM?
3. ¿Qué fuentes o variables aportaron mayor valor?
4. ¿Se alcanzaron los criterios de éxito definidos?
5. Por par: ON1/OM1 (Gini/AUC), ON2/OM2 (Gini Stability), ON3/OM3 (importancia), ON4/OM4 (FPR a recall fijo) y ON5/OM5 (regla de alerta).
--> 

## Recomendaciones

<!-- Recomendar, según resultados medidos: modelo a utilizar como prototipo, esquema de monitoreo, necesidad de recalibración/reentrenamiento y mejoras prioritarias. -->

## Lecciones aprendidas

<!-- Documentar aprendizajes reales sobre: CRISP-DM, múltiples granularidades, agregaciones, fuga temporal, validación OOT, estabilidad y trabajo en equipo. -->

## Siglario

| Término | Definición usada en el proyecto |
|---|---|
| AUC | Área bajo la curva ROC; mide capacidad de ordenación. |
| Gini | Transformación del AUC usada para evaluar poder discriminatorio. |
| Gini Stability | Métrica de la competición: media del Gini semanal penalizada por su tendencia negativa y su variabilidad. |
| ON / OM | Objetivo de negocio / objetivo de minería de datos; cada ON1-ON5 se asocia con un OM1-OM5. |
| CRISP-DM | Metodología de minería de datos usada en el proyecto. |
| OOT | Validación fuera de tiempo (*out-of-time*). |
| `WEEK_NUM` | Variable de semana de observación utilizada para evaluación temporal. |
| *Drift* | Cambio en la población o relación entre variables y objetivo a través del tiempo. |
| PSI | Índice para medir cambio en la distribución de una población. |
| Fuga de información | Uso de información no disponible al momento real de realizar la predicción. |
| Calibración | Correspondencia entre probabilidades predichas y frecuencia observada del evento. |

## Referencias

<!-- Incluir referencias completas y consistentes. Como mínimo:
- Competición y datos de Kaggle Home Credit – Credit Risk Model Stability.
- Material docente de CRISP-DM.
- Documentación de las librerías/modelos utilizados.
- Documentos regulatorios o técnicos citados en el caso de negocio.
-->
