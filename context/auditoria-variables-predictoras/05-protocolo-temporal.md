# Protocolo temporal de selección, ajuste y evaluación

Fecha: 14/11/2026. Estado: **vigente** desde esta fecha para todo experimento de modelado. Cierra la sección 2 del [plan de revisión](plan-revision.md).

## Por qué hace falta

Los modelos del avance 2 tienen dos problemas de diseño:

1. **Preprocesamiento contaminado.** Se entrenó con las semanas 0 a 63 y se validó con las 64 a 73, pero usando las vistas E6, que ajustaron imputación, escalado, vocabulario y descarte de constantes con las semanas 0 a 73. La validación ya había influido en el preprocesamiento. Ver [reconciliación de vistas](03-reconciliacion-vistas-y-pruebas.md).
2. **OOT consultado.** Las semanas 74 a 91 se usaron para comparar modelos y analizar ablaciones. Ya no son una evaluación ciega.

Por eso esos resultados (Gini Stability OOT 0,494 de LightGBM) quedan como **históricos** y no se reportan como resultado final.

## Contrato de predicción

| Elemento | Definición |
| --- | --- |
| Unidad | Una solicitud por `case_id`. |
| Objetivo | `target` binario: 1 incumplimiento, 0 pago. El horizonte de incumplimiento no está publicado en el dataset; se registra como limitación. |
| Momento de predicción | `date_decision`. Toda variable debe estar disponible en esa fecha ([evidencia de disponibilidad](04-evidencia-disponibilidad-temporal.md)). |
| Columnas de control, nunca predictoras | `case_id`, `WEEK_NUM`, `date_decision`, `MONTH`, `target`. |
| Matriz de partida | Matriz maestra E5 (sin transformaciones ajustadas). Las vistas E6 publicadas no se reutilizan para validación interna. |

## Particiones

Desarrollo: semanas 0 a 73 (1 332 001 casos). Evaluación final: OOT, semanas 74 a 91 (194 658 casos).

El volumen cae después de la semana 60 (las semanas 70 a 73 tienen 654 positivos), así que las particiones internas se definen por bloques con suficientes positivos, en ventana creciente:

| Partición | Ajuste | Validación | Casos validación | Positivos validación |
| --- | --- | --- | --- | --- |
| P1 | 0 a 39 | 40 a 49 | 248 961 | 9 012 |
| P2 | 0 a 49 | 50 a 59 | 215 392 | 8 771 |
| P3 | 0 a 59 | 60 a 73 | 112 852 | 4 230 |

Reglas:

1. **En cada partición, todo lo aprendido se ajusta solo con su bloque de ajuste.** Imputación, escalado, vocabulario categórico, descarte de constantes, selección de variables e hiperparámetros.
2. **Selección de variables, ablaciones e hiperparámetros:** se decide por el **Gini Stability fuera de partición**. Se unen las predicciones de validación de P1, P2 y P3, que forman una serie continua de las semanas 40 a 73 (34 semanas), y cada semana la predice un modelo que no la usó en su ajuste. Además, la diferencia en el Gini semanal medio debe repetirse en al menos dos de las tres particiones.

   *Corrección del 14/11/2026, anterior a cualquier consulta del OOT sobre la versión v2:* la versión inicial promediaba el Gini Stability de cada partición. Con 10 a 14 semanas por bloque, la penalización por pendiente (88 × pendiente) llegaba a −1,8 frente a un Gini medio de 0,40 en una prueba con 5 % de los datos; la selección habría respondido al ruido de la pendiente.
3. **Calibración y umbrales:** se aprenden con las predicciones de P3 sobre las semanas 60 a 73, el periodo más cercano al OOT.
4. **Modelo final:** el modelo de P3 (ajuste 0 a 59), con su calibrador y umbrales de 60 a 73. Se elige no reajustar con 0 a 73 para que el calibrador corresponda exactamente al modelo evaluado. Costo: se pierden 112 852 casos recientes del ajuste. Como contraste se reporta también un reajuste 0 a 73 sin recalibrar, identificado como variante.

## Uso del OOT

- Se evalúa **una sola vez**, después de congelar variables, hiperparámetros, calibrador y umbrales con las particiones internas.
- No se elige nada mirando el OOT: ni modelo, ni variables, ni umbral.
- El informe debe declarar que el OOT fue consultado en el avance 2. Es la única evaluación fuera de tiempo disponible: el test local tiene 10 casos y sirve solo para comprobar compatibilidad.

## Modelos

| Rol | Modelo | Vista |
| --- | --- | --- |
| Referencia | Regresión logística | Lineal, ajustada por partición |
| Candidato | LightGBM | Árboles, ajustada por partición |

Ambos con presupuesto de búsqueda comparable y configuración registrada. Otros modelos (árbol, bosque, MLP) son complementarios y no cambian la selección.

## Métricas y criterios de aceptación

| Objetivo | Métrica | Aceptación |
| --- | --- | --- |
| OM1 discriminación | Gini Stability (`baseline.gini_stability`), AUC, Gini semanal | Candidato con Gini Stability OOT mayor que la referencia y ventaja en la mayoría de las semanas OOT |
| OM1 calibración | Brier, log loss, curva de calibración por deciles, calibración global por periodo | Brier menor que el de predecir la tasa base; error de calibración esperado menor que 0,01; desvío relativo entre tasa predicha y observada menor que 20 % en OOT |
| OM2 interpretación | SHAP global en P3 y en OOT; casos de referencia | Ranking reproducible con semilla fija; cambios de rango entre periodos reportados |
| OM3 falsos positivos | FPR a recall 70 %, 80 % y 90 % | Umbral fijado en 60 a 73 y aplicado al OOT; candidato con menor FPR que la referencia al mismo recall. Se reporta además la comparación sobre la curva OOT, separada de la de umbral fijo |
| OM4 cambios | PSI semanal del score y de las variables principales; detector ADWIN sobre pérdida semanal | Cada alerta ubicada en una semana y contrastada con el Gini y la calibración de ese periodo |

## Segmento de historial limitado (ON2)

Definición operativa: **sin historial observado** = sin solicitud previa válida (`applprev_1__source_present` falso) y sin registros de bureau (`credit_bureau_a_1__source_present` y `credit_bureau_b_1__source_present` falsos).

En desarrollo son 154 599 casos (11,6 %), con tasa de incumplimiento 2,64 % frente a 3,37 % del resto.

Limitación: la ausencia de una fuente no prueba ausencia de historial crediticio. Es un indicador de historial no observado por estas fuentes. Para ON2 se reporta el FPR del segmento a igual recall global, comparando candidato contra referencia.

## Pruebas de quitar variables

Bajo este mismo protocolo (P1 a P3), con el candidato y la referencia:

1. Sin variables de cobertura (`*__n_records`, `*__source_present`).
2. Sin `person_2__*`, por la caída de presencia en OOT ([D03](04-evidencia-disponibilidad-temporal.md)).
3. Contactos: conteo de rol `PE` frente a ninguna variable de contactos.
4. Sin `tax_registry_b_1__*`, por su salto de importancia en el OOT del avance 2.

Un grupo se retira si quitarlo no empeora el Gini Stability fuera de partición (semanas 40 a 73), no empeora el Gini semanal medio en al menos dos de tres particiones y no aumenta la desviación del Gini semanal fuera de partición.

## Registro

Cada experimento guarda en `results/modeling/` la configuración, la semilla, la partición, las métricas por semana y el hash de la matriz E5 usada.
