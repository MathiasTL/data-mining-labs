# Plan de trabajo: de la auditoría al cierre del proyecto

**Estado:** plan operativo inicial del 14/11/2026. **Fecha límite vigente:** 05/12/2026, según [Entendimiento empresarial](../development/01-entendimiento-empresarial.md#14-producir-el-plan-del-proyecto). Las fechas intermedias son metas propuestas de coordinación; los criterios de salida determinan cuándo avanzar.

## Ruta de trabajo

1. Acordar el protocolo temporal y cerrar las decisiones pendientes de variables.
2. Corregir la preparación, reconstruir las matrices afectadas y verificar su trazabilidad.
3. Comparar modelos bajo el mismo protocolo; después, evaluar los cuatro objetivos de minería.
4. Seleccionar un modelo, construir un prototipo académico reproducible y cerrar el informe con evidencia.

La [revisión de variables predictoras](03-revision-variables-predictoras.md) y su [auditoría](auditoria-variables-predictoras/README.md) son el punto de partida. El cierre anterior del capítulo 03 sigue siendo evidencia histórica; las correcciones producirán una versión nueva, identificable y verificable.

## Hitos y responsables

Los responsables siguen las funciones registradas en [Entendimiento empresarial](../development/01-entendimiento-empresarial.md#recursos-y-responsables). Nick coordina el seguimiento y las dependencias. Andre lidera las decisiones de modelado; Mathias y Gabriel apoyan las correcciones de fuentes que les corresponden; Giancarlo lidera la evaluación. Las decisiones que cambien alcance o fecha se registran con el equipo.

| Hito | Meta interna | Responsable principal | Estado | Entregable y criterio de salida |
| --- | --- | --- | --- | --- |
| **H1. Contrato y protocolo** | 17/11 | Andre, con Giancarlo y Nick | Pendiente | Unidad de predicción, significado y límite conocido de `target`, predictores reservados, particiones internas, ajuste de transformaciones, uso previo del OOT, referencia, métricas y reglas de selección documentados **antes** de nuevos experimentos. |
| **H2. Características corregidas** | 21/11 | Andre, con Mathias y Gabriel | Pendiente | Decisiones G2 implementadas; catálogo, configuración, matrices y manifiestos de la misma versión; pruebas de reglas modificadas y controles de integridad ejecutados; incidencias críticas resueltas y demás limitaciones documentadas. |
| **H3. Modelos comparables** | 27/11 | Andre | Pendiente | Probabilidad constante, regresión logística y al menos un candidato no lineal comparados con la misma partición; parámetros, semilla, versión de datos, tiempo y métricas por periodo registrados en [capítulo 04](../development/04-modelado.md). |
| **H4. Evaluación y decisión** | 30/11 | Giancarlo, con Andre | Pendiente | Comparación OOT y temporal, calibración, falsos positivos a igual detección, interpretación y alertas contrastadas con desempeño; modelo elegido o límite de selección explicado en [capítulo 05](../development/05-evaluacion.md). |
| **H5. Prototipo y cierre** | 05/12 | Andre (prototipo), Nick (informe), Giancarlo (revisión) | Pendiente | Inferencia reproducible y demostración ejecutable; [capítulo 06](../development/06-implementacion.md), conclusiones e informe con resultados, límites, referencias y comandos verificables. |

## Tareas por hito

### H1. Fijar el diseño antes de medir

- [ ] Confirmar una fila por `case_id`, `date_decision` como corte de disponibilidad y el significado oficial conocido de `target`; si el horizonte del incumplimiento no está publicado, declararlo como límite.
- [ ] Separar selección, ajuste, calibración y elección de umbrales en particiones temporales **dentro de train** (`WEEK_NUM < 74`). Ajustar imputación, vocabulario, escalado y descarte de constantes con el entrenamiento de cada partición interna.
- [ ] Registrar que OOT (`WEEK_NUM >= 74`) ya se consultó en ablaciones. Definir su uso restante y si existe algún periodo realmente no examinado; si no lo hay, limitar explícitamente la interpretación de la evaluación final.
- [ ] Fijar antes de comparar: referencia, métricas, nivel de recall para OM3, criterio de calibración y reglas para interpretar alertas de OM4. Definir un segmento de historial limitado solo si existe una regla verificable para identificarlo.

**Salida:** protocolo versionado y revisado por modelado y evaluación. Referencia: [plan de revisión, §2](auditoria-variables-predictoras/plan-revision.md#2-fijar-el-contrato-de-predicción-y-evaluación).

### H2. Corregir y congelar una versión de datos

- [ ] Cambiar `person_1__n_contacts__all` para contar únicamente filas de rol `PE`, o conservar el conteo anterior con nombre correcto si la comparación justifica ambas variables.
- [ ] Retirar de la selección los indicadores de cobertura de `static_cb_0`; evaluar por ablación su efecto y el de `person_2` y las variantes de contactos.
- [ ] Completar en el catálogo el denominador de `applprev_2` y los supuestos de disponibilidad temporal; mantener bloqueadas las columnas sin evidencia suficiente.
- [ ] Reconstruir E4–E7 desde los Parquet originales, volver a ejecutar los controles sobre train y OOT, y guardar hashes, manifiestos y resultados separados de la versión histórica.
- [ ] Comprobar una fila por caso, particiones sin solapamiento, predictores sin controles ni valores infinitos, y paridad de esquema. Registrar cualquier limitación que siga abierta.

**Salida:** una versión de características y matrices apta para los experimentos, con código y documentación concordantes. Referencia: [decisiones y correcciones de la auditoría](auditoria-variables-predictoras/plan-revision.md#registro-de-correcciones).

### H3. Entrenar y comparar

- [ ] Entrenar una referencia constante, regresión logística y al menos un modelo no lineal con el mismo protocolo temporal y conjuntos de variables declarados.
- [ ] Usar las particiones internas para seleccionar variables, hiperparámetros, calibrador y umbral; no aprovechar parámetros ajustados con todo train en validaciones internas.
- [ ] Registrar Gini Stability, Gini/AUC por semana, PR-AUC, Brier score, log loss y costes de entrenamiento. Guardar predicciones y configuración para reproducir cada comparación.
- [ ] Probar las ablaciones pendientes con el mismo diseño. Explicar si una diferencia observada es robusta o incierta; evitar elegir variables por una única métrica.

**Salida:** al menos una referencia y dos candidatos comparados, con resultados por periodo y trazabilidad completa, como exige la [guía de modelado](04-modelado.md#criterio-para-avanzar).

### H4. Comprobar los objetivos de minería

| Objetivo | Evidencia mínima para decidir si se cumplió |
| --- | --- |
| **OM1. Predicción estable** | Comparación OOT de Gini Stability frente a la referencia; discriminación por semana y calibración de probabilidades. |
| **OM2. Explicación** | Factores globales y de casos concretos, con comparación entre train y periodo posterior; describir asociaciones, sin atribuir causalidad. |
| **OM3. Menos falsos positivos** | FPR de los modelos al mismo recall acordado; umbrales elegidos sin usar el periodo de evaluación. Evaluación de historial limitado solo si el segmento fue definido válidamente. |
| **OM4. Cambio temporal** | Alertas fechadas de desempeño o población, contrastadas con Gini/AUC, cobertura y faltantes. Una alerta por sí sola no demuestra que reentrenar mejore el modelo. |

**Salida:** decisión de modelo final basada principalmente en Gini Stability OOT y, como criterios secundarios, interpretabilidad, coste y reproducibilidad; límites documentados. Referencia: [objetivos y criterio de selección](../development/01-entendimiento-empresarial.md#13-determinar-los-objetivos-de-la-minería-de-datos).

### H5. Demostrar y cerrar

- [ ] Reentrenar o cargar de forma reproducible el modelo seleccionado y aplicar exactamente su preprocesamiento a nuevos casos.
- [ ] Entregar un prototipo mínimo ejecutable; un script de inferencia es suficiente si valida entrada y produce probabilidades en el formato esperado. Documentar instalación, ejecución y limitaciones.
- [ ] Producir predicciones de prueba y validar claves, nulos y rango de probabilidades. El test local de diez casos comprueba compatibilidad, no desempeño poblacional.
- [ ] Completar informe y resumen ejecutivo con resultados verificados, método, referencias, límites y criterios de éxito cumplidos o incumplidos. Hacer una reproducción de principio a fin por otra persona del equipo.

**Salida:** entregables de las seis fases de CRISP-DM trazables a código, datos y resultados; prototipo académico verificable. Referencia: [guía de implementación](06-implementacion.md#criterio-de-cierre-de-implementación).

## Seguimiento y decisiones

En cada hito, registrar **estado, responsable, fecha, evidencia y bloqueo**. Un hito se marca terminado solo cuando cumple su criterio de salida. Si la corrección de datos o la evaluación cambia el resultado esperado, actualizar el protocolo y los capítulos afectados antes de afirmar cumplimiento de ON1–ON3 u OM1–OM4.

**Primer paso:** completar H1 y abrir las tareas pendientes de H2 usando el registro existente de la [auditoría](auditoria-variables-predictoras/plan-revision.md#registro-de-correcciones).
