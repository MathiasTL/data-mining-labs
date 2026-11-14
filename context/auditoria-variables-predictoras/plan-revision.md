# Revisión de variables predictoras antes del modelado

## Propósito y uso

Validar y corregir las características utilizadas para estimar `target`, aprovechando la comprensión y preparación existentes. El resultado será una primera versión de características con disponibilidad temporal justificada, construcción reproducible y utilidad evaluada.

Este documento es un plan, no una certificación del estado actual. Marcar cada tarea únicamente cuando exista evidencia; registrar los bloqueos y las limitaciones. No es necesario rehacer todas las fases ni recuperar variables bloqueadas sin justificación.

**Fuente de objetivos:** [Entendimiento empresarial](../../development/01-entendimiento-empresarial.md).

| Objetivo de minería                               | Aporte de la revisión                                                                                                            |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| OM1: probabilidades y evaluación OOT              | Características válidas y comparación predictiva y probabilística.                                                               |
| OM2: interpretación global, individual y temporal | Catálogo, linaje y significado de características.                                                                               |
| OM3: falsos positivos a igual detección           | Datos y protocolo para comparar puntos de operación.                                                                             |
| OM4: cambios y ventanas de reentrenamiento        | Diagnóstico temporal de cobertura, ausencia y distribución. Las alertas y el reentrenamiento requieren experimentos posteriores. |

## 1. Auditar el punto de partida

- [ ] Revisar [Comprensión de datos](../../development/02-comprension-datos.md) y [Preparación de datos](../../development/03-preparacion-datos.md).
- [ ] Revisar `results/preparation/C8_feature_catalog.csv` y las selecciones G2.
- [ ] Revisar `results/preparation/E5_master_tables.csv`, `E6_views.csv`, `F3_final_checks.csv` y los manifiestos.
- [ ] Localizar la especificación de transformaciones de las vistas.
- [ ] Separar características de contenido, indicadores de cobertura y columnas de control.
- [ ] Reconciliar las cifras archivadas: 124 características ensambladas, 234 columnas lineales y 114 para árboles. No son cantidades que deban mantenerse obligatoriamente.
- [ ] Identificar constantes y columnas transformadas descartadas.
- [ ] Verificar qué matrices existen y cuáles deben regenerarse.
- [ ] Comprobar correspondencia entre configuración, catálogo, matrices y manifiestos.
- [ ] Separar reportes vigentes de históricos, especialmente `products-v1` y `products-v2`.

**Cierre:** inventario único de características y versiones, con correspondencia entre fuente, matriz y vista.

## 2. Fijar el contrato de predicción y evaluación

### Predicción

- [ ] Confirmar una solicitud por `case_id` como unidad de predicción.
- [ ] Documentar la definición oficial disponible de `target` y su codificación.
- [ ] Verificar horizonte de incumplimiento y disponibilidad de etiquetas; registrar si no están publicados.
- [ ] Confirmar `date_decision` como referencia adoptada para la información disponible al predecir.
- [ ] Excluir de los predictores las columnas de control del diseño actual: `case_id`, `WEEK_NUM`, `date_decision`, `MONTH` y `target`.
- [ ] Definir cómo identificar historial crediticio limitado si se realizarán conclusiones específicas sobre ese segmento. No equiparar automáticamente ausencia de una fuente con ausencia de historial.

### Evaluación

- [ ] Registrar que el OOT desde la semana 74 ya fue consultado en ablaciones.
- [ ] Definir particiones temporales internas de entrenamiento para seleccionar variables, ajustar modelos, calibradores y umbrales.
- [ ] Mantener separado el ajuste de transformaciones en cada partición interna: ajustar con su entrenamiento, no con toda la matriz preprocesada previamente.
- [ ] Definir el uso restante del OOT y cualquier periodo final reservado, reconociendo qué datos ya se examinaron.
- [ ] Acordar referencia, métricas principales y condiciones de aceptación antes de los siguientes experimentos.

**Cierre:** protocolo temporal documentado que evita utilizar información futura para seleccionar características.

## 3. Revisar la elegibilidad de variables fuente

### Estados

| Estado       | Significado                                                            |
| ------------ | ---------------------------------------------------------------------- |
| Aprobada     | Disponibilidad y construcción justificadas.                            |
| Condicionada | Utilizable con una regla o transformación explícita.                   |
| Bloqueada    | Falta evidencia para justificar su uso.                                |
| Excluida     | Fuga, identificación, fuera de alcance u otro impedimento documentado. |

### Orden de revisión

- [ ] Validar las características actuales de solicitudes previas y personas.
- [ ] Revisar presencia y conteos de fuentes con los mismos controles temporales que sus valores.
- [ ] Reconsiderar variables estáticas y bureau bloqueadas solo cuando exista evidencia adicional de disponibilidad.
- [ ] Revisar productos y registros conservando los filtros temporales corregidos.

### Para cada variable candidata

- [ ] Identificar tabla, significado y definición oficial disponible.
- [ ] Comprobar si el valor estaba disponible al decidir, no solo si existe una fecha anterior en la fila.
- [ ] Revisar posibles actualizaciones posteriores: pagos, mora, estados y saldos.
- [ ] Verificar que no sea identificador, cuasi-identificador o atributo excluido por la política del proyecto.
- [ ] Registrar estado, regla temporal, evidencia y motivo.
- [ ] Usar asociaciones con `target` como señales de investigación, no como prueba automática de fuga o seguridad.

**Cierre:** toda variable candidata tiene decisión y justificación; las bloqueadas no entran al modelo.

## 4. Corregir y completar la construcción

- [ ] Documentar por característica: origen, significado, agregación, ventana, regla temporal y tratamiento de ausencia.
- [ ] Aplicar filtros temporales antes de agregar.
- [ ] Validar relaciones padre-hijo antes de utilizar registros dependientes.
- [ ] Diferenciar ausencia de historial, ausencia de fuente y dato desconocido cuando sea posible.
- [ ] Revisar características redundantes y constantes sin eliminarlas únicamente por correlación.
- [ ] Comprobar vocabularios, categorías nuevas y parámetros aprendidos solo con entrenamiento.
- [ ] Incorporar nuevas agregaciones o ventanas solo con hipótesis de utilidad y fechas suficientes.
- [ ] Actualizar catálogo y configuración junto con cualquier cambio de construcción.
- [ ] Añadir o ajustar pruebas para las reglas modificadas.

**Cierre:** características reproducibles y trazables desde las fuentes hasta la matriz.

## 5. Validar matrices y comportamiento temporal

### Integridad técnica

- [ ] Comprobar una fila por caso y alineación de características, etiquetas y semanas.
- [ ] Verificar particiones sin solapamiento de solicitudes.
- [ ] Comprobar esquema, orden, tipos y ausencia de infinitos.
- [ ] Verificar que ninguna columna reservada sea predictor.
- [ ] Comprobar que transformar OOT o test no cambia el estado del preprocesamiento.
- [ ] Ejecutar controles sobre la versión actual; no dar por suficientes los 29 controles archivados.

### Diagnóstico

- [ ] Medir ausencia, cobertura, constantes, categorías y rangos por característica.
- [ ] Analizar distribución y ausencia por semana, con soporte de casos y positivos.
- [ ] Revisar redundancia entre características.
- [ ] Priorizar cambios ya observados en cobertura de registros fiscales y bureau.
- [ ] No interpretar cambios de cobertura automáticamente como fuga o deterioro predictivo.
- [ ] Utilizar el test local de diez casos solo para compatibilidad, no para inferencia poblacional.

**Cierre:** informe de calidad con incidencias resueltas o limitaciones aceptadas explícitamente.

## 6. Evaluar utilidad con experimentos controlados

La utilidad no se resuelve solo con EDA: requiere entrenar modelos. Estos experimentos forman parte de la selección de características, no equivalen al cierre del modelado.

- [ ] Establecer regresión logística como referencia y un candidato de árboles, con recursos comparables y configuración registrada.
- [ ] Comparar el conjunto actual aprobado con variantes bajo el mismo protocolo temporal.
- [ ] Evaluar ablaciones por grupos, especialmente indicadores de cobertura.
- [ ] Evaluar ampliaciones justificadas de variables fuente.
- [ ] Registrar discriminación y Gini Stability.
- [ ] Registrar Brier score, log loss y curvas de calibración; definir qué significa calibración aceptable.
- [ ] Comparar FPR a niveles acordados y comparables de recall, distinguiendo evaluación de curvas y umbrales fijados con validación.
- [ ] Revisar importancia e interpretaciones globales y de casos de referencia.
- [ ] Analizar dependencia de fuentes cuya cobertura cambia entre periodos.
- [ ] Registrar variabilidad o incertidumbre de las diferencias cuando corresponda.
- [ ] Conservar o retirar grupos según evidencia conjunta, no por una única importancia o correlación.

**Cierre:** selección justificada de una primera versión de características. No se exige que cada variable mejore todas las métricas.

## 7. Consolidar correcciones

- [ ] Actualizar selección de columnas, reglas y pruebas.
- [ ] Regenerar matrices afectadas y ejecutar los controles correspondientes.
- [ ] Actualizar catálogos, esquemas, manifiestos y resultados.
- [ ] Mantener separados los resultados históricos y los corregidos.
- [ ] Actualizar comprensión y preparación cuando cambien sus afirmaciones.
- [ ] Documentar el protocolo y experimentos en [Modelado](../../development/04-modelado.md).
- [ ] Registrar pendientes de interpretación, alertas y reentrenamiento que no queden resueltos por esta revisión.

**Cierre:** código, matrices, resultados y documentación describen la misma versión.

## Registro de decisiones

Añadir filas según se realice la revisión.

| Variable o grupo | Hallazgo | Evidencia | Decisión y motivo | Responsable | Estado    |
| ---------------- | -------- | --------- | ----------------- | ----------- | --------- |
| `applprev_2__*` | Sin fecha propia; conteo compatible con corte en `date_decision` | [04](04-evidencia-disponibilidad-temporal.md), D01 | Condicionada: se usa con el supuesto registrado | Andre (G2) | Decidida |
| `person_1` valores del solicitante, `person_2__*` | Datos del formulario de la solicitud actual | [04](04-evidencia-disponibilidad-temporal.md), D02 | Aprobada para valores; `person_2` entra en ablación de cobertura por su caída en OOT | Andre (G2) | Decidida |
| `person_1__n_contacts__all` | Cuenta canales (`PE`, `CL`, `EM`), no personas; su caída en OOT viene de `CL` y `EM` | [04](04-evidencia-disponibilidad-temporal.md), D02 | Reemplazar por conteo de rol `PE`; motivo: semántica correcta y estabilidad temporal | Andre (G2) | Decidida, por implementar |
| `static_cb_0__n_records`, `static_cb_0__source_present` | Presencia solo varía en semanas 0 a 30: indicador de época | [04](04-evidencia-disponibilidad-temporal.md), D03 | Excluir; motivo: no describe al cliente y degrada la estabilidad | Andre (G2) | Decidida, por implementar |
| `static_0__*`, `other_1__*` | Constantes en train; `other_1` sin fecha y aparece desde semana 31 | [04](04-evidencia-disponibilidad-temporal.md), D03 | Se mantienen fuera del modelo | Andre (G2) | Decidida |

## Registro de correcciones

| Incidencia    | Corrección requerida | Artefactos afectados | Prueba o validación | Responsable | Estado    |
| ------------- | -------------------- | -------------------- | ------------------- | ----------- | --------- |
| `n_contacts` mal definido | Conteo de filas `num_group1 >= 1` con rol `PE` | `families/applications_people.py`, catálogo C8, E4 a E6 | Prueba unitaria por rol; regenerar con el protocolo temporal | Andre | Pendiente |
| Cobertura de `static_cb_0` | Retirar de la selección G2 | Selección G2, catálogo C8, E5 y E6 | Prueba de ausencia en vistas; ablación confirmatoria | Andre | Pendiente |
| Catálogo incompleto | Denominador de `applprev_2` y supuestos de disponibilidad | `C8_feature_catalog.csv` | Revisión del catálogo regenerado | Andre | Pendiente |

## Criterio final de aceptación

- [ ] Para cada característica se puede explicar qué representa, de dónde sale, por qué estaba disponible al predecir y cómo se transforma.
- [ ] Las características seleccionadas tienen una justificación experimental o funcional explícita.
- [ ] No hay incidencias críticas de fuga, alineación o integridad sin resolver.
- [ ] Se documentaron limitaciones y variables bloqueadas, sin presentarlas como verificadas.
- [ ] Existe un protocolo temporal que separa selección, ajuste y evaluación.
- [ ] Las ampliaciones de alcance se identificaron explícitamente: detectar cambios no demuestra por sí solo la necesidad ni la utilidad de reentrenar.

**Orden recomendado:** inventario → contrato → elegibilidad → construcción → calidad → experimentos → consolidación.
