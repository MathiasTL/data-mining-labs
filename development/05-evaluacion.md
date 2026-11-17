# 5. Evaluación

> **Estado (14/11/2026):** evaluación del modelo final sobre el OOT (semanas 74 a 91), consultado **una sola vez** sobre la versión v2. Fuente: `results/modeling/v2-run1`. Se cumplen OM1 en discriminación, OM2 y OM3; **no se cumplen** la calibración de OM1 ni la ventaja en el segmento de historial limitado de ON2; OM4 cumple solo con el detector de calibración agregado en esta evaluación.

## 5.1 Evaluar resultados

### Comparación global

| Modelo | Gini Stability | ROC-AUC | PR-AUC | Brier | Brier de la tasa base | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Constante | 0,000 | 0,500 | 0,022 | 0,02152 | 0,02152 | 0,1092 |
| Regresión logística | 0,438 | 0,731 | 0,062 | 0,02136 | 0,02152 | 0,1013 |
| **LightGBM** | **0,484** | **0,752** | **0,075** | 0,02113 | 0,02152 | 0,0993 |

OOT: 194 658 casos, 4 260 incumplimientos (2,19 %). El Brier y el log loss corresponden a la probabilidad calibrada.

LightGBM supera a la regresión logística en Gini Stability (+0,046) y **gana en las 18 semanas del OOT**. Ambos superan ampliamente a la predicción constante. Con esto se cumple el criterio de OM1 en discriminación y, por extensión, el de ON1: el modelo ordena el riesgo mejor que una referencia simple en un periodo posterior al entrenamiento.

### Estabilidad temporal

| Semana | Casos | Tasa de incumplimiento | Gini, regresión logística | Gini, LightGBM |
|---:|---:|---:|---:|---:|
| 74 | 11 280 | 2,7 % | 0,470 | 0,509 |
| 75 | 13 656 | 2,1 % | 0,436 | 0,502 |
| 76 | 10 261 | 2,5 % | 0,492 | 0,534 |
| 77 | 7 754 | 2,5 % | 0,412 | **0,437** |
| 78 | 8 755 | 2,2 % | 0,490 | 0,532 |
| 79 | 5 158 | 2,2 % | 0,509 | 0,535 |
| 80 | 6 246 | 1,9 % | 0,527 | 0,569 |
| 81 | 6 743 | 2,8 % | **0,374** | 0,439 |
| 82 | 5 423 | 2,2 % | 0,377 | 0,443 |
| 83 | 7 295 | 2,1 % | 0,451 | 0,523 |
| 84 | 9 721 | 2,2 % | 0,436 | 0,459 |
| 85 | 16 239 | 1,9 % | 0,441 | 0,485 |
| 86 | 15 630 | 1,8 % | 0,511 | 0,515 |
| 87 | 17 886 | 1,8 % | 0,493 | 0,517 |
| 88 | 14 234 | 1,9 % | 0,442 | 0,472 |
| 89 | 13 600 | 2,3 % | 0,495 | 0,531 |
| 90 | 12 103 | 2,7 % | 0,451 | 0,520 |
| 91 | 12 674 | 2,3 % | 0,466 | 0,523 |

En LightGBM, el Gini semanal medio es 0,503 y la peor semana es la 77 (0,437). No hay tendencia descendente: el Gini Stability supera al Gini semanal medio menos su variabilidad, sin penalización por pendiente. La regresión logística cae más en sus peores semanas (81 y 82, 0,374 y 0,377).

### Calibración (OM1): no se cumple

| Modelo | ECE | Predicho / observado | Criterio |
|---|---:|---:|---|
| Regresión logística | 0,0143 | 1,65 | ECE < 0,01 y desvío < 20 % |
| LightGBM | 0,0136 | 1,62 | ECE < 0,01 y desvío < 20 % |
| LightGBM, reajuste 0 a 73 | 0,0058 | 1,26 | Cumple el ECE, no el desvío |

El modelo **sobreestima el riesgo en un 62 %**. El Brier apenas mejora al de predecir la tasa base. La causa es temporal: la tasa de incumplimiento observada cae de 4,8 % (semanas 60 a 63) a 2,2 % desde la semana 66, mientras el calibrador isotónico se ajustó con todas las semanas 60 a 73 mezcladas.

Bellotti y Crook [3] comparan tasas de incumplimiento estimadas y observadas en meses posteriores, un antecedente para examinar la escala del riesgo por periodo. Su modelo de supervivencia con cuentas de tarjetas no es equivalente al clasificador utilizado aquí.

| Semanas | Tasa observada | Riesgo predicho, LightGBM calibrado | Predicho / observado |
|---|---:|---:|---:|
| 60 a 63 | 4,8 % | 4,0 % | 0,83 |
| 64 a 65 | 2,9 % | 3,6 % | 1,25 |
| 66 a 73 | 2,2 % | 3,4 % | 1,52 |
| 74 a 91 (OOT) | 2,2 % | 3,5 % | 1,62 |

El orden del riesgo se mantiene y la escala no. Para aprobar o rechazar con umbrales relativos (OM3) esto no es un problema. Para usar la probabilidad como valor absoluto, por ejemplo en provisiones, el modelo **no es utilizable sin recalibrar**.

**Hipótesis, no verificada:** madurez de la etiqueta. Las solicitudes más recientes habrían tenido menos tiempo para caer en mora antes del corte del dataset. El horizonte de `target` no está publicado, así que no se puede confirmar ni descartar.

### Falsos positivos a igual detección (OM3 y ON2)

Umbrales fijados en la validación de P3 y aplicados sin cambios al OOT. Hay 190 398 buenos pagadores en el OOT.

| Recall objetivo | Recall logrado, regresión logística | FPR, regresión logística | Recall logrado, LightGBM | FPR, LightGBM | Buenos pagadores rechazados de menos |
|---:|---:|---:|---:|---:|---:|
| 70 % | 71,0 % | 37,1 % | 71,5 % | 34,0 % | ~5 800 |
| 80 % | 81,0 % | 49,5 % | 80,7 % | **45,4 %** | **~7 700** |
| 90 % | 89,6 % | 66,0 % | 90,4 % | 63,1 % | ~5 600 |

Con el umbral fijo, ambos modelos alcanzan en el OOT el recall pedido; la caída de la tasa base no lo afecta, porque los umbrales actúan sobre el orden. LightGBM rechaza menos buenos pagadores en los tres niveles. **OM3 se cumple** y el criterio general de ON2, menos solventes rechazados a igual detección, también.

### Segmento sin historial observado (ON2): no se cumple

Definición operativa: sin solicitud previa válida y sin registros de bureau A1 ni B1. Es un indicador de historial no observado por estas fuentes, no prueba ausencia de historial crediticio.

| Segmento OOT | Casos | Incumplimientos | AUC, regresión logística | AUC, LightGBM | FPR a recall 80 % (curva del segmento), regresión logística | FPR a recall 80 % (curva del segmento), LightGBM |
|---|---:|---:|---:|---:|---:|---:|
| Sin historial observado | 11 223 | 234 | **0,709** | 0,707 | **51,5 %** | 53,6 % |
| Resto | 183 435 | 4 026 | 0,735 | **0,756** | 47,8 % | **43,5 %** |

En el segmento que ON2 menciona explícitamente, **LightGBM no supera a la regresión logística**: toda su ventaja proviene de los clientes con historial. Con 234 incumplimientos la diferencia es pequeña frente a la incertidumbre, pero no hay evidencia de mejora. Además, el segmento se reduce de 11,6 % de los casos en desarrollo a 5,8 % en el OOT.

### Interpretación (OM2)

- El ranking SHAP de LightGBM es **estable**. Con cinco muestras independientes, la correlación de Spearman mínima es 0,9997 y las 20 variables principales coinciden en todas, tanto en P3 como en el OOT. Esto responde a la preocupación de Chen et al. [5] sobre explicaciones inestables con clases desbalanceadas.
- Las cinco variables principales son las mismas en P3 y OOT: tipo de ingreso, edad, plazo medio de solicitudes previas, educación registrada en solicitudes previas y registros del registro fiscal C.
- Los cambios de importancia entre periodos se concentran en `person_2`, que sube del puesto 28 al 11, coherente con la caída de su presencia.

OM2 se cumple: el ranking es reproducible y los cambios entre periodos están identificados. Las importancias describen asociación, no causalidad.

### Detección de cambios (OM4)

| Detector | Resultado | Lectura |
|---|---|---|
| PSI semanal del score | Máximo 0,04, sin alertas | La distribución del score no cambia |
| ADWIN sobre log loss semanal | Sin alertas | Con 32 puntos semanales tiene poco poder [10] |
| PSI de variables | Idioma y educación de `person_1` con PSI de 1,6 a 2,9 desde la semana 79 | Cambio fuerte de población **sin pérdida de Gini** |
| Fuente `tax_registry_b_1` | Ausente en las semanas 0 a 59, presente desde la 60 | Cambio de fuentes fechado; el modelo de P3 no puede usarla |
| **Calibración semanal** (agregado en esta evaluación) | Primera alerta en la **semana 66**; 26 de 32 semanas en alerta | Detecta el cambio de tasa base que los otros detectores no ven |

**Regla de la alerta de calibración:** la razón entre riesgo predicho y tasa observada sale del intervalo [0,8; 1,25] dos semanas seguidas. Se calculó sobre las predicciones ya congeladas, sin reentrenar ni elegir nada, así que no constituye una nueva selección con el OOT. Requiere etiquetas, por lo que en operación tendría el retraso del horizonte de `target`.

**Resultado de OM4.** Los detectores definidos antes de la corrida (PSI del score y ADWIN) **no** produjeron alertas, aunque hubo un cambio real. El detector de calibración sí lo ubica en la semana 66, contrastado con el desempeño: la discriminación se mantiene y la escala se rompe. La ventana de reentrenamiento que sugiere la evidencia es **recalibrar con semanas posteriores a la 65**, no reentrenar el ranking.

## 5.2 Proceso de revisión

### Controles efectuados

- [x] Se revisó la fuga de información temporal: auditoría de variables, protocolo, tests de no lectura del OOT y revisión independiente del código.
- [x] Se revisó la unicidad del caso después de las integraciones: una fila por `case_id` en E5-v2, con 29/29 controles de integración.
- [x] Se verificó que todas las métricas reportadas como evidencia corresponden a datos no usados para entrenar. Las métricas de P3 están marcadas `in_sample` porque esa validación eligió calibrador y umbrales.
- [x] Se revisaron las variables importantes y su coherencia temporal: estabilidad SHAP y cambios por periodo.
- [x] Se evaluaron las limitaciones y los riesgos de sesgo con la información disponible: segmento sin historial y exclusión de sexo y género desde G2.

### Cambios tras la revisión

1. Los resultados del avance 2 se declararon históricos: vistas contaminadas y OOT ya consultado.
2. `person_1__n_contacts__all` se reemplazó por el conteo de personas de contacto, y la cobertura de `static_cb_0` se excluyó por funcionar como indicador de época.
3. La selección pasó de Gini Stability por partición a Gini Stability fuera de partición, antes de consultar el OOT.
4. La revisión independiente obligó a aplicar las pruebas de quitar variables en la etapa final, a corregir el ECE ante empates de la calibración isotónica y a impedir reconsultas del OOT sin registro.
5. Se agregó el detector de calibración semanal después de ver que los detectores previstos no capturaban el cambio. Queda declarado como agregado posterior.

## 5.3 Determinar los próximos pasos

### Modelo seleccionado

**LightGBM** (63 hojas, 300 árboles, tasa de aprendizaje 0,05, semilla 42), ajustado con las semanas 0 a 59 sobre la vista de árboles de `E5-v2`, con 110 predictores efectivos, calibración isotónica y umbrales de la validación 60 a 73.

- **Motivo:** mayor Gini Stability OOT (criterio principal, sección 1.3) y ventaja en todas las semanas.
- **Criterios secundarios:**
  - Ajuste en 17 segundos.
  - Explicaciones estables.
  - Reproducción con un comando por etapa.

Su probabilidad **no debe usarse como valor absoluto** hasta recalibrarla.

### Cumplimiento de objetivos

| Objetivo | Estado | Evidencia |
|---|---|---|
| OM1, discriminación | Cumplido | Gini Stability 0,484 frente a 0,438; 18 de 18 semanas |
| OM1, calibración | No cumplido | Predicho / observado 1,62; ECE 0,0136 |
| OM2 | Cumplido | SHAP estable; cambios por periodo identificados |
| OM3 | Cumplido | −4,1 puntos de FPR a recall 80 % |
| OM4 | Parcial | Solo el detector de calibración, agregado después, ubica el cambio |
| ON1 | Cumplido en orden de riesgo | Vía OM1 |
| ON2 | Parcial | Cumple en la población total; no en el segmento sin historial |
| ON3 | Parcial | Se detecta cuándo y por qué cambia la confiabilidad (escala, no orden), con un detector posterior |

### Limitaciones

- **OOT ya examinado** en el avance 2; es la única evaluación fuera de tiempo y no existe un periodo intacto.
- **Horizonte de `target` no publicado:** la hipótesis de madurez de etiqueta no se puede verificar.
- **Segmento sin historial pequeño** (234 incumplimientos en OOT) y definido por ausencia de fuentes, no por historial real.
- **Una sola semilla de entrenamiento**; la estabilidad se midió para SHAP, no para el modelo.
- **Detector de calibración diseñado después de ver el resultado:** su validez se demuestra en la semana 66, que está dentro de desarrollo, pero su umbral no se fijó de antemano.

### Mejoras futuras

1. **Recalibrar** con las semanas posteriores a la 65 y validar el resultado en un periodo nuevo.
2. **Incorporar la alerta de calibración semanal** al monitoreo, con umbral fijado de antemano.
3. **Modelos específicos o variables alternativas** para el segmento sin historial (ON2).
4. **Verificar con el proveedor el horizonte de `target`** para confirmar o descartar la madurez de etiqueta.
5. **Repetir con varias semillas** y estimar intervalos de las diferencias entre modelos.
