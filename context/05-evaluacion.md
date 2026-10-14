# 5. Evaluación

## Objetivo de la fase

**Objetivos cubiertos:** OM1 a OM5. Cada resultado se reporta contra su objetivo (ON/OM); la Gini Stability (OM2) es la métrica de la competición y el Gini/AUC global (OM1) la complementa.

Decidir si el modelo satisface el objetivo del proyecto: distinguir riesgo de incumplimiento y mantener un desempeño razonablemente estable en periodos posteriores.

## Tareas

### 5.1 Evaluar desempeño predictivo

1. Comparar candidatos contra la referencia y entre sí con Gini Stability (métrica de la competición) y con Gini/AUC global.
2. Reportar métricas complementarias relevantes para la clase positiva: PR-AUC, recall, precisión y calibración.
3. Analizar distribución de puntuaciones y errores para evitar conclusiones basadas en una sola métrica.
4. Evaluar si la mejora frente al baseline justifica el aumento de complejidad.

### 5.2 Evaluar estabilidad temporal

1. Calcular Gini/AUC para cada semana o bloque de semanas de validación mediante `WEEK_NUM`.
2. Medir variación, tendencia y peor periodo, no solo el promedio global.
3. Comparar la distribución temporal de variables relevantes y, si aporta valor, calcular PSI u otra medida de cambio poblacional.
4. Investigar degradaciones: cambio de población, faltantes, menor cobertura, variables inestables o sobreajuste.
5. Validar que los resultados no se expliquen por fuga de información o cambios en la partición.

### 5.3 Revisión de riesgos y utilidad

1. Revisar variables con posible sesgo, baja trazabilidad o uso temporalmente inválido.
2. Identificar segmentos donde el modelo funcione peor, siempre que las variables disponibles permitan el análisis.
3. Determinar si el modelo requiere calibración, simplificación, más ingeniería de características o un corte temporal distinto.
4. Documentar limitaciones: datos de competición, falta de variables operativas reales y ausencia de validación externa.

### 5.4 Selección del modelo

Antes de seleccionar, verificar la regla de alerta del OM5 sobre las semanas OOT: debe detectar la degradación observada (tamaño de la caída y número de semanas k: *supuesto por acordar*).

El modelo final se selecciona por un balance explícito de:

- desempeño global;
- estabilidad por `WEEK_NUM`;
- diferencia entre entrenamiento y validación;
- interpretabilidad y trazabilidad;
- coste computacional y reproducibilidad.

Si un modelo tiene el mejor AUC global, pero cae marcadamente en periodos posteriores, no debe elegirse sin justificar por qué ese riesgo es aceptable.

## Evidencias y entregable

- Tabla comparativa final de modelos.
- Gráficos ROC/PR y calibración si aplica.
- Tabla y gráfico de Gini/AUC por periodo.
- Análisis de estabilidad, errores, segmentos y limitaciones.
- Decisión documentada del modelo final y próximos pasos.

## Criterio para avanzar

La implementación comienza cuando el equipo haya seleccionado un modelo de forma justificada y haya documentado tanto su rendimiento como sus límites de estabilidad temporal.
