# 5. Evaluación

> **Estado:** esta sección decide el modelo final. Debe distinguir rendimiento global de estabilidad temporal.

## 5.1 Evaluar resultados

### Comparación global

<!-- Insertar tabla final de modelos y explicación de su desempeño frente a la línea base. -->

### Estabilidad temporal

| Modelo | Periodo (`WEEK_NUM`) | Gini/AUC | Tamaño | Tasa de evento | Observación |
|---|---|---:|---:|---:|---|
| <!-- Completar --> |  |  |  |  |  |

<!-- Insertar gráfico temporal, tendencia, variabilidad, peor periodo y posibles causas de degradación. -->

### Análisis complementario

<!-- Documentar calibración, segmentos, drift/PSI, errores y hallazgos relevantes si fueron implementados. -->

## 5.2 Proceso de revisión

### Controles efectuados

- [ ] Se revisó fuga de información temporal.
- [ ] Se revisó unicidad del caso después de las integraciones.
- [ ] Se verificó que todas las métricas corresponden a validación no utilizada para entrenar.
- [ ] Se revisaron variables importantes y su coherencia temporal.
- [ ] Se evaluaron limitaciones y riesgos de sesgo con la información disponible.

### Cambios tras la revisión

<!-- Registrar correcciones, experimentos descartados o cambios de decisión. -->

## 5.3 Determinar los próximos pasos

### Modelo seleccionado

<!-- Indicar nombre, versión, variables, parámetros y razón de selección. -->

### Limitaciones

<!-- Documentar límites de datos, estabilidad, interpretabilidad, cómputo y generalización. -->

### Mejoras futuras

<!-- Priorizar acciones: variables, calibración, ensambles, monitoreo o reentrenamiento. -->
