# 6. Implementación

> El alcance es un prototipo académico. Esta sección no debe afirmar que existe un despliegue financiero real.

## 6.1 Implementación del plan

### Artefactos entregados

| Artefacto | Ubicación/comando | Descripción | Estado |
|---|---|---|---|
| Pipeline de preparación | <!-- Completar --> | Genera características desde fuentes originales | Pendiente |
| Modelo final | <!-- Completar --> | Modelo entrenado y configuración | Pendiente |
| Predicciones de prueba | <!-- Completar --> | Archivo compatible con sample_submission | Pendiente |
| Prototipo demostrativo | <!-- Completar --> | Script, API o dashboard | Pendiente |

### Generación de predicciones

<!-- Documentar comando, validaciones de salida y conformidad con sample_submission.csv. -->

### Uso del prototipo

<!-- Añadir instrucciones breves: requisitos, entrada, comando para ejecutar, salida y limitaciones. -->

## 6.2 Monitoreo y mantenimiento del plan

En un escenario real se propone monitorear:

| Indicador | Frecuencia propuesta | Señal de alerta | Acción sugerida |
|---|---|---|---|
| Gini semanal y su tendencia (OM5) | Semanal | Caída del Gini semanal mayor que un umbral durante k semanas consecutivas, o tendencia negativa sostenida (umbral y k: *supuesto por acordar*) | Investigación/recalibración/reentrenamiento |
| Distribución de variables / PSI | <!-- Completar --> | <!-- Completar --> | Revisar drift y calidad de datos |
| Porcentaje de faltantes | <!-- Completar --> | <!-- Completar --> | Revisar fuente y pipeline |
| Calibración / tasa de eventos | <!-- Completar --> | <!-- Completar --> | Recalibrar o revisar modelo |

## 6.3 Elaborar el informe final

<!-- Registrar estado de figuras, tablas, referencias y anexos que sustentan este informe. -->

## 6.4 Proyecto de revisión

### Reproducibilidad

- [ ] Pipeline ejecutado desde las fuentes documentadas.
- [ ] Dependencias y versiones registradas.
- [ ] Semillas y parámetros disponibles.
- [ ] Modelo/predicciones generados sin pasos manuales ocultos.
- [ ] Prototipo probado por al menos un integrante distinto de quien lo construyó.

### Resultado de revisión

<!-- Completar con fecha, responsables, incidencias y resolución. -->
