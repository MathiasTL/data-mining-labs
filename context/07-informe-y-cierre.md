# 7. Informe final y cierre

## Objetivo

Consolidar el trabajo técnico y las decisiones del proyecto en un informe coherente con el índice entregado por el profesor, respaldado por código, métricas y artefactos reproducibles.

## Índice que debe respetarse

1. Resumen Ejecutivo.
2. Entendimiento empresarial:
   - Determinar los objetivos comerciales.
   - Evaluar la situación.
   - Determinar los objetivos de la minería de datos.
   - Producir el plan del proyecto.
3. Comprensión de los datos:
   - Recopilar datos iniciales.
   - Describir datos.
   - Explorar datos.
   - Verificar la calidad de los datos.
4. Preparación de datos:
   - Seleccionar datos.
   - Datos limpios.
   - Datos de construcción.
   - Integrar datos.
   - Formatear datos.
5. Modelado:
   - Seleccionar técnica de modelado.
   - Generar diseño de prueba.
   - Modelo de construcción.
   - Evaluar el modelo.
6. Evaluación:
   - Evaluar resultados.
   - Proceso de revisión.
   - Determinar los próximos pasos.
7. Implementación:
   - Implementación del plan.
   - Monitoreo y mantenimiento del plan.
   - Elaborar el informe final.
   - Proyecto de revisión.
8. Conclusiones.
9. Recomendaciones.
10. Lecciones aprendidas.
11. Siglario.
12. Referencias.

## Cómo redactar cada parte

- Incluir solo afirmaciones respaldadas por datos, gráficos, código o documentos del proyecto.
- Distinguir resultados medidos, hipótesis y recomendaciones futuras.
- Describir la metodología de forma suficiente para reproducir la preparación, partición, entrenamiento y evaluación.
- Reportar métricas globales y temporales; no presentar únicamente el mejor resultado puntual.
- Justificar exclusiones de datos, agregaciones, técnicas y modelo final.
- Explicitar limitaciones y riesgos: datos de competición, posibles cambios de población, falta de validación externa y carácter demostrativo del prototipo.
- Referenciar la competición, el material CRISP-DM, documentación de librerías y fuentes regulatorias citadas en el caso de negocio.

## Lista de verificación de cierre

### Código y datos

- [ ] El pipeline se ejecuta desde fuentes originales o sus rutas documentadas.
- [ ] Las transformaciones de entrenamiento y prueba son idénticas.
- [ ] Las semillas, librerías y parámetros están registrados.
- [ ] No existen pasos manuales necesarios que no estén documentados.

### Modelo y evaluación

- [ ] Se comparó una línea base y al menos dos modelos candidatos.
- [ ] Se utilizó validación OOT con `WEEK_NUM`.
- [ ] Se reportaron Gini/AUC globales y por periodo.
- [ ] La selección del modelo considera estabilidad, no solo desempeño global.
- [ ] Se documentaron limitaciones, errores y riesgos de fuga de información.

### Entrega

- [ ] El informe respeta el índice solicitado.
- [ ] El prototipo demostrativo funciona y tiene instrucciones de uso.
- [ ] Las predicciones de prueba respetan el formato de `sample_submission.csv`.
- [ ] La presentación es consistente con el informe.
- [ ] El equipo revisó roles, aportes y lecciones aprendidas.

## Resultado esperado

El cierre se logra con un informe final trazable, un modelo seleccionado de forma justificada, evidencia de estabilidad temporal, una demostración funcional y material suficiente para reproducir y sustentar las decisiones del proyecto.
