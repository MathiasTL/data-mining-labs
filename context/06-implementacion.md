# 6. Implementación

## Objetivo de la fase

Entregar un prototipo académico funcional que aplique el pipeline seleccionado, genere probabilidades de riesgo y demuestre cómo se utilizaría el modelo sin conectarse a sistemas financieros reales.

## Tareas

### 6.1 Preparar el artefacto final

1. Reentrenar el modelo seleccionado siguiendo el protocolo aprobado.
2. Guardar o hacer regenerables el modelo, las transformaciones y el orden de las variables.
3. Aplicar el mismo pipeline al conjunto de prueba.
4. Generar predicciones de probabilidad en el formato definido por `../data/sample_submission.csv`.
5. Validar que la salida no tenga claves duplicadas, valores faltantes ni probabilidades fuera del rango válido.

### 6.2 Construir el prototipo demostrativo

Elegir una opción viable dentro del plazo:

- **script:** recibe datos preprocesados o un archivo y devuelve predicciones;
- **API:** expone una operación de predicción con entrada validada;
- **dashboard:** permite cargar o seleccionar una observación y visualizar probabilidad y factores principales.

El prototipo debe dejar claro que es demostrativo. No debe presentar decisiones de aprobación/rechazo como automáticas ni usar datos reales externos.

### 6.3 Proponer monitoreo y mantenimiento

Documentar una propuesta para un escenario real, sin implementarlo como producción:

- seguimiento de Gini/AUC por semana;
- distribución de puntuaciones y de variables;
- porcentaje de faltantes;
- PSI o indicadores de *drift*;
- tasa observada de incumplimiento cuando esté disponible;
- reglas para investigar, recalibrar o reentrenar.

## Evidencias y entregable

- Pipeline de inferencia reproducible.
- Modelo final y configuración.
- Archivo de predicciones compatible con la muestra de la competición.
- Prototipo demostrativo ejecutable.
- Manual breve de instalación y uso.
- Propuesta de monitoreo, mantenimiento y limitaciones.

## Criterio de cierre de implementación

Cualquier integrante debe poder seguir las instrucciones y reproducir la predicción o demostración en un entorno preparado, sin pasos manuales no documentados.
