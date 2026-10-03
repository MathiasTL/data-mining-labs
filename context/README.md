# Mapa de avance del proyecto

Esta carpeta contiene la guía operativa del proyecto. Cada archivo corresponde a una fase de CRISP-DM y define qué hacer, qué dejar documentado y cuándo se puede avanzar. Debe leerse junto con los documentos vigentes del equipo en `../docs/us/`.

## Orden de trabajo

| Orden | Guía | Resultado que habilita |
|---:|---|---|
| 1 | [Entendimiento empresarial](01-entendimiento-empresarial.md) | Problema, objetivos, métricas, alcance y plan confirmados. |
| 2 | [Comprensión de datos](02-comprension-datos.md) | Inventario, EDA, calidad y mapa de granularidades. |
| 3 | [Preparación de datos](03-preparacion-datos.md) | Tabla maestra y pipeline reproducible sin fuga de información. |
| 4 | [Modelado](04-modelado.md) | Modelos comparados con validación fuera de tiempo. |
| 5 | [Evaluación](05-evaluacion.md) | Modelo seleccionado por desempeño y estabilidad temporal. |
| 6 | [Implementación](06-implementacion.md) | Predicciones, prototipo demostrativo y propuesta de monitoreo. |
| 7 | [Informe y cierre](07-informe-y-cierre.md) | Informe final y evidencias de reproducibilidad listos para entrega. |

## Reglas transversales

- La unidad final de modelado debe tener **una fila por caso** de la tabla base.
- Las tablas con múltiples registros por caso deben agregarse antes de unirse a la tabla base, salvo que se justifique otra cardinalidad.
- Toda variable debe estar disponible al momento de la predicción; está prohibido usar información futura o derivada del resultado.
- La validación temporal por `WEEK_NUM` es obligatoria. Una partición aleatoria puede servir como diagnóstico, pero no reemplaza la validación fuera de tiempo.
- Cada experimento debe registrar versión de datos, variables, partición, modelo, hiperparámetros, semilla, métricas y fecha.
- Los documentos de `../docs/` fuera de `../docs/us/` son material del profesor y no se modifican.

## Hitos

| Fase | Fecha límite |
|---|---:|
| Entendimiento empresarial | 03/10/2026 |
| Comprensión de datos | 24/10/2026 |
| Preparación de datos | 14/11/2026 |
| Modelado | 28/11/2026 |
| Evaluación | 05/12/2026 |
| Implementación, informe y presentación | 19/12/2026 |
