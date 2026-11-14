# Auditoría de variables predictoras

## Estado

Auditoría iniciada el 13/11/2026. Inventario, revisión focalizada de linaje y reconciliación de vistas completados en sus alcances documentados. Segunda evaluación: 39 controles satisfactorios, tres no ejecutados por ausencia de train y 159 pruebas aprobadas. Elegibilidad semántica completa, reconstrucción de entrenamiento y aceptación para modelado pendientes.

## Archivos

| Archivo                                                           | Contenido                                                                                                                           |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [Plan de revisión](plan-revision.md)                              | Copia de trabajo del plan con checklist. Los avances parciales se registran en los informes, sin dar por cerradas tareas completas. |
| [Primera evaluación](01-inventario-y-hallazgos.md)                | Hallazgos, límites, acciones y siguiente revisión.                                                                                  |
| [Resumen](resumen-inventario.json)                                | Conteos de la ejecución automatizada y fecha UTC.                                                                                   |
| [Inventario de evidencias](inventario-evidencias.csv)             | Rutas, tamaños y SHA-256 de resultados, documentos y configuraciones inventariados.                                                 |
| [Inventario de características](inventario-caracteristicas.csv)   | Copia del catálogo examinado.                                                                                                       |
| [Disponibilidad de matrices](disponibilidad-matrices.csv)         | Presencia local y contraste de hashes E5/E6.                                                                                        |
| [Entradas de manifiestos](verificacion-entradas-manifiestos.csv)  | Correspondencia de entradas declaradas con archivos actuales.                                                                       |
| [Script](auditar_inventario.py)                                   | Reproduce los inventarios sin modificar datos ni entrenar modelos.                                                                  |
| [Linaje y temporalidad](02-linaje-y-reglas-temporales.md)         | Contraste del catálogo con el código y supuestos pendientes.                                                                        |
| [Reconciliación y pruebas](03-reconciliacion-vistas-y-pruebas.md) | Controles ejecutados, límites y prioridades.                                                                                        |
| [Controles de vistas](controles-vistas.csv)                       | Resultado individual de comprobaciones.                                                                                             |
| [Resumen de vistas](resumen-vistas.json)                          | Conteos, matrices examinadas y huellas.                                                                                             |
| [Script de vistas](auditar_vistas.py)                             | Reconciliación del catálogo, especificación y matrices locales.                                                                     |
| [Salida de pruebas](pruebas-segunda-evaluacion.txt)               | Registro de 159 pruebas aprobadas.                                                                                                  |
| [Disponibilidad temporal](04-evidencia-disponibilidad-temporal.md) | Evidencia empírica para applprev_2, person_1/2 y cobertura estática; corrección de A01.                                            |
| [Script de disponibilidad](auditar_disponibilidad.py)             | Genera los cuatro CSV `disponibilidad-*.csv` desde los parquet crudos de train.                                                     |

## Convenciones

- Las rutas dentro de los CSV son relativas a la raíz del repositorio, salvo los campos archivados explícitamente identificados.
- Un control histórico no cuenta como reejecutado.
- Una coincidencia SHA-256 acredita identidad, no ausencia de fuga.
- Las hipótesis se mantienen separadas de los hallazgos verificados.
- Las correcciones de código o datos se efectuarán después de justificar su necesidad; esta primera evaluación no las realiza.

## Siguiente paso

Resolver los supuestos de disponibilidad temporal, completar el linaje documentado y acordar el protocolo de selección dentro de entrenamiento. Ver pendientes y consulta al equipo en la segunda evaluación.
