# Primera evaluación: inventario y correspondencia de evidencias

Fecha: 13/11/2026. Estado: auditoría iniciada; no autoriza todavía el modelado formal.

## Alcance ejecutado

Revisión programática de archivos locales mediante Python estándar: inventario y SHA-256 de evidencias, lectura del catálogo, disponibilidad y hashes de matrices E5/E6, contraste de entradas de dos manifiestos y lectura de la especificación de vistas.

No se regeneraron matrices, no se entrenaron modelos y no se ejecutaron nuevamente los 29 controles de integración. La coincidencia binaria de una matriz con su hash archivado acredita identidad de archivo, no corrección semántica ni ausencia de fuga.

## Checklist de esta evaluación

- [x] Inventariar archivos de resultados, documentos de desarrollo y configuraciones: 194 archivos con ruta, tamaño y SHA-256.
- [x] Copiar el catálogo vigente a un inventario de auditoría: 124 características.
- [x] Contar roles: 94 predictores de contenido y 30 características de cobertura.
- [x] Verificar unicidad de nombres: sin duplicados en el catálogo.
- [x] Revisar completitud de campos del catálogo.
- [x] Contrastar las 12 entradas declaradas entre `E7_manifest.json` y `F3_final_run_manifest.json`: todas coinciden con los archivos locales. Son 12 comparaciones entre dos manifiestos, no necesariamente 12 archivos distintos.
- [x] Comprobar disponibilidad y SHA-256 de nueve matrices declaradas en E5/E6.
- [x] Localizar `data/artifacts/preparation/E6-v1/matrix_views_spec.csv`.
- [ ] Revisar semánticamente cada regla temporal y cada característica.
- [ ] Verificar relaciones completas entre fuente, característica y columnas transformadas.
- [ ] Reejecutar controles técnicos sobre matrices regeneradas de entrenamiento.
- [ ] Aprobar el protocolo temporal antes de nuevos experimentos.

## Hallazgos

### A01. Las matrices de entrenamiento no están en las rutas locales esperadas

**Prioridad:** alta para reproducir y entrenar. **Estado:** pendiente.

No están disponibles en las rutas declaradas, adaptadas a este repositorio:

- `data/artifacts/preparation/E5-v1/feature_matrix__train.parquet`.
- `data/artifacts/preparation/E6-v1/linear__train.parquet`.
- `data/artifacts/preparation/E6-v1/tree__train.parquet`.

Esto es consistente con la documentación que indica que los Parquet grandes no se versionan. No se ha probado todavía si existen copias en otra ubicación ni si están todas las entradas necesarias para regenerarlos.

**Acción:** verificar fuentes y espacio/recursos antes de regenerar mediante el procedimiento existente. No modificar ni sustituir evidencias históricas inadvertidamente.

### A02. Las seis matrices de OOT y test coinciden con la ejecución archivada

**Prioridad:** informativa. **Estado:** identidad de archivos verificada.

Las matrices maestra, lineal y de árboles de OOT y test están presentes. Sus seis SHA-256 coinciden con E5/E6.

**Evidencia:** `disponibilidad-matrices.csv`.

**Límite:** todavía no se inspeccionó su contenido tabular ni su validez temporal en esta auditoría.

### A03. El catálogo contiene 33 campos `source_columns` vacíos

**Prioridad:** media; revisión de linaje. **Estado:** pendiente de clasificación.

- 30 corresponden a cobertura.
- Tres corresponden a predictores estructurales:
  - `applprev_2__children_per_parent_max__all`.
  - `applprev_2__children_per_parent_mean__all`.
  - `person_1__n_contacts__all`.

No son automáticamente defectos: los conteos pueden derivarse de filas y claves, no de una variable de medida. Debe comprobarse que las claves, filtros y población contada estén documentados. Si no lo están, completar el linaje o precisar la convención del catálogo.

Los campos de tabla fuente, agregación, ventana, regla temporal y descripción no presentan vacíos. Esto verifica completitud textual, no suficiencia de las explicaciones.

### A04. La especificación permite comenzar la reconciliación de vistas

**Prioridad:** media. **Estado:** conteo verificado; mapeo pendiente.

`matrix_views_spec.csv` contiene 411 filas:

- 348 salidas con rol predictor, suma de 234 lineales y 114 de árboles según los reportes.
- 53 entradas con rol `dropped`.
- 10 entradas de control entre las dos vistas.

La igualdad de totales es un primer control, no demuestra todavía que todas las transformaciones y descartes estén correctamente relacionados con el catálogo.

**Acción:** revisar por vista unicidad de salidas, características de origen, constantes descartadas y parámetros de ajuste.

### A05. Los 29 controles disponibles son históricos

**Prioridad:** alta para la aceptación final. **Estado:** pendientes de reejecución.

`F3_final_checks.csv` contiene 29 estados `ok`. La auditoría verificó su existencia y contenido, pero no repitió esos controles. Deben distinguirse de las verificaciones ejecutadas en esta carpeta.

## Decisiones de esta primera evaluación

1. Conservar los artefactos y reportes originales sin cambios.
2. No declarar características seguras por tener campos completos o asociaciones débiles con `target`.
3. No regenerar matrices todavía: comprobar antes insumos, protocolo y reglas que podrían requerir corrección.
4. Continuar con el contrato de predicción y la elegibilidad temporal; priorizar linaje de conteos y cambios de cobertura.

## Próxima evaluación

- [ ] Leer y contrastar objetivos actuales, contratos y decisiones G2 con el código de construcción.
- [ ] Auditar por vista el mapeo del catálogo a las columnas transformadas.
- [ ] Revisar las tres características estructurales y la convención de linaje de cobertura.
- [ ] Verificar entradas necesarias para regenerar entrenamiento sin ejecutar aún el proceso completo.
- [ ] Registrar un protocolo para selección de variables dentro de entrenamiento, reconociendo el uso previo de OOT.

## Reproducción

Desde la raíz del repositorio:

```sh
python3 context/auditoria-variables-predictoras/auditar_inventario.py
```

El comando escribe solo los inventarios y el resumen de esta carpeta. Lee archivos de resultados, documentos, configuraciones y matrices declaradas; no altera las fuentes. Al repetirlo actualiza la fecha y las evidencias de este primer inventario, por lo que se debe conservar una copia si se requiere comparar ejecuciones.
