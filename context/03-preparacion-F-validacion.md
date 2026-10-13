# Épica F — Validación, documentación y cierre

> **Estado (13/10/2026):** E4 y E5 están cerradas; E6 y E7 siguen pendientes. F1 y F2 tienen pruebas; F3, F4, F5 y F7 tienen entregables adelantados, pero su cierre formal requiere contrastarlos con E6/E7 y con los criterios del backlog. F6 tiene una revisión de Andre y una revisión adicional de integración/documentación (abajo); faltan las revisiones asignadas al resto del equipo. G4 sigue pendiente.

## Resumen

| Tarea                                       | Estado                      | Evidencia                                                                                                                                                                                          |
| ------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F1** Pruebas de adaptadores y catálogos   | Hecho                       | Auditoría de las cinco familias; 12 pruebas nuevas en `tests/test_preparation_f1_gaps.py` (una marcada como fallo esperado, ver abajo)                                                             |
| **F2** Pruebas end-to-end sintéticas        | Hecho                       | `tests/test_preparation_g3_integration.py` comprueba partición, descartes y agregados contra valores conocidos, join, ajuste/transformación y manifiestos de familias y matrices                   |
| **F3** Validaciones sobre artefactos reales | Adelantado, pendiente de E7 | 26/26 sobre E4 y 29/29 sobre E5 regenerado localmente; train E5 original ausente aquí, E7 pendiente                                                                                                |
| **F4** Resultados por fuente                | Evidencia parcial           | `results/preparation/family_results.csv` reúne casos, cobertura y roles por familia y split; no incluye comando ni fecha por fuente como exige el backlog                                          |
| **F5** Capítulo del informe                 | Borrador adelantado         | `development/03-preparacion-datos.md` tiene 3.1–3.5, pero describe vistas de E6 aún no ejecutadas y depende del cierre de F3, F4 y F7                                                              |
| **F6** Revisión cruzada                     | Parcial                     | Revisión de Andre al frente de bureau y revisión adicional de integración/documentación (abajo); faltan las revisiones del resto del equipo                                                        |
| **F7** Informe de fuga temporal             | Evidencia parcial           | `results/preparation/temporal_leakage_report.csv` contiene métricas de exclusiones por familia y split; falta explicitar reglas de disponibilidad, columnas excluidas y evidencia según el backlog |

F3 se ejecutó primero sobre las piezas publicadas de E4 (26/26). Ahora `validation.py --matrix-dir` coteja los bytes, manifiestos y contenido de las matrices E5 publicadas contra las matrices ensambladas desde E4 (29/29). Como train E5 no estaba versionado, se regeneró antes de ejecutar la auditoría real:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e5 --report data/artifacts/preparation/E5-v1/F3_rebuild_report.csv
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.validation --matrix-dir data/artifacts/preparation/E5-v1 --out data/artifacts/preparation/F3-checks
```

La ejecución pasó el 13/10/2026: 29 controles correctos, cero fallidos; `data/artifacts/preparation/F3-checks/integration_checks.csv` registra los detalles sin sobrescribir el informe anterior. **Alcance:** se compararon matrices E5 regeneradas localmente con E4 local; los bytes resultantes difieren de los checksums E5 versionados (dependen del entorno) y aquí no existe el Parquet train E5 original. Después de medir se restauraron los artefactos versionados y se retiró train regenerado; OOT y test originales sí coinciden con sus propios manifiestos. Para cerrar F3 falta validar los artefactos finales de E7, incluido el estado ajustado y las vistas, cuando existan.

## F1 — Huecos cubiertos

La auditoría revisó, por familia, casos temporales, nulos, claves y agregaciones calculadas a mano. Los huecos se cubrieron en un archivo propio, sin editar las pruebas de cada responsable:

| Familia                | Caso nuevo                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| Solicitudes y personas | Columna toda nula, tipo físico `Null`, `NaN` en un importe y `case_id` nulo en la fuente |
| Profundidad 0          | Valor nulo en la fuente con `first` y `case_id` `NaN`                                    |
| Registros fiscales     | Claves repetidas, nulas y `NaN`, y conteos calculados a mano                             |
| Bureau                 | Ancla igual a la fecha de decisión (se conserva) y clave repetida                        |
| Catálogo C8            | Consolidación de tres familias con conteo literal y esquema distinto rechazado           |

**Error encontrado:** bureau no valida claves. Un `(case_id, num_group1)` repetido se cuenta dos veces (`n_records` pasa de 2 a 3). La prueba queda como fallo esperado estricto (`xfail`), así que avisará en cuanto se corrija. Los datos reales no se ven afectados: el capítulo 02 verificó que esas claves son únicas.

## F6 — Revisión de Andre al frente de bureau (Mathias)

**Lo que está bien:** el adaptador lee por grupos de filas con memoria acotada, fusiona parciales con una fórmula estable de varianza, valida el padre antes que el hijo y publica artefactos con manifiesto y suma de verificación. Sus agregados están probados contra valores calculados a mano.

**Hallazgo principal: el ancla de existencia descarta la mayoría de las filas.**

| Tabla               | Filas fuente (capítulo 02) |                                         Excluidas en train + OOT | Porcentaje |
| ------------------- | -------------------------: | ---------------------------------------------------------------: | ---------: |
| `credit_bureau_a_1` |                 15 940 537 | 10 193 265 (8 360 380 sin ancla y 1 832 885 con ancla posterior) |     63,9 % |
| `credit_bureau_b_1` |                     85 791 |                                                           76 645 |     89,3 % |
| `credit_bureau_b_2` |                  1 286 755 |                        1 166 558 (casi todas por padre inválido) |     90,7 % |

La regla usa `lastupdate_*`, la fecha de **última actualización** del registro, como prueba de que existía al decidir. Es conservadora y evita la fuga de valores actualizados después de la decisión, así que es correcta para las características de valor. Para la **cobertura**, en cambio, cambia el significado: `source_present` pasa a ser "tiene registros actualizados antes de la decisión", no "tiene historial en el bureau". Por eso la cobertura de bureau A cae del 90,8 % bruto al 71,0 % en train, y sube al 87,6 % en OOT, donde las actualizaciones posteriores quedan más lejos.

**Recomendación (no bloquea G4):** para los indicadores de cobertura, usar la fecha de inicio del contrato (`dateofcredstart_739D` en A, `contractdate_551D` en B) como prueba de existencia, y mantener `lastupdate_*` para las características de valor cuando G2 apruebe alguna. Mientras bureau solo aporte cobertura, la diferencia afecta a la estabilidad entre train y OOT.

**Otros puntos:**

- La clave repetida de F1.
- `credit_bureau_a_2` se omite por falta de ancla aprobada; queda documentado en C4-C5.

## F6 — Revisión adicional de integración y documentación (13/10/2026)

Revisión independiente del ensamblado E5 (`e5.py`, `assembly.py`, `schema.py`, `tests/test_preparation_e5.py`) y de las evidencias F3–F7. No sustituye las revisiones cruzadas asignadas a personas del equipo.

- **Correcto:** E5 exige las 12 familias, conserva los casos y el orden tras cada join, verifica las sumas de las piezas y comprueba la paridad de esquema entre splits. Las pruebas de E5, F1 y G3/F2 pasan: 34 correctas y 1 fallo esperado (`uv run --extra dev pytest -q tests/test_preparation_e5.py tests/test_preparation_f1_gaps.py tests/test_preparation_g3_integration.py`).
- **Publicación antes de validación global (a resolver en E5/E7):** `assemble_master_tables` publica cada Parquet y manifiesto dentro de `assemble_feature_matrix`, y solo después llama a `validate_matrix_schemas`. Si falla la paridad entre splits, quedan tablas aparentemente publicadas sin informe E5 válido. Validar antes de publicar o publicar el conjunto como una unidad; agregar prueba de fallo tardío.
- **Portabilidad del informe E5 (a documentar o corregir):** `results/preparation/E5_master_tables.csv` contiene rutas absolutas de la computadora donde se generó (`/Users/andre/...`), no válidas en otras máquinas. Las sumas sí permiten comprobar el contenido; los consumidores no deberían depender de esas rutas.
- **Dependencias F:** F3 comprueba piezas E4 y una matriz temporal, no salidas E7; el capítulo F5 aún anticipa la ejecución real de vistas E6, y F4/F7 no contienen todos los campos pedidos en el backlog. Estos entregables se pueden usar como borrador, no cerrar todavía por mera existencia de archivos.

## Qué falta para G4

1. E6 y E7: ajuste/transformación finales y comando único (Gabriel y Nick según el backlog).
2. F6: las revisiones cruzadas de Nick, Giancarlo, Mathias y Gabriel; resolver o documentar los hallazgos de la revisión adicional.
3. Los arreglos pendientes: columnas constantes (Gabriel), `pl.concat` en `views.py` (Giancarlo) y claves repetidas en bureau (Mathias).
4. Completar la evidencia F4/F7, comprobar las salidas finales de E7 en F3, actualizar F5 con resultados efectivos y cerrar G4 con la suite completa y `git diff --check`.
