# Épica F — Validación, documentación y cierre

> **Estado (13/10/2026):** E4–E7 están cerradas según sus actas. F1 y F2 tienen pruebas; F3 está cerrada tras repetir el comando E7 con cotejo de E5 y verificar las salidas de E6. F4, F5 y F7 requieren contrastar su evidencia con los criterios del backlog. F6 tiene una revisión de Andre y una revisión adicional de integración/documentación (abajo); faltan las revisiones asignadas al resto del equipo. G4 sigue pendiente.

## Resumen

| Tarea                                       | Estado              | Evidencia                                                                                                                                                                                          |
| ------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F1** Pruebas de adaptadores y catálogos   | Hecho               | Auditoría de las cinco familias; 12 pruebas nuevas en `tests/test_preparation_f1_gaps.py` (una marcada como fallo esperado, ver abajo)                                                             |
| **F2** Pruebas end-to-end sintéticas        | Hecho               | `tests/test_preparation_g3_integration.py` comprueba partición, descartes y agregados contra valores conocidos, join, ajuste/transformación y manifiestos de familias y matrices                   |
| **F3** Validaciones sobre artefactos reales | Hecho               | E7 ejecutado localmente: 29/29 controles, incluidos los 3 cotejos E5; evidencia archivada en `F3_final_checks.csv` y `F3_final_run_manifest.json`                                                  |
| **F4** Resultados por fuente                | Evidencia parcial   | `results/preparation/family_results.csv` reúne casos, cobertura y roles por familia y split; no incluye comando ni fecha por fuente como exige el backlog                                          |
| **F5** Capítulo del informe                 | Borrador adelantado | `development/03-preparacion-datos.md` tiene 3.1–3.5; debe contrastarse con E6/E7 y con el cierre de F3, F4 y F7                                                                                    |
| **F6** Revisión cruzada                     | Parcial             | Revisión de Andre al frente de bureau y revisión adicional de integración/documentación (abajo); faltan las revisiones del resto del equipo                                                        |
| **F7** Informe de fuga temporal             | Evidencia parcial   | `results/preparation/temporal_leakage_report.csv` contiene métricas de exclusiones por familia y split; falta explicitar reglas de disponibilidad, columnas excluidas y evidencia según el backlog |

## F3 — Validación real tras E7

El 13/10/2026 se ejecutó en esta máquina, sin `--from-scratch` y con `POLARS_MAX_THREADS=1`:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json
```

E4 y sus checkpoints pasaron la auditoría; E5 regeneró las matrices de train, OOT y test y E6 generó las seis vistas y el estado ajustado. El paso `validation` de E7 pasó **29 de 29 controles**, incluidos los tres nuevos `published_matrix`, y el contrato de entrada al baseline. Evidencia conservada: `results/preparation/F3_final_checks.csv` y `results/preparation/F3_final_run_manifest.json`. Mientras los artefactos locales aún estaban presentes, se cotejaron los SHA-256 de las tres matrices E5, seis vistas E6, estado y especificación con los 11 registros del manifiesto: todos coincidieron.

Para no incorporar a Git hashes y rutas absolutos específicos de esta máquina, se archivaron **solo los dos informes F3** y se restauró el conjunto versionado E5/E6/E7. Se retiraron las tres salidas de train regeneradas, que se pueden volver a producir con el comando de arriba. `F3_final_run_manifest.json` documenta la ejecución local pasada y referencia esos archivos ya retirados; **no debe usarse como manifiesto de los artefactos actualmente presentes**. `E7_manifest.json` vuelve a ser el manifiesto histórico de la ejecución desde cero (26 controles), coherente con los artefactos versionados.

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
- **Dependencias F (al momento de la revisión):** F3 solo comprobaba piezas E4 y una matriz temporal; eso se resolvió con la ejecución F3 descrita arriba. F5 debe contrastarse con los resultados finales de E6/E7, y F4/F7 aún no contienen todos los campos pedidos en el backlog.

## Qué falta para G4

1. F3 completada: conservar y revisar la evidencia de la ejecución E7 local (29/29 controles y 11 hashes verificados).
2. F6: las revisiones cruzadas de Nick, Giancarlo, Mathias y Gabriel; resolver o documentar los hallazgos de la revisión adicional.
3. Los arreglos pendientes: columnas constantes (Gabriel), `pl.concat` en `views.py` (Giancarlo) y claves repetidas en bureau (Mathias).
4. Completar la evidencia F4/F7, actualizar F5 con resultados efectivos y cerrar G4 con la suite completa y `git diff --check`.
