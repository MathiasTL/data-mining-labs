# Épica F — Validación, documentación y cierre

> **Estado (13/10/2026):** F1, F2, F3, F4, F5 y F7 avanzados por Andre (P3) a pedido del equipo; F6 tiene la revisión de Andre y faltan las de los demás. G4 queda pendiente de E4 a E7 y de F6.

## Resumen

| Tarea | Estado | Evidencia |
| --- | --- | --- |
| **F1** Pruebas de adaptadores y catálogos | Hecho | Auditoría de las cinco familias; 12 pruebas nuevas en `tests/test_preparation_f1_gaps.py` (una marcada como fallo esperado, ver abajo) |
| **F2** Pruebas end-to-end sintéticas | Hecho | `tests/test_preparation_g3_integration.py` cubre partición, filtrado, agregado, join, ajuste y transformación, y manifiestos verificados |
| **F3** Validaciones sobre artefactos reales | Hecho (adelantado a E7) | `uv run -m dm_credit.preparation.validation`: 26 de 26 controles correctos; `results/preparation/integration_checks.csv` |
| **F4** Resultados por fuente | Hecho | `results/preparation/family_results.csv` (casos, cobertura y roles por familia y split) |
| **F5** Capítulo del informe | Hecho | `development/03-preparacion-datos.md`, sin marcadores pendientes |
| **F6** Revisión cruzada | Parcial | Revisión de Andre al frente de bureau (abajo); faltan las de los demás |
| **F7** Informe de fuga temporal | Hecho | `results/preparation/temporal_leakage_report.csv`, consolidado de los manifiestos de las doce familias |

F3 se ejecutó sobre los artefactos publicados antes de que exista el comando único (E7). Cuando E7 esté listo, basta con repetir el mismo comando sobre sus salidas.

## F1 — Huecos cubiertos

La auditoría revisó, por familia, casos temporales, nulos, claves y agregaciones calculadas a mano. Los huecos se cubrieron en un archivo propio, sin editar las pruebas de cada responsable:

| Familia | Caso nuevo |
| --- | --- |
| Solicitudes y personas | Columna toda nula, tipo físico `Null`, `NaN` en un importe y `case_id` nulo en la fuente |
| Profundidad 0 | Valor nulo en la fuente con `first` y `case_id` `NaN` |
| Registros fiscales | Claves repetidas, nulas y `NaN`, y conteos calculados a mano |
| Bureau | Ancla igual a la fecha de decisión (se conserva) y clave repetida |
| Catálogo C8 | Consolidación de tres familias con conteo literal y esquema distinto rechazado |

**Error encontrado:** bureau no valida claves. Un `(case_id, num_group1)` repetido se cuenta dos veces (`n_records` pasa de 2 a 3). La prueba queda como fallo esperado estricto (`xfail`), así que avisará en cuanto se corrija. Los datos reales no se ven afectados: el capítulo 02 verificó que esas claves son únicas.

## F6 — Revisión de Andre al frente de bureau (Mathias)

**Lo que está bien:** el adaptador lee por grupos de filas con memoria acotada, fusiona parciales con una fórmula estable de varianza, valida el padre antes que el hijo y publica artefactos con manifiesto y suma de verificación. Sus agregados están probados contra valores calculados a mano.

**Hallazgo principal: el ancla de existencia descarta la mayoría de las filas.**

| Tabla | Filas fuente (capítulo 02) | Excluidas en train + OOT | Porcentaje |
| --- | ---: | ---: | ---: |
| `credit_bureau_a_1` | 15 940 537 | 10 193 265 (8 360 380 sin ancla y 1 832 885 con ancla posterior) | 63,9 % |
| `credit_bureau_b_1` | 85 791 | 76 645 | 89,3 % |
| `credit_bureau_b_2` | 1 286 755 | 1 166 558 (casi todas por padre inválido) | 90,7 % |

La regla usa `lastupdate_*`, la fecha de **última actualización** del registro, como prueba de que existía al decidir. Es conservadora y evita la fuga de valores actualizados después de la decisión, así que es correcta para las características de valor. Para la **cobertura**, en cambio, cambia el significado: `source_present` pasa a ser "tiene registros actualizados antes de la decisión", no "tiene historial en el bureau". Por eso la cobertura de bureau A cae del 90,8 % bruto al 71,0 % en train, y sube al 87,6 % en OOT, donde las actualizaciones posteriores quedan más lejos.

**Recomendación (no bloquea G4):** para los indicadores de cobertura, usar la fecha de inicio del contrato (`dateofcredstart_739D` en A, `contractdate_551D` en B) como prueba de existencia, y mantener `lastupdate_*` para las características de valor cuando G2 apruebe alguna. Mientras bureau solo aporte cobertura, la diferencia afecta a la estabilidad entre train y OOT.

**Otros puntos:**
- La clave repetida de F1.
- `credit_bureau_a_2` se omite por falta de ancla aprobada; queda documentado en C4-C5.

## Qué falta para G4

1. E4 a E7: ejecución oficial en cola, ensamblado y ajuste finales y comando único (Nick y Gabriel).
2. F6: las revisiones cruzadas de Nick, Giancarlo, Mathias y Gabriel.
3. Los arreglos pendientes: columnas constantes (Gabriel), `pl.concat` en `views.py` (Giancarlo) y claves repetidas en bureau (Mathias).
4. Repetir F3 sobre las salidas de E7 y cerrar G4 con la suite completa y `git diff --check`.
