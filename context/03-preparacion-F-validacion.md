# Épica F — Validación, documentación y cierre

> **Estado (13/10/2026):** E4–E7 y F1–F7 están documentadas; F3 conserva 29/29 controles archivados. F6 se cierra con revisión técnica de frentes ajenos, pruebas y hallazgos trazables documentados como límites. No se presupone aprobación o confirmación de otras personas. Los riesgos identificados se resolvieron en el cierre: ver [Resolución en G4](#resolución-en-g4) y [G4 — cierre](03-preparacion-G4-cierre.md).

## Resumen

| Tarea                                       | Estado                       | Evidencia                                                                                                                                                                                                                                                                    |
| ------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F1** Pruebas de adaptadores y catálogos   | Hecho                        | Auditoría de las cinco familias; 12 pruebas nuevas en `tests/test_preparation_f1_gaps.py` (el fallo esperado de bureau ya se corrigió)                                                                                                                                       |
| **F2** Pruebas end-to-end sintéticas        | Hecho                        | `tests/test_preparation_g3_integration.py` comprueba partición, descartes y agregados contra valores conocidos, join, ajuste/transformación y manifiestos de familias y matrices                                                                                             |
| **F3** Validaciones sobre artefactos reales | Hecho                        | E7 ejecutado localmente: 29/29 controles, incluidos los 3 cotejos E5; evidencia archivada en `F3_final_checks.csv` y `F3_final_run_manifest.json`                                                                                                                            |
| **F4** Resultados por fuente                | Hecho                        | `results/preparation/family_results.csv` reúne casos, cobertura y roles por familia y split, con comando, informe y fecha por fuente; ver `context/03-preparacion-F4-resultados.md`                                                                                          |
| **F5** Capítulo del informe                 | Hecho                        | `development/03-preparacion-datos.md`, 3.1–3.5: cifras con informes E4–E7 y F3/F4/F7, distinción E7 26/26 frente a F3 29/29 y límites de disponibilidad temporal y artefactos train ausentes                                                                                 |
| **F6** Revisión cruzada                     | Hecho (límites documentados) | Revisión registrada de bureau y revisión técnica de integración, profundidad 0/vistas y productos/preprocesador (abajo); pruebas focalizadas y hallazgos trazables, sin atribuir revisiones nuevas a integrantes concretos. Las decisiones y correcciones se verifican en G4 |
| **F7** Informe de fuga temporal             | Hecho                        | `context/03-preparacion-F7-informe-fuga.md`, `results/preparation/F7_temporal_leakage_by_family.csv`, `temporal_leakage_report.csv` y hojas G2: 12 reglas, decisiones y descartes por split; la cobertura de productos y registros se filtra por fecha desde `products-v2`                    |

## F3 — Validación real tras E7

El 13/10/2026 se ejecutó en esta máquina, sin `--from-scratch` y con `POLARS_MAX_THREADS=1`:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json
```

E4 y sus checkpoints pasaron la auditoría; E5 regeneró las matrices de train, OOT y test y E6 generó las seis vistas y el estado ajustado. El paso `validation` de E7 pasó **29 de 29 controles**, incluidos los tres nuevos `published_matrix`, y el contrato de entrada al baseline. Evidencia conservada: `results/preparation/F3_final_checks.csv` y `results/preparation/F3_final_run_manifest.json`. Mientras los artefactos locales aún estaban presentes, se cotejaron los SHA-256 de las tres matrices E5, seis vistas E6, estado y especificación con los 11 registros del manifiesto: todos coincidieron.

Para no incorporar a Git hashes y rutas absolutos específicos de esta máquina, se archivaron **solo los dos informes F3** y se restauró el conjunto versionado E5/E6/E7. Se retiraron las tres salidas de train regeneradas, que se pueden volver a producir con el comando de arriba. `F3_final_run_manifest.json` documenta la ejecución local pasada y referencia esos archivos ya retirados; **no debe usarse como manifiesto de los artefactos actualmente presentes**. `E7_manifest.json` vuelve a ser el manifiesto histórico de la ejecución desde cero (26 controles), coherente con los artefactos versionados.

**Actualización en G4 (13/10/2026):** tras las correcciones del cierre se ejecutó el comando **con** `--from-scratch` (101,7 s). E4 se reconstruyó entero, E5 y E6 se regeneraron y `validation` pasó **29 de 29 controles**. `F3_final_checks.csv` y `F3_final_run_manifest.json` son ahora copias de `integration_checks.csv` y `E7_manifest.json` de esa corrida, coherentes con los artefactos versionados.

## F1 — Huecos cubiertos

La auditoría revisó, por familia, casos temporales, nulos, claves y agregaciones calculadas a mano. Los huecos se cubrieron en un archivo propio, sin editar las pruebas de cada responsable:

| Familia                | Caso nuevo                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| Solicitudes y personas | Columna toda nula, tipo físico `Null`, `NaN` en un importe y `case_id` nulo en la fuente |
| Profundidad 0          | Valor nulo en la fuente con `first` y `case_id` `NaN`                                    |
| Registros fiscales     | Claves repetidas, nulas y `NaN`, y conteos calculados a mano                             |
| Bureau                 | Ancla igual a la fecha de decisión (se conserva) y clave repetida                        |
| Catálogo C8            | Consolidación de tres familias con conteo literal y esquema distinto rechazado           |

**Error encontrado inicialmente:** bureau no validaba claves. Un `(case_id, num_group1)` repetido se contaba dos veces (`n_records` pasaba de 2 a 3). La prueba se marcó inicialmente como fallo esperado estricto (`xfail`); ver la corrección abajo. Los datos reales no se veían afectados: el capítulo 02 verificó que esas claves son únicas.

**Corregido (13/10/2026):** bureau lee siempre la clave de fila —`(case_id, num_group1)` en las tablas de profundidad 1 y `(case_id, num_group1, num_group2)` en las hijas— y rechaza con `ValueError` cualquier clave repetida, dentro de un lote o entre lotes del mismo shard. La prueba dejó de ser un fallo esperado. Reconstruidas desde los shards reales con caché vacía, las seis piezas de bureau no tienen claves repetidas y su contenido es igual al publicado. Límite: una clave repetida entre dos shards distintos no se detecta.

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

- La clave repetida de F1 (corregida el 13/10/2026, ver arriba).
- `credit_bureau_a_2` se omite por falta de ancla aprobada; queda documentado en C4-C5.

## F6 — Revisión adicional de integración y documentación (13/10/2026)

Revisión independiente del ensamblado E5 (`e5.py`, `assembly.py`, `schema.py`, `tests/test_preparation_e5.py`) y de las evidencias F3–F7. Es evidencia técnica de F6; no se atribuye a otros integrantes del equipo.

- **Correcto:** E5 exige las 12 familias, conserva los casos y el orden tras cada join, verifica las sumas de las piezas y comprueba la paridad de esquema entre splits. En esa revisión, las pruebas de E5, F1 y G3/F2 pasaron: 34 correctas y 1 fallo esperado de bureau (corregido después, ver F1) (`uv run --extra dev pytest -q tests/test_preparation_e5.py tests/test_preparation_f1_gaps.py tests/test_preparation_g3_integration.py`).
- **Publicación antes de validación global (a resolver en E5/E7):** `assemble_master_tables` publica cada Parquet y manifiesto dentro de `assemble_feature_matrix`, y solo después llama a `validate_matrix_schemas`. Si falla la paridad entre splits, quedan tablas aparentemente publicadas sin informe E5 válido. Validar antes de publicar o publicar el conjunto como una unidad; agregar prueba de fallo tardío.
- **Portabilidad del informe E5 (a documentar o corregir):** `results/preparation/E5_master_tables.csv` contiene rutas absolutas de la computadora donde se generó (`/Users/andre/...`), no válidas en otras máquinas. Las sumas sí permiten comprobar el contenido; los consumidores no deberían depender de esas rutas.
- **Dependencias F (al momento de la revisión):** F3 solo comprobaba piezas E4 y una matriz temporal; se resolvió con la ejecución F3 descrita arriba. F4 se completó después (`context/03-preparacion-F4-resultados.md`); F7 ya entrega reglas, decisiones y descartes por familia (`context/03-preparacion-F7-informe-fuga.md`); F5 consolida las evidencias en `development/03-preparacion-datos.md`. Sigue abierta la decisión de uso de cobertura sin as-of.

## F6 — Revisión técnica independiente complementaria (13/10/2026)

**Alcance:** lectura de código, catálogos y pruebas de profundidad 0/vistas, productos/preprocesador e integración E5/E7. Junto con la revisión registrada de bureau, cubre frentes ajenos con evidencia verificable, sin atribuir a Nick, Giancarlo, Mathias ni Gabriel revisiones que no constan. El reparto de personas del backlog organiza el trabajo, pero aquí no se exige una confirmación externa para cerrar la revisión técnica. No se reconstruyeron Parquet reales. Pruebas sintéticas ejecutadas durante las revisiones: profundidad 0/vistas (32 correctas), C6/preprocesamiento/F1/E6 (46 correctas y 1 fallo esperado), E5/assembly/schema/E7 (34 correctas). Son ejecuciones de conjuntos que pueden solaparse; no se deben sumar como número de pruebas únicas.

| Frente                 | Hallazgo y evidencia                                                                                                                                                                                                                                                                                                                                                                                                                     | Tratamiento pendiente                                                                                                                                                                                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E6 y cobertura         | `preprocessing.build_spec_from_catalog` selecciona por `logical_type`, sin filtrar `role` (`preprocessing.py`, `build_spec_from_catalog`; `e6.py`, `fit_and_transform`). Por tanto, las 30 columnas de cobertura del catálogo C8 entran en las vistas como características, incluidas las 12 de productos/registros sin as-of que F7 identifica. No se ha probado fuga efectiva, pero sí falta de garantía de disponibilidad al decidir. | **Decisión previa al modelado/G4:** excluir estas columnas de las vistas o justificar y aplicar una regla temporal; después regenerar y validar E5/E6/F3 si cambia el contrato. No ajustar silenciosamente las cifras 303/124 del informe histórico.                                      |
| E5 y publicación       | `e5.assemble_master_tables` publica las tres matrices mediante `assembly.assemble_feature_matrix` antes de `validate_matrix_schemas`. Si falla la paridad entre splits, los archivos quedan publicados aunque no haya informe E5 válido; el hallazgo anterior de §F6 sigue vigente.                                                                                                                                                      | Validar antes de publicar o publicar como conjunto; añadir prueba de fallo tardío. Hasta entonces, tratar una ejecución fallida como potencialmente contaminante y no consumir sus salidas.                                                                                               |
| E7 parametrización     | `build.plan_steps` no pasa `cutoff_week` ni `results_dir` configurados a E5/E6/validation, que usan sus valores por defecto; el manifiesto se calcula con la configuración. Con la configuración actual (corte 74 y directorio predeterminado) no se observó divergencia.                                                                                                                                                                | Corregir propagación y probar configuración no predeterminada antes de prometer reproducibilidad con otros valores; mantener documentado el límite del comando E7 actual.                                                                                                                 |
| Productos/registros    | `_selected_columns` en `families/products_registry.py` acepta un G2 sin filas de una familia como si hubiera cero aprobaciones. `_read_source` hace semi-join con casos de la base antes de validar claves; los nulos y filas fuera de la base pueden desaparecer de los controles/`source_rows_outside_base`. Los agregados reciben una sola fila por caso, pero la auditoría de origen no es completa.                                 | Exigir decisión G2 por familia y probar la lectura de fuente con claves nulas y ajenas a la base, o documentar explícitamente el alcance de esa métrica.                                                                                                                                  |
| Vistas y profundidad 0 | Las columnas numéricas constantes siguen en las vistas; la escala lineal usa 1 para varianza cero, sin división por cero (`views.py`). Las coberturas de profundidad 0 no usan una fecha de fuente as-of (`families/depth0.py`); G2 no aprueba allí predictores de valor.                                                                                                                                                                | Decidir si descartar constantes antes de cerrar G4; evaluar disponibilidad de coberturas estáticas antes de modelar. No se confirmó una advertencia real de `pl.concat` en la versión comprobada: el aviso previo de §Pendientes de F5 queda por verificar, no es fallo reproducido aquí. |

**Aspectos correctos:** profundidad 0 valida claves y partición de base; el ensamblador verifica sumas, esquema y orden por join; las vistas reutilizan el ajuste de train; productos/registros rechaza predictores G2 aprobados que no sabe agregar. Estos controles no resuelven los riesgos de disponibilidad temporal. Los hallazgos de cobertura, publicación y configuración son problemas de contrato/seguridad de salida, no evidencia de que la ejecución histórica E7 o F3 fallase con la configuración versionada.

## Resolución en G4

| Hallazgo | Resolución |
| --- | --- |
| Cobertura de productos y registros sin as-of (E6 y cobertura, F7) | `products-v2`: solo cuentan filas con fecha `<= date_decision`; `other_1`, sin fecha, queda en cero. La ablación de F7 muestra que la ganancia restante es legítima. |
| Coberturas de profundidad 0 sin fecha as-of | Aceptado y documentado: son una fila estática por caso, que describe la solicitud actual (regla G1). `static_0__source_present` es constante y se descarta; `static_cb_0` (98,1 % / 99,9 %) se conserva. |
| E5 publica antes de la validación global | Los tres splits se publican como unidad; prueba de fallo tardío en `tests/test_preparation_e5.py`. |
| E7 no propaga `cutoff_week` ni `results_dir` | `plan_steps` pasa corte, directorio de resultados y catálogo a E4, E5, E6 y validation; la configuración rechaza valores que E4 no puede respetar (corte distinto de 74 u otro `results_dir`). |
| Productos: G2 sin filas de una familia | Es un error, ya no equivale a cero aprobadas. |
| Productos: semi-join antes de validar claves | Se validan nulos y la clave `(case_id, num_group1)` del archivo completo, y 1:0..1 de `other_1` sobre los casos de la base; `source_rows_outside_base` cuenta las filas ajenas al split. La muestra real de test trae un caso ajeno a la base con dos filas en `other_1`. |
| Columnas constantes en las vistas | Se descartan y se registran con `role = dropped`. |
| `pl.concat(how="horizontal")` en `views.py` | Comprobado: polars emitía `DeprecationWarning`; reemplazado por `hstack`. |
| Rutas absolutas en informes | Documentado: solo en campos informativos; la carga usa rutas relativas y SHA-256. |

## Qué faltaba para G4 (antes del cierre)

1. F3 completada: conservar y revisar la evidencia de la ejecución E7 local (29/29 controles y 11 hashes verificados).
2. F6 cerrada con hallazgos trazables; en G4 decidir y comprobar las correcciones de contrato y disponibilidad temporal identificadas arriba, o registrar una aceptación explícita del riesgo. No quedan confirmaciones personales como condición de F6.
3. Pendientes: decidir sobre las columnas constantes, corregir la publicación E5 y la parametrización E7; comprobar si el aviso sobre `pl.concat` en `views.py` sigue aplicando antes de tratarlo como defecto. Las claves repetidas en bureau ya están corregidas.
4. Decidir si se excluyen o justifican temporalmente los indicadores de cobertura de productos y registros sin as-of (riesgo F7); luego verificar la suite completa y `git diff --check` para G4. Los Parquet grandes de train E5/E6 no están presentes actualmente; F3 registra una validación pasada, no una nueva validación local.
