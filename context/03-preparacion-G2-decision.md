# G2 — Decisión de fuentes, columnas y reglas temporales

> **Estado:** decisión **vigente** de Andre (P3), única autoridad de G1 y G2 según lo acordado con el equipo, redactada con apoyo de Claude el 13/10/2026. Se aplica desde ya: el despacho `build_family_features` de `applprev` y `person` solo construye las columnas aprobadas aquí.
> **Antecedentes:** [G1](03-preparacion-G1-decision.md), [A4](03-preparacion-A4-solicitudes-personas.md), y las selecciones B3 (profundidad 0), B4 (solicitudes y personas), B5 (bureau, incorporada después de esta decisión inicial) y B6 (productos y registros). Véase la ampliación de la sección 6.
> **Qué cierra:** la lista inicial de columnas de `applprev` y `person` con evidencia de datos. Fija la regla que aplica a las demás familias, cuyas listas finales dependen de sus responsables.

## 1. Política de aprobación

Una columna entra al primer corte solo si cumple **todo** lo siguiente:

1. **No es clave, cuasi-identificador ni atributo sensible.** Sexo y género quedan excluidos por defecto; entrarán únicamente si el equipo decide sobre equidad y regulación.
2. **Su disponibilidad temporal no depende del nombre.** Un nombre limpio no prueba que el dato existiera al decidir. Se exige la regla `<= date_decision` ya aplicada por el núcleo, y que no sea un estado, mora, pago, deuda o motivo que pudo cambiar después (esos quedan **bloqueados**).
3. **No parece un calendario o una proyección.** Una fecha que cae después de la decisión en una fracción relevante de filas válidas (por ejemplo `dtlastpmtallstes_3545839D`, 5,4 %) se bloquea.
4. **Tiene cobertura mínima:** al menos 5 % de valores en su tabla fuente (en `person`, en la persona principal o en los contactos). Con menos, no aporta información suficiente en el primer corte.
5. **Pasa el filtro empírico de fuga**, calculado solo con entrenamiento (`WEEK_NUM < 74`; OOT y test no se usan jamás):
    - Gini univariado de cada característica derivada no mayor a **0,5** en valor absoluto.
    - Gini del indicador de ausencia no mayor a 0,5.
    - El signo del Gini no cambia entre tres bloques de semanas.
6. **El núcleo sabe agregarla.** Una columna válida que el núcleo todavía no agrega queda **aplazada**, no descartada.

Decisiones posibles: `aprobada`, `excluida` (definitiva en el primer corte), `bloqueada` (exclusión preventiva, se reevalúa en una segunda iteración si el modelo lo justifica), `aplazada` y `pendiente_filtro`.

## 2. Resultado para `applprev` y `person`

Fuente de verdad: `results/preparation/G2_applications_people_columns.csv` (una fila por columna, con la decisión y su motivo). El filtro completo está en `G2_applications_people_screen.csv` y el vocabulario de categorías en `G2_applications_people_vocabulary.csv`.

| Tabla          | Aprobadas | Bloqueadas | Excluidas | Aplazadas |
| -------------- | --------: | ---------: | --------: | --------: |
| `applprev_1`   |        19 |         15 |         7 |         0 |
| `applprev_2`   |         0 |          2 |         4 |         0 |
| `person_1`     |        16 |          3 |        16 |         2 |
| `person_2`     |         0 |          1 |         7 |         3 |
| **Total (95)** |    **35** |     **21** |    **34** |     **5** |

**Aprobadas de `applprev_1`:** importes y plazos (`annuity_853A`, `credamount_590A`, `credacc_credlmt_575A`, `tenor_203L`, `mainoccupationinc_437A`, `byoccupationinc_3656910L`, `childnum_21L`), categorías (`credtype_587L`, `education_1138M`, `familystate_726L`, `inittransactioncode_279L`, `profession_152M` solo con el número de categorías distintas), indicadores (`isbidproduct_390L`, `isdebitcard_527L`), fechas como antigüedad o recencia (`approvaldate_319D`, `dateactivated_425D`, `dtlastpmt_581D`, `employedfrom_700D`) y `creationdate_885D` como fecha de referencia.

**Aprobadas de `person_1`:** atributos de la persona principal (`num_group1 = 0`): ingresos, tipo de ingreso, educación, estado civil, idioma, empleo y rol, edad (desde `birth_259D`) y antigüedad laboral (desde `empl_employedfrom_271D`); y proporciones de la relación con el cliente entre los contactos (`relationshiptoclient_415T` y `_642T`).

**No entran, y por qué:**

- `dtlastpmtallstes_3545839D` y `firstnonzeroinstldate_307D`: fechas con aspecto de calendario (5,4 % y 3,3 % de filas con fecha posterior a la decisión). Bloqueadas.
- `gender_992L` y `sex_738L`: atributos sensibles. Excluidos.
- `birthdate_87D`: 99 % nulo y duplica `birth_259D`. Excluida.
- `credacc_maxhisbal_375A`, `credacc_minhisbal_90A`, `credacc_transactions_402L` y `revolvingaccount_394A`: 4,3 % a 4,7 % de cobertura, justo bajo el mínimo. Excluidas.
- Estados, mora, pagos, deuda y motivos de rechazo de `applprev_1`: bloqueados desde B4, sin evidencia de una instantánea válida a la decisión.
- `applprev_2`: solo aporta conteos a través del padre válido; sus columnas quedan bloqueadas.

### Evidencia del filtro de fuga

El filtro evaluó 102 características de train (94 numéricas o booleanas y 8 de texto):

| Resultado                                           | Valor |
| --------------------------------------------------- | ----: |
| Características con bandera de fuga o inestabilidad | **0** |
| Gini absoluto máximo de una variable sola           | 0,184 (`applprev_1__employedfrom_700d__days_since_last__all`) |
| Gini absoluto máximo del indicador de ausencia      | 0,070 |
| Características de texto no medidas                 |     8 |

**Cómo leerlo:** descarta una fuga fuerte, la que haría que una variable sola casi adivine el objetivo. No demuestra que no haya fugas moderadas, y para las 8 columnas de texto (educación, tipo de ingreso, etc.) no hay medición: se aprobaron por regla. Es un criterio mínimo, no una garantía.

### Resultado de la construcción final

Con solo lo aprobado, la ejecución real de train, OOT y test produce **65 características de `applprev` y 37 de `person`**, idénticas en los tres splits (`results/preparation/applications_people_feature_catalog.csv`, 102 filas). Tarda 17 s y alcanza un pico de unos 4,6 GB de memoria.

Algunas características aprobadas casi no varían (por ejemplo `persontype_1072L` y `role_1084L` de la persona principal): el preprocesador (D1) debe descartar las constantes. (Implementado en G4: `views.fit_matrix_views` descarta y registra toda columna constante en train.)

## 3. Regla para las demás familias

Las selecciones B3 (Giancarlo, profundidad 0), B5 (Mathias, bureau, posterior a esta redacción inicial) y B6 (Gabriel, productos y registros) usan etiquetas distintas. La equivalencia orienta la revisión; para el estado por columna y la resolución vigente del primer corte prevalece la sección 6:

| Etiqueta del frente                                           | Decisión de G2                                                                                                                                                              |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exclude_predictor` / `excluir` / `control_only`              | **Excluida** (control solo para trazabilidad).                                                                                                                              |
| `manual_review` / `bloquear_hasta_G2`                         | **Bloqueada**: no entra al primer corte.                                                                                                                                    |
| `transformar_as_of`                                           | **Candidata a aprobación** como antigüedad o recencia solo si cumple todas las reglas 1–6; puede quedar bloqueada por calendario, semántica o disponibilidad sin demostrar. |
| `provisional_candidate` / `candidato_G2`                      | **Aprobada solo si** cumple las reglas 1 a 5, con el filtro de fuga sobre su artefacto de train.                                                                            |
| `riesgo explícito de fuga` (por ejemplo `riskassesment_302T`) | **Excluida**.                                                                                                                                                               |

El filtro se puede ejecutar sobre cualquier familia ya construida:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.leakage_screen \
    <ruta>/<familia>__train.manifest.json --out results/preparation/G2_<familia>_screen.csv
```

**Estado al redactar esta decisión inicial:** estaban pendientes Giancarlo (profundidad 0) y Gabriel (productos y registros); bureau no entraba al primer corte porque aún no existía B5. **Actualización posterior:** B5 sí existe como selección preliminar de Mathias. La evidencia y las decisiones históricas de filtrado no sustituyen la resolución vigente por columna: la sección 6 determina el estado actual de B3, B5 y B6 y no aprueba predictores de esas familias en el primer corte.

## 4. Reproducir

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.applications_people_g2   # decide y escribe los tres CSV de G2
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.applications_people_run   # construye train, OOT y test con lo aprobado
uv run --extra dev pytest -q tests/test_preparation_g2.py tests/test_preparation_leakage_screen.py
```

Sin los CSV de G2, el despacho falla con un error (no hay valores por defecto sin aprobar).

## 5. Límites

Esta decisión no aprueba predictores sensibles, no evalúa el desempeño del modelo ni usa OOT para elegir columnas. Si una segunda iteración quiere recuperar columnas bloqueadas, debe aportar evidencia de su disponibilidad al decidir.

## 6. Ampliación de la decisión: B3, B5 y B6 (13/10/2026)

**Alcance y ratificación.** Esta ampliación registra la decisión conservadora del primer corte sobre la evidencia actualmente disponible. El usuario confirma que P2, P4 y P5 ya han dado su conformidad con las matrices de sus frentes; tomo esa conformidad como ratificación para comenzar las tareas C. **No cambia ninguna de las 95 decisiones vigentes** de `applprev`/`person` (§2). Los archivos `results/preparation/G2_depth0_columns.csv`, `G2_bureau_columns.csv` y `G2_products_registry_columns.csv` son la matriz por `(family, column)`: conservan la acción B3/B5/B6, motivo, regla temporal propuesta, procedencia, cobertura con su ámbito, estado del filtro y responsable. `g2_decision` es la resolución operativa, no la etiqueta del selector. Solo se asignará `pendiente_filtro` cuando la disponibilidad temporal ya esté acreditada y reste únicamente el filtro empírico. En las tres matrices actuales ninguna alcanza todavía ese estado: `bloqueada` incluye candidatas que podrían revisarse si aparece evidencia temporal. Ni siquiera `pendiente_filtro` autoriza un predictor.

| Selección                              |   Filas | Excluidas como predictor | Bloqueadas | `pendiente_filtro` | Aprobadas nuevas |
| -------------------------------------- | ------: | -----------------------: | ---------: | -----------------: | ---------------: |
| B3 (`base`, `static_0`, `static_cb_0`) |     226 |                       31 |        195 |                  0 |            **0** |
| B5 (bureau A y B, padre e hijo)        |     149 |                       23 |        126 |                  0 |            **0** |
| B6 (productos y registros fiscales)    |      33 |                       15 |         18 |                  0 |            **0** |
| **Total**                              | **408** |                   **69** |    **339** |              **0** |            **0** |

**Decisiones específicas y excepciones de clasificación:**

- B3: los cinco campos de `base` son controles/objetivo excluidos **como predictores**, no se borran de la base. Los `manual_review` siguen bloqueados. Los candidatos con semántica de estado, pagos o deuda, como `static_0/numinstpaid_4499208L` y `sumoutstandtotal_3546847A`, quedan bloqueados; los restantes siguen bloqueados por falta de evidencia temporal individual y de filtro train; no se etiquetan `pendiente_filtro` antes de acreditar lo primero. `riskassesment_302T` sigue excluida. Las coberturas del perfil de `static_*` incluyen OOT: son descriptivas y **no** se utilizaron para aprobar o excluir por el umbral del train; por ejemplo `equalityempfrom_62L` requiere medir cobertura en train antes de decidir. Responsable de completar evidencia: P2.
- B5: sus **cuatro** `referencia_as_of` se conservan solo como posibles anclas/control, nunca como predictores; su validez para filtrar filas sigue sujeta a probar su semántica. `credit_bureau_a_2` continúa sin columnas candidatas a inclusión: el emparejamiento de los meses y años de pago de B5 difiere de A5 y no se resuelve comparando solo porcentajes de nulos. Estados, pagos, DPD, deuda y fechas de fin con aspecto de calendario permanecen bloqueados. Algunas exclusiones de B5 por cobertura pasan aquí a `bloqueada` cuando el riesgo temporal prima; el resto de candidatas B5 también se bloquea hasta probar temporalidad y entonces podrán pasar a `pendiente_filtro`; esta matriz es la autoridad vigente para ese estado, y preserva la acción B5 como procedencia. La cobertura anotada de bureau es porcentaje no nulo en filas fuente train **antes** del corte `as-of`, no cobertura de casos tras filtrarlo. Responsable: P4.
- B6: los tres resúmenes `last*` de `debitcard_1` tienen cobertura bruta de **7,7660 %, 7,0444 % y 7,0444 %**, respectivamente (`results/variable_catalog.csv`), por encima del 5 %. **No** se excluyen por cobertura: están bloqueados porque `source_defined` no acredita cuándo se calculó/publicó el valor. `contractenddate_991D` puede ser calendario, las demás fechas requieren probar disponibilidad y los importes de `other_1` no tienen referencia temporal; nombres fiscales y claves siguen excluidos como predictores. Responsable: P5.

**Decisión de construcción:** el primer corte autorizado de predictores sigue limitado a `applprev`/`person`. Los catálogos y reportes C1 de profundidad 0 publicados antes de esta ampliación reflejan 57 candidatos de B3 como predictores **sin aprobación G2**: son evidencia histórica, no el catálogo autorizado para ensamblar la matriz final. P2 puede continuar C1/C8 en profundidad 0 con valores y métricas de cobertura (`n_records`, `source_present`) y **cero predictores**; `families/depth0.py` permite esa construcción si no hay aprobaciones explícitas. No se deben reutilizar los artefactos C1 anteriores como matriz de predictores. Al cierre de G2, bureau y productos/registros aún no tenían adaptadores registrados en `build_family_features`; hoy ambos están registrados (C4–C5 y C6–C7) y, con esta matriz, solo producen cobertura (`n_records`, `source_present`) y cero predictores hasta que una aprobación posterior habilite columnas.

**Para reconsiderar una columna:** P2/P4/P5 documentan su procedencia y momento de disponibilidad para cada valor (no solo el nombre de la columna), resuelven fechas futuras, coberturas y linaje padre-hijo, y proponen las derivaciones que realmente sabe construir el adaptador. Solo entonces generan un artefacto **de evaluación aislado** de `train` (`WEEK_NUM < 74`), sin publicarlo como catálogo aprobado, y ejecutan el filtro de §3 sobre su manifiesto real, con `POLARS_MAX_THREADS=1`. El filtro carga la matriz derivada y la base train: tiene coste de memoria proporcional a filas por características; bureau A exige además el procesamiento por lotes previsto en A5 (su tabla hija ronda 188 millones de filas). No se inventan rutas de manifiesto ni se ejecutan Parquet reales sin datos y presupuesto disponibles. `not_screened` (texto), `insufficient`, cobertura baja o ausencia de evidencia temporal **no** equivalen a `ok`. Solo una nueva decisión explícita actualiza `g2_decision` a `aprobada` y habilita regenerar catálogo, manifiestos y matrices. OOT/test se reservan para validar esquemas/estabilidad _después_, nunca para seleccionar columnas.

**Criterio de cierre de G2 para iniciar C:** ninguna columna nueva de B3/B5/B6 como predictor sin justificación temporal contrastable y filtro train; ninguna reservada o sensible como predictor; anclas fuera de X; bloqueos/exclusiones trazables; agregados de cobertura permitidos. Con las confirmaciones de P2/P4/P5 comunicadas por el usuario, **G2 queda cerrado para el primer corte conservador y las tareas C pueden comenzar**. G2 se reabre si se propone incorporar una columna bloqueada; eso requerirá evidencia temporal, filtro exclusivamente train y una decisión explícita nueva.
