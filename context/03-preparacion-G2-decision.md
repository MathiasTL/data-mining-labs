# G2 — Decisión de fuentes, columnas y reglas temporales

> **Estado:** decisión **vigente** de Andre (P3), única autoridad de G1 y G2 según lo acordado con el equipo, redactada con apoyo de Claude el 13/10/2026. Se aplica desde ya: el despacho `build_family_features` de `applprev` y `person` solo construye las columnas aprobadas aquí.
> **Antecedentes:** [G1](03-preparacion-G1-decision.md), [A4](03-preparacion-A4-solicitudes-personas.md), y las selecciones B3 (profundidad 0), B4 (solicitudes y personas) y B6 (productos y registros).
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

| Tabla | Aprobadas | Bloqueadas | Excluidas | Aplazadas |
| --- | ---: | ---: | ---: | ---: |
| `applprev_1` | 19 | 15 | 7 | 0 |
| `applprev_2` | 0 | 2 | 4 | 0 |
| `person_1` | 16 | 3 | 16 | 2 |
| `person_2` | 0 | 1 | 7 | 3 |
| **Total (95)** | **35** | **21** | **34** | **5** |

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

| Resultado | Valor |
| --- | ---: |
| Características con bandera de fuga o inestabilidad | **0** |
| |Gini| máximo (por sí sola) | 0,184 (`applprev_1__employedfrom_700d__days_since_last__all`) |
| |Gini| máximo del indicador de ausencia | 0,070 |
| Características de texto no medidas | 8 |

**Cómo leerlo:** descarta una fuga fuerte, la que haría que una variable sola casi adivine el objetivo. No demuestra que no haya fugas moderadas, y para las 8 columnas de texto (educación, tipo de ingreso, etc.) no hay medición: se aprobaron por regla. Es un criterio mínimo, no una garantía.

### Resultado de la construcción final

Con solo lo aprobado, la ejecución real de train, OOT y test produce **65 características de `applprev` y 37 de `person`**, idénticas en los tres splits (`results/preparation/applications_people_feature_catalog.csv`, 102 filas). Tarda 17 s y alcanza un pico de unos 4,6 GB de memoria.

Algunas características aprobadas casi no varían (por ejemplo `persontype_1072L` y `role_1084L` de la persona principal): el preprocesador (D1) debe descartar las constantes.

## 3. Regla para las demás familias

Las selecciones B3 (Giancarlo, profundidad 0) y B6 (Gabriel, productos y registros) usan etiquetas distintas. La equivalencia, ya decidida, es:

| Etiqueta del frente | Decisión de G2 |
| --- | --- |
| `exclude_predictor` / `excluir` / `control_only` | **Excluida** (control solo para trazabilidad). |
| `manual_review` / `bloquear_hasta_G2` | **Bloqueada**: no entra al primer corte. |
| `transformar_as_of` | **Aprobada como antigüedad o recencia** si no parece calendario (regla 3). |
| `provisional_candidate` / `candidato_G2` | **Aprobada solo si** cumple las reglas 1 a 5, con el filtro de fuga sobre su artefacto de train. |
| `riesgo explícito de fuga` (por ejemplo `riskassesment_302T`) | **Excluida**. |

El filtro se puede ejecutar sobre cualquier familia ya construida:

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.leakage_screen \
    <ruta>/<familia>__train.manifest.json --out results/preparation/G2_<familia>_screen.csv
```

**Pendientes de sus responsables:** Giancarlo (profundidad 0) y Gabriel (productos y registros) ejecutan el filtro sobre sus artefactos de train y dejan su lista final de aprobadas. **Bureau** no entra al primer corte: no existe todavía la selección B5 de Mathias. Cuando exista, se le aplica esta misma política.

## 4. Reproducir

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.applications_people_g2   # decide y escribe los tres CSV de G2
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.applications_people_run   # construye train, OOT y test con lo aprobado
uv run --extra dev pytest -q tests/test_preparation_g2.py tests/test_preparation_leakage_screen.py
```

Sin los CSV de G2, el despacho falla con un error (no hay valores por defecto sin aprobar).

## 5. Límites

Esta decisión no aprueba predictores sensibles, no evalúa el desempeño del modelo ni usa OOT para elegir columnas. Si una segunda iteración quiere recuperar columnas bloqueadas, debe aportar evidencia de su disponibilidad al decidir.
