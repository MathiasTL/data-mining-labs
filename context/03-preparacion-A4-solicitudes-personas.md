# A4 — Propuesta de agregaciones: solicitudes previas y personas

> **Responsable:** Andre (P3). **Estado:** propuesta para G1/G2; nada de esto está aprobado todavía.
> **Alcance:** `applprev_1`, `applprev_2`, `person_1`, `person_2`, según el [backlog del capítulo 03](03-preparacion-datos-backlog.md).
> **Complementos:** fixtures `tests/preparation_fixtures.py`, selección de columnas B4 en `results/preparation/B4_applications_people_columns.csv`, coerción D4 y extremos D5 en `src/dm_credit/preparation/`.

## 1. Evidencia verificada con los datos de entrenamiento

Cifras producidas por `src/dm_credit/preparation/applications_people_evidence.py` (resultado en `results/preparation/A4_evidence.csv`): solo lectura sobre `train`, con un hilo y sin materializar joins entre familias. El módulo está probado contra los fixtures, cuyos resultados se calcularon a mano. Las cifras marcadas con † **no** salen de ese script: se obtuvieron con consultas de solo lectura ad hoc el 12/10/2026 y todavía no tienen script reproducible (pendiente); la cobertura viene de `results/integration_analysis/family_cardinality.csv`.

### Solicitudes previas

| Hecho                                                                   |                                                                                                                  Valor |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------: |
| Filas de `applprev_1` / casos cubiertos                                 |                                                                                                  6 525 979 / 1 221 522 |
| Filas con `creationdate_885D` **posterior** a `date_decision`           |                                                                                                       140 040 (2,15 %) |
| Anticipación máxima de esas filas †                                     |                                                                                         14 días después de la decisión |
| Filas con `creationdate_885D` nula                                      |                                                                                                                     66 |
| Casos que conservan al menos una fila tras el filtro `<= date_decision` |                                                                             1 208 039 (se pierden 13 483 casos, 1,1 %) |
| Hijos de `applprev_2` sin padre en `applprev_1`                         |                                                                                                                      0 |
| Hijos de `applprev_2` totales / con padre temporalmente válido          |                                                                  14 075 487 / 13 862 275 (se descartan 213 212, 1,5 %) |
| Filas de `applprev_1` con otras fechas posteriores a la decisión †      | `approvaldate_319D` 37 383; `dateactivated_425D` 16 669; `dtlastpmt_581D` 42 009; `firstnonzeroinstldate_307D` 212 957 |

El filtro temporal **no es un detalle**: sin él entran 140 040 filas del futuro y 213 212 hijos colgados de padres que no existían.

### Personas

| Hecho                                                                          |                                   Valor |
| ------------------------------------------------------------------------------ | --------------------------------------: |
| Filas de `person_1` / casos                                                    | 2 973 991 / 1 526 659 (cobertura 100 %) |
| Casos con una fila `num_group1 = 0`                                            |                       1 526 659 (todos) |
| `num_group1 = 0`: ingresos, sexo y nacimiento informados †                     |                                   100 % |
| `num_group1 >= 1`: ingresos, sexo y nacimiento informados †                    |                                     0 % |
| `empl_employedfrom_271D` posterior a la decisión                               |                               316 filas |
| `birth_259D` posterior a la decisión                                           |                                       0 |
| Hijos de `person_2` sin padre en `person_1`                                    |                                       0 |
| `person_2`: filas con inicio de empleo informado † / posterior a la decisión † |                      5 757 (0,35 %) / 0 |

Casos por número de personas †: 1 → 769 339; 2 → 273 106; 3 → 302 446; 4 → 159 315; 5 → 20 987; 6 o más → 1 466.

**Inferencia respaldada por datos:** `num_group1 = 0` es la persona principal de la solicitud; las demás filas son contactos o personas relacionadas sin atributos demográficos propios. Es una inferencia por patrón de nulos, no una definición oficial: se marca como decisión de G1 (sección 6).

## 2. Regla temporal propuesta

1. La referencia es `date_decision` de `train_base`.
2. `applprev_1`: una fila existe para el caso solo si `creationdate_885D <= date_decision`. Fecha nula = se descarta (no se puede demostrar disponibilidad). Igualdad (mismo día) = se conserva.
3. `applprev_2`: un hijo solo se usa si su padre `(case_id, num_group1)` pasó el filtro anterior. Se valida el padre **antes** de agregar.
4. `person_1` y `person_2`: no tienen fecha por fila; son atributos de la solicitud actual. No hay filtro por fila. Las fechas internas (`empl_employedfrom_271D`, `empls_employedfrom_796D`, `birth_259D`) se transforman en antigüedad solo si son `<= date_decision`; un valor posterior se anula y solo se registra como total en el manifiesto de la familia (nunca como característica por caso).
5. Las otras fechas de `applprev_1` siguen la misma regla por campo: valor posterior a la decisión = nulo; el total de anulados va solo al manifiesto.
6. Ventana retrospectiva: depende de la política común de A7 (Giancarlo). Hasta que G1 la fije, el borrador usa **todo el historial `<= date_decision`**.

## 3. Catálogo inicial de agregaciones

Convención (A6 y A7): `<familia>__<columna>__<agregación>__<ventana>`, en minúsculas y con ventana `all` hasta que G1 fije otras; la cobertura se llama `<familia>__n_records` y `<familia>__source_present`. Una característica entra solo si se describe con familia + columnas fuente + filtro temporal + agregación + significado.

### 3.1 `applprev_1` (filtrada, una fila por caso)

| Fuente                                                                                                        | Regla                                                               | Característica                                              | Justificación temporal                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| todas las filas válidas                                                                                       | conteo                                                              | `applprev_1__n_records`                                     | filas con `creationdate_885D <= date_decision`                                                                                                                                      |
| presencia                                                                                                     | indicador                                                           | `applprev_1__source_present`                                | distingue "sin historial" (80,01 % de cobertura previa al filtro) de "historial vacío"                                                                                              |
| `creationdate_885D`                                                                                           | recencia y antigüedad                                               | `__days_since_last_creation`, `__days_since_first_creation` | fecha filtrada; nunca futura                                                                                                                                                        |
| `credamount_590A`, `annuity_853A`                                                                             | conteo válido, mínimo, máximo, media, suma                          | `__credamount__{count,min,max,mean,sum}` y análogas         | importes del contrato previo ya creado                                                                                                                                              |
| `tenor_203L`, `mainoccupationinc_437A`, `byoccupationinc_3656910L`, `childnum_21L`                            | conteo válido, mínimo, máximo, media                                | `__tenor__*`, etc.                                          | atributos de la solicitud previa                                                                                                                                                    |
| `credtype_587L`, `inittransactioncode_279L`, `education_1138M`, `familystate_726L`                            | conteo de categorías distintas y proporción de categorías aprobadas | `__credtype__n_distinct`, `__credtype__share_<cat>`         | categorías con vocabulario ajustado solo en train                                                                                                                                   |
| `isbidproduct_390L`, `isdebitcard_527L`                                                                       | conteo y proporción verdadera                                       | `__isbidproduct__share_true`                                | indicadores de la solicitud previa                                                                                                                                                  |
| `approvaldate_319D`, `dateactivated_425D`, `dtlastpmt_581D`, `dtlastpmtallstes_3545839D`, `employedfrom_700D` | recencia del máximo válido                                          | `__days_since_last_<campo>`                                 | valor `<= date_decision`; el resto se anula. Los anulados se cuentan solo en el manifiesto: una característica que cuente fechas futuras usaría información posterior a la decisión |

### 3.2 `applprev_2` (solo vía padre válido)

| Fuente                 | Regla                                                                             | Característica                                                 | Justificación temporal                         |
| ---------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------- |
| hijos con padre válido | conteo total, hijos por padre válido (media, máximo; un padre sin hijos cuenta 0) | `applprev_2__n_records`, `__children_per_parent_{mean,max}__all` | el padre pasó el filtro de `creationdate_885D` |
| `conts_type_509L`      | proporción de categorías aprobadas                                                | `__conts_type__share_<cat>`                                    | **bloqueada hasta G2**: sin fecha propia       |

`credacc_cards_status_52L` queda excluida (estado sin linaje); `cacccardblochreas_147M` queda bloqueada hasta G2, igual que `conts_type_509L`.

### 3.3 `person_1`

| Fuente                                    | Regla                                               | Característica                                                                       | Justificación temporal                                      |
| ----------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| fila `num_group1 = 0` (persona principal) | valor directo de cada atributo aprobado en B4       | `person_1__main__education`, `__main__incometype`, `__main__mainoccupationinc`, etc. | atributos de la solicitud actual; sin fecha por fila        |
| `birth_259D` (principal)                  | edad en años a `date_decision`                      | `person_1__main__age_years`                                                          | fecha de nacimiento nunca posterior a la decisión (0 casos) |
| `empl_employedfrom_271D` (principal)      | antigüedad laboral en días; anulada si es posterior | `person_1__main__employment_days`                                                    | 316 filas posteriores se anulan                             |
| personas con `num_group1 >= 1`            | conteo de contactos, conteo por rol/tipo aprobado   | `person_1__n_contacts`, `__contacts__share_role_<cat>`                               | atributos de la solicitud actual                            |

`gender_992L` y `sex_738L` son atributos **sensibles**: B4 los deja como `candidato_G2` con una nota, y no deben entrar al catálogo de características hasta que el equipo decida (sección 6).

### 3.4 `person_2`

| Fuente                                       | Regla                                      | Característica                      | Justificación temporal             |
| -------------------------------------------- | ------------------------------------------ | ----------------------------------- | ---------------------------------- |
| hijos con padre en `person_1`                | conteo                                     | `person_2__n_records`               | atributos de la solicitud actual   |
| `conts_role_79M`, `relatedpersons_role_762T` | proporción de categorías aprobadas         | `person_2__conts_role__share_<cat>` | vocabulario ajustado solo en train |
| `empls_employedfrom_796D`                    | antigüedad máxima si es `<= date_decision` | `person_2__max_employment_days`     | 0 valores posteriores observados   |

Quedan excluidos todos los cuasi-identificadores (direcciones, distritos, códigos postales, nombre del empleador).

## 4. Resolución de la profundidad 2

- **`applprev_2`:** se resuelve primero el padre. Se filtra `applprev_1` por fecha, se obtienen las claves válidas `(case_id, num_group1)` y solo entonces se agregan los hijos de esos padres. Hacerlo al revés mezclaría hijos de padres inexistentes en la decisión.
- **`person_2`:** el padre `(case_id, num_group1)` existe siempre en `person_1`; la unión padre-hijo no pierde filas.
- En ambas, la clave del hijo es `case_id + num_group1 + num_group2`. `num_group1` y `num_group2` nunca son predictores.

## 5. Qué ya está listo

| Entregable                                                               | Dónde                                                                                                                                  | Estado                   |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Fixtures padre-hijo con eventos posteriores, nulos y borde del mismo día | `tests/preparation_fixtures.py` y `tests/test_preparation_fixtures.py`                                                                 | Hecho                    |
| Evidencia reproducible de esta propuesta                                 | `src/dm_credit/preparation/applications_people_evidence.py`                                                                            | Hecho, con pruebas       |
| Selección de columnas B4 (borrador)                                      | `results/preparation/B4_applications_people_columns.csv`, generado con `uv run -m dm_credit.preparation.applications_people_selection` | Borrador a aprobar en G2 |
| Coerción estricta D4                                                     | `src/dm_credit/preparation/coercion.py`                                                                                                | Hecho, con pruebas       |
| Política de extremos D5                                                  | `src/dm_credit/preparation/outliers.py`                                                                                                | Hecho, con pruebas       |

Resultado de B4 para las 95 columnas: 43 candidatas a G2, 19 bloqueadas hasta G2, 22 excluidas, 10 fechas para transformar en antigüedad/recencia y 1 fecha de referencia (`creationdate_885D`).

## 6. Casos ambiguos para decidir en G1/G2

1. **Persona principal:** ¿se acepta `num_group1 = 0` como solicitante principal? Está respaldado por el patrón de nulos, no por una definición oficial.
2. **Atributos sensibles** (`gender_992L`, `sex_738L`): ¿entran al modelo? Implica equidad y regulación, no solo desempeño.
3. **`firstnonzeroinstldate_307D`:** 212 957 filas tienen la fecha de la primera cuota posterior a la decisión. Puede ser calendario contractual conocido; no prueba disponibilidad del resto del estado. Propuesta: anular los valores posteriores y no derivar nada más de ella.
4. **`profession_152M`** (11 558 valores distintos): alta cardinalidad. Propuesta: solo conteo de distintos, sin categorías libres.
5. **Estados, mora, deuda y motivos de rechazo** de `applprev_1` (`status_219L`, `rejectreason*`, `outstandingdebt_522A`, `currdebt_94A`, `credacc_status_367L`, `credacc_actualbalance_314A`, etc.): ¿existe una instantánea válida a `date_decision`? Sin evidencia, siguen bloqueadas.
6. **Fecha nula en `creationdate_885D`** (66 filas): se descartan. Es una regla conservadora; son pocas filas.
7. **Ventana retrospectiva:** A7 (Giancarlo) propone `all` por defecto, igual que este borrador; falta que G1 la apruebe.
8. **`dtlastpmtallstes_3545839D`:** al ejecutar el núcleo sobre datos reales, 351 803 filas **ya válidas** (5,4 %) tienen esta fecha posterior a la decisión, mucho más que `dtlastpmt_581D` (41 684). Una fecha de "último pago" que cae en el futuro sugiere un calendario o una proyección, no un hecho ocurrido. Propuesta: bloquearla hasta G2 en lugar de transformarla en recencia.

## 7. Reproducir la evidencia

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.applications_people_evidence   # cifras de la sección 1
uv run -m dm_credit.preparation.applications_people_selection                      # B4
uv run --extra dev pytest -q tests/test_preparation_*.py
```

## 8. Compatibilidad con los contratos A1/A2 (`03-preparation-A1-A2-contracts.md`)

Revisión del 12/10/2026 contra la decisión A1 y la propuesta de interfaces A2. Este documento es compatible con ambos; quedan estos puntos para G1.

**Coincide:** columnas reservadas idénticas a `dm_credit.baseline.RESERVED`; `FeatureDefinition` (tipo lógico, rol predictor/cobertura, columnas fuente, regla temporal) equivale al catálogo de la sección 3; `source_present` es una característica de cobertura; `coercion.py` encaja como "error de entrada/datos".

**Decisiones que G1 debe tomar:**

1. **Qué es una "familia".** `build_family_features` recibe una sola familia y no recibe tablas de otras. Pero `applprev_2` necesita el padre válido de `applprev_1` y `person_2` el de `person_1`. Propuesta: familia = grupo padre-hijo (`applprev_1`+`applprev_2`, `person_1`+`person_2`, bureau A y B).
2. **Base en la firma.** El split `"oot"` lee archivos de train y el filtro necesita `date_decision`: el adaptador requiere `case_id`, `WEEK_NUM` y `date_decision`. Propuesta: recibir `base` de forma explícita.
3. **Casos sin cobertura.** Falta definir quién completa `n_records = 0` y `source_present = False`. Propuesta: el ensamblador, tras el left join; el resto de características quedan nulas. Afecta a 13 483 casos de `applprev_1` que pierden toda su cobertura tras el filtro, además de los casos que nunca tuvieron historial.
4. **Personas e invariante 5.** `person_*` no tiene fecha por fila; leído estrictamente, el contrato las bloquearía. Se propone aprobar la regla "atributo de la solicitud actual", respaldada por `num_group1 = 0` presente en el 100 % de los casos.
5. **Excepciones.** Propuesta: `CoercionError` y `DomainViolationError` heredan del error de entrada/datos del contrato.
6. **Tipos lógicos.** `FeatureDefinition` debe reutilizar `coercion.LOGICAL_TYPES` (`date`, `int`, `float`, `bool`, `string`).
7. **Orden.** Cada agregado se escribe ordenado por `case_id`.

**Corrección propia:** una versión previa de este documento proponía características `__n_future_<campo>`. Se retiraron: contar fechas posteriores a la decisión es fuga de información.

## 9. Núcleo de C2 y C3 (avance previo a G1)

`src/dm_credit/preparation/families/applications_people.py` implementa la agregación como funciones puras sobre tablas ya leídas: `aggregate_applications` y `aggregate_people` (y sus variantes `*_with_exclusions`, que además devuelven los totales para el manifiesto). No leen disco ni escriben manifiestos, de modo que no dependen de la interfaz `build_family_features` que G1 todavía no congela; cuando se congele, el adaptador será una envoltura delgada.

- El núcleo devuelve una fila por `case_id` solo para los casos con cobertura. Los adaptadores completan una fila por **cada caso de la base**, con `n_records = 0` y `source_present = False` en cada tabla física (`applprev_1`, `applprev_2`, `person_1`, `person_2`), igual que bureau y productos: el ensamblador de Nick no rellena cobertura. El resto de características queda nulo para que lo trate D2.
- La lista de columnas está en `ApplicationsConfig` y `PeopleConfig`. Los valores por defecto son los candidatos de B4 y **no están aprobados**: dependen de G2. Los vocabularios de categorías (`category_shares`) vienen vacíos porque deben ajustarse solo con train (D3).
- Reutilizan la coerción estricta D4. Validan claves únicas, que todo caso exista en la base, que cada caso tenga persona principal y que ninguna característica use filas posteriores a la decisión.

**Verificación con datos reales de train** (un hilo, `POLARS_MAX_THREADS=1`, lectura con proyección de columnas):

| Resultado | Valor | Coincide con la evidencia (sección 1) |
| --- | ---: | --- |
| Casos con cobertura de solicitudes | 1 208 039 | Sí |
| Filas posteriores a la decisión excluidas | 140 040 | Sí |
| Filas con fecha de creación nula excluidas | 66 | Sí |
| Hijos de `applprev_2` con padre inválido | 213 212 | Sí |
| Casos con persona | 1 526 659 | Sí (cobertura 100 %) |
| Empleos posteriores a la decisión anulados | 316 | Sí |
| Edad de la persona principal | 20 a 76 años, media 44 | Plausible |

Aviso: los conteos de fechas internas posteriores (por ejemplo `dtlastpmt_581D` 41 684) se calculan entre filas ya válidas, por eso difieren de los † de la sección 1, que cuentan todas las filas.

**Recursos:** 11 s y un pico de memoria de unos 5,1 GB leyendo las tablas completas en memoria. Eso excede el objetivo de 2 GB por proceso de A5; es aceptable en una máquina de 16 GB pero conviene leer por lotes o por shard en la ejecución real E4.


## 10. Adaptadores de familia (bajo la decisión G1)

`build_applprev_features` y `build_person_features` (en `families/applications_people.py`, accesibles con `build.build_family_features`) envuelven el núcleo: leen con el lector compartido solo los casos de la base, aplican el núcleo y publican el Parquet y su manifiesto de forma atómica. El manifiesto guarda las exclusiones (totales, nunca por caso), la versión de reglas (`A4-v1`) y las huellas de entrada: corte, base y shards.

Ejecución real con train y OOT: ver la sección 7 del documento de [decisión G1](03-preparacion-G1-decision.md). Las columnas por defecto siguen siendo los candidatos de B4, **pendientes de G2**.

## 11. Aporte de `applprev` y `person` al informe de fuga temporal (F7)

Totales por regla, producidos por `uv run -m dm_credit.preparation.families.applications_people_run` y escritos en `results/preparation/F7_applications_people_exclusions.csv`. Son solo para el manifiesto y el informe: ninguna característica por caso cuenta fechas futuras. Los conteos de fechas internas se calculan entre filas ya válidas y solo para las fechas que G2 aprobó: `dtlastpmtallstes_3545839D` (351 803 valores posteriores en train + OOT) quedó **bloqueada** y ya no se construye, por lo que no figura aquí. Con los datos reales:

| Regla (familia) | Train (`< 74`) | OOT (`>= 74`) | Train + OOT |
| --- | ---: | ---: | ---: |
| Solicitud creada después de la decisión, fila excluida (`applprev`) | 117 977 | 22 063 | 140 040 |
| Fecha de creación nula, fila excluida (`applprev`) | 65 | 1 | 66 |
| Hijo de `applprev_2` con padre inválido, excluido | 178 626 | 34 586 | 213 212 |
| `approvaldate_319D` posterior, anulada | 89 | 5 | 94 |
| `dateactivated_425D` posterior, anulada | 10 491 | 1 508 | 11 999 |
| `dtlastpmt_581D` posterior, anulada | 35 089 | 6 595 | 41 684 |
| `employedfrom_700D` posterior, anulada | 0 | 0 | 0 |
| `empl_employedfrom_271D` posterior (`person`), anulada | 312 | 4 | 316 |
| `birth_259D` posterior (`person`) | 0 | 0 | 0 |
| `date_decision` nula | 0 | 0 | 0 |

Las sumas train + OOT coinciden con la evidencia de la sección 1 (140 040, 66, 213 212 y 316), de modo que la partición no pierde ni inventa exclusiones.

**Cobertura** (`results/preparation/C2_C3_applications_people_run.csv`; no cambia con G2, porque depende de las filas válidas y no de las columnas):

| Split | Familia | Casos de la base | Con cobertura | Sin cobertura |
| --- | --- | ---: | ---: | ---: |
| train | `applprev` | 1 332 001 | 1 042 139 | 289 862 |
| OOT | `applprev` | 194 658 | 165 900 | 28 758 |
| train | `person` | 1 332 001 | 1 332 001 | 0 |
| OOT | `person` | 194 658 | 194 658 | 0 |

Los casos sin cobertura se completan en los adaptadores con `n_records = 0` y `source_present = False`; verificado al ensamblar el OOT real con el ensamblador de Nick (194 658 filas, 0 nulos en las columnas de cobertura). El test local (10 casos) pasó la misma construcción y tiene exactamente las mismas características que train (65 de `applprev` y 37 de `person` tras G2): la compatibilidad de esquema con el test real queda comprobada.
