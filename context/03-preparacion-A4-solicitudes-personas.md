# A4 — Propuesta de agregaciones: solicitudes previas y personas

> **Responsable:** Andre (P3). **Estado:** propuesta para G1/G2; nada de esto está aprobado todavía.
> **Alcance:** `applprev_1`, `applprev_2`, `person_1`, `person_2`, según el [backlog del capítulo 03](03-preparacion-datos-backlog.md).
> **Complementos:** fixtures `tests/preparation_fixtures.py`, selección de columnas B4 en `results/preparation/B4_applications_people_columns.csv`, coerción D4 y extremos D5 en `src/dm_credit/preparation/`.

## 1. Evidencia verificada con los datos de entrenamiento

Cifras producidas por `src/dm_credit/preparation/applications_people_evidence.py` (resultado en `results/preparation/A4_evidence.csv`): solo lectura sobre `train`, con un hilo y sin materializar joins entre familias. El módulo está probado contra los fixtures, cuyos resultados se calcularon a mano. Las cifras marcadas con † **no** salen de ese script: se obtuvieron con consultas de solo lectura ad hoc el 12/10/2026 y todavía no tienen script reproducible (pendiente); la cobertura viene de `results/integration_analysis/family_cardinality.csv`.

### Solicitudes previas

| Hecho | Valor |
| --- | ---: |
| Filas de `applprev_1` / casos cubiertos | 6 525 979 / 1 221 522 |
| Filas con `creationdate_885D` **posterior** a `date_decision` | 140 040 (2,15 %) |
| Anticipación máxima de esas filas † | 14 días después de la decisión |
| Filas con `creationdate_885D` nula | 66 |
| Casos que conservan al menos una fila tras el filtro `<= date_decision` | 1 208 039 (se pierden 13 483 casos, 1,1 %) |
| Hijos de `applprev_2` sin padre en `applprev_1` | 0 |
| Hijos de `applprev_2` totales / con padre temporalmente válido | 14 075 487 / 13 862 275 (se descartan 213 212, 1,5 %) |
| Filas de `applprev_1` con otras fechas posteriores a la decisión † | `approvaldate_319D` 37 383; `dateactivated_425D` 16 669; `dtlastpmt_581D` 42 009; `firstnonzeroinstldate_307D` 212 957 |

El filtro temporal **no es un detalle**: sin él entran 140 040 filas del futuro y 213 212 hijos colgados de padres que no existían.

### Personas

| Hecho | Valor |
| --- | ---: |
| Filas de `person_1` / casos | 2 973 991 / 1 526 659 (cobertura 100 %) |
| Casos con una fila `num_group1 = 0` | 1 526 659 (todos) |
| `num_group1 = 0`: ingresos, sexo y nacimiento informados † | 100 % |
| `num_group1 >= 1`: ingresos, sexo y nacimiento informados † | 0 % |
| `empl_employedfrom_271D` posterior a la decisión | 316 filas |
| `birth_259D` posterior a la decisión | 0 |
| Hijos de `person_2` sin padre en `person_1` | 0 |
| `person_2`: filas con inicio de empleo informado † / posterior a la decisión † | 5 757 (0,35 %) / 0 |

Casos por número de personas †: 1 → 769 339; 2 → 273 106; 3 → 302 446; 4 → 159 315; 5 → 20 987; 6 o más → 1 466.

**Inferencia respaldada por datos:** `num_group1 = 0` es la persona principal de la solicitud; las demás filas son contactos o personas relacionadas sin atributos demográficos propios. Es una inferencia por patrón de nulos, no una definición oficial: se marca como decisión de G1 (sección 6).

## 2. Regla temporal propuesta

1. La referencia es `date_decision` de `train_base`.
2. `applprev_1`: una fila existe para el caso solo si `creationdate_885D <= date_decision`. Fecha nula = se descarta (no se puede demostrar disponibilidad). Igualdad (mismo día) = se conserva.
3. `applprev_2`: un hijo solo se usa si su padre `(case_id, num_group1)` pasó el filtro anterior. Se valida el padre **antes** de agregar.
4. `person_1` y `person_2`: no tienen fecha por fila; son atributos de la solicitud actual. No hay filtro por fila. Las fechas internas (`empl_employedfrom_271D`, `empls_employedfrom_796D`, `birth_259D`) se transforman en antigüedad solo si son `<= date_decision`; un valor posterior se anula y se cuenta.
5. Las otras fechas de `applprev_1` siguen la misma regla por campo: valor posterior a la decisión = nulo y contador de anulados.
6. Ventana retrospectiva: depende de la política común de A7 (Giancarlo). Hasta que G1 la fije, el borrador usa **todo el historial `<= date_decision`**.

## 3. Catálogo inicial de agregaciones

Convención: `<familia>__<columna>__<agregación>`. Una característica entra solo si se describe con familia + columnas fuente + filtro temporal + agregación + significado.

### 3.1 `applprev_1` (filtrada, una fila por caso)

| Fuente | Regla | Característica | Justificación temporal |
| --- | --- | --- | --- |
| todas las filas válidas | conteo | `applprev_1__n_records` | filas con `creationdate_885D <= date_decision` |
| presencia | indicador | `applprev_1__source_present` | distingue "sin historial" (80,01 % de cobertura previa al filtro) de "historial vacío" |
| `creationdate_885D` | recencia y antigüedad | `__days_since_last_creation`, `__days_since_first_creation` | fecha filtrada; nunca futura |
| `credamount_590A`, `annuity_853A` | conteo válido, mínimo, máximo, media, suma | `__credamount__{count,min,max,mean,sum}` y análogas | importes del contrato previo ya creado |
| `tenor_203L`, `mainoccupationinc_437A`, `byoccupationinc_3656910L`, `childnum_21L` | conteo válido, mínimo, máximo, media | `__tenor__*`, etc. | atributos de la solicitud previa |
| `credtype_587L`, `inittransactioncode_279L`, `education_1138M`, `familystate_726L` | conteo de categorías distintas y proporción de categorías aprobadas | `__credtype__n_distinct`, `__credtype__share_<cat>` | categorías con vocabulario ajustado solo en train |
| `isbidproduct_390L`, `isdebitcard_527L` | conteo y proporción verdadera | `__isbidproduct__share_true` | indicadores de la solicitud previa |
| `approvaldate_319D`, `dateactivated_425D`, `dtlastpmt_581D`, `dtlastpmtallstes_3545839D`, `employedfrom_700D` | recencia del máximo válido y conteo de valores anulados por ser posteriores | `__days_since_last_<campo>`, `__n_future_<campo>` | valor `<= date_decision`; el resto se anula |

### 3.2 `applprev_2` (solo vía padre válido)

| Fuente | Regla | Característica | Justificación temporal |
| --- | --- | --- | --- |
| hijos con padre válido | conteo total, hijos por padre (media, máximo) | `applprev_2__n_children`, `__children_per_parent_{mean,max}` | el padre pasó el filtro de `creationdate_885D` |
| `conts_type_509L` | proporción de categorías aprobadas | `__conts_type__share_<cat>` | **bloqueada hasta G2**: sin fecha propia |

`credacc_cards_status_52L` queda excluida (estado sin linaje); `cacccardblochreas_147M` queda bloqueada hasta G2, igual que `conts_type_509L`.

### 3.3 `person_1`

| Fuente | Regla | Característica | Justificación temporal |
| --- | --- | --- | --- |
| fila `num_group1 = 0` (persona principal) | valor directo de cada atributo aprobado en B4 | `person_1__main__education`, `__main__incometype`, `__main__mainoccupationinc`, etc. | atributos de la solicitud actual; sin fecha por fila |
| `birth_259D` (principal) | edad en años a `date_decision` | `person_1__main__age_years` | fecha de nacimiento nunca posterior a la decisión (0 casos) |
| `empl_employedfrom_271D` (principal) | antigüedad laboral en días; anulada si es posterior | `person_1__main__employment_days` | 316 filas posteriores se anulan |
| personas con `num_group1 >= 1` | conteo de contactos, conteo por rol/tipo aprobado | `person_1__n_contacts`, `__contacts__share_role_<cat>` | atributos de la solicitud actual |

`gender_992L` y `sex_738L` son atributos **sensibles**: B4 los deja como `candidato_G2` con una nota, y no deben entrar al catálogo de características hasta que el equipo decida (sección 6).

### 3.4 `person_2`

| Fuente | Regla | Característica | Justificación temporal |
| --- | --- | --- | --- |
| hijos con padre en `person_1` | conteo | `person_2__n_records` | atributos de la solicitud actual |
| `conts_role_79M`, `relatedpersons_role_762T` | proporción de categorías aprobadas | `person_2__conts_role__share_<cat>` | vocabulario ajustado solo en train |
| `empls_employedfrom_796D` | antigüedad máxima si es `<= date_decision` | `person_2__max_employment_days` | 0 valores posteriores observados |

Quedan excluidos todos los cuasi-identificadores (direcciones, distritos, códigos postales, nombre del empleador).

## 4. Resolución de la profundidad 2

- **`applprev_2`:** se resuelve primero el padre. Se filtra `applprev_1` por fecha, se obtienen las claves válidas `(case_id, num_group1)` y solo entonces se agregan los hijos de esos padres. Hacerlo al revés mezclaría hijos de padres inexistentes en la decisión.
- **`person_2`:** el padre `(case_id, num_group1)` existe siempre en `person_1`; la unión padre-hijo no pierde filas.
- En ambas, la clave del hijo es `case_id + num_group1 + num_group2`. `num_group1` y `num_group2` nunca son predictores.

## 5. Qué ya está listo

| Entregable | Dónde | Estado |
| --- | --- | --- |
| Fixtures padre-hijo con eventos posteriores, nulos y borde del mismo día | `tests/preparation_fixtures.py` y `tests/test_preparation_fixtures.py` | Hecho |
| Evidencia reproducible de esta propuesta | `src/dm_credit/preparation/applications_people_evidence.py` | Hecho, con pruebas |
| Selección de columnas B4 (borrador) | `results/preparation/B4_applications_people_columns.csv`, generado con `uv run -m dm_credit.preparation.applications_people_selection` | Borrador a aprobar en G2 |
| Coerción estricta D4 | `src/dm_credit/preparation/coercion.py` | Hecho, con pruebas |
| Política de extremos D5 | `src/dm_credit/preparation/outliers.py` | Hecho, con pruebas |

Resultado de B4 para las 95 columnas: 43 candidatas a G2, 19 bloqueadas hasta G2, 22 excluidas, 10 fechas para transformar en antigüedad/recencia y 1 fecha de referencia (`creationdate_885D`).

## 6. Casos ambiguos para decidir en G1/G2

1. **Persona principal:** ¿se acepta `num_group1 = 0` como solicitante principal? Está respaldado por el patrón de nulos, no por una definición oficial.
2. **Atributos sensibles** (`gender_992L`, `sex_738L`): ¿entran al modelo? Implica equidad y regulación, no solo desempeño.
3. **`firstnonzeroinstldate_307D`:** 212 957 filas tienen la fecha de la primera cuota posterior a la decisión. Puede ser calendario contractual conocido; no prueba disponibilidad del resto del estado. Propuesta: anular los valores posteriores y no derivar nada más de ella.
4. **`profession_152M`** (11 558 valores distintos): alta cardinalidad. Propuesta: solo conteo de distintos, sin categorías libres.
5. **Estados, mora, deuda y motivos de rechazo** de `applprev_1` (`status_219L`, `rejectreason*`, `outstandingdebt_522A`, `currdebt_94A`, `credacc_status_367L`, `credacc_actualbalance_314A`, etc.): ¿existe una instantánea válida a `date_decision`? Sin evidencia, siguen bloqueadas.
6. **Fecha nula en `creationdate_885D`** (66 filas): se descartan. Es una regla conservadora; son pocas filas.
7. **Ventana retrospectiva:** depende de A7.

## 7. Reproducir la evidencia

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.applications_people_evidence   # cifras de la sección 1
uv run -m dm_credit.preparation.applications_people_selection                      # B4
uv run --extra dev pytest -q tests/test_preparation_*.py
```
