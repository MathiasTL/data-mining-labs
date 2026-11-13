# Segunda evaluación: linaje y reglas temporales

Fecha: 13/11/2026. Estado: revisión estática parcial; no autoriza todavía el modelado formal.

## Alcance y límites

Continuación de [01-inventario-y-hallazgos.md](01-inventario-y-hallazgos.md), centrada en las tres características estructurales sin `source_columns` y las reglas de las 30 características de cobertura de [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv).

Se leyeron los [objetivos empresariales](../../development/01-entendimiento-empresarial.md): evaluación para originación (línea 15), estabilidad temporal y revisión de disponibilidad como mitigación de fuga (líneas 66–71). Se contrastó el catálogo con los constructores actuales de `src`.

**Límite: revisión estática.** No se inspeccionaron valores de matrices, no se regeneraron artefactos, no se entrenaron modelos ni se reejecutaron controles históricos. No se acredita que los artefactos archivados se hayan construido con la versión actual del código. Las referencias a líneas corresponden al código leído durante esta revisión.

La ausencia de `source_columns` no demuestra fuga. La existencia de un filtro tampoco demuestra por sí sola que su ancla represente disponibilidad de información en originación.

## Checklist parcial

- [x] Leer el inventario inicial y los objetivos empresariales actuales.
- [x] Localizar las implementaciones de las tres características estructurales.
- [x] Identificar claves, población contada, filtros y tratamiento de padres sin hijos.
- [x] Explicar la generación de los `source_columns` vacíos.
- [x] Contrastar estáticamente las reglas de cobertura del catálogo con sus constructores.
- [x] Separar filtros ejecutables, supuestos de procedencia y garantías temporales pendientes.
- [ ] Acreditar con documentación fuente la disponibilidad de hijos, contactos y filas estáticas en originación.
- [ ] Justificar que las anclas elegidas representan disponibilidad y no solo fecha de evento.
- [ ] Revisar exhaustivamente decisiones G2 y sus evidencias de aprobación; esta evaluación identifica su uso en el código, no ratifica esas decisiones.
- [ ] Comprobar valores y exclusiones en artefactos y reconciliarlos con las fuentes.
- [ ] Demostrar correspondencia entre código actual y ejecución histórica.
- [ ] Aprobar el protocolo temporal antes de nuevos experimentos.

## H01. El vacío de `source_columns` responde a una convención del generador

**Clasificación:** explicación técnica verificada; suficiencia del linaje pendiente.

En [contracts.py](../../src/dm_credit/preparation/contracts.py), líneas 55–65 y 70–75, los nombres de cobertura y los nombres `fuente__métrica__ventana` se interpretan con `variable=""`. En las líneas 149–153, `describe_features()` asigna `source_columns=()` cuando no hay variable.

Así se explican las tres entradas estructurales y las 30 de cobertura. El campo enumera una variable de medida cuando el nombre la incluye; no captura automáticamente claves, predicados, anclas o dependencias entre tablas. Las descripciones también se generan de forma genérica (líneas 154–169).

**Conclusión:** no son 33 pruebas de defecto ni de fuga. Sí existe una limitación documental: esos campos no bastan para reconstruir el linaje completo de los conteos.

## H02. `applprev_2`: máximo y media de hijos por padre

Entradas: [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv), líneas 63–64.

Implementación: [applications_people.py](../../src/dm_credit/preparation/families/applications_people.py), líneas 230–292.

1. Unicidad exigida: padres `(case_id, num_group1)`; hijos `(case_id, num_group1, num_group2)` (233–234).
2. Se incorpora `date_decision` desde la base y se conservan padres con `creationdate_885D <= date_decision`; la fecha de creación nula no cumple el filtro (247–253).
3. Los hijos se conservan mediante un semi-join con las claves de padres válidos (273–276).
4. Se cuentan hijos por padre y se completa con cero a los padres válidos sin hijos (277–280).
5. Por `case_id`, se calcula la media y el máximo de esos conteos (281–284).

**Población y denominador:** la media incluye todos los padres válidos, también los que tienen cero hijos. No es una media únicamente sobre padres con hijos. Con padres válidos pero sin hijos, ambas métricas son cero. Sin padres válidos, no se genera una fila agregada para estas métricas; el adaptador completa únicamente la cobertura y deja los demás valores nulos (406–414 y 476–477).

**Discrepancia documental:** el catálogo no explica el denominador, los ceros ni la dependencia de `applprev_1` y de la base. `source_table=applprev_2` identifica la tabla contada, no el linaje completo. La regla catalogada sí coincide con el filtro de creación del padre y la exigencia de padre válido.

**Límite temporal:** no hay filtro temporal independiente sobre los hijos. El código garantiza asociación con un padre temporalmente elegible según su fecha de creación; no garantiza por sí mismo que cada fila hija ya estuviera disponible al decidir la solicitud actual.

## H03. `person_1__n_contacts__all`: conteo por índice de grupo

Entrada: [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv), línea 88.

Implementación: [applications_people.py](../../src/dm_credit/preparation/families/applications_people.py), líneas 315–359.

- Exige unicidad por `(case_id, num_group1)` (318).
- Exige exactamente una persona principal con `num_group1=0` por caso presente en `person_1` (332–334).
- Calcula por `case_id` la suma de `(num_group1 >= 1)` (355–357).

**Qué cuenta:** filas de `person_1` cuyo índice de grupo es al menos uno. No es un conteo de teléfonos, canales ni personas deduplicadas por un identificador personal.

**Discrepancia documental:** la descripción genérica no declara el predicado. El significado empresarial de «contacto» depende de respaldar esa interpretación del índice con documentación del dataset.

**Límite temporal:** las fechas internas futuras se anulan, pero no se eliminan sus filas (207–210 y 338–340). El conteo de contactos no está condicionado a una fecha de alta o disponibilidad de cada fila.

## H04. Reglas de cobertura: poblaciones distintas bajo un mismo rol

`n_records` y `source_present` no describen siempre presencia física bruta. Según la familia, describen filas temporalmente elegibles, presencia de la solicitud o una política de exclusión.

### Solicitudes previas

Catálogo: [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv), líneas 56, 58 y 65–66.

Código: [applications_people.py](../../src/dm_credit/preparation/families/applications_people.py), líneas 253–260 y 273–292.

- `applprev_1__n_records`: número de padres que superan el filtro de creación.
- `applprev_2__n_records`: suma de hijos asociados a padres válidos.
- La presencia de hijos es `n_records > 0`.
- La cobertura ausente se completa con cero y falso (406–414).

Coincide con la regla general del catálogo. Este omite la exclusión de fechas de creación nulas y no explicita la ausencia de comprobación temporal independiente del hijo.

### Personas

Catálogo: [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv), líneas 89, 112 y 114–115.

Código: [applications_people.py](../../src/dm_credit/preparation/families/applications_people.py), líneas 330–369.

- Cuenta las filas de `person_1` y `person_2`.
- Exige que los hijos de `person_2` tengan padre existente (335–336).
- Anula fechas internas futuras sin eliminar filas (338–340).
- Calcula cobertura después de ese tratamiento, sin filtro temporal de población (355–369).

**Discrepancia de alcance:** «atributos de la solicitud actual; fechas internas <= date_decision» no equivale a demostrar la fecha de disponibilidad de contactos o relaciones. Es una declaración de procedencia acompañada de un tratamiento de valores.

### Bureau

Catálogo: [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv), líneas 67–72.

Código: [bureau.py](../../src/dm_credit/preparation/families/bureau.py).

Anclas declaradas (101–108):

- `credit_bureau_a_1`: `lastupdate_1112D` y `lastupdate_388D`.
- `credit_bureau_b_1`: `lastupdate_260D`.
- `credit_bureau_b_2`: `pmts_date_1107D`.

El filtro exige al menos un ancla presente y ninguna ancla presente que incumpla la regla; después exige padre válido para hijos (293–311). Los conteos se calculan tras preparar las filas (318–320 y 453–455), y la cobertura se obtiene de esos conteos (382–385).

**Ambigüedad verificable:** el catálogo y `_ROW_RULE` dicen a la vez «al menos un ancla no nula» y «ancla nula: fila descartada» (666–677). El código descarta por ausencia cuando no hay ninguna ancla presente (300–307). En `credit_bureau_a_1`, una ancla nula y otra válida permiten conservar la fila. La frase final no debe interpretarse como descarte por cualquier ancla nula.

El plan permite omitir tablas sin anclas aprobadas como referencias temporales y omitir hijos si no se procesa el padre (238–255). La omisión no debe confundirse con cobertura cero de una tabla procesada. El catálogo revisado no incluye `credit_bureau_a_2`.

### Productos y registros fiscales

Catálogo: [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv), líneas 73–78 y 120–125.

Código: [products_registry.py](../../src/dm_credit/preparation/families/products_registry.py).

| Fuente | Ancla usada |
| --- | --- |
| `debitcard_1` | `openingdate_857D` |
| `deposit_1` | `openingdate_313D` |
| `tax_registry_a_1` | `recorddate_4527225D` |
| `tax_registry_b_1` | `deductiondate_4917603D` |
| `tax_registry_c_1` | `processingdate_168D` |

Las anclas están en 49–55. Se excluyen filas con ancla nula o posterior a la decisión (181–200). El constructor ejecuta el filtro antes del agregado (309–310); el conteo y la presencia están en 220–229. Hay coincidencia entre catálogo y código.

Para `other_1`, sin ancla, el código devuelve todas las filas descartadas (188–190) y la regla lo declara explícitamente (59–69). **Su cobertura cero significa ninguna fila admitida por la política, no necesariamente ausencia física de registros.**

### Estáticas

Catálogo: [C8_feature_catalog.csv](../../results/preparation/C8_feature_catalog.csv), líneas 116–119.

Código: [depth0.py](../../src/dm_credit/preparation/families/depth0.py), líneas 51–67 y 133–165.

La lectura restringe filas a casos de la base y exige unicidad de `case_id`, sin filtro de fecha. El conteo se calcula sobre las filas fuente y la presencia como `n_records > 0` (147–154).

**Discrepancia de alcance:** la regla compartida (19–22) habla de columnas aprobadas G2 y revisión temporal de predictores. No describe una comprobación temporal de existencia de la fila. La cobertura no exige que haya alguna columna predictora aprobada: el conteo depende de las filas, no de sus medidas.

## Supuesto, decisión G2 y garantía temporal: no son equivalentes

- **Supuesto de procedencia:** llamar a `person` «atributos de la solicitud actual» expresa una interpretación del dataset; no añade al código una prueba de disponibilidad de cada fila.
- **Decisión G2:** habilitar columnas o referencias temporales es una decisión de gobierno del pipeline. Su uso está localizado en el código, pero esta revisión no examinó exhaustivamente las evidencias que justifican cada aprobación.
- **Garantía operativa del filtro:** el código puede garantizar que una fecha seleccionada no sea posterior a `date_decision`, o que un hijo tenga padre elegible.
- **Garantía temporal empresarial:** exige además justificar que esa fecha o relación demuestra que la información estaba disponible en originación. Una fecha de evento, creación o apertura no debe equipararse automáticamente con fecha de disponibilidad de toda la fila.

Por tanto, ni una aprobación G2 ni un campo textual completo bastan por sí solos para certificar ausencia de fuga. Tampoco la ausencia de una fecha o de `source_columns` basta para afirmar que existe fuga.

## Preguntas genuinas pendientes

1. ¿Qué evidencia garantiza que las filas de `applprev_2` ya existían al decidir la solicitud actual, además de pertenecer a un padre con creación anterior?
2. ¿Está documentado que `person_1/2` son una fotografía disponible en originación y que `num_group1 >= 1` representa los contactos que se pretende contar?
3. ¿Las anclas de productos y registros acreditan disponibilidad de la fila completa o solo fecha del evento?
4. ¿Qué evidencia respalda que la presencia de `static_0/static_cb_0` era observable al decidir, independientemente de las columnas aprobadas?
5. ¿El contrato del catálogo pretende que `source_columns` enumere solo medidas o todas las dependencias? El generador actual implementa la primera interpretación para estos nombres.

## Acciones propuestas, sin alterar código productivo

1. Registrar en documentación de auditoría el linaje estructural: claves, tablas relacionadas, predicados, denominadores y distinción entre cero y nulo.
2. Acordar y documentar la convención de `source_columns`; no rellenarlo mecánicamente como si todos los campos fueran medidas.
3. Precisar documentalmente que bureau descarta por ausencia de todas las anclas, no por cualquier ancla nula.
4. Distinguir cobertura física, cobertura temporalmente elegible y cobertura cero por política de exclusión.
5. Solicitar o localizar evidencia del dataset para responder las preguntas temporales antes de aprobar estas características para originación.
6. Registrar por separado la aprobación G2, su fundamento y la garantía concreta que implementa cada filtro.
7. Planificar posteriormente controles de valores, exclusiones y procedencia de artefactos, sin ejecutarlos ni sustituir evidencias históricas inadvertidamente.

## Conclusión

Las tres características estructurales están localizadas y sus fórmulas son verificables. Los vacíos de `source_columns` son explicables por el generador, aunque el catálogo no expresa suficientemente sus dependencias. Las reglas de cobertura presentan una ambigüedad sobre anclas nulas en bureau y diferencias de alcance en personas y estáticas.

Esta evaluación no demuestra fuga ni ausencia de fuga. Tampoco acredita equivalencia entre código actual y matrices históricas. Las cuestiones pendientes son de evidencia temporal y documentación de linaje, no una autorización para cambiar el código productivo o regenerar matrices.
