# Tercera evaluación: evidencia de disponibilidad temporal

Fecha: 14/11/2026. Responde la consulta al equipo de [reconciliación y pruebas](03-reconciliacion-vistas-y-pruebas.md) y las preguntas 1, 2 y 4 de [linaje y reglas temporales](02-linaje-y-reglas-temporales.md).

## Alcance y límites

Prueba empírica sobre los parquet crudos de train (semanas 0 a 91, incluido el OOT 74 a 91), más las descripciones de `data/feature_definitions.csv`. No hay README oficial del dataset en el repositorio; las descripciones de Kaggle se usan como evidencia de significado, no de disponibilidad.

Una prueba empírica puede **refutar** que una fuente sea una foto posterior a la decisión, pero no **demostrar** disponibilidad. Las conclusiones se clasifican como evidencia compatible, supuesto de trabajo o bloqueo.

Tampoco se tocó código productivo, matrices ni modelos. Las propuestas de cambio quedan para la decisión G2.

## Corrección a A01

En el equipo de Andre las tres matrices de entrenamiento existen en las rutas declaradas y sus SHA-256 coinciden con E5/E6:

| Matriz | SHA-256 |
| --- | --- |
| `E5-v1/feature_matrix__train.parquet` | `bc073c99…22ac` |
| `E6-v1/linear__train.parquet` | `a1a670fe…0d98` |
| `E6-v1/tree__train.parquet` | `8d0b37f9…ceb` |

A01 describe la máquina donde se ejecutó la auditoría, no la ausencia del artefacto. Los tres controles no ejecutados pueden correrse en ese equipo.

## Evidencia nueva

- [Script reproducible](auditar_disponibilidad.py)
- [Hijos de applprev_2 por antigüedad y periodo](disponibilidad-applprev2-hijos.csv)
- [Roles de person_1](disponibilidad-person1-roles.csv)
- [Contactos de person_1 por periodo y rol](disponibilidad-person1-contactos-periodo.csv)
- [Cobertura de fuentes por periodo](disponibilidad-cobertura-periodo.csv)

## Qué variables están en juego

En el catálogo, de `applprev_2`, `person_2`, `static_0`, `static_cb_0` y `other_1` solo hay **conteos y banderas de presencia**, no valores de medida. De `person_1` hay valores de la persona principal y el conteo `person_1__n_contacts__all`.

Según `matrix_views_spec.csv`, `static_0__*` y `other_1__*` ya se descartan por constantes en train. **No entran al modelo**, así que las preguntas sobre ellas no bloquean el modelado.

## D01. applprev_2: compatible con un conteo cortado en la decisión

**Pregunta 1.** `applprev_2` no tiene ninguna columna de fecha: `cacccardblochreas_147M`, `conts_type_509L`, `credacc_cards_status_52L` y las claves. La única garantía posible es indirecta.

Hechos:

1. Todo padre válido tiene al menos un hijo (`zero_kids_share = 0`) y no hay hijos huérfanos.
2. Los hijos crecen con la antigüedad del padre: 1,52 con 0 días, 2,66 con más de tres años. Los hijos se acumulan durante la vida de la solicitud previa.
3. **Prueba discriminante.** Se fija la cohorte por año de creación del padre y se compara según cuándo se decidió la solicitud actual:

| Año del padre | w00-30 | w31-60 | w61-73 | OOT |
| --- | --- | --- | --- | --- |
| 2017 | 1,70 | 2,17 | 2,20 | 2,27 |
| 2018 | 1,53 | 1,95 | 1,96 | 2,00 |
| 2019 | 1,50 | 1,76 | 1,78 | 1,81 |

Si `applprev_2` fuera una foto tomada al extraer el dataset, una misma cohorte de padres tendría la misma cantidad de hijos con independencia de la fecha de decisión actual. Ocurre lo contrario: las decisiones posteriores ven más hijos. Eso es lo esperado si el proveedor cortó los registros en `date_decision`.

**Límite.** El salto se concentra entre w00-30 y w31-60 y luego se aplana; parte puede deberse a cambios de composición de la cartera. La prueba refuta la hipótesis de foto en extracción, pero no certifica cada fila.

**Clasificación: condicionada.** Uso permitido con el supuesto registrado: «applprev_2 está cortado en date_decision por el proveedor; evidencia indirecta D01». El catálogo debe declarar el denominador (todos los padres válidos) y que no existen padres sin hijos.

## D02. person_1: datos del formulario; `n_contacts` mide otra cosa

**Pregunta 2, disponibilidad.** `personindex_1023L` se define como «Order of the person specified on the application form». Las filas con `num_group1 = 0` traen siempre fecha de nacimiento e ingreso y nunca relación con el cliente; las demás traen relación y nunca ingreso. La estructura es la de un formulario: un solicitante y sus referencias. Solo el 0,056 % de `empl_employedfrom_271D` es posterior a la decisión, y el pipeline ya lo anula. `person_2` no tiene fechas posteriores.

**Clasificación de los valores de la persona principal: aprobada**, con el supuesto «person_1 y person_2 son datos declarados en la solicitud actual».

**Pregunta 2, semántica de `n_contacts`.** Las filas con `num_group1 >= 1` no son todas personas de contacto:

| Rol | Filas | `type_25L` más frecuente | Lectura |
| --- | --- | --- | --- |
| `PE` | 805 942 | `PHONE` | Persona de contacto |
| `CL` | 345 529 | `PRIMARY_MOBILE` | Otro teléfono del propio cliente |
| `EM` | 289 744 | `PHONE` | Teléfono del empleador |

`person_1__n_contacts__all` cuenta **canales de contacto**, no personas. Además, esa mezcla es la causa de su inestabilidad temporal:

| Periodo | `n_contacts` | `PE` | `CL` | `EM` |
| --- | --- | --- | --- | --- |
| w00-30 | 1,05 | 0,57 | 0,26 | 0,22 |
| w31-60 | 1,01 | 0,52 | 0,26 | 0,23 |
| w61-73 | 0,69 | 0,41 | 0,16 | 0,11 |
| OOT | 0,56 | 0,48 | 0,06 | 0,01 |

Las filas `CL` y `EM` casi desaparecen en el OOT, lo que sugiere un cambio en el formulario o en el sistema de captura; las personas `PE` se mantienen. No es fuga, pero perjudica directamente la Gini Stability (OM1) y es un cambio de población que OM4 debe detectar.

**Clasificación: condicionada.** Propuesta para G2: reemplazar o complementar `n_contacts` por un conteo de rol `PE`, y renombrar el actual como conteo de canales. Validar con ablación bajo el protocolo temporal.

## D03. Fuentes estáticas y person_2: sin fuga, pero con cobertura que marca la época

**Pregunta 4.** Presencia por periodo, de [cobertura por periodo](disponibilidad-cobertura-periodo.csv):

| Fuente | w00-30 | w31-60 | w61-73 | OOT | En el modelo |
| --- | --- | --- | --- | --- | --- |
| `static_0` | 100 % | 100 % | 100 % | 100 % | No, constante |
| `static_cb_0` | 95,3 % | 99,98 % | 99,98 % | 99,85 % | Sí |
| `other_1` bruto | 0 % | 2,0 % | 12,9 % | 13,0 % | No, la regla lo anula |
| `person_2` | 99,0 % | 97,4 % | 90,5 % | 69,8 % | Sí |
| `applprev_1` padre válido | 76,7 % | 78,5 % | 85,1 % | 85,2 % | Sí |

- **`static_cb_0`.** Solo varía en las primeras 30 semanas. Su ausencia funciona como un indicador de «periodo antiguo» y no como información del cliente. **Propuesta: excluir** `static_cb_0__n_records` y `static_cb_0__source_present`, motivo indicador de época; la ablación debe confirmar que no se pierde señal.
- **`other_1`.** La regla actual es correcta: la fuente aparece recién desde la semana 31, sin fecha, y usarla introduciría un artefacto de época. **Excluida**, se mantiene.
- **`person_2`.** La presencia cae de 99 % a 70 % en el OOT. La tasa de target de los ausentes pasa de 2,2 veces la de los presentes a 1,1 veces: la relación con `target` cambia. **Condicionada**; entra en la ablación de cobertura y en el diagnóstico de OM4.
- **`applprev_1`.** La presencia sube de forma gradual y la relación con `target` se conserva. Sin observaciones.

## Respuesta a la consulta al equipo

No hay documentación oficial en el repositorio que acredite la disponibilidad. Con la evidencia empírica se registran estos supuestos de trabajo:

| Supuesto | Evidencia | Estado |
| --- | --- | --- |
| applprev_2 está cortado en `date_decision` | D01, prueba de cohortes | Condicionada |
| person_1/2 son datos de la solicitud actual | D02, estructura de formulario y descripciones | Aprobada para valores |
| `n_contacts` cuenta canales, no personas | D02, roles `PE`, `CL`, `EM` | Condicionada, propuesta de cambio |
| Cobertura de `static_cb_0` marca la época | D03 | Propuesta de exclusión |
| `other_1` sin fecha queda excluida | D03 | Excluida, se mantiene |

## Pendientes para la decisión G2

1. Decidir la exclusión de la cobertura de `static_cb_0`.
2. Decidir el conteo de rol `PE` frente al `n_contacts` actual.
3. Actualizar el catálogo C8: denominador de applprev_2, significado de `n_contacts` y los supuestos de esta tabla.
4. Incluir `person_2__*` y las variantes de contactos en la ablación de cobertura del protocolo temporal.
5. La pregunta 3, sobre anclas de productos y registros, queda fuera de esta evaluación.

## Reproducción

```sh
POLARS_MAX_THREADS=4 .venv/bin/python context/auditoria-variables-predictoras/auditar_disponibilidad.py
```

Escribe solo los cuatro CSV `disponibilidad-*.csv` de esta carpeta.
