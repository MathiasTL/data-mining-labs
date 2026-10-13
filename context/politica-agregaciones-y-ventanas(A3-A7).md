# Propuesta de política de agregaciones y ventanas retrospectivas

> **Estado:** borrador para discutir en G1; no es una política aprobada ni autoriza todavía la inclusión de variables.
>
> **Responsable:** Giancarlo (P2).  
> **Tareas:** A3, catálogo inicial para `static_0` y `static_cb_0`; A7, política común de ventana retrospectiva.

## 1. Qué significa «agregar» en A3

Una agregación resume varias filas relacionadas con el mismo caso en una sola fila. Por ejemplo, de cinco pagos de un caso se podría calcular el número de pagos, el importe promedio o la fecha del pago más reciente. Se hace antes de unir esa familia a la tabla base para evitar duplicar el caso.

Sin embargo, los perfiles de cardinalidad del proyecto muestran que estas dos fuentes no son familias típicas de muchas filas por caso:

- `static_0`: una fila por cada `case_id` de la base (relación 1:1).
- `static_cb_0`: cero o una fila por `case_id` (relación 1:0..1; cobertura aproximada de 98,28 %).

Por tanto, para estas fuentes, «proponer agregaciones» significa sobre todo **proponer qué columnas usar y cómo convertirlas en características**. No hace falta agrupar muchas filas por caso. Si durante la integración aparecieran duplicados, no se deben resolver con un `mean` o un `first` automático: hay que detenerse y aclarar su causa.

Ejemplos del tipo de operaciones que se podrían proponer:

- Mantener un importe ya calculado por la fuente como una característica numérica.
- Mantener una categoría o convertir una bandera booleana, de acuerdo con el preprocesamiento común.
- Convertir una fecha válida en antigüedad respecto de `date_decision`, pero solo si la fecha representa un evento ocurrido y disponible a esa fecha.
- Crear un indicador de presencia de fila para `static_cb_0`, pues la fuente no cubre todos los casos.

## 2. Propuesta inicial para A3

| Fuente / grupo de columnas | Característica candidata | Regla propuesta | Cautela y decisión pendiente |
|---|---|---|---|
| `static_0`: importes, saldos, cuotas y montos, por ejemplo `annuity_780A` | El valor numérico de origen | Conservar como valor por caso; no sumar ni volver a promediar entre filas | Confirmar que el valor estaba disponible en `date_decision`; especialmente revisar importes descritos como «next month» o «current» |
| `static_0`: métricas de mora/pago con ventana en el nombre o definición, por ejemplo `avgdbddpdlast3m_4187120P`, `maxdpdlast12m_727P` | Métrica de mora/pago de la fuente | Conservar por separado y registrar la ventana que ya define la fuente | No recalcular desde los mismos campos ni asumir que el nombre prueba cuándo se calculó el dato |
| `static_0`: fechas de pagos, solicitudes, aprobaciones, rechazos o activación | Recencia en días o meses a la decisión, si aplica | `date_decision - fecha_evento`, solo cuando `fecha_evento <= date_decision` | Si la fecha es posterior o no se puede interpretar, excluirla y registrar el motivo; no usar fechas crudas como predictor por defecto |
| `static_0`: categorías y booleanos de producto/contrato | Categoría o indicador | Mantener el valor lógico/categórico para que lo trate el preprocesador común | Revisar estados actuales/finales y campos con categorías de alta cardinalidad antes de aprobarlos |
| `static_cb_0`: consultas ya resumidas por período, por ejemplo `days30_165L`, `days120_123L`, `days360_512L` | Número de consultas de la ventana indicada | Conservar cada período como característica separada | Confirmar el significado de los nombres y que la consulta/dato estaba disponible al decidir |
| `static_cb_0`: fechas de asignación o respuesta | Recencia o indicador de presencia, solo con semántica verificada | Comparar la fecha con `date_decision`; una fecha futura no se usa | Una fecha de respuesta no demuestra por sí sola que la respuesta se conocía al decidir; verificar el tiempo de disponibilidad |
| `static_cb_0`: fila ausente | `source_present = 0`; fila presente: `source_present = 1` | Crear indicador de cobertura y hacer un `left join` 1:1 | La ausencia no equivale necesariamente a cero; preservar los faltantes de las columnas por separado |
| Ambas fuentes: `case_id` | Ninguna característica predictora | Usarlo solo como clave de unión y control | Excluirlo de predictores |
| `static_cb_0`: `riskassesment_302T` | Excluir por ahora | No incluir en el primer catálogo aprobado | La descripción indica probabilidad de incumplimiento en el futuro; revisar explícitamente por riesgo de fuga del objetivo |

### Reglas generales propuestas

1. Empezar con características simples y trazables: valor de origen, indicador de presencia y recencia derivada de una fecha verificable.
2. No inventar agregaciones estadísticas (`sum`, `mean`, `max`) sobre una fuente que ya trae una fila por caso.
3. No considerar una columna segura solo por ser estática, venir en `static_0`/`static_cb_0` o tener una ventana mencionada en el nombre.
4. Marcar como **pendiente de aprobación** cualquier columna cuya disponibilidad al decidir, semántica o fecha de snapshot no se haya verificado. La revisión temporal existente advierte que los nombres/descripciones por sí solos no verifican el momento en que el dato estuvo disponible.
5. Dejar el preprocesamiento de faltantes, categorías y escalado al diseño común correspondiente; A3 decide la característica y su justificación, no cómo ajustar el preprocesador.

## 3. Qué significa «ventana» en A7

La ventana retrospectiva responde a **cuánto del pasado de un caso se permite usar** para construir una característica.

Supongamos que la decisión de crédito de un caso se toma el **30 de junio**:

- **Todo el historial:** usar los eventos anteriores o iguales al 30 de junio, sin límite de antigüedad.
- **Últimos 3 meses:** usar eventos dentro de los tres meses previos a la decisión y hasta el 30 de junio.
- **As-of:** es el límite que prohíbe mirar al futuro. En cualquier ventana, la fecha del evento debe ser `<= date_decision`.

Una ventana no es lo mismo que la partición train/OOT: `WEEK_NUM < 74` y `WEEK_NUM >= 74` separan los conjuntos para evaluación; la ventana y el límite *as-of* controlan qué información histórica se puede usar para cada caso.

## 4. Política común propuesta para A7

Propongo acordar en G1 estas reglas como punto de partida:

1. **Límite obligatorio:** la fecha del evento debe ser `<= date_decision` y, si se conoce por separado, la fecha en que el dato estuvo disponible también debe ser `<= date_decision`. Un evento futuro o un dato recibido/publicado después se excluye. Si no se puede comprobar la disponibilidad, la característica queda pendiente de revisión en vez de asumirse segura.
2. **Regla por defecto para fuentes de eventos con fechas fiables:** usar todo el historial disponible hasta la decisión (`all`). Así se establece una referencia sencilla sin crear decenas de ventanas arbitrarias.
3. **Ventanas recientes opcionales:** usar ventanas de 3, 6 o 12 meses solo si hay justificación de dominio y una fecha fiable. Aplicar límites de calendario uniformes: `(date_decision - N meses, date_decision]`. La frontera inferior es exclusiva y la fecha de decisión inclusiva.
4. **Campos ya resumidos por la fuente:** conservar la ventana documentada por la fuente (por ejemplo, «últimos 12 meses») como una ventana de origen; no aplicarles una segunda ventana sin evidencia de que se puede reconstruir correctamente.
5. **Fechas nulas, ambiguas o sin fecha de disponibilidad verificable:** no tratarlas como si fueran eventos seguros. Excluir la característica temporal o dejarla pendiente de revisión, y contar la exclusión en el informe de fuga.
6. **Fuentes tipo snapshot o tablas de profundidad 0:** no filtrar ni descartar un valor solo porque no haya varias filas históricas. Primero se debe verificar que el valor o snapshot existía al momento de la decisión. Sin esa evidencia, queda pendiente o excluido preventivamente.

| Tipo de fuente/característica | Ventana propuesta | Condición temporal | Justificación |
|---|---|---|---|
| Eventos con fecha de ocurrencia y disponibilidad fiable | `all` por defecto | Evento `<= date_decision` | Referencia simple que usa el historial conocido hasta la decisión |
| Eventos donde la recencia importe y esté justificada | `3m`, `6m` o `12m` | `(date_decision - N meses, date_decision]` | Permite distinguir actividad reciente sin mirar eventos futuros |
| Resúmenes que ya incluyen un período en su definición | `source_defined` (documentar período exacto) | Verificar disponibilidad del resumen en la decisión | Evita duplicar o alterar la ventana original |
| Snapshot/campo sin fecha de disponibilidad fiable | `pending` (no usar hasta resolver) | No se puede verificar *as-of* | Evita asumir que «estático» significa «conocido en el pasado» |

### Convención de nombres sugerida

Registrar siempre la ventana en el catálogo y, para características nuevas derivadas de eventos, usar un sufijo consistente:

- `{familia}__{caracteristica}__all`
- `{familia}__{caracteristica}__3m`
- `{familia}__{caracteristica}__6m`
- `{familia}__{caracteristica}__12m`
- `{familia}__{caracteristica}__source_defined`

Ejemplos ilustrativos: `credit_bureau_a_1__record_count__all`, `credit_bureau_a_1__record_count__3m`. Para un resumen preexistente, el catálogo debe guardar la ventana explícita de origen; el nombre del campo original puede preservarse si cambiarlo no está aprobado. Esta convención queda sujeta al acuerdo de nombres de G1.

## 5. Decisiones para llevar a G1

- Aprobar o cambiar la regla por defecto `all` para eventos fiables.
- Confirmar si las ventanas recientes serán de calendario (propuesta) o días fijos.
- Acordar si `source_defined` se representa como sufijo, como metadato del catálogo o ambos.
- Definir qué evidencia basta para considerar que un snapshot/campo estaba disponible en `date_decision`.
- Aprobar individualmente las columnas ambiguas o mantenerlas excluidas/pendientes.

## Referencias del repositorio

- [Backlog de preparación de datos](03-preparacion-datos-backlog.md), tareas A3/A7 y compuerta G1.
- [Guía de preparación de datos](03-preparacion-datos.md), reglas de construcción de características y temporalidad.
- `results/integration_analysis/family_cardinality.csv`, cardinalidad observada por familia.
- `results/integration_analysis/temporal_leakage_review.csv`, revisión inicial de variables que requieren regla temporal.
