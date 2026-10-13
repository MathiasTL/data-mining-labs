# G1 — Revisión técnica de contratos e integración

**Estado:** recomendaciones de Nick para revisión por Claude y aprobación conjunta; **G1 no está aprobada**.  
**Insumos:** [A1/A2](03-preparation-A1-A2-contracts.md), [A3/A7](<politica-agregaciones-y-ventanas(A3-A7).md>), [A4](03-preparacion-A4-solicitudes-personas.md), [A5](03-preparacion-A5-bureau-streaming.md), [A6](03-preparacion-A6-faltantes-categorias.md), [backlog](03-preparacion-datos-backlog.md), `dm_credit.io`, `dm_credit.baseline`, `preparation.coercion` y `preparation.outliers`.

## Resolución recomendada de las doce decisiones de G1

1. **Interfaz de familia.** Una familia lógica agrupa las tablas que necesitan validar una relación padre-hijo: `applprev` (`applprev_1/_2`), `person` (`person_1/_2`), `credit_bureau_a` (`a_1/_2`) y `credit_bureau_b` (`b_1/_2`). La familia posee sus claves intermedias y entrega **un** agregado por `case_id`; nunca cruza con otra familia. Para la regla temporal necesita la base del split, no solo el nombre del split. Firma orientativa ya incorporada como propuesta revisada en A2, pendiente de aprobación:

    ```python
    build_family_features(
        family: str,
        base: pl.DataFrame,  # case_id, WEEK_NUM, date_decision; target no se usa
        split: Literal["train", "oot", "test"],
        *,
        cutoff_week: int = 74,
        output_dir: Path,
    ) -> FamilyFeatureArtifact
    ```

    El llamador valida base única, sin claves/semanas/fechas nulas, con train `< 74`, OOT `>= 74` y test separado y sin `target`. El adaptador valida de nuevo los supuestos necesarios. `FamilyFeatureArtifact` contiene familia, split, rutas Parquet/JSON, conteos y esquema **ordenado**; el manifiesto es la fuente persistida de auditoría, no un segundo estado independiente. El agregado vacío es válido solo con esquema lógico declarado, no ante un shard requerido ausente. `assemble_feature_matrix(base, family_artifacts, split, *, output_dir)` acepta solo artefactos compatibles y valida claves, filas, casos y orden después de cada left join. La forma exacta de las dataclasses es trabajo de P1, pero estos invariantes no son opcionales.

2. **Nombres.** Para características derivadas, `<fuente_fisica>__<variable_o_metrica>__<agregacion>__<ventana>` en minúsculas; ventana `all`, `3m`, `6m`, `12m` o `source_defined`. Excepciones explícitas: `<fuente_fisica>__n_records`, `<fuente_fisica>__source_present` y `<feature>__missing`; recencia usa agregación `days_since_last`/`days_since_first`. El prefijo físico conserva linaje dentro de la familia lógica. Revisar los nombres ilustrativos de A4/A7 antes de implementarlos: hoy no siguen una única gramática. Categorías incrustadas en nombres solo si están aprobadas y su normalización no colisiona; de lo contrario, fallar o asignar un identificador estable documentado. `case_id`, índices de grupo y columnas temporales/objetivo nunca son características.

3. **Catálogo/manifiesto.** CSV versionado de características (nombre, rol, tipo lógico, familia, fuente/columnas, agregación, ventana, regla temporal, descripción); manifiesto JSON versionado por agregado con ruta relativa, split, reglas/configuración, orden/tipos, conteos, exclusiones temporales por motivo, validaciones, huellas de entradas y métricas de ejecución. Ninguno incluye filas/muestras sensibles. La especificación de preprocesamiento conserva parámetros y orden separados de los manifiestos de fuente. Salidas temporales se validan antes de publicación atómica; no sobrescribir una salida válida antes de aprobar el reemplazo.

4. **Agregaciones iniciales.** Conteos y cobertura para toda fuente; para numéricas `n_valid`, `n_null`, `sum`, `min`, `max`, media (`sum/n`) y desviación con parciales combinables cuando corresponda; booleanas `n_true` y proporción; categóricas solo frecuencias/proporciones de categorías aprobadas. Fuentes 1:1/0:1 conservan valor y fallan ante duplicados. El backlog también prevé `n_distinct`: **no prohibirlo globalmente**, pero en bureau/registro masivo posponer el exacto salvo que se demuestre coste acotado o se apruebe un algoritmo aproximado y su error; para fuentes pequeñas, admitirlo con prueba. Mediana/percentiles no entran en el camino streaming inicial. Las columnas concretas se aprueban en G2.

5. **Tipos, nulos y categorías.** Tipos de `coercion.LOGICAL_TYPES` (`date/int/float/bool/string`). Convertir `Null` o columnas enteramente nulas al tipo lógico declarado; `NaN` → nulo, infinito/no convertible → error. Fechas futuras no producen característica ni conteo por caso: excluir la fila cuando determine su existencia temporal, o anular **solo el campo** cuando sea una fecha interna. Registrar conteos agregados en el manifiesto. `__missing__` representa ausencia; `__UNKNOWN__` representa categoría nueva/rara; el vocabulario y el orden se ajustan solo en train `< 74`. `min_frequency=0.01` puede ser default **configurable** de la vista lineal, no prueba de que una categoría/columna sea temporalmente válida ni parámetro universal de las agregaciones.

6. **Errores y advertencias.** Error por contrato (split/clave/manifest/schema/colisiones), disponibilidad temporal no demostrada para una característica solicitada, coerción imposible, infinito y dominio aprobado violado. No fallar por un evento futuro esperado: se excluye/anula y se audita. Fuera del rango observado de train produce `RangeWarning`, no recorte automático. `CoercionError` y `DomainViolationError` ya heredan de `ValueError`: no exigir cambiar su jerarquía para G1; P1 puede envolverlos conservando causa/contexto. `fit_range_policy` solo comprueba el corte cuando recibe `WEEK_NUM`; D1 debe validarlo obligatoriamente antes de `fit`.

7. **Rutas.** `data/artifacts/preparation/` para salidas pesadas y checkpoints (el directorio `data/` ya está ignorado por git); una subruta por configuración/versión, fuente o familia y split. Cada parcial lleva huella de shards, versión de reglas, **base de referencia y corte**: cambiar `date_decision`, asignación train/OOT o cutoff invalida su reutilización. Un fichero físico vacío con esquema esperado se acepta; falta de shard no. Catálogo/configuración/versiones pequeñas pueden estar bajo control de versiones, sin datos crudos.

8. **Comando reproducible.** Proponer `uv run -m dm_credit.preparation.build --config configs/preparation.json`; JSON evita añadir PyYAML al `pyproject.toml` actual. Es un **comando futuro**, no existe aún. Configuración versionada con corte 74, familias, ventanas aprobadas, reglas y opciones de lectura; rutas locales/salidas y huellas efectivas se registran en la ejecución. P1 posee CLI/config y documenta uso de `DM_DATA_DIR` de `dm_credit.io`.

9. **Orden y presupuesto real.** Aplicar cola única `[R]`, `POLARS_MAX_THREADS=1`, medición de tiempo/pico y límites por lote. Seguir el orden recomendado en el backlog §7: fixture/fuente pequeña → depth 0 → productos → solicitudes/personas → bureau B → registros → bureau A, con `a_2` al final → ensamblado/preprocesamiento. Dentro de bureau, padre antes de hijo. Train/OOT pueden reutilizar una pasada por shards train **solo si** los parciales son por caso, no aprenden entre casos y coinciden huellas de datos/base/reglas/corte; publicar matrices separadas. El objetivo de 2 GB de A5 es una estimación, no una garantía medida; si no cabe el semijoin de claves padre, particionar/volcar a disco antes que hacer un join masivo.

10. **Propiedad.** P1: contrato compartido, split, validador de esquema, ensamblado, CLI/config, orquestación y adaptación de `dm_credit.io` para iterar por lotes/row group con proyección; P4 revisa ese seam por las necesidades de bureau. P2: depth 0, vistas y política de ventana/informe de fuga; P3: solicitudes/personas, coerción y extremos; P4: bureau; P5: productos/registros y preprocesador. Nadie modifica el archivo de otro sin coordinar. Los módulos de familia no incorporan lectores privados: `io.load_parquet` hoy materializa la tabla completa y **no sirve** para A5.

11. **Ventana común.** Para eventos con fecha y disponibilidad verificadas, `all` por defecto hasta la decisión (inclusiva); ventanas opcionales `(date_decision - N meses, date_decision]` solo con justificación y calendario uniforme. Resúmenes ya calculados conservan su ventana `source_defined` si se demuestra su disponibilidad; snapshots no son automáticamente seguros por ser estáticos. Para fuentes mensuales, la regla conservadora es mes **estrictamente anterior** al mes de decisión, pero el emparejamiento de campos año/mes de `credit_bureau_a_2` y su aceptación son de B5/G2. Sin fecha fiable, no incluir por defecto.

12. **Controles y vistas.** `case_id`, `WEEK_NUM`, `date_decision` y `target` en train/OOT permanecen como controles; `MONTH` se conserva como control si viene en base. `num_group1/num_group2` son claves técnicas de las fuentes y **no** se llevan a la matriz final como predictores. Cada vista preserva el mismo conjunto/orden de casos y traza las características de origen: lineal imputada/escalada/codificada, árboles con nulos/categóricas nativos donde la implementación lo permita. El esquema lógico y orden de cada vista quedan congelados tras ajuste en train; test no tiene `target`. `baseline.validate_contract` no obliga `date_decision` ni verifica explícitamente ausencia de `target` en test: D6 debe comprobar ambos; para probar `run_baseline` sin cambios se necesita entregarle train y OOT juntos con el corte 74, no cada partición aisladamente.

## Contradicciones resueltas y límites de evidencia

- **A2 vs A4/A5 (padre-hijo, base):** el límite de familia es el grupo padre-hijo; el adaptador recibe base mínima ya particionada. Los datos padre son internos de esa familia. La firma original de A2 sin `base` no podía evaluar `date_decision` ni separar OOT de los archivos físicos train; **A2 ya se revisó** para reflejar la evidencia de A4/A5. Sigue pendiente de ratificación en G1; no hay código implementado.
- **Lectura central vs streaming:** `dm_credit.io.load_parquet` usa `pl.read_parquet`; A5 documenta 188 298 452 filas de `credit_bureau_a_2`. Se necesita un lector central de row groups con proyección antes de C4; P1 lo implementaría con revisión P4.
- **Persona principal y atributos sin fecha:** A4 verifica que `num_group1=0` existe para todos los casos de train y observa patrones de nulos, pero lo denomina explícitamente **inferencia**, no definición oficial. Se puede usar `num_group1=0` como selección estructural para ensayos; **no** aprobar por eso edad, empleo, sexo ni ningún atributo como predictor hasta comprobar disponibilidad/uso permitido en B4/G2. Los datos sensibles se excluyen por defecto, pendiente de decisión de equidad/uso.
- **`n_distinct`/ventanas vs memoria:** el backlog admite conteos distintos y ventanas opcionales, A5 muestra por qué ambos pueden disparar los parciales. Separar contrato general de capacidades por familia, sin exigir un cómputo exacto no acotado a bureau.
- **Imputación y recorte:** los valores sin cobertura pasan a `n_records=0` y `source_present=False` **por fuente física** tras el left join; otras características quedan nulas, no cero. `outliers.py` ya exige `clip_columns` explícitas, por tanto no activar winsorización general ni afirmar que solo va en una vista sin decisión posterior de D7.
- **Enlaces a A1/A2:** A4–A6 ya apuntan al documento canónico `03-preparation-A1-A2-contracts.md`. El antiguo `03-preparation-contracts.md` no existe en el árbol actual.

## Para cerrar G1 tras esta revisión

Claude y el equipo deben validar o corregir estas doce recomendaciones, especialmente **firma con base**, **familias padre-hijo**, **dueño/seam IO**, **gramática de nombres**, **cache de parciales** y **manejo conservador de personas/snapshots**. Después, P1 congela la propuesta revisada de A2 en el contrato canónico y se acuerdan formato exacto de `FeatureDefinition`/manifiestos, errores y mapa de archivos. La lista/semántica de columnas sigue B3–B6/G2: esta acta no autoriza su inclusión ni declara G1 aprobada.

## Revisión de Andre (P3), 12/10/2026

Revisión técnica de las doce recomendaciones contra el código y los datos. **Este apartado no aprueba G1**: la aprobación es conjunta (Nick, Giancarlo, Mathias, Gabriel y Andre).

### Afirmaciones del acta verificadas

- `baseline.validate_contract` exige `case_id` y `WEEK_NUM`, y `target` solo si `require_target=True`: **no** exige `date_decision` ni comprueba que `target` falte en test. Correcto; D6 debe verificar ambos.
- `CoercionError` y `DomainViolationError` heredan de `ValueError`; `RangeWarning` de `UserWarning`. Correcto.
- `fit_range_policy` solo aplica la guarda de semana 74 si la tabla trae `WEEK_NUM`. Correcto; D1 debe exigirlo antes de `fit`.
- `io.load_parquet` usa `pl.read_parquet` y materializa la tabla completa. Correcto: no sirve para `credit_bureau_a_2`.
- `data/` está en `.gitignore`. Correcto.

### Veredicto por decisión

| # | Decisión | Veredicto | Nota |
| -: | --- | --- | --- |
| 1 | Familia padre-hijo y firma con `base` | De acuerdo | Es lo que necesita `applprev` y `person` (A4 §8). El núcleo de C2/C3 ya recibe tablas y `base`, sin leer disco, y se envolverá cuando P1 congele las dataclasses. |
| 2 | Gramática de nombres | De acuerdo, con dos ajustes | Ver "Ajustes" 1 y 2. |
| 3 | Catálogo y manifiesto | De acuerdo | El núcleo ya devuelve los totales de exclusión por separado (`exclusions`) para el manifiesto. |
| 4 | Agregaciones iniciales | De acuerdo | `n_distinct` en `applprev` es barato (el núcleo completo tarda 11 s); la cautela para bureau es correcta. |
| 5 | Tipos, nulos y categorías | De acuerdo | El núcleo anula fechas internas futuras y no las cuenta por caso. |
| 6 | Errores y advertencias | De acuerdo | |
| 7 | Rutas | De acuerdo, con un ajuste | Ver "Ajustes" 3. |
| 8 | Comando reproducible | De acuerdo | |
| 9 | Orden y presupuesto | De acuerdo, con una medición | Ver "Ajustes" 4. |
| 10 | Propiedad | De acuerdo | P3 envolverá el núcleo en `build_family_features`; no habrá lector privado. |
| 11 | Ventana común | De acuerdo | Coincide con el borrador de A4 (`all`). |
| 12 | Controles y vistas | De acuerdo | El núcleo no emite `num_group1` ni `num_group2`. |

### Ajustes propuestos

1. **Nombre de recencia.** La gramática del acta (§2) manda `days_since_last` como agregación: `applprev_1__creationdate_885d__days_since_last__all`. El núcleo y A6 usan `applprev_1__days_since_last_creationdate_885d__all`. Es un renombrado mecánico; se hace cuando G1 fije la gramática.
2. **Expresión regular de A6.** `^[a-z][a-z0-9_]*(__[a-z0-9_]+){1,3}$` no admite `<feature>__missing` cuando la característica ya tiene cuatro segmentos (`applprev_1__credamount_590a__mean__all__missing` tiene cuatro tras el prefijo). Ampliar a `{1,4}` o permitir un sufijo opcional `__missing`.
3. **Ruta de artefactos.** Derivarla de `io.data_dir() / "artifacts" / "preparation"` y no de un `data/` literal, porque `DM_DATA_DIR` puede apuntar fuera del repositorio.
4. **Presupuesto de memoria.** Medido con datos reales, un hilo y lectura completa en memoria: `applprev_1` + `applprev_2` + `person_1` + `person_2` tardan 11 s y alcanzan un pico de unos 5,1 GB. Cabe en 16 GB pero supera los 2 GB de A5. Para `applprev` y `person` es aceptable mientras llega el lector por lotes de P1; no debe tomarse como medida válida para bureau.

### Brecha corregida en el núcleo

El acta pide cobertura "por fuente física". El núcleo solo emitía `source_present` para `applprev_1` y `person_1`; ahora también lo emite para `applprev_2` y `person_2` (verdadero si hay al menos un hijo). Tiene prueba propia.

### Hallazgo de datos para G2

`dtlastpmtallstes_3545839D` tiene 351 803 filas ya válidas (5,4 %) con fecha posterior a la decisión, frente a 41 684 de `dtlastpmt_581D`. Parece un calendario, no un hecho ocurrido. Propuesta: bloquearla hasta G2.

### Qué falta para cerrar G1

Reunión conjunta para un sí o no sobre cada uno de estos puntos del acta: firma con `base`, familias padre-hijo, dueño del lector por lotes en `io`, gramática de nombres (con los ajustes 1 y 2), caché de parciales y manejo de personas y snapshots. Después, P1 congela A2 en el contrato canónico y se acuerdan el formato de `FeatureDefinition` y los manifiestos. G1 sigue sin estar aprobada.

## Revisión de Mathias (P4), 13/10/2026

Revisión del lector compartido de `dm_credit.io` (`shard_paths`, `scan_parquet_shards`, `iter_row_groups`), asignada a P4 en la decisión 10 del acta y en el punto 3 de la [decisión G1](03-preparacion-G1-decision.md). Se comprobó contra los archivos reales de bureau, leyendo como máximo un grupo de filas por fuente. **No modifica `io.py`**: son hallazgos para su responsable.

### Comprobado y conforme

| Aspecto | Resultado |
| --- | --- |
| Descubrimiento de shards | `credit_bureau_a_1`: 4 en train y 5 en test. `credit_bureau_a_2`: 11 en train y 12 en test. `credit_bureau_b_1` y `credit_bureau_b_2`: 1 en cada split. |
| Orden numérico | `train_credit_bureau_a_2_10.parquet` queda al final, después de `_9`. |
| Memoria por lote | Un grupo de filas de `train_credit_bureau_a_2_5.parquet` (1 048 576 filas) ocupa unos 153 MB con las 19 columnas y unos 89 MB con 11 columnas numéricas. Confirma las estimaciones de A5 (~160 MB y ~90 MB). |
| Tiempo por lote | Entre 0,02 s y 0,08 s por grupo de filas. |
| Orden interno | El lote llega ordenado por `case_id`, como asume A5. |
| Archivo vacío | `test_tax_registry_c_1.parquet` produce un lote de 0 filas y 5 columnas, sin error. Las columnas llegan con tipo físico `Null`: la coerción al tipo lógico corresponde a D4. |

### Hallazgos

| # | Gravedad | Hallazgo | Evidencia | Corrección propuesta |
| -: | --- | --- | --- | --- |
| 1 | Alta | Un nombre de familia lógica pasado como `source` devuelve tablas distintas como si fueran shards de una misma fuente, sin error. | `shard_paths("train", "credit_bureau_b")` devuelve `b_1` y `b_2`; `"person"` devuelve `person_1` y `person_2`; `"applprev"` devuelve solo `applprev_2`. `scan_parquet_shards("train", "credit_bureau_b")` produce una tabla de 49 columnas, cuando `b_1` tiene 45 y `b_2` tiene 6. Solo `"credit_bureau_a"` falla. | Validar `source` contra la lista de las 17 fuentes físicas y rechazar cualquier otro valor. G1 nombra las familias `credit_bureau_b`, `person` y `applprev`, por lo que la confusión es previsible. |
| 2 | Media | `iter_row_groups` descarta en silencio las columnas pedidas que no existen. | Con `columns=["case_id", "no_such_column"]` devuelve solo `case_id`. `scan_parquet_shards` sí rechaza columnas desconocidas: las dos funciones se comportan distinto. | Comparar `columns` con el esquema del archivo antes de leer y lanzar `ValueError` con las columnas ausentes. |
| 3 | Media | `iter_row_groups` lee con varios hilos. | `ParquetFile.read_row_group` usa `use_threads=True` por defecto y el grupo de hilos de Arrow tenía 10. `POLARS_MAX_THREADS=1` no lo controla. | Leer con `use_threads=False`, o exponer el parámetro con ese valor por defecto, para cumplir la decisión 9 (un hilo en ejecuciones `[R]`). |
| 4 | Baja | El error por archivo inexistente se lanza al pedir el primer lote, no al llamar a la función. | `iter_row_groups("/ruta/inexistente")` devuelve un generador; `FileNotFoundError` aparece en el primer `next()`. | Validar la ruta antes de crear el generador. |
| 5 | Baja | `scan_parquet_shards` concatena con `diagonal_relaxed`, que rellena con nulos una columna ausente en un shard y unifica tipos distintos sin avisar. | Lectura del código; es lo que permite el hallazgo 1. No afecta a bureau, que usa `iter_row_groups`. | Comprobar que todos los shards de una fuente comparten nombres de columnas y fallar si no; conservar la relajación solo para tipos físicos `Null`. |

### Veredicto

El lector cubre lo que A5 necesita para `credit_bureau_a` y `credit_bureau_b`. Los hallazgos 1 a 3 deberían corregirse antes de la ejecución real (E4); el adaptador de bureau puede desarrollarse mientras tanto pasando siempre nombres físicos completos y validando por su cuenta las columnas recibidas.

### Reproducir

```bash
uv run python -c "from dm_credit import io; print([p.name for p in io.shard_paths('train', 'credit_bureau_b')])"
uv run python -c "from dm_credit import io; p = io.shard_paths('train', 'credit_bureau_b_1')[0]; print(next(io.iter_row_groups(p, columns=['case_id', 'no_such_column'])).columns)"
```
