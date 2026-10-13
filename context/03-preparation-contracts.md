# Decisión A1 — contratos ejecutables de preparación

Estado: propuesta inicial para revisión en G1.

## Invariantes

1. **Partición temporal:** con corte `74`, train contiene exclusivamente filas con `WEEK_NUM < 74` y OOT exclusivamente filas con `WEEK_NUM >= 74`. Las particiones son disjuntas y su unión conserva todas las filas. Test es una partición independiente, no se mezcla con train ni OOT. El corte coincide con `context/02a-etapa-inicial-tabla-base.md`.
2. **Clave y granularidad:** `case_id` debe existir, no contener nulos y ser único en la tabla base y en cada agregado/matriz final: exactamente una fila por caso.
3. **Columnas reservadas:** `case_id`, `num_group1`, `num_group2`, `date_decision`, `MONTH`, `WEEK_NUM` y `target` son columnas de control, nunca predictores. La lista coincide con `dm_credit.baseline.RESERVED`.
4. **Objetivo:** train y OOT requieren `target` binario no nulo (0/1); test no debe incluir `target`. La ausencia de `target` en test no se interpreta como nulo ni se imputa.
5. **Disponibilidad temporal:** cada registro fuente empleado para un caso debe satisfacer la regla de disponibilidad acordada para su familia respecto de `date_decision`. Si no hay evidencia suficiente para establecer esa regla, la columna se excluye o queda bloqueada; no se presume que la fecha del registro equivale a disponibilidad.
6. **Esquema:** train, OOT y test deben conservar las mismas características, en el mismo orden y con los mismos tipos lógicos. Las columnas de control se mantienen separadas de los predictores; `target` es la única diferencia permitida y solo aparece en train/OOT.

## Evidencia y límites de esta decisión

`dm_credit.baseline` ya valida clave única/no nula, semana, objetivo binario, exclusión de columnas reservadas y corte OOT parametrizable. Las pruebas sintéticas en `tests/test_preparation_invariants.py` fijan esos comportamientos y el contrato con/sin objetivo.

La validación de disponibilidad temporal y la comparación del esquema entre particiones requieren módulos de preparación todavía no implementados. En esta etapa se fijan como criterios de aceptación: las pruebas de esas reglas se añadirán junto con los adaptadores y el validador de esquema (D6), usando fixtures sintéticos. No se modifica el contrato de `dm_credit.baseline`.

## A2 — propuesta de interfaces compartidas

Estado: propuesta para revisión en G1; no constituye todavía una API congelada.

### Artefacto de familia y manifiesto

Cada familia escribe un agregado Parquet y un manifiesto JSON correspondiente. El manifiesto es el registro auditable y contiene versión de formato/reglas, split, familia, ruta relativa del agregado, número de filas, número de `case_id` distintos, lista ordenada de características con tipo lógico, rol (predictor/cobertura), columnas fuente, reglas temporales y sus exclusiones, y validaciones ejecutadas. No se guardan muestras de datos sensibles en el manifiesto.

```python
@dataclass(frozen=True)
class FamilyFeatureArtifact:
    family: str
    split: Literal["train", "oot", "test"]
    data_path: Path
    manifest_path: Path
    row_count: int
    case_count: int
    feature_schema: tuple[FeatureDefinition, ...]


def build_family_features(
    family: str,
    split: Literal["train", "oot", "test"],
    *,
    cutoff_week: int = 74,
    output_dir: Path,
) -> FamilyFeatureArtifact: ...
```

La función resuelve sus shards mediante el inventario o patrón documentado, usa la capa de lectura compartida y mantiene dentro del módulo la lógica de fechas, agregación y streaming. No recibe ni devuelve tablas de otras familias. El `case_id` del agregado debe ser no nulo y único. Para una familia vacía (p. ej. shard vacío), devuelve un Parquet vacío pero tipado, con el esquema acordado, más manifiesto válido.

### Ensamblado

```python
def assemble_feature_matrix(
    base: pl.DataFrame,
    family_artifacts: Sequence[FamilyFeatureArtifact],
    split: Literal["train", "oot", "test"],
    *,
    output_dir: Path,
) -> FeatureMatrixArtifact: ...
```

El ensamblador valida que cada artefacto corresponda al mismo split, que cada agregado tenga clave válida y manifiesto legible, y que no haya nombres de características duplicados ni colisión con columnas de control. Realiza joins `1:1` por `case_id`, preservando filas, unicidad y conjunto/orden de casos de la base después de cada join. Conserva controles (`case_id`, `WEEK_NUM`, `date_decision`, `MONTH` si está presente y `target` únicamente fuera de test) aparte de los predictores. El resultado registra el orden final del esquema y los manifiestos de entrada.

### Errores y rendimiento

- Error de contrato para split inválido, esquema incompatible, manifiesto ausente/malformado, claves nulas/duplicadas, colisiones de nombres o violación de invariantes tras join.
- Error de entrada/datos para shards requeridos ausentes o datos que no se pueden coercionar al esquema lógico; familia explícitamente vacía no es error.
- Error de escritura/IO conserva causa y ruta para hacer el fallo accionable.
- Los módulos de familias pueden procesar por lotes/chunks y emitir checkpoints; no deben materializar shards pesados ni hacer joins entre familias. Las ejecuciones pesadas reales se serializan según el backlog.
- La serialización de Parquet/JSON debe evitar publicar un artefacto parcial: escribir temporalmente y renombrar al completar validaciones. Esto no reemplaza el registro de configuración/hash y métricas de ejecución exigido para la reproducción final.

### Decisiones abiertas para G1

Aprobar (o cambiar) dataclasses y nombres, JSON como formato del manifiesto, definición de `FeatureDefinition`/`FeatureMatrixArtifact`, si `base` se entrega como DataFrame o se resuelve por el seam IO, semántica de orden de casos y estrategia de publicación atómica/checkpoint. El contrato no prescribe biblioteca de streaming ni fuerza una API de preprocesamiento; esta última ya tiene su seam conceptual en el backlog.

## Pendientes para G1

- Confirmar si los nombres y tipos lógicos del esquema final se declaran en un manifiesto versionado.
- Aprobar las reglas `as-of` específicas por familia; este documento solo fija el invariante común.
- Confirmar el manejo de filas cuya semana o fecha de decisión sea nula (propuesta: fallo explícito, no asignación implícita a una partición).
