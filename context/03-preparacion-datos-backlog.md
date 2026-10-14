# Backlog del capítulo 03: preparación de datos

> **Objetivo:** construir un pipeline reproducible que genere matrices de entrenamiento, OOT y test con una fila por `case_id`, sin fuga temporal y con un esquema de características estable.
>
> **Equipo:** Nick, Giancarlo, Andre, Mathias y Gabriel trabajando en paralelo, con propiedad de archivos disjunta y compuertas secuenciales explícitas.
>
> **Fuera de alcance:** entrenamiento y selección de modelos, ajuste con OOT/test, optimización de Gini Stability y avance al capítulo 04.

## 1. Definición de terminado de la fase

El capítulo 03 se considera terminado cuando:

1. un único comando puede reconstruir las matrices desde los Parquet originales;
2. entrenamiento usa `WEEK_NUM < 74` y OOT usa `WEEK_NUM >= 74`;
3. ninguna transformación aprende parámetros de OOT o test;
4. cada agregado y cada matriz final contiene una fila por `case_id`;
5. todo evento utilizado cumple su regla de disponibilidad temporal respecto de `date_decision`;
6. los joins conservan número, unicidad y conjunto de casos de la tabla base correspondiente;
7. train, OOT y test comparten características, orden y tipos, salvo `target`;
8. categorías desconocidas se manejan sin ampliar vocabularios ajustados;
9. faltantes, extremos y coerciones tienen reglas explícitas y probadas;
10. existe un catálogo de características con origen, transformación y regla temporal;
11. la suite completa pasa y la documentación enlaza código, comandos y resultados;
12. existen dos vistas de la matriz final —lineal (imputada, escalada, codificada) y de árboles (nulos y categóricas nativos)— con las mismas filas y la misma trazabilidad de columnas;
13. cada matriz conserva `case_id`, `WEEK_NUM`, `date_decision` y, en train/OOT, `target` como columnas de control fuera del conjunto predictor;
14. la matriz de train y OOT entra sin cambios a `dm_credit.baseline` (contrato de una fila por caso y columnas reservadas);
15. existe un informe de fuga temporal por familia con las filas excluidas por regla de disponibilidad.

## 2. Principios de coordinación

### 2.1 Etiquetas

- **`[P]` Paralelo:** puede desarrollarse al mismo tiempo que otros trabajos `[P]`.
- **`[S]` Secuencial:** depende de una compuerta previa o modifica un artefacto compartido.
- **`[R]` Ejecución restringida:** puede programarse en paralelo, pero su ejecución sobre datos reales debe serializarse por memoria o E/S.

### 2.2 Reglas para evitar bloqueos y conflictos

1. Cada persona trabaja únicamente en su conjunto de archivos asignado hasta la integración.
2. Las implementaciones por familia deben satisfacer la misma interfaz congelada en la compuerta G1.
3. Las pruebas unitarias usan datos sintéticos pequeños. No necesitan esperar a los Parquet reales.
4. No se crean lectores propios: toda lectura cruza el seam de `dm_credit.io`.
5. Ningún módulo de familia realiza joins con otra familia.
6. Cada módulo devuelve o escribe un agregado con una fila por `case_id` y un manifiesto de características.
7. Las ejecuciones pesadas no se realizan simultáneamente, aunque el desarrollo sí sea paralelo.
8. Los cambios a interfaces compartidas después de G1 requieren acuerdo del equipo y actualización de todos los adaptadores afectados.
9. `development/03-preparacion-datos.md` tiene una única persona editora durante cada corte documental para evitar conflictos.

## 3. Diseño de módulos y seams

El objetivo es ocultar la complejidad de cada fuente detrás de interfaces pequeñas y estables.

### 3.1 Seam de agregación por familia

Interfaz conceptual que debe congelarse en G1:

```python
build_family_features(
    family: str,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
) -> FamilyFeatureArtifact
```

`FamilyFeatureArtifact` debe permitir auditar como mínimo:

- ruta del agregado;
- familia y versión de reglas;
- número de filas y `case_id` distintos;
- lista ordenada de características y tipos;
- columnas fuente utilizadas;
- reglas temporales aplicadas;
- conteos de filas excluidas por temporalidad;
- indicadores de cobertura y ausencia;
- validaciones ejecutadas.

La interfaz exacta puede cambiar durante G1, pero debe quedar pequeña. La lógica de fechas, agregaciones y lectura de shards pertenece a la implementación de cada módulo, no a sus llamadores.

### 3.2 Seam de ajuste y transformación

Interfaz conceptual:

```python
fitted = fit_preprocessor(train_features, specification)
X_train = fitted.transform(train_features)
X_oot = fitted.transform(oot_features)
X_test = fitted.transform(test_features)
```

Invariantes:

- `fit` recibe exclusivamente semanas `< 74`;
- `transform` no modifica vocabularios, imputadores ni columnas;
- una categoría desconocida se representa mediante `__UNKNOWN__` o mecanismo equivalente;
- valores no convertibles, infinitos y pérdida de columnas requeridas producen error;
- valores numéricos fuera del rango de train generan advertencia, no recorte automático.

### 3.3 Seam de ensamblado

Interfaz conceptual:

```python
assemble_feature_matrix(base, family_artifacts, split) -> FeatureMatrixArtifact
```

El ensamblador solo acepta agregados con `case_id` único. Después de cada join valida número de filas, unicidad y conjunto de casos.

La matriz resultante conserva `case_id`, `WEEK_NUM`, `date_decision` y `target` (solo train/OOT) como **columnas de control**: no son predictores, pero `dm_credit.baseline` y la evaluación semanal las necesitan. El manifiesto distingue explícitamente control de predictores.

## 4. Reparto entre cinco personas

| Persona   | Identificador | Frente principal                                       | Propiedad inicial de archivos                                                                                             | Dependencias principales                    |
| --------- | ------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Nick      | P1            | Contratos, partición, orquestación e integración final | módulos compartidos nuevos de contrato/orquestación; pruebas asociadas                                                    | Lidera G1, G2, G3 y G4                      |
| Giancarlo | P2            | Base, fuentes de profundidad 0, ventana y vistas       | adaptador de `base`, `static_0`, `static_cb_0`; política de ventana; vistas de matriz; informe de fuga; pruebas propias   | G1 para implementar interfaz definitiva     |
| Andre     | P3            | Solicitudes previas, personas, coerción y extremos     | adaptador de `applprev_1`, `applprev_2`, `person_1`, `person_2`; módulos de coerción de tipos y extremos; pruebas propias | G1; reglas padre-hijo y temporales          |
| Mathias   | P4            | Bureau de crédito                                      | adaptador de `credit_bureau_a_1`, `credit_bureau_a_2`, `credit_bureau_b_1`, `credit_bureau_b_2`; pruebas propias          | G1; ejecución real restringida              |
| Gabriel   | P5            | Productos/registros, faltantes y categorías            | adaptador de `debitcard_1`, `deposit_1`, `other_1`, `tax_registry_[a-c]_1`; módulo de preprocesamiento; pruebas propias   | G1 para adaptadores; G2 para cerrar formato |

En las tablas de tareas se conservan los identificadores breves: P1 = Nick, P2 = Giancarlo, P3 = Andre, P4 = Mathias y P5 = Gabriel. Los nombres concretos de módulos se deciden en G1. Una distribución recomendada es:

```text
src/dm_credit/preparation/
├── contracts.py                 # Nick
├── split.py                     # Nick
├── orchestrator.py              # Nick
├── assembly.py                  # Nick
├── coercion.py                  # Andre (D4)
├── outliers.py                  # Andre (D5)
├── views.py                     # Giancarlo (D7)
├── families/
│   ├── depth0.py                # Giancarlo
│   ├── applications_people.py   # Andre
│   ├── bureau.py                # Mathias
│   └── products_registry.py     # Gabriel
└── preprocessing.py             # Gabriel
```

No se debe crear esta estructura de forma automática si durante G1 se demuestra que una interfaz más pequeña encaja mejor con el código existente.

## 5. Backlog ejecutable

### Épica A — Contratos y esqueleto compartido

| ID  | Tipo      | Responsable       | Trabajo                                                                                                                                                    | Dependencias | Evidencia / criterio de aceptación                                                    |
| --- | --------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------- |
| A1  | `[P]`     | P1                | Traducir los contratos de los capítulos 02 y 03 a invariantes ejecutables: partición, columnas reservadas, una fila por caso, temporalidad y esquema.      | Ninguna      | Documento corto de decisión y pruebas sintéticas de invariantes.                      |
| A2  | `[P]`     | P1                | Diseñar las interfaces de agregado por familia, artefacto, manifiesto y ensamblado.                                                                        | Ninguna      | Propuesta revisable, con errores y características de rendimiento explícitas.         |
| A3  | `[P]`     | P2                | Proponer catálogo inicial de agregaciones para `static_0` y `static_cb_0`, incluyendo reglas de fechas y exclusiones.                                      | Ninguna      | Tabla fuente → regla → característica → justificación temporal.                       |
| A4  | `[P]`     | P3                | Proponer agregaciones de solicitudes/personas y resolver relación de profundidad 2 con su padre.                                                           | Ninguna      | Catálogo inicial y casos ambiguos marcados para decisión.                             |
| A5  | `[P] [R]` | P4                | Proponer estrategia streaming/checkpoint para bureau sin materializar 188 M de filas ni joins masivos.                                                     | Ninguna      | Plan de memoria, orden de lectura, agregaciones soportadas y recuperación.            |
| A6  | `[P]`     | P5                | Proponer política de faltantes, categorías, extremos, coerción y nombres de características.                                                               | Ninguna      | Especificación de `fit`/`transform` sin aprendizaje desde OOT/test.                   |
| A7  | `[P]`     | P2                | Proponer la política común de ventana retrospectiva: todo el historial `<= date_decision` o ventanas fijas (por ejemplo 3, 6, 12 meses) y cómo se nombran. | Ninguna      | Tabla familia → ventana → justificación; evita que cada frente invente la suya.       |
| G1  | `[S]`     | Equipo; lidera P1 | Congelar interfaces, convención de nombres, formato de manifiesto, errores, política de ventana y propiedad de archivos.                                   | A1–A7        | Decisión registrada; todos los frentes pueden implementar sin editar archivos ajenos. |

**Duración objetivo de G1:** una sesión corta. Mientras Nick prepara la decisión, Giancarlo, Andre, Mathias y Gabriel pueden completar A3–A7 y crear fixtures/pruebas de caja negra contra la interfaz conceptual.

### Épica B — Selección y partición de datos

| ID  | Tipo  | Responsable       | Trabajo                                                                                                                                       | Dependencias | Evidencia / criterio de aceptación                                         |
| --- | ----- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------- |
| B1  | `[P]` | P1                | Implementar partición inmutable train/OOT con corte 74 y separación de test.                                                                  | G1           | Casos de borde en semana 73/74; particiones disjuntas y exhaustivas.       |
| B2  | `[P]` | P1                | Implementar validación central de columnas reservadas: `case_id`, `num_group1`, `num_group2`, `date_decision`, `MONTH`, `WEEK_NUM`, `target`. | G1           | Ninguna reservada entra al conjunto predictor; `target` solo en train.     |
| B3  | `[P]` | P2                | Seleccionar columnas permitidas/excluidas para profundidad 0.                                                                                 | G1           | Manifiesto con motivo y evidencia para cada exclusión.                     |
| B4  | `[P]` | P3                | Seleccionar columnas permitidas/excluidas para solicitudes/personas.                                                                          | G1           | Campos sin referencia temporal suficiente quedan excluidos o bloqueados.   |
| B5  | `[P]` | P4                | Seleccionar columnas permitidas/excluidas para bureau.                                                                                        | G1           | Estados, pagos, DPD y fechas con regla `_as-of_` explícita.                |
| B6  | `[P]` | P5                | Seleccionar columnas permitidas/excluidas para productos/registros fiscales.                                                                  | G1           | Identificadores y cuasi-identificadores excluidos de predictores directos. |
| G2  | `[S]` | Equipo; lidera P1 | Aprobar lista inicial de fuentes/columnas y reglas temporales.                                                                                | B1–B6        | No quedan columnas incluidas sin justificación temporal.                   |

### Épica C — Agregadores por familia

Todos los agregadores deben producir conteo de registros, indicador de cobertura y características con nombres estables. Deben filtrar temporalmente antes de agregar.

Todos descubren los shards de cada familia por `results/parquet_inventory.csv` o por patrón de nombre, nunca con una lista fija: train tiene 32 archivos físicos y test 36 (difieren `static_0`, `applprev_1`, `credit_bureau_a_1` y `credit_bureau_a_2`). `test_tax_registry_c_1.parquet` está **vacío** y sus columnas son de tipo físico `Null`: el adaptador debe devolver un agregado con el esquema correcto y sin error.

| ID  | Tipo      | Responsable                   | Trabajo                                                                                              | Dependencias                | Evidencia / criterio de aceptación                                                                                              |
| --- | --------- | ----------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| C1  | `[P]`     | P2                            | Adaptador de `base`, `static_0` y `static_cb_0`.                                                     | G1; cierre funcional con G2 | Una fila por caso; fechas y columnas estáticas tratadas según contrato.                                                         |
| C2  | `[P]`     | P3                            | Adaptador de `applprev_1` y `applprev_2`, enlazando profundidad 2 a padres válidos antes de agregar. | G1; cierre funcional con G2 | No se usa una fila hija cuyo padre no sea temporalmente válido.                                                                 |
| C3  | `[P]`     | P3                            | Adaptador de `person_1` y `person_2`.                                                                | G1; cierre funcional con G2 | Roles/relaciones agregados sin usar identificadores directos.                                                                   |
| C4  | `[P] [R]` | P4                            | Adaptador streaming de `credit_bureau_a_1` y `credit_bureau_a_2`.                                    | G1; cierre funcional con G2 | Memoria acotada, checkpoints, una fila por caso, agregados verificables por muestra sintética.                                  |
| C5  | `[P] [R]` | P4                            | Adaptador de `credit_bureau_b_1` y `credit_bureau_b_2`.                                              | G1; cierre funcional con G2 | Relación padre-hijo respetada; sin joins gigantes.                                                                              |
| C6  | `[P]`     | P5                            | Adaptador de `debitcard_1`, `deposit_1` y `other_1`.                                                 | G1; cierre funcional con G2 | `other_1` solo se usa si su disponibilidad temporal está demostrada; en caso contrario se excluye con evidencia.                |
| C7  | `[P] [R]` | P5                            | Adaptador streaming de `tax_registry_a_1`, `tax_registry_b_1` y `tax_registry_c_1`.                  | G1; cierre funcional con G2 | Alta cardinalidad manejada sin nombres directos como categorías libres; prueba con el archivo test vacío de `tax_registry_c_1`. |
| C8  | `[P]`     | P2–P5, cada cual en su módulo | Catálogo de características por familia.                                                             | C1–C7 según frente          | Nombre, tipo, fuente, columna, agregación, ventana, regla temporal y descripción.                                               |

### Épica D — Limpieza y formateo

| ID  | Tipo  | Responsable | Trabajo                                                                                                                             | Dependencias               | Evidencia / criterio de aceptación                                                                                           |
| --- | ----- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| D1  | `[P]` | P5          | Implementar preprocesador ajustable solo con train temporal.                                                                        | G1                         | `fit` rechaza OOT/test o recibe una partición validada; estado serializable.                                                 |
| D2  | `[P]` | P5          | Política de faltantes: indicadores de ausencia y estrategias por tipo/familia.                                                      | G1; se completa tras C1–C7 | Ninguna imputación implícita; parámetros aprendidos solo de train.                                                           |
| D3  | `[P]` | P5          | Política categórica: raras, desconocidas y `__UNKNOWN__`.                                                                           | G1                         | OOT/test no amplían vocabulario; orden estable.                                                                              |
| D4  | `[P]` | P3          | Coerción de tipos y control de infinitos/valores no convertibles (`coercion.py`).                                                   | G1                         | Errores explícitos; columnas físicas `Null` adoptan el esquema lógico; las cuatro fechas `Float64` nulas de test se validan. |
| D5  | `[P]` | P3          | Política de extremos (`outliers.py`).                                                                                               | G1                         | Fuera de rango genera advertencia; recorte solo con regla de dominio aprobada y ajustada en train.                           |
| D6  | `[P]` | P1          | Especificación y validación del esquema final ordenado.                                                                             | G1                         | Misma lista, orden y tipos en train, OOT y test; columnas de control separadas de predictores; `target` ausente en test.     |
| D7  | `[P]` | P2          | Vistas de la matriz final (`views.py`): **lineal** (imputada, escalada, codificada) y **de árboles** (nulos y categóricas nativos). | G1; se completa tras D1–D3 | Mismas filas y columnas de origen en ambas vistas; especificación `matrix_views.csv`; ajustadas solo con train `< 74`.       |

D1–D7 pueden desarrollarse con tablas sintéticas mientras se implementan los agregadores. Su validación final depende de G3. La vista lineal y la de árboles responden a `context/03` (3.4.4): el capítulo 04 compara regresión logística y boosting sobre la misma partición.

### Épica E — Integración y tabla maestra

| ID  | Tipo      | Responsable                           | Trabajo                                                                     | Dependencias             | Evidencia / criterio de aceptación                                            |
| --- | --------- | ------------------------------------- | --------------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| E1  | `[P]`     | P1                                    | Implementar ensamblador que solo acepte agregados con `case_id` único.      | G1                       | Falla ante claves nulas/repetidas o familias sin manifiesto.                  |
| E2  | `[P]`     | P1                                    | Implementar validaciones posteriores a cada join.                           | G1                       | Filas, unicidad y conjunto de casos permanecen invariantes.                   |
| E3  | `[P]`     | P2–P5                                 | Entregar fixtures y artefactos pequeños de cada adaptador al ensamblador.   | C1–C7 parciales          | P1 puede probar integración sin esperar ejecuciones completas.                |
| G3  | `[S]`     | Equipo; lidera P1                     | Integración sintética de los cuatro adaptadores de familia y preprocesador. | E1–E3, D1–D7 funcionales | Pipeline completo sobre fixtures; contratos y nombres congelados.             |
| E4  | `[S] [R]` | P1 coordina; P2–P5 ejecutan su frente | Generar agregados reales de forma secuencial.                               | G2, G3, C1–C7            | Cada ejecución produce manifiesto, métricas de memoria/tiempo y validaciones. |
| E5  | `[S] [R]` | P1                                    | Ensamblar tablas maestras sin ejecutar familias pesadas en paralelo.        | E4                       | Una fila por caso; invariantes después de cada join.                          |
| E6  | `[S] [R]` | P5                                    | Ajustar preprocesador con train `< 74` y transformar train/OOT/test.        | E5                       | Esquemas idénticos; OOT/test no alteran estado ajustado.                      |
| E7  | `[S]`     | P1                                    | Escribir manifiesto final y comando único de reproducción.                  | E6                       | Rutas, hash/configuración, columnas, particiones y artefactos registrados.    |

**Avance E4 (13/10/2026, histórico; superado: E4 cerrada con 36/36 piezas, ver su documento):** parcial, no habilitaba E5. Auditoría real de 36 artefactos: 30 reutilizados, 5 regenerados y validados, `applprev/train` obsoleto y bloqueado tras superar el presupuesto de 5.120 MiB. Bureau A2 sigue omitida por B5/G2. Recursos históricos no medidos de las ejecuciones reutilizadas quedan explícitamente como desconocidos. Evidencia, limitaciones y reproducción en [E4 — ejecución secuencial](03-preparacion-E4-ejecucion.md) y `results/preparation/E4_*`.

### Épica F — Validación, documentación y cierre

| ID  | Tipo  | Responsable                     | Trabajo                                                                      | Dependencias | Evidencia / criterio de aceptación                                                                        |
| --- | ----- | ------------------------------- | ---------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------- |
| F1  | `[P]` | P2–P5                           | Pruebas específicas de sus adaptadores y catálogos.                          | C1–C8        | Casos temporales, nulos, claves y agregaciones calculables manualmente.                                   |
| F2  | `[P]` | P1                              | Pruebas end-to-end sintéticas del pipeline.                                  | G3           | Partición, filtrado, agregado, join, fit/transform y manifiesto.                                          |
| F3  | `[S]` | P1                              | Ejecutar validaciones sobre artefactos reales.                               | E7           | Conteos, unicidad, esquema, finitud, cobertura por familia y entrada sin cambios a `dm_credit.baseline`.  |
| F4  | `[P]` | P2–P5                           | Redactar resultados de sus fuentes en notas disjuntas o tablas de evidencia. | E4           | Cifras acompañadas de comando, reporte y fecha.                                                           |
| F5  | `[S]` | Editor acordado; recomendado P1 | Consolidar `development/03-preparacion-datos.md` sin placeholders.           | F3, F4, F7   | Secciones 3.1–3.5 completas y trazables.                                                                  |
| F6  | `[S]` | Equipo                          | Revisión cruzada: cada persona revisa un frente ajeno.                       | F5           | Hallazgos resueltos o documentados como limitación.                                                       |
| F7  | `[S]` | P2 consolida; P3–P5 aportan     | Informe de fuga temporal por familia (`temporal_leakage_report.csv`).        | B3–B6, E4    | Por familia: regla de disponibilidad, columnas excluidas, filas descartadas por temporalidad y evidencia. |
| G4  | `[S]` | Equipo                          | Cierre del capítulo 03.                                                      | F1–F7        | Suite completa, `git diff --check`, comando reproducible y definición de terminado satisfecha.            |

**Cierre G4 (13/10/2026): hecho.** Definición de terminado 15/15, reconstrucción desde cero con el comando único (29/29 controles), suite completa sin fallos esperados y `git diff --check` limpio. Cambios, entregables equivalentes y riesgos aceptados en [G4 — cierre](03-preparacion-G4-cierre.md).

## 6. Plan de arranque simultáneo

Las cinco personas pueden comenzar inmediatamente así:

| Persona   | Primera tarea | Segunda tarea mientras espera G1                                                                                               |
| --------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Nick      | A1 y A2       | Preparar fixtures comunes para B1, B2, E1 y E2.                                                                                |
| Giancarlo | A3 y A7       | Fixtures y resultados manuales para fuentes 1:1 y 1:0..1.                                                                      |
| Andre     | A4            | Fixtures padre-hijo de profundidades 1 y 2, incluidos eventos posteriores a decisión; casos de coerción y extremos para D4/D5. |
| Mathias   | A5            | Prototipo sintético de agregación por lotes/checkpoints, sin leer Parquet grandes.                                             |
| Gabriel   | A6            | Fixtures de faltantes, desconocidos, infinitos, coerción y rangos nuevos.                                                      |

Después de G1, B1–B6, C1–C7, D1–D7, E1–E3 y F1/F2 avanzan en paralelo dentro de sus dependencias. El camino crítico es:

```text
A1–A7 → G1 → B1–B6 → G2 → cierre de C1–C7 → G3
→ E4 secuencial → E5 → E6 → E7 → F3/F5/F6 → G4
```

## 7. Política de ejecución sobre datos reales

El trabajo de código y las pruebas sintéticas sí pueden ejecutarse en paralelo. Las tareas marcadas `[R]` deben usar una cola compartida:

1. una sola ejecución pesada activa;
2. `POLARS_MAX_THREADS=1`;
3. DuckDB, si se reutiliza, limitado a un hilo y memoria acotada;
4. directorios de salida temporales por familia y publicación atómica al completar;
5. checkpoints para `credit_bureau_a_2` y otras fuentes fragmentadas grandes;
6. tiempos límite y registro de duración/memoria;
7. nunca materializar joins entre fuentes `1:N`;
8. no sobrescribir un artefacto válido hasta que el reemplazo supere sus validaciones.

Orden sugerido de ejecución real para detectar problemas temprano:

1. fixtures y fuente pequeña;
2. profundidad 0;
3. productos pequeños;
4. solicitudes/personas;
5. bureau B;
6. registros fiscales;
7. bureau A, dejando `credit_bureau_a_2` al final;
8. ensamblado y preprocesamiento.

## 8. Estrategia mínima de características

Para evitar que cinco frentes inventen reglas incompatibles, el primer corte debe ser conservador:

- todas las familias: cobertura y conteo de registros;
- numéricas: conteo válido, nulos, mínimo, máximo, media, suma y desviación cuando tengan sentido;
- categóricas: conteo distinto y frecuencia/proporción de categorías aprobadas; no usar identificadores libres;
- booleanas: conteo y proporción verdadera;
- fechas válidas: recencia y antigüedad respecto de `date_decision`, nunca fecha futura;
- profundidad 2: agregar primero al padre lógico o demostrar que la agregación directa por caso conserva la semántica;
- no generar combinaciones cruzadas entre familias en el capítulo 03.

Una característica solo entra si puede describirse con:

```text
familia + columnas fuente + filtro temporal + agregación + significado
```

## 9. Pruebas obligatorias

### Unitarias por módulo

- partición exacta en semana 74;
- exclusión de columnas reservadas;
- fecha anterior, igual y posterior a `date_decision`;
- profundidad 1 y 2;
- fuente sin cobertura;
- nulos y categorías desconocidas;
- tipos `Null`, coercibles y no convertibles;
- infinitos y extremos;
- orden determinista de columnas;
- estado de preprocesamiento inmutable durante `transform`.

### Integración sintética

- dos o más familias `1:N` agregadas antes del join;
- join que no modifica filas ni casos;
- rechazo de agregado con `case_id` duplicado;
- train/OOT/test con idénticas características finales;
- test sin `target`;
- manifiesto reproducible y catálogo completo.

### Validación sobre datos reales

- train + OOT suma exactamente 1 526 659 casos;
- train conserva 1 332 001 casos y 43 734 positivos; OOT conserva 194 658 casos y 4 260 positivos (verificado con `train_base` y corte 74);
- la cobertura `source_present` de cada familia coincide con la documentada en `results/integration_analysis/` (por ejemplo `applprev_1` 80,01 %, `credit_bureau_a_1` 90,80 %, `tax_registry_a_1` 30,00 %); una diferencia indica un filtro temporal o un join defectuoso;
- las matrices de train y OOT pasan `dm_credit.baseline.validate_contract` y se pueden usar con `run_baseline` sin modificar código;
- la matriz de test tiene las mismas características, orden y tipos que train, sin `target` (la muestra local solo tiene 10 filas por archivo: la validación es estructural, no distribucional);
- ninguna matriz final duplica `case_id`;
- ninguna característica contiene infinitos;
- cada característica aparece en el catálogo;
- el preprocesador fue ajustado solo con `WEEK_NUM < 74`;
- ejecución repetida con misma configuración conserva esquema y resultados deterministas.

## 10. Entregables esperados

Las rutas definitivas se congelan en G1. Como mínimo deben existir equivalentes a:

```text
src/dm_credit/preparation/
tests/test_preparation_*.py
results/preparation/
├── family_manifests/
├── feature_catalog.csv
├── integration_checks.csv
├── preprocessing_spec.csv
├── matrix_views.csv
├── temporal_leakage_report.csv
├── run_metadata.csv
└── final_schema.csv
```

Las matrices potencialmente grandes pueden ubicarse fuera de git, pero su ruta, configuración, esquema y comando de regeneración deben constar en `run_metadata.csv` y en el README.

## 11. Tablero resumido

| Estado inicial                    | IDs                                                          |
| --------------------------------- | ------------------------------------------------------------ |
| Listo para comenzar en paralelo   | A1–A7                                                        |
| Listo después de G1               | B1–B6, C1–C7 en esqueleto, D1–D7, E1–E3, F1–F2               |
| Requiere decisión temporal G2     | Inclusión definitiva de columnas y cierre funcional de C1–C7 |
| Requiere integración sintética G3 | Ejecución real E4–E7                                         |
| Requiere artefactos reales        | F3–F5, F7                                                    |
| Cierre secuencial                 | F6 y G4                                                      |

## 12. Riesgos del plan y mitigaciones

| Riesgo                                           | Mitigación                                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Cinco personas cambian interfaces compartidas    | Congelar G1 temprano y asignar propiedad de archivos.                                                                           |
| Fuga por fechas ambiguas                         | Exclusión preventiva hasta G2; no inferir seguridad por nombre.                                                                 |
| Agregaciones semánticamente inconsistentes       | Catálogo obligatorio y revisión cruzada por otra persona.                                                                       |
| Saturación de memoria/E/S                        | Desarrollo paralelo, ejecución pesada serial y checkpoints.                                                                     |
| Explosión de columnas                            | Primer corte conservador y convención común de agregaciones.                                                                    |
| Drift entre esquemas                             | Manifiesto y esquema final único; `transform` no aprende.                                                                       |
| Test contamina decisiones                        | Test solo valida contrato; no aporta estadísticas ajustadas.                                                                    |
| Conflictos de documentación                      | Un editor por corte; contribuciones en notas/tablas disjuntas.                                                                  |
| Bloqueo por `credit_bureau_a_2`                  | Adaptador streaming aislado; el resto del pipeline se valida con fixture y otras familias.                                      |
| Test local mínimo (10 filas, un archivo vacío)   | Validar solo estructura y esquema con test; no inferir distribución ni drift; prueba explícita con `tax_registry_c_1` vacío.    |
| Shards distintos entre train y test              | Descubrimiento de shards por inventario o patrón; nunca listas fijas.                                                           |
| Ventanas retrospectivas inconsistentes           | Política común en A7 y G1; cada característica registra su ventana en el catálogo.                                              |
| Carga desigual o indisponibilidad de una persona | D4–D5 con Andre y A7/D7/F7 con Giancarlo para aliviar a Gabriel y Nick; cada frente documenta su estado para poder traspasarlo. |
| Calendario sin holgura antes del 14/11           | Fechas por compuerta (sección 14); la ejecución real E4 es la línea roja y empieza con `credit_bureau_a_2` al final.            |

## 13. Primera reunión de sincronización (G1)

Nick facilita esta compuerta. Giancarlo, Andre, Mathias y Gabriel presentan respectivamente A3 y A7, A4, A5 y A6; la decisión final es conjunta.

Debe resolver únicamente estas doce decisiones para no bloquear el paralelismo:

1. interfaz y estructura de `FamilyFeatureArtifact`;
2. convención de nombres de características;
3. formato del catálogo y manifiestos;
4. lista inicial de agregaciones permitidas;
5. representación de fechas, booleanos, nulos y categorías desconocidas;
6. errores frente a advertencias;
7. rutas de artefactos grandes ignorados por git;
8. comando único y configuración versionada;
9. orden de ejecución pesada;
10. propiedad definitiva de archivos;
11. política común de ventana retrospectiva (A7);
12. columnas de control en las matrices y formato de las dos vistas (lineal y de árboles).

Todo lo demás debe resolverse dentro de cada módulo mientras respete esas interfaces e invariantes.

## 14. Calendario propuesto

Fechas **propuestas**, a confirmar o ajustar en G1. El hito oficial de preparación de datos es el 14/11/2026 (sábado); el plan deja un día de holgura.

| Compuerta o bloque                               | Fecha propuesta | Responsables                 |
| ------------------------------------------------ | --------------: | ---------------------------- |
| A1–A7 (propuestas iniciales)                     |       jue 15/10 | Todos                        |
| **G1** (interfaces y decisiones)                 |       vie 16/10 | Equipo; lidera Nick          |
| B1–B6, C en esqueleto, D1–D7, E1–E3, F1–F2       |       vie 23/10 | Cada frente en paralelo      |
| **G2** (columnas y reglas temporales)            |       lun 26/10 | Equipo; lidera Nick          |
| C1–C8 con cierre funcional                       |       vie 30/10 | P2–P5                        |
| **G3** (integración sintética)                   |       lun 02/11 | Equipo; lidera Nick          |
| E4 ejecución real serial (A2 de bureau al final) |     03/11–06/11 | P2–P5 en cola; coordina Nick |
| E5 ensamblado → E6 preprocesado → E7 manifiesto  |     09/11–11/11 | Nick y Gabriel               |
| F3, F4, F7                                       |       jue 12/11 | Nick, P2–P5                  |
| F5, F6 y **G4** (cierre)                         |       vie 13/11 | Equipo                       |
