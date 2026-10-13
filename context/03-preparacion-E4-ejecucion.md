# E4 — Auditoría y ejecución real secuencial

**Estado: cerrada el 13/10/2026** (Andre, P3): las 36 piezas se reconstruyeron en serie y los dos auditores pasan. Las secciones 1 a 5 conservan el intento anterior de Nick, bloqueado en `applprev/train` por memoria; el cierre está en la **sección 6**.
Auditoría y reconstrucciones realizadas el 13/10/2026, después de G3 (`d40daba`),
sobre el checkout `f59d5ac` más los módulos E4 de este cambio. **No se ejecutaron
E5/E6, ensamblado de matrices maestras, ajuste de preprocesadores ni modelos.**
No se modificó G2 ni se incorporaron predictores.

## 1. Auditoría previa y decisión de reutilización

Se inspeccionaron backlog, G2/G3, A5, C1–C7, runners públicos, los cuatro catálogos
por frente, C8, reportes de cobertura/exclusiones y `results/integration_analysis/`.
Se encontraron **36 artefactos reales: 12 familias × train/OOT/test**. Las notas
que todavía indicaban que bureau v2 no tenía ejecución real estaban desactualizadas.

`results/preparation/E4_initial_audit.json` conserva el resultado **anterior a
reconstruir**:

| Frente               | Artefactos | Decisión inicial | Evidencia                                                           |
| -------------------- | ---------: | ---------------- | ------------------------------------------------------------------- |
| Profundidad 0        |          6 | Reutilizar       | `depth0-v3`, dos coberturas por familia, cero predictores           |
| Productos y fiscales |         18 | Reutilizar       | `products-v1`, dos coberturas por familia, cero predictores         |
| Bureau A/B           |          6 | Reutilizar       | `C4-C5-v2`, plan aprobado vigente, 2/4 coberturas, cero predictores |
| Solicitudes/personas |          6 | Obsoletos        | Las huellas `nombre:tamaño:mtime_ns` de los shards no coinciden     |

En C2/C3 coincidían nombres y tamaños de entrada, pero no `mtime_ns`. No se dispone
de SHA-256 **histórico** del contenido fuente para demostrar equivalencia: no se
reescribieron las huellas antiguas para hacerlas pasar. Las salidas se contrastaron
con G2, su vocabulario, código/reglas vigentes y catálogos, además de comprobar sus
SHA-256. Que un esquema coincida no hace vigente una entrada obsoleta.

Las versiones pre-G2 (`C4-C5-v1`, C1 con 57 candidatos) **no son entradas
autorizadas**. `C8_feature_catalog_partial.csv` es histórico; el catálogo vigente
es `C8_feature_catalog.csv`: **124 características, 94 predictores y 30 coberturas**.
Solo `applprev` y `person` tienen predictores (61 y 33, respectivamente); sus otras
cuatro características por familia son coberturas.

## 2. Trabajo ejecutado, recursos y bloqueo

Todo se ejecutó serialmente con `POLARS_MAX_THREADS=1`; primero el test pequeño,
luego las reconstrucciones de C2/C3. No hubo nuevas ejecuciones de profundidad 0,
productos, bureau B, fiscales ni bureau A: **se reutilizaron**, no se atribuye una
construcción nueva a esos frentes. Su reutilización evita repetir las fuentes más
costosas del orden A5. No se consultaron los shards de bureau A2: el plan B5/G2 la
omite por ausencia de ancla temporal autorizada.

`e4_rebuild.py` invoca `build_family_features` en procesos separados. El supervisor
Linux mide externamente el RSS del árbol de procesos cada 100 ms y usa GNU `time`
para duración y máximo RSS del proceso constructor. Presupuesto: **5.120 MiB**, 120 s
por ejecución; requiere al comenzar memoria disponible para ese límite más 1.536
MiB de reserva, y detiene si la reserva durante ejecución cae por debajo de 1 GiB.
La máquina tenía aproximadamente 8,2 GiB disponibles al inicio. El objetivo A5 de
2 GB no es una medición ni una garantía del builder C2/C3: `person/train` lo supera,
aunque quedó dentro del presupuesto explícito de esta ejecución.

| Familia/split    | Acción               | Duración externa (s) | Pico externo (KiB) | Resultado                                    |
| ---------------- | -------------------- | -------------------: | -----------------: | -------------------------------------------- |
| `applprev/test`  | Generado             |                 3,12 |            398.316 | Validado, 10 casos                           |
| `person/test`    | Generado             |                 2,16 |            403.076 | Validado, 10 casos                           |
| `applprev/train` | Intentado y detenido |        No disponible |      No disponible | Supervisor observó RSS del árbol > 5.120 MiB |
| `applprev/oot`   | Generado             |                 8,82 |          2.009.500 | Validado, 194.658 casos                      |
| `person/train`   | Generado             |                11,89 |          2.711.960 | Validado, 1.332.001 casos                    |
| `person/oot`     | Generado             |                 3,35 |          1.139.224 | Validado, 194.658 casos                      |

El intento detenido no publicó un reemplazo. Al matar el grupo también se detuvo
GNU `time`; esa primera versión del supervisor no persistía las muestras antes
del aborto. Solo está acreditado el cruce del umbral, **no** una duración ni un
pico exactos. El supervisor ahora conserva las muestras para eventuales abortos
futuros; no se rellenaron retrospectivamente las métricas faltantes.

No se elevó el presupuesto ni se reintentó `applprev/train`. Para resolverlo hace
falta un builder/runner C2 de memoria acotada (por lotes con validación de equivalencia)
o una decisión explícita de recursos. No se sustituyeron datos por fixtures.

Las cinco salidas nuevas viven en `data/artifacts/preparation/E4-v1/<familia>/<split>/`.
Se conservaron intactas las anteriores. El auditor prefiere un reemplazo E4 si existe
y lo valida: no cae silenciosamente al anterior si el reemplazo es inválido.
`E4_runs.csv` es el inventario con rutas, checksums y origen de cada resultado.
**30 `reused`, 5 `generated_validated`, 1 `invalid` con bloqueo de construcción.**
El catálogo estructural incluye `applprev/train`, pero ese artefacto antiguo **no se
puede consumir como resultado vigente de E4**.

## 3. Validaciones observadas

- Partición real: train **1.332.001**, OOT **194.658**; total **1.526.659**; test local
  **10 casos**. Corte inmutable: `< 74` / `>= 74`.
- Los 36 Parquet inspeccionados tienen claves no nulas, únicas, ordenadas y exactamente
  iguales a la base de su split. En 35 se verificó también vigencia de entradas; en
  `applprev/train` la comprobación estructural no resuelve su huella obsoleta.
- Manifiestos con versión/formato, checksum de datos, esquema físico y definiciones
  completas iguales entre splits. Ausencia de infinitos comprobada columna por columna.
- Correspondencia completa con los cuatro catálogos y C8, y con los reportes C1,
  C2/C3, C6/C7 y exclusiones F7. No se regeneraron catálogos ni decisiones G2.
- Configuración y vocabulario de G2 concordantes para solicitudes/personas. Cero
  predictores de profundidad 0, bureau, productos y fiscales.
- Test fiscal C vacío: agregado de **10 casos con cobertura cero**, no cero casos.
- **19 checkpoints bureau** validados: hashes de parciales/claves padre, claves de
  caché, versión, plan, huellas del shard/padre/base y corte. Sus sumas de filas,
  unión de casos y exclusiones concuerdan con los agregados y manifiestos finales.

### Cobertura frente a `integration_analysis`

La referencia `family_cardinality.csv` describe train físico completo, es decir,
**train + OOT antes de filtros**. No se exige igualdad de cobertura bruta y filtrada.
El informe `E4_coverage.csv` conserva cada tabla/split y la conciliación combinada.

| Tabla        | Casos brutos | Casos tras filtro | Filas descartadas | Interpretación                                              |
| ------------ | -----------: | ----------------: | ----------------: | ----------------------------------------------------------- |
| Bureau A1    |    1.386.273 |         1.116.767 |        10.193.265 | Ancla nula o posterior; cobertura baja de 90,80 % a 73,15 % |
| Bureau B1    |       36.500 |             7.384 |            76.645 | Ancla nula o posterior                                      |
| Bureau B2    |       36.447 |             7.379 |         1.166.558 | Fecha posterior y/o padre inválido, contadores disjuntos    |
| `applprev_1` |    1.221.522 |         1.208.039 |           140.106 | **Balance histórico/parcial**, incluye train obsoleto       |
| `applprev_2` |    1.221.522 |         1.208.039 |           213.212 | **Balance histórico/parcial**, incluye train obsoleto       |

En cada tabla temporal, filas válidas + exclusiones disjuntas = filas brutas
(la conciliación de solicitudes no autoriza reutilizar su train). Los 140.106 de
solicitudes son 140.040 posteriores + 66 fechas de creación nulas. Los contadores de
**valores** anulados no se suman como filas descartadas ni se convierten en predictores.

Profundidad 0, productos, fiscales y personas mantienen los conteos/coberturas físicas
documentados. Por ejemplo, fiscal A = 457.934 casos (29,9958 %), fiscal B = 150.732,
fiscal C = 482.265. Fiscal C tiene cero cobertura OOT y test; fiscal B tiene 24.424
casos train y 126.308 OOT. No se corrige este cambio de cobertura ni se interpreta
como disponibilidad temporal: los productos/fiscales aportan **solo cobertura física**.
`other_1` no aporta sus importes como predictores. Para A2, los ceros del informe
significan **omitida**, no medición de ausencia de registros.

## 4. Evidencia y límites de trazabilidad

Bajo `results/preparation/`:

- `E4_initial_audit.json`: auditoría previa preservada, seis huellas obsoletas.
- `E4_audit.json`: comprobaciones finales, hashes actuales de código/configuración,
  bases, catálogos, reportes, esquemas y errores. `ok=false` es intencional mientras
  siga bloqueado train de solicitudes.
- `E4_runs.csv`: familia/split, filas/casos, columnas/roles, origen, hashes,
  recursos medidos y bloqueos. Los tiempos de auditoría no son tiempos de construcción.
- `E4_builds.json`: comandos reales, recursos GNU `time`, hashes de entradas/código
  al construir, salidas y aborto; los registros `generated_pending_audit` son el
  evento de construcción, su validación posterior está en `E4_runs.csv`.
- `E4_coverage.csv`: cobertura y conciliación; `current_inputs_valid=false` identifica
  métricas de un artefacto no vigente, aunque el balance aritmético sea correcto.
- `E4_exclusions.csv`: exclusiones por split/familia, con unidad filas/valores/columnas.
- `E4_shards.csv` / `E4_shards.json`: evidencia de los 19 checkpoints y conciliación.

Las ejecuciones reutilizadas no registraron duración ni pico RSS de construcción,
ni recursos por shard. Esos campos permanecen vacíos/null. No se rehizo trabajo
pesado únicamente para fabricar una medición histórica. Los RSS del auditor son
máximos **acumulados de ese proceso** (KiB Linux), no máximos individuales del builder.
Tampoco existen conteos históricos de filas crudas leídas por shard; se informan
filas válidas/casos de sus parciales y exclusiones, sin inventar lecturas.

Los manifiestos antiguos tienen huellas fuente de tamaño/mtime, no hashes del
contenido ni commit/hash histórico de código/configuración. La reutilización se
fundamenta en esas huellas, checksums de salida, versión de reglas, plan bureau,
contrato/selección G2 y evidencia concordante. Los hashes actuales del auditor no
prueban retrospectivamente qué código se ejecutó. No se afirma equivalencia
numérica con un rebuild completo ni ausencia de toda posible fuga temporal.

## 5. Reproducción y pruebas

Desde la raíz, con los Parquet locales y GNU `time` disponibles:

```bash
# Solo audita; exit 1 esperado mientras applprev/train esté obsoleto.
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4

# Solo trabajos C2/C3 con huella obsoleta identificados por el auditor.
# Omite ya validados y conserva bloqueos anteriores: NO reintenta train.
# Exit 1 si quedan bloqueos. Las demás causas/familias exigen revisión manual.
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4_rebuild \
    --max-rss-mib 5120 --timeout-seconds 120

POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4_shards

POLARS_MAX_THREADS=1 uv run --extra dev pytest -q tests/test_preparation_*.py
git --no-pager diff --check
```

**Resultados de validación ejecutada:** suite de preparación **514 passed**, 5
`RangeWarning` esperados de las pruebas sintéticas G3 (política D5, sin recorte),
en 28,08 s. `git diff --check` sin errores. Auditor final: exit 1, exclusivamente
por `applprev/train/metadata_current_inputs`; 30 reutilizados y 5 generados válidos.
Auditor de shards: exit 0, 19 checkpoints válidos. La última auditoría de artefactos
tardó 3,26 s, pico acumulado 494.416 KiB; la de checkpoints 0,19 s y 297.516 KiB.
Estas últimas mediciones corresponden **a los auditores**, no a los builds históricos.

Las pruebas sintéticas no reemplazan datos reales ni resuelven el bloqueo de recursos.
(Histórico) **No cerrar E4 ni iniciar E5** hasta reconstruir y validar `applprev/train`: resuelto en la sección 6. La falta
de mediciones históricas de las familias reutilizadas queda como limitación explícita
incluso después de resolver ese bloqueo.

## 6. Cierre de E4 (13/10/2026, Andre)

### Qué bloqueaba y cómo se resolvió

| Problema | Causa | Solución |
| --- | --- | --- |
| `applprev/train` superaba 5 120 MiB | El adaptador leía y agregaba todos los casos a la vez | El adaptador de C2 y C3 procesa los casos en **bloques** (4 por defecto). Como toda agregación es por caso, el resultado es idéntico: se comprobó con pruebas y con el OOT real de Nick (mismas 194 658 filas y mismas exclusiones). |
| Las piezas reutilizadas solo eran válidas en la computadora que las construyó | La huella de los shards incluía la **fecha de modificación** del archivo, que cambia en cada copia de los datos | La huella pasó a ser `nombre:tamaño:sha256:<contenido>` (`io.file_fingerprint`), una sola función para las cuatro familias y el auditor. La fecha ya no participa. |
| `e4_rebuild` solo funcionaba en Linux | Usaba `/proc` y GNU time | En macOS mide con `ps`, `memory_pressure` y `/usr/bin/time -l`. Se agregó `--all` para reconstruir las 36 piezas en el orden del backlog. |
| El auditor de checkpoints leía manifiestos de `C4-C5-v2` | No conocía los reemplazos de E4 | Usa el reemplazo de `E4-v1` cuando existe, igual que el auditor principal. |

### Ejecución

Macintosh de 16 GB, un hilo (`POLARS_MAX_THREADS=1`), presupuesto de **5 120 MiB** y 900 s por pieza. Orden del backlog §7: primero la prueba pequeña de cada familia, luego profundidad 0, productos, solicitudes y personas, bureau B, registros fiscales y bureau A al final (`credit_bureau_a_2` no se lee).

| Familia | Train (s) | Train (MiB) | OOT (s) | OOT (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `static_0` | 1,28 | 414 | 1,03 | 358 |
| `static_cb_0` | 1,24 | 431 | 0,93 | 358 |
| `debitcard_1` | 1,04 | 407 | 0,88 | 358 |
| `deposit_1` | 1,04 | 406 | 0,87 | 358 |
| `other_1` | 1,05 | 399 | 0,88 | 358 |
| `applprev` | 11,83 | 2 104 | 4,08 | 1 903 |
| `person` | 3,81 | 1 670 | 1,87 | 1 072 |
| `credit_bureau_b` | 1,16 | 502 | 0,94 | 384 |
| `tax_registry_a_1` | 1,26 | 682 | 0,90 | 358 |
| `tax_registry_b_1` | 1,02 | 410 | 0,93 | 432 |
| `tax_registry_c_1` | 1,28 | 688 | 0,90 | 357 |
| `credit_bureau_a` | 2,72 | 642 | 2,01 | 508 |

Las 12 piezas de test tardaron menos de 1 s cada una. Total: **55,6 s** para las 36 piezas; pico máximo **2 104 MiB**, que cabe en la máquina de 8 GiB de Nick. Las exclusiones de bureau A en train (7 528 249 filas sin ancla y 1 583 190 con ancla posterior) coinciden con la ejecución C4-C5, así que se leyeron todas las filas.

### Validaciones

| Auditor | Resultado |
| --- | --- |
| `uv run -m dm_credit.preparation.e4` | **ok**: 36 de 36 piezas `generated_validated`, salida 0 |
| `uv run -m dm_credit.preparation.e4_shards --checkpoint-dir data/artifacts/preparation/E4-v1/checkpoints` | **ok**: 26 controles y 19 checkpoints vigentes |
| `uv run -m dm_credit.preparation.validation` (F3, ahora sobre `E4-v1`) | **26 de 26** controles correctos |
| Suite completa | 645 pruebas correctas y 1 fallo esperado (claves repetidas en bureau, ver F1) |

### Reproducir

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4_rebuild --all --max-rss-mib 5120 --timeout-seconds 900
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e4_shards --checkpoint-dir data/artifacts/preparation/E4-v1/checkpoints
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.validation
```

`--all` exige que `data/artifacts/preparation/E4-v1/` no tenga manifiestos previos: hay que moverla o borrarla antes. Con E4 cerrada, el siguiente paso es E5 (ensamblar las tablas maestras a partir de `E4-v1`).
