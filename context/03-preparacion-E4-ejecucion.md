# E4 — Auditoría y ejecución real secuencial

**Estado: parcial, bloqueada en `applprev/train` por presupuesto de memoria.**
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

`e4_rebuild.py` invoca constructores en procesos separados. El supervisor Linux mide
externamente el RSS del árbol cada 100 ms y GNU `time` registra duración y RSS máximo
del proceso constructor. Presupuesto: **5.120 MiB**, 120 s por ejecución; exige al
inicio presupuesto más 1.536 MiB de reserva y detiene si la reserva cae por debajo de
1 GiB. Para `applprev` se añadió `build_applprev_features_batched`: particiona la base
por bloques de 100.000 casos, agrega cada bloque con las mismas reglas y combina las
filas/exclusiones. La prueba sintética compara exactamente el Parquet y exclusiones
con el builder de una sola pasada. No cambió G2 ni el contrato del artefacto.

| Familia/split    | Acción             | Duración externa (s) | Pico externo (KiB) | Resultado                                       |
| ---------------- | ------------------ | -------------------: | -----------------: | ----------------------------------------------- |
| `applprev/test`  | Generado           |                 3,12 |            398.316 | Validado, 10 casos                              |
| `person/test`    | Generado           |                 2,16 |            403.076 | Validado, 10 casos                              |
| `applprev/train` | Generado por lotes |                 44,2 |          2.685.224 | Validado, 1.332.001 casos; árbol bajo 5.120 MiB |
| `applprev/oot`   | Generado           |                 8,82 |          2.009.500 | Validado, 194.658 casos                         |
| `person/train`   | Generado           |                11,89 |          2.711.960 | Validado, 1.332.001 casos                       |
| `person/oot`     | Generado           |                 3,35 |          1.139.224 | Validado, 194.658 casos                         |

El intento inicial sin lotes se mantuvo registrado como bloqueado; el nuevo registro
identifica `applprev_case_batches_v1` y lo reemplaza como estado más reciente. La nueva
construcción no elevó el presupuesto ni sustituyó datos por fixtures.

Las cinco salidas nuevas viven en `data/artifacts/preparation/E4-v1/<familia>/<split>/`.
Se conservaron intactas las anteriores. El auditor prefiere un reemplazo E4 si existe
y lo valida: no cae silenciosamente al anterior si el reemplazo es inválido.
`E4_runs.csv` es el inventario con rutas, checksums y origen de cada resultado.
**30 `reused`, 6 `generated_validated`, 0 artefactos inválidos.** `applprev/train`
se publicó en E4-v1 y pasó la auditoría final.

## 3. Validaciones observadas

- Partición real: train **1.332.001**, OOT **194.658**; total **1.526.659**; test local
  **10 casos**. Corte inmutable: `< 74` / `>= 74`.
- Los 36 Parquet inspeccionados tienen claves no nulas, únicas, ordenadas y exactamente
  iguales a la base de su split; todos pasaron la comprobación de vigencia de entradas.
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

**Resultados de validación ejecutada:** suite de preparación **536 passed, 1 xfailed**,
5 `RangeWarning` esperados de las pruebas sintéticas G3 (política D5, sin recorte),
en 29,93 s. Auditor E4: exit 0, 36 artefactos vigentes. Auditor de shards: exit 0,
19 checkpoints válidos. La última auditoría de artefactos
tardó 3,26 s, pico acumulado 494.416 KiB; la de checkpoints 0,19 s y 297.516 KiB.
Estas últimas mediciones corresponden **a los auditores**, no a los builds históricos.

Las pruebas sintéticas no reemplazan los datos reales; la reconstrucción real por lotes
quedó validada con el auditor E4. E4 ya no está bloqueada por `applprev/train`. La falta
de mediciones históricas de las familias reutilizadas queda como limitación explícita.
