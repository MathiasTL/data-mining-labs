# E7 — Manifiesto final y comando único de reproducción

> **Estado: cerrada el 13/10/2026** (Andre, P3, en lugar de Nick).

## El comando

```bash
POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json --from-scratch
```

Reconstruye toda la preparación desde los Parquet originales, en serie y cada paso en su propio proceso:

| Paso | Qué hace | Segundos |
| --- | --- | ---: |
| `e4_rebuild` | Las 36 piezas de familia (12 familias × train, OOT y test) | 65,4 |
| `e4_checkpoints` | Junta los checkpoints de bureau para su auditor | 0,0 |
| `e4_audit` | Auditor de E4: 36 de 36 piezas válidas | 2,7 |
| `e4_checkpoint_audit` | Auditor de checkpoints de bureau | 1,2 |
| `e5` | Tablas maestras de train, OOT y test | 5,9 |
| `e6` | Preprocesador y vistas lineal y de árboles, ajustados con train | 11,5 |
| `validation` | Los 26 controles de F3 | 10,3 |

Total: **98,6 s** y un pico de unos **3,8 GB**. Sin `--from-scratch`, el comando audita las piezas de E4 existentes y regenera desde E5. Se detiene en el primer paso que falle; los registros de cada paso quedan en `data/artifacts/preparation/E7-logs/`.

## Reproducibilidad comprobada

Ejecutado desde cero, el comando produjo las mismas tablas maestras de E5 y las mismas vistas de E6, **byte por byte** (mismos SHA-256 en `E5_master_tables.csv` y `E6_views.csv`), y las mismas piezas de E4. Solo cambian los tiempos medidos y el orden interno de un índice de checkpoints de bureau B (mismas 2 163 claves).

## Lo que queda registrado

| Archivo | Contenido |
| --- | --- |
| `configs/preparation.json` | Configuración versionada: corte, hilos, versiones de artefactos, presupuesto de E4, catálogo y decisiones G2 |
| `results/preparation/E7_manifest.json` | Manifiesto final: commit de git, versiones de Python y polars, SHA-256 de la configuración y de las decisiones, particiones (casos, positivos, semanas y huella de base), SHA-256 de las 36 piezas de E4, de las tablas maestras y de las vistas y su estado ajustado, y la duración de cada paso |
| `results/preparation/run_metadata.csv` | Resumen clave–valor del manifiesto, con rutas y comando de regeneración (pedido por el backlog) |
| `results/preparation/final_schema.csv` | Esquema final ordenado: 129 columnas con su tipo, rol (control, predictor o cobertura) y tipo lógico |

Las tablas y vistas de train no se versionan en git por su tamaño (93, 132 y 103 MiB); su ruta, su SHA-256 y el comando para regenerarlas están en el manifiesto y en `run_metadata.csv`, como pide el backlog.

## Corrección incluida

`e4_rebuild` terminaba con código 1 aunque todas las piezas se generaran: revisaba todo el historial de `E4_builds.json`, que incluye el bloqueo antiguo de `applprev/train`. Ahora solo cuenta el último registro de cada pieza (`any_blocked`).

Con E7 cerrada, para G4 faltan las revisiones cruzadas (F6) y los tres arreglos menores pendientes.
