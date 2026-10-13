# G3 — Integración sintética del pipeline de preparación

> **Estado:** **cerrada** el 13/10/2026 por Andre (P3), con apoyo de Claude. Se construye sobre la prueba E3 de Nick (`tests/test_preparation_e3_integration.py`) y la extiende hasta el preprocesador y las vistas.
> **Qué exige el backlog:** integración sintética de los adaptadores de familia y el preprocesador, con E1 a E3 y D1 a D7 funcionales; pipeline completo sobre fixtures; contratos y nombres congelados.
> **Qué habilita:** la ejecución real en cola (E4) y, después, E5 a E7.

## 1. Precondiciones

| Pieza | Módulo | Responsable | Estado |
| --- | --- | --- | --- |
| Adaptadores de todas las familias (C1 a C7) | `families/depth0.py`, `applications_people.py`, `bureau.py`, `products_registry.py` | Giancarlo, Andre, Mathias, Gabriel | Hechos y registrados en `build_family_features` |
| Ensamblador y validaciones tras cada join (E1, E2) | `assembly.py` | Nick | Hecho |
| Artefactos pequeños para el ensamblador (E3) | `tests/preparation_*_artifacts.py` y prueba E3 | Todos | Hecho |
| Preprocesador, faltantes y categorías (D1 a D3) | `preprocessing.py` | Gabriel | Hecho; usa la coerción (D4) y los extremos (D5) |
| Coerción y extremos (D4, D5) | `coercion.py`, `outliers.py` | Andre | Hecho |
| Esquema final (D6) | `schema.py` | Nick | Hecho |
| Vistas lineal y de árboles (D7) | `views.py` | Giancarlo | Hecho |

## 2. La prueba

`tests/test_preparation_g3_integration.py` arma, con datos sintéticos y las decisiones G2 publicadas, la cadena completa: **adaptadores → ensamblador → preprocesador → vistas** para train, OOT y test. Los 14 controles pasan:

1. Las once familias (`static_0`, `static_cb_0`, `applprev`, `person`, bureau A y B, y los seis productos y registros) llegan a la matriz ensamblada, con los casos en el orden de la base.
2. Las matrices ensambladas de train, OOT y test cumplen el contrato de esquema final (`validate_matrix_schemas`).
3. Todos los nombres siguen la gramática congelada y ninguno es una columna reservada.
4. Solo `applprev` y `person` aportan predictores; las demás familias aportan cobertura (G2).
5. Las vistas lineal y de árboles conservan filas, orden de casos y columnas de control; test no tiene `target`.
6. Cada vista tiene exactamente los mismos predictores, en el mismo orden y con los mismos tipos, en los tres splits.
7. La vista lineal no tiene nulos y todos sus números son finitos.
8. La vista de árboles conserva los nulos nativos.
9. Una categoría que train nunca vio llega como `__UNKNOWN__` en las dos vistas.
10. Ajustar con OOT falla, y transformar OOT y test no cambia el estado ajustado.
11. El estado ajustado se guarda y se recarga, y reproduce exactamente las dos vistas.
12. Cada artefacto publicado se verifica por su suma de verificación.

## 3. Contratos congelados

A partir de G3 estos contratos no cambian sin una decisión explícita, porque los usan todos los frentes:

| Contrato | Dónde | Versión |
| --- | --- | --- |
| Gramática de nombres `<fuente>__<variable>__<agregación>__<ventana>` | `contracts.parse_feature_name` | Hasta cuatro segmentos tras la fuente |
| `FeatureDefinition` y manifiesto de familia | `contracts.py` | `format_version` 1, con suma de verificación del Parquet |
| Partición train / OOT / test | `split.py` | Corte 74 |
| Columnas de control y reservadas | `columns.py`, `baseline.RESERVED` | `case_id`, `WEEK_NUM`, `date_decision`, `MONTH`, `target`; `num_group1` y `num_group2` nunca llegan a la matriz |
| Estado del preprocesador | `preprocessing.py` | Versión 1 |
| Estado de las vistas | `views.py` | Versión 1 |
| Columnas aprobadas | Archivos `results/preparation/G2_*` | Decisión G2 del primer corte |

## 4. Evidencia con datos reales

Además de la prueba sintética, el primer corte real (solicitudes previas y personas, las únicas familias con predictores aprobados) pasó por el ensamblador y las dos vistas:

| Split | Vista lineal | Nulos en predictores | Vista de árboles |
| --- | --- | ---: | --- |
| train | 1 332 001 × 245 | 0 | 1 332 001 × 107 |
| OOT | 194 658 × 245 | 0 | 194 658 × 107 |
| test | 10 × 244 (sin `target`) | 0 | 10 × 106 |

La vista lineal tiene 240 predictores y la de árboles 102; las cinco columnas de control se conservan. Tardó 8 s, con un pico de unos 3,3 GB.

## 5. Hallazgos para seguimiento

No bloquean G3, pero tienen responsable:

| Hallazgo | Impacto | Responsable |
| --- | --- | --- |
| 23 de 204 columnas del preprocesador son constantes en train (19 indicadores `__missing` de columnas que nunca faltan y 4 características como `persontype_1072L`). | No rompe nada: la vista lineal las deja en cero y los árboles las ignoran, pero inflan la matriz. G2 ya pedía descartarlas. | Gabriel (D1) |
| `views.py` usa `pl.concat(..., how="horizontal")`, que polars marca como obsoleto: cambiará de comportamiento en una versión mayor. | Hoy funciona porque `uv.lock` fija la versión; romperá al actualizar polars. | Giancarlo (D7) |
| `RangeWarning` en OOT y test por valores fuera del rango de train. | Esperado: es la política de D5 (avisar sin recortar). | Ninguno |

## 6. Reproducir

```bash
uv run --extra dev pytest -q tests/test_preparation_e3_integration.py tests/test_preparation_g3_integration.py
```
