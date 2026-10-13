# B3 — Selección de columnas de profundidad 0

## Objetivo y alcance

Este entregable documenta una propuesta reproducible de selección de columnas para las fuentes de profundidad 0:

- `base`: tabla principal, una fila por solicitud/caso.
- `static_0`: atributos asociados directamente a cada caso, con una fila por `case_id`.
- `static_cb_0`: información complementaria asociada directamente al caso, con cero o una fila por `case_id`.

La cardinalidad se comprobó en `results/integration_analysis/family_cardinality.csv`. La base aporta cinco columnas de control/objetivo; las dos tablas fuente aportan 168 y 53 columnas, respectivamente. Por tanto, el catálogo actual contiene 226 decisiones para B3.

B3 decide y justifica el tratamiento propuesto por columna. **No transforma datos ni crea todavía las matrices de entrenamiento.** `control_only`, `exclude_predictor`, `manual_review` y `provisional_candidate` describen la propuesta, no el resultado de C1 ni una lista de predictores ya aprobada.

## Cómo se reproduce

El selector usa los reportes ya versionados y no vuelve a leer los Parquet originales:

```powershell
uv run -m dm_credit.preparation.depth0_selection
```

Parámetros disponibles:

```text
--catalog         results/variable_catalog.csv
--temporal-review results/integration_analysis/temporal_leakage_review.csv
--out             results/preparation/B3_depth0_columns.csv
```

La ejecución falla explícitamente si faltan columnas requeridas de los reportes, si no hay una revisión temporal para alguna familia o si el catálogo contiene decisiones duplicadas. La salida es determinista y se ordena por familia y nombre de columna.

## Criterios de decisión

| Acción propuesta | Significado |
|---|---|
| `control_only` | Se conserva para identificar el caso, controlar tiempo/partición o como etiqueta; nunca se usa como predictor. |
| `exclude_predictor` | No se propone como predictor por ser clave/cuasi-identificador o por riesgo explícito de fuga del objetivo. |
| `manual_review` | La información existente no basta para justificar su uso; la propuesta actual es no usarla hasta contar con evidencia concreta de significado/disponibilidad. |
| `provisional_candidate` | La definición oficial no muestra una señal automática de fuga en los reportes actuales. Es una candidata para evaluación, no prueba de que estuvo disponible al tomar la decisión. |

Reglas principales:

1. Las cinco columnas de `base` son controles/objetivo, según el contrato de G1.
2. `case_id` en las fuentes estáticas se excluye de predictores, pero se conserva como clave de unión.
3. Los identificadores/cuasi-identificadores señalados por los catálogos se excluyen de predictores.
4. Fechas y variables marcadas por la revisión temporal como dudosas se clasifican para revisión. La ventana común fijada por G1 es `all` hasta `date_decision`, inclusive; un nombre de ventana en una definición no demuestra disponibilidad del dato.
5. `riskassesment_302T` se excluye porque su definición describe una probabilidad de incumplimiento futuro. Los scores `riskassesment_*` se revisan por procedencia y momento de cálculo antes de considerarlos.
6. Una definición vacía o no oficial no se interpreta por conjetura: se marca para revisión.
7. Una fila ausente de `static_cb_0` es falta de cobertura; no equivale automáticamente a valores cero. C1 debe conservar un indicador de cobertura de la fuente.

## Resultado generado

El CSV `results/preparation/B3_depth0_columns.csv` contiene una fila por columna del catálogo para las tres familias, con:

`family`, `column`, `dtype`, `description`, `proposed_action`, `reason`, `temporal_rule`, `evidence`.

Resumen reproducible por familia y acción:

| Fuente | Columnas | Decisión |
|---|---:|---|
| `base` | 5 | 5 `control_only`; no son predictores. |
| `static_0` | 168 | 24 `exclude_predictor`, 104 `manual_review`, 40 `provisional_candidate`. |
| `static_cb_0` | 53 | 2 `exclude_predictor`, 34 `manual_review`, 17 `provisional_candidate`. |
| **Total** | **226** | 5 controles, 26 exclusiones, 138 revisiones manuales y 57 candidatas provisionales. |

El comando anterior imprime el mismo resumen agrupado; los detalles individuales y sus justificaciones están en el CSV.

## Fuentes y límites de evidencia

- `results/variable_catalog.csv`: nombres, tipos, definiciones oficiales disponibles, clasificación funcional, riesgos de identificación y notas iniciales de revisión.
- `results/integration_analysis/temporal_leakage_review.csv`: resumen de familias, columnas temporalmente sensibles y reglas preventivas.
- `results/integration_analysis/family_cardinality.csv`: relación observada de cada fuente con `case_id`.
- `context/03-preparacion-G1-decision.md`: columnas de control, convención y ventana temporal común.

Los perfiles y reportes fueron generados por las herramientas existentes del proyecto. El selector B3 los reutiliza para que la decisión pueda regenerarse y probarse.

El catálogo y el análisis temporal no contienen evidencia independiente del instante de extracción de cada campo. Por ello, las candidatas provisionales no se tratan como predictores definitivamente seguros; las columnas dudosas quedan fuera del uso hasta documentar la evidencia que resuelva la duda. B3 queda completada como **selección propuesta y auditable**; C1 aplicará posteriormente las decisiones que correspondan al construir los agregados.
