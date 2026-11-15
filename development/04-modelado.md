# 4. Modelado

> **Estado (14/11/2026):** experimentos ejecutados sobre la matriz `E5-v2` con el [protocolo temporal](../context/auditoria-variables-predictoras/05-protocolo-temporal.md). Corrida de referencia: `results/modeling/v2-run1`. Los resultados del avance 2 (Gini Stability OOT 0,494) quedan como **históricos**: usaban vistas ajustadas con las semanas 0 a 73 mientras validaban en 64 a 73.

## 4.1 Seleccionar técnica de modelado

| Modelo | Rol | Motivo de inclusión | Tratamiento de nulos/categóricas |
|---|---|---|---|
| Predicción constante | Referencia mínima | Predice la tasa de incumplimiento del bloque de ajuste; fija el piso (Gini 0) | No aplica |
| Regresión logística | Referencia interpretable | Estándar en scoring de crédito; coeficientes legibles [1] | Vista lineal: imputación por mediana con indicador de faltante, escalado estándar, codificación one-hot con vocabulario congelado |
| LightGBM | Candidato no lineal | Captura interacciones y no linealidades en datos tabulares grandes; usado en scoring con explicaciones SHAP [4], [6] | Vista de árboles: nulos nativos y categóricas nativas con vocabulario congelado |

### Criterio de elección de técnicas

Los datos son tabulares, con 1,33 millones de solicitudes en desarrollo, 112 a 231 predictores según la vista, muchos faltantes estructurales y una tasa de incumplimiento cercana al 3,4 %. La literatura recomienda comparar una regresión logística interpretable con un modelo que capture efectos no lineales [1], [4]. LightGBM cumple ese papel con bajo costo de cómputo en una máquina de 16 GB. Una red neuronal se descartó para esta versión: aporta poca ventaja esperada en datos tabulares y complica la interpretación [9].

Las referencias numeradas están en [referencias técnicas](../docs/referencias-tecnicas-modelado.md).

## 4.2 Generar diseño de prueba

### Validación fuera de tiempo

| Partición | Ajuste (semanas) | Validación (semanas) | Casos de validación | Positivos |
|---|---|---|---:|---:|
| P1 | 0 a 39 | 40 a 49 | 248 961 | 9 012 |
| P2 | 0 a 49 | 50 a 59 | 215 392 | 8 771 |
| P3 | 0 a 59 | 60 a 73 | 112 852 | 4 230 |
| OOT | (modelo de P3) | 74 a 91 | 194 658 | 4 260 |

- **Selección de hiperparámetros y de grupos de variables:** Gini Stability fuera de partición, calculado sobre las validaciones unidas de P1 a P3 (semanas 40 a 73, 34 semanas). Una mejora del Gini semanal medio también debe repetirse en al menos dos de las tres particiones.
- **Calibración (isotónica) y umbrales:** se aprenden con la validación de P3.
- **Modelo final:** el de P3, sin reajuste, para que el calibrador corresponda al modelo evaluado. Como contraste se reporta un reajuste con las semanas 0 a 73.
- **OOT:** se consultó **una sola vez** sobre v2. Un marcador `final.done` impide repetir la consulta sin dejar registro. El OOT ya había sido examinado en el avance 2; es la única evaluación fuera de tiempo disponible, porque el test local tiene 10 casos.

### Controles de validez

- **Transformaciones por partición.** Imputación, escalado, vocabulario y descarte de constantes se reajustan en cada partición, solo con su bloque de ajuste. No se reutilizan las vistas E6 publicadas.
- **Columnas de control.** `case_id`, `WEEK_NUM`, `date_decision`, `MONTH` y `target` nunca entran como predictores; hay una aserción en el código.
- **Selección y pruebas de quitar variables sin OOT.** Un test ejecuta esas etapas con el archivo OOT ausente.
- **Etapa final sin efecto del OOT.** Un test altera etiquetas y scores del OOT y comprueba que calibrador, umbrales, selección, exclusiones y variables vigiladas no cambian.
- **Disponibilidad temporal.** Revisada en la [auditoría de variables](../context/auditoria-variables-predictoras/README.md).
- **Revisión independiente** del código antes de la corrida: sin caminos de fuga; los siete hallazgos se corrigieron.

Una corrección del protocolo se hizo **antes** de consultar el OOT. El Gini Stability calculado por partición, con 10 a 14 semanas, quedaba dominado por la penalización de pendiente (−1,8 frente a un Gini medio de 0,40 en una prueba con el 5 % de los datos). Se reemplazó por el cálculo fuera de partición.

## 4.3 Modelo de construcción

### Registro de experimentos

Datos: `E5-v2`, 122 características. Semilla 42. Máquina: 10 núcleos, 16 GB. Duración total de la corrida: 11 minutos, con un pico de memoria de 6,6 GB.

| Experimento | Variables | Partición | Hiperparámetros | Semilla | Tiempo de ajuste | Estado |
|---|---|---|---|---:|---:|---|
| Constante | Ninguna | P1 a P3, OOT | Tasa base del ajuste | — | 0 s | Referencia |
| Regresión logística | Vista lineal (hasta 228) | P1 a P3 | C ∈ {0,1; 1} | 42 | 11 s (6 ajustes) | Seleccionada C = 0,1 |
| LightGBM | Vista de árboles (hasta 110) | P1 a P3 | hojas ∈ {31; 63}, árboles ∈ {300; 600}, tasa 0,05 | 42 | 184 s | Seleccionada 63 hojas, 300 árboles |
| Pruebas de quitar variables | 4 grupos | P1 a P3 | Configuración seleccionada | 42 | 311 s (etapa) | Ningún grupo retirado |
| Final | Todas | P3 y OOT | Seleccionada | 42 | LR 2,1 s; LightGBM 17,4 s | Ejecutado |

### Ajuste de hiperparámetros

La búsqueda fue una grilla pequeña y comparable entre modelos, sin early stopping, para no ajustar el número de árboles con la validación. Se parte de la configuración más simple. Una más compleja la reemplaza solo si mejora el Gini Stability fuera de partición y gana en al menos dos de las tres particiones.

| Modelo | Configuración | Gini Stability fuera de partición | Gini semanal medio | AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Constante | — | 0,000 | 0,000 | 0,508 | 0,039 |
| Regresión logística | **C = 0,1** | **0,288** | 0,397 | 0,705 | 0,086 |
| Regresión logística | C = 1 | 0,286 | 0,397 | 0,705 | 0,086 |
| LightGBM | 31 hojas, 300 árboles | 0,426 | 0,451 | 0,727 | 0,099 |
| LightGBM | 31 hojas, 600 árboles | 0,427 | 0,452 | 0,728 | 0,100 |
| LightGBM | **63 hojas, 300 árboles** | **0,428** | 0,453 | 0,728 | 0,100 |
| LightGBM | 63 hojas, 600 árboles | 0,425 | 0,451 | 0,727 | 0,100 |

### Pruebas de quitar variables

Un grupo se retira si, sin él, no empeora el Gini Stability fuera de partición, no empeora el Gini semanal medio en al menos dos de tres particiones y no aumenta la desviación semanal.

| Grupo | Columnas que llegan al modelo | Resultado |
|---|---:|---|
| Cobertura (`*__n_records`, `*__source_present`) | 19 de 28; 9 son constantes | Se conserva en ambos modelos |
| `person_2__*` | 2 | Se conserva |
| Contactos (`person_1__n_contact_persons__all`) | 1 | Se conserva |
| `tax_registry_b_1__*` | 0 | **Sin efecto**: la fuente vale cero en todas las semanas 0 a 59 y aparece recién desde la semana 60 |

Evidencia: `results/modeling/v2-run1/ablation_decisions.csv` y `oof_metrics.csv`.

## 4.4 Evaluar el modelo

Métricas en OOT (semanas 74 a 91) del modelo final de P3. La evaluación completa está en el [capítulo 5](05-evaluacion.md).

| Modelo | Gini Stability | Gini semanal medio | ROC-AUC | PR-AUC | Recall con umbral de P3 (objetivo 80 %) | Observaciones |
|---|---:|---:|---:|---:|---:|---|
| Constante | 0,000 | 0,000 | 0,500 | 0,022 | — | Piso |
| Regresión logística | 0,438 | 0,460 | 0,731 | 0,062 | 81,0 % | Referencia |
| **LightGBM** | **0,484** | **0,503** | **0,752** | **0,075** | 80,7 % | Gana en 18 de 18 semanas |
| LightGBM, reajuste 0 a 73 | 0,493 | — | 0,757 | 0,076 | 72,8 % | Contraste; el umbral de P3 ya no corresponde |

### Interpretación

La importancia de variables se midió con SHAP (TreeExplainer) sobre 20 000 casos de P3 y del OOT [5], [6]. Para ver si las explicaciones son estables con pocos incumplimientos [5], el ranking se repitió con cinco muestras independientes (semillas 42 a 46). La correlación de Spearman entre rankings fue 0,9998 como mínimo y las 20 variables principales coincidieron en todas las muestras.

| Puesto | Variable | Puesto en P3 | Puesto en OOT |
|---:|---|---:|---:|
| 1 | Tipo de ingreso del solicitante | 1 | 1 |
| 2 | Edad del solicitante | 2 | 2 |
| 3 | Plazo medio de solicitudes previas | 3 | 3 |
| 4 | Nivel educativo en solicitudes previas | 4 | 4 |
| 5 | Registros del registro fiscal C | 5 | 5 |

Los mayores cambios entre P3 y OOT son `person_2__n_records` (del puesto 28 al 11) y `person_2__source_present` (del 73 al 50), coherentes con la caída de presencia de `person_2` en el OOT (de 99 % a 70 %). En la regresión logística, los coeficientes estandarizados más altos corresponden al número de filas de `person_1`, al número de personas de contacto y a la edad.

**La importancia describe asociación con la predicción, no causalidad.**

## Reproducción

```sh
export POLARS_MAX_THREADS=1
M=data/artifacts/preparation/E5-v2; O=results/modeling/<corrida>
.venv/bin/python -m dm_credit.modeling.run --matrix-dir $M --out $O --stage select
.venv/bin/python -m dm_credit.modeling.run --matrix-dir $M --out $O --stage ablate
.venv/bin/python -m dm_credit.modeling.run --matrix-dir $M --out $O --stage final
```

La matriz `E5-v2` se regenera con `POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --from-scratch`, agregando `--force` si las carpetas v2 ya existen (ver [capítulo 3](03-preparacion-datos.md)). La etapa `final` se niega a ejecutarse de nuevo sobre la misma carpeta salvo con `--force-reconsult`, que queda registrado.
