# Etapa A — Construir el terreno con la tabla base

## Propósito

Esta es la primera etapa técnica del proyecto, posterior a la consolidación de la comprensión del negocio. Su finalidad es construir un flujo mínimo y verificable antes de trabajar con las demás fuentes Parquet.

El equipo debe poder responder:

> ¿Qué contiene la tabla base, cuál es la unidad de análisis, cómo se separarán los datos en el tiempo y cómo se entrenará el primer modelo de referencia?

## Alcance de esta etapa

### Datos que se analizan

```text
data/parquet_files/train/train_base.parquet
```

Se revisará también el archivo base equivalente de `test`, si está disponible, únicamente para comprobar compatibilidad de esquema con entrenamiento.

### Datos que aún no se analizan

No se integran ni se procesan todavía las fuentes de:

- solicitudes previas (`applprev`);
- bureau de crédito;
- personas;
- estáticos;
- depósitos;
- tarjetas;
- registros fiscales;
- otras tablas relacionadas.

Estas fuentes se incorporarán una a una después de contar con el baseline de la tabla base.

## Resultado esperado

Al finalizar, debe existir una tabla base entendida y un contrato mínimo para modelado:

```text
case_id
WEEK_NUM
target                 # disponible solo en entrenamiento
variables_predictoras  # candidatas y temporalmente válidas
```

Además, el equipo contará con:

- un reporte de calidad;
- un catálogo funcional de columnas;
- una partición de validación fuera de tiempo;
- un entorno capaz de leer Parquet;
- un baseline listo para entrenar;
- documentación parcial del capítulo de comprensión de datos.

---

# Plan de trabajo

## Paso 0 — Reunión de arranque

**Duración sugerida:** 45 a 60 minutos.

### Objetivos

1. Revisar y aprobar `development/01-entendimiento-empresarial.md`.
2. Confirmar que el primer objetivo técnico es un baseline con la tabla base, no el modelo final.
3. Crear las tareas de esta etapa en el tablero del equipo.
4. Acordar una fecha de revisión conjunta de resultados.
5. Confirmar que cualquier resultado se registrará en `development/` y será reproducible.

### Acuerdos mínimos

- La salida final debe ser una probabilidad `score` por caso.
- La métrica oficial de Kaggle debe confirmarse y documentarse antes de seleccionar el modelo final.
- `WEEK_NUM` se utilizará inicialmente para definir la validación fuera de tiempo (OOT).
- No se incluyen de inicio identificadores técnicos, el `target` ni variables con posible fuga de información.

---

## Paso 1 — Preparar el entorno común

**Responsable principal:** Andre.

### Tareas

1. Crear o documentar el entorno Python del proyecto.
2. Instalar una biblioteca para leer Parquet eficientemente (`polars` o `pyarrow`) y las dependencias mínimas de análisis y métricas.
3. Definir una única forma común de cargar datos, evitando que cada notebook implemente una lectura distinta.
4. Preparar una función o script de perfilamiento que reciba un DataFrame y produzca:
    - número de filas y columnas;
    - nombres y tipos de variables;
    - nulos por columna;
    - valores únicos;
    - duplicados de filas;
    - estadísticos numéricos;
    - categorías principales.
5. Preparar el esqueleto del baseline. Debe poder recibir una tabla con `case_id`, `WEEK_NUM`, `target` y predictores, separar los datos temporalmente y calcular métricas.

### Entregable

```text
Entorno documentado + lector de Parquet + perfilador común + esqueleto de baseline
```

### Criterio de terminado

Otro integrante puede cargar `train_base.parquet` usando las instrucciones documentadas y ejecutar el perfilamiento sin modificar el código.

---

## Paso 2 — Integridad estructural de `train_base`

**Responsable principal:** Mathias.

### Alcance preciso

Mathias analiza solo `train_base.parquet` en esta etapa. No limpia definitivamente los datos ni revisa los otros Parquet todavía. Su foco es confirmar que la tabla puede utilizarse como eje de un primer modelo.

### Tareas

#### 2.1 Tamaño y estructura

Registrar:

- filas y columnas;
- nombres de columnas;
- tipos de datos originales;
- columnas numéricas, categóricas y temporales aparentes;
- presencia de clave, objetivo y semana;
- consumo aproximado de memoria.

#### 2.2 Integridad de la clave

Confirmar la clave candidata del caso y medir:

| Control                      | Pregunta                                          |
| ---------------------------- | ------------------------------------------------- |
| Clave nula                   | ¿Existe algún caso sin identificador?             |
| Clave repetida               | ¿Existe más de una fila para el mismo `case_id`?  |
| Filas exactamente duplicadas | ¿Existen dos filas iguales en todas sus columnas? |

La tabla base debe aspirar a tener una sola fila por `case_id`. Si aparecen claves repetidas, Mathias debe mostrar una muestra de los casos y reportar la frecuencia antes de que el equipo decida cómo tratarlos.

#### 2.3 Validación estructural mínima

Identificar y reportar:

- columnas cuyo tipo sea incompatible con su contenido aparente;
- campos vacíos por completo;
- campos que no puedan leerse correctamente;
- nombres de columnas duplicados o inconsistentes, si existieran.

Los nulos, distribuciones, negativos, cardinalidad y variables constantes serán analizados por Nick en el paso 5 para equilibrar la carga.

### Entregable

Un reporte de integridad estructural: tamaño, esquema, tipos, clave candidata, duplicados y errores de lectura o estructura.

### Criterio de terminado

El equipo puede afirmar si `train_base` tiene una unidad de análisis identificable, si `case_id` es una clave utilizable y si existen problemas estructurales que impidan crear el baseline.

---

## Paso 3 — Catálogo funcional de columnas

**Responsable principal:** Gabriel.

### Insumos

```text
train_base.parquet
data/feature_definitions.csv
```

### Objetivo

Clasificar cada columna de la tabla base por función técnica. Esta clasificación no decide todavía cuáles son las variables más predictivas.

### Tareas

Para cada columna, documentar:

| Campo             | Descripción                     | Grupo                 | Acción inicial               | Justificación            |
| ----------------- | ------------------------------- | --------------------- | ---------------------------- | ------------------------ |
| Nombre de columna | Desde `feature_definitions.csv` | Ver grupos siguientes | Mantener / excluir / revisar | Razón técnica o temporal |

Usar los siguientes grupos:

| Grupo                                     | Acción inicial                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| Identificador técnico                     | Excluir del modelo, conservar para trazabilidad.                                 |
| Variable objetivo                         | Usar solo como etiqueta de entrenamiento.                                        |
| Variable temporal                         | Usar primero para validación OOT; evaluar con cuidado si se considera predictor. |
| Predictor potencial                       | Mantener para el baseline si está disponible al momento de predicción.           |
| Posible fuga de información               | Excluir temporalmente hasta confirmar disponibilidad.                            |
| Constante o técnica                       | Excluir.                                                                         |
| Alta cardinalidad / posible identificador | Revisar y normalmente excluir o transformar.                                     |
| Sin definición suficiente                 | Mantener en revisión.                                                            |

### Trabajo conjunto con Nick

Gabriel interpreta los valores que Nick haya marcado como sospechosos. Por ejemplo, debe determinar si un negativo representa un error o una convención válida del dataset.

### Entregable

Catálogo de todas las columnas de la tabla base con su grupo y acción inicial.

### Criterio de terminado

El equipo puede construir una lista explícita de columnas candidatas para el baseline y explicar por qué se excluye cada columna no permitida.

---

## Paso 4 — Objetivo, tiempo y validación OOT

**Responsable principal:** Giancarlo.

### Tareas

#### 4.1 Analizar el objetivo

Calcular distribución de `target`:

| Clase (`target`) | Casos     | Porcentaje |
| ---------------: | --------: | ---------: |
|                0 | 1 478 665 | 96,8563 %  |
|                1 |    47 994 | 3,1437 %   |

Existe un desbalance marcado: la clase positiva representa 3,1437 % de los casos, frente a 96,8563 % de la clase 0 (aproximadamente 1 positivo por cada 30,8 negativos). No hay etiquetas faltantes ni valores distintos de 0/1.

#### 4.2 Analizar `WEEK_NUM`

Crear una tabla por semana:

| `WEEK_NUM` | Casos | Positivos | Tasa de incumplimiento |
| ---------: | ----: | --------: | ---------------------: |
| 0 | 16 735 | 400 | 2,3902 % |
| 1 | 18 841 | 472 | 2,5052 % |
| 2 | 17 476 | 465 | 2,6608 % |
| 3 | 16 108 | 419 | 2,6012 % |
| 4 | 14 309 | 404 | 2,8234 % |
| 5 | 15 335 | 407 | 2,6541 % |
| 6 | 16 385 | 445 | 2,7159 % |
| 7 | 17 238 | 462 | 2,6801 % |
| 8 | 13 076 | 424 | 3,2426 % |
| 9 | 14 716 | 390 | 2,6502 % |
| 10 | 16 650 | 471 | 2,8288 % |
| 11 | 16 101 | 367 | 2,2794 % |
| 12 | 17 105 | 479 | 2,8004 % |
| 13 | 15 791 | 410 | 2,5964 % |
| 14 | 17 534 | 525 | 2,9942 % |
| 15 | 17 713 | 502 | 2,8341 % |
| 16 | 16 598 | 535 | 3,2233 % |
| 17 | 12 146 | 440 | 3,6226 % |
| 18 | 11 729 | 420 | 3,5809 % |
| 19 | 17 194 | 529 | 3,0767 % |
| 20 | 16 366 | 501 | 3,0612 % |
| 21 | 16 962 | 543 | 3,2013 % |
| 22 | 23 261 | 559 | 2,4032 % |
| 23 | 26 734 | 633 | 2,3678 % |
| 24 | 19 503 | 535 | 2,7432 % |
| 25 | 20 744 | 588 | 2,8346 % |
| 26 | 20 621 | 572 | 2,7739 % |
| 27 | 22 450 | 630 | 2,8062 % |
| 28 | 22 013 | 591 | 2,6848 % |
| 29 | 22 841 | 613 | 2,6838 % |
| 30 | 20 688 | 602 | 2,9099 % |
| 31 | 21 275 | 544 | 2,5570 % |
| 32 | 23 889 | 572 | 2,3944 % |
| 33 | 21 525 | 632 | 2,9361 % |
| 34 | 23 477 | 638 | 2,7176 % |
| 35 | 24 137 | 677 | 2,8048 % |
| 36 | 23 812 | 810 | 3,4016 % |
| 37 | 22 255 | 834 | 3,7475 % |
| 38 | 22 424 | 861 | 3,8396 % |
| 39 | 21 039 | 820 | 3,8975 % |
| 40 | 23 907 | 833 | 3,4843 % |
| 41 | 21 125 | 843 | 3,9905 % |
| 42 | 20 643 | 801 | 3,8802 % |
| 43 | 21 029 | 841 | 3,9992 % |
| 44 | 25 887 | 938 | 3,6234 % |
| 45 | 28 947 | 1 001 | 3,4580 % |
| 46 | 22 299 | 915 | 4,1033 % |
| 47 | 31 888 | 936 | 2,9353 % |
| 48 | 22 298 | 890 | 3,9914 % |
| 49 | 30 938 | 1 014 | 3,2775 % |
| 50 | 26 244 | 1 060 | 4,0390 % |
| 51 | 35 920 | 1 136 | 3,1626 % |
| 52 | 22 915 | 689 | 3,0068 % |
| 53 | 22 194 | 823 | 3,7082 % |
| 54 | 18 740 | 869 | 4,6371 % |
| 55 | 18 538 | 789 | 4,2561 % |
| 56 | 17 072 | 811 | 4,7505 % |
| 57 | 18 221 | 838 | 4,5991 % |
| 58 | 17 819 | 867 | 4,8656 % |
| 59 | 17 729 | 889 | 5,0144 % |
| 60 | 15 864 | 729 | 4,5953 % |
| 61 | 15 568 | 807 | 5,1837 % |
| 62 | 17 164 | 895 | 5,2144 % |
| 63 | 14 337 | 589 | 4,1083 % |
| 64 | 10 268 | 322 | 3,1360 % |
| 65 | 4 258 | 96 | 2,2546 % |
| 66 | 2 278 | 52 | 2,2827 % |
| 67 | 1 434 | 30 | 2,0921 % |
| 68 | 825 | 21 | 2,5455 % |
| 69 | 1 582 | 35 | 2,2124 % |
| 70 | 3 808 | 82 | 2,1534 % |
| 71 | 7 921 | 176 | 2,2219 % |
| 72 | 8 858 | 207 | 2,3369 % |
| 73 | 8 687 | 189 | 2,1757 % |
| 74 | 11 280 | 303 | 2,6862 % |
| 75 | 13 656 | 290 | 2,1236 % |
| 76 | 10 261 | 255 | 2,4851 % |
| 77 | 7 754 | 197 | 2,5406 % |
| 78 | 8 755 | 195 | 2,2273 % |
| 79 | 5 158 | 112 | 2,1714 % |
| 80 | 6 246 | 120 | 1,9212 % |
| 81 | 6 743 | 188 | 2,7881 % |
| 82 | 5 423 | 122 | 2,2497 % |
| 83 | 7 295 | 156 | 2,1385 % |
| 84 | 9 721 | 216 | 2,2220 % |
| 85 | 16 239 | 308 | 1,8967 % |
| 86 | 15 630 | 277 | 1,7722 % |
| 87 | 17 886 | 329 | 1,8394 % |
| 88 | 14 234 | 265 | 1,8617 % |
| 89 | 13 600 | 309 | 2,2721 % |
| 90 | 12 103 | 330 | 2,7266 % |
| 91 | 12 674 | 288 | 2,2724 % |

Determinar:

- **Rango:** `WEEK_NUM` va de 0 a 91 (92 semanas); no hay semanas faltantes en la columna.
- **Volumen:** la mediana semanal es 17 088,5 casos. El volumen cae especialmente en las semanas 65–70; la menor es la 68, con 825 casos y 21 positivos. Las semanas 67 y 69 también tienen menos de 1 600 casos. Sus tasas semanales son más variables por el menor tamaño.
- **Variación de la tasa:** en todo el periodo la tasa semanal va de 1,7722 % (semana 86) a 5,2144 % (semana 62). En las semanas 0–35 suele estar alrededor de 2,4–3,2 %; entre las semanas 36–63 predominan tasas más altas (aproximadamente 3,0–5,2 %); luego baja y en las semanas 72–91 se sitúa entre 1,7722 % y 2,7881 %. Esto evidencia variación temporal de la prevalencia, que deberá considerarse al evaluar estabilidad.
- **Uso de semanas de bajo volumen:** no se eliminan del conjunto solo por tener pocos casos; son parte del orden temporal. Se señalan porque sus tasas tienen mayor incertidumbre y el informe posterior debe considerar el soporte semanal.

#### 4.3 Proponer la validación temporal

Se fija como división OOT primaria el **corte `WEEK_NUM = 74`**:

```text
Entrenamiento: WEEK_NUM < 74  (semanas 0–73)
Validación OOT: WEEK_NUM >= 74 (semanas 74–91)
```

Con este corte, entrenamiento contiene 1 332 001 casos y 43 734 positivos (tasa 3,2833 %); validación contiene 194 658 casos y 4 260 positivos (tasa 2,1885 %) distribuidos en 18 semanas. En la validación, cada semana tiene al menos 5 158 casos y 112 positivos. El corte resulta de la regla reproducible que reserva aproximadamente el último 20 % de las semanas observadas (`results/target_analysis/oot_recommendation.csv`); conserva un periodo OOT con positivos semanales suficientes y deja las semanas de menor volumen 65–70 fuera de la ventana de evaluación.

#### 4.4 Confirmar métricas

La competición usa la **Gini Stability Metric** (ver [Evaluation de Home Credit - Credit Risk Model Stability en Kaggle](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/evaluation)). Para cada semana se calcula:

```text
Gini semanal = 2 × ROC-AUC semanal − 1
Gini Stability = media(Gini semanal)
                 + 88 × min(0, pendiente)
                 − 0,5 × desviación estándar(residuos)
```

La pendiente se obtiene de una regresión lineal del Gini semanal respecto de `WEEK_NUM`; los residuos son la diferencia entre cada Gini semanal y el valor ajustado por esa regresión. La penalización de pendiente solo se aplica cuando la tendencia es decreciente.

Como métricas internas de apoyo se reportarán ROC-AUC y Gini globales en el periodo OOT, PR-AUC por el desbalance de clases, y Gini por semana. Para comparar modelos se conservará el mismo corte `WEEK_NUM = 74` y se presentará una tabla temporal con semana, cantidad de casos y positivos, tasa de incumplimiento, ROC-AUC/Gini semanal; se resumirán además Gini Stability, media y mínimo de Gini semanal. Si una semana no contiene ambas clases, su ROC-AUC/Gini no se define y se marcará como no disponible, no como cero.

### Entregable

Protocolo de validación y evaluación: distribución del objetivo, análisis temporal, corte OOT propuesto y métricas.

### Criterio de terminado

El equipo sabe exactamente qué datos se usarán para entrenar, cuáles para evaluar y cómo se medirá la estabilidad del modelo.

---

## Paso 5 — Perfil descriptivo y compatibilidad `train`/`test`

**Responsable principal:** Nick.

### Alcance preciso

Nick asume una tarea técnica de perfilamiento para reducir la carga de Mathias. Trabaja con `train_base.parquet` y con el archivo base equivalente de `test`, si está disponible. No selecciona el modelo ni coordina los entregables de los demás en esta etapa.

### Tareas

#### 5.1 Cobertura y valores faltantes

Para cada columna de entrenamiento, calcular cantidad y porcentaje de nulos. También identificar si existen filas con ausencia masiva de datos o grupos de variables que faltan simultáneamente.

| Columna            | Nulos | % nulos | Patrón u observación inicial |
| ------------------ | ----: | ------: | ---------------------------- |
| <!-- completar --> |       |         |                              |

No eliminar una variable solo por tener nulos. La ausencia puede representar falta de producto, historial o información disponible.

#### 5.2 Distribuciones y valores sospechosos

Para variables numéricas, reportar mínimo, máximo, media, mediana, percentiles, ceros y conteo de negativos.

Para categóricas, reportar valores únicos, categorías más frecuentes, categorías raras y porcentaje del valor dominante.

Identificar:

- variables constantes o casi constantes;
- variables de cardinalidad muy alta;
- valores extremos;
- valores negativos o codificaciones que requieran interpretación.

Un valor sospechoso no se modifica ni se elimina en esta etapa. Nick lo registra y Gabriel confirma su significado con `feature_definitions.csv`.

#### 5.3 Comparación de esquema entre entrenamiento y prueba

Comparar el archivo base de `train` con el correspondiente de `test`:

- columnas presentes en ambos;
- columna objetivo presente solo en entrenamiento, si corresponde;
- tipos compatibles para cada predictor común;
- columnas adicionales o ausentes;
- categorías o rangos claramente incompatibles, si se detectan.

Esta revisión no utiliza el test para aprender imputaciones, escalados, codificaciones ni parámetros de selección. Su único fin es prevenir que el pipeline falle al generar predicciones.

### Entregable

Un reporte técnico de cobertura y distribución de la tabla base, acompañado de una comparación de esquema `train`/`test` y una lista de valores que requieren interpretación de Gabriel.

### Criterio de terminado

El equipo conoce el patrón de faltantes, la variación de cada predictor, las columnas constantes o de alta cardinalidad y las diferencias estructurales que deben considerarse para aplicar el mismo pipeline a entrenamiento y prueba.

---

# Reunión de cierre de etapa A

Cada integrante presenta únicamente evidencia concreta:

| Persona   | Debe presentar                                                              |
| --------- | --------------------------------------------------------------------------- |
| Mathias   | Reporte de calidad y problemas detectados.                                  |
| Gabriel   | Catálogo y clasificación de columnas.                                       |
| Giancarlo | Distribución de target, semanas y propuesta OOT.                            |
| Andre     | Entorno, perfilador y contrato técnico del baseline.                        |
| Nick      | Reporte de nulos, distribuciones, cardinalidad y compatibilidad train/test. |

## Decisiones que deben aprobarse

```text
Tabla de entrenamiento: train_base.parquet
Clave: [por confirmar]
Objetivo: [por confirmar]
Variable temporal: [por confirmar]
Columnas excluidas: [lista]
Columnas candidatas: [lista]
Corte OOT: [semanas]
Métrica principal: [confirmada]
Modelo baseline: [definido]
```

# Criterio para pasar a la siguiente etapa

La etapa A termina cuando se cumplan todos estos puntos:

- [ ] La tabla base está perfilada y su clave fue verificada.
- [ ] Los nulos, duplicados y valores sospechosos están documentados.
- [ ] Las columnas tienen clasificación funcional.
- [ ] El `target` y `WEEK_NUM` están confirmados.
- [ ] Existe una partición OOT documentada.
- [ ] El entorno carga Parquet de forma reproducible.
- [ ] Andre puede entrenar el baseline con el contrato acordado.
- [ ] Los hallazgos relevantes se registraron en `development/`.

Una vez cerrada, la siguiente tarea es entrenar el baseline y después añadir **una sola fuente adicional** de manera controlada.
