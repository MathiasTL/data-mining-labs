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

| Clase |              Casos | Porcentaje |
| ----: | -----------------: | ---------: |
|     0 | <!-- completar --> |            |
|     1 | <!-- completar --> |            |

Identificar si existe desbalance y si hay valores faltantes o inesperados en la etiqueta.

#### 4.2 Analizar `WEEK_NUM`

Crear una tabla por semana:

|         `WEEK_NUM` | Casos | Casos positivos | Tasa de incumplimiento |
| -----------------: | ----: | --------------: | ---------------------: |
| <!-- completar --> |       |                 |                        |

Determinar:

- rango de semanas disponible;
- semanas con volumen insuficiente;
- cambios en tasa de incumplimiento;
- posibles periodos para entrenamiento y validación.

#### 4.3 Proponer la validación temporal

Proponer y justificar una primera división OOT:

```text
Entrenamiento: semanas antiguas
Validación OOT: semanas posteriores
```

El corte debe sustentarse en el volumen de datos y representación de la clase positiva, no en una elección arbitraria.

#### 4.4 Confirmar métricas

Confirmar en la documentación oficial de Kaggle la definición de la métrica de la competición y documentar:

- fórmula o criterio de Gini Stability;
- métricas internas de apoyo;
- reporte temporal que se usará para comparar modelos.

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
