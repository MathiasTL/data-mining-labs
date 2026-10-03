# 2. Comprensión de los datos

## Objetivo de la fase

Conocer la estructura, granularidad, calidad y dimensión temporal de cada fuente antes de construir características o entrenar modelos.

## Archivos de partida

- `../data/parquet_files/train/`
- `../data/parquet_files/test/`
- `../data/feature_definitions.csv`
- `../data/sample_submission.csv`

## Tareas

### 2.1 Inventario y granularidad

1. Listar todos los archivos de entrenamiento y prueba.
2. Para cada tabla, registrar filas, columnas, tipos, claves candidatas, porcentaje de nulos y memoria utilizada.
3. Identificar la tabla base y confirmar la clave del caso, el objetivo y `WEEK_NUM`.
4. Clasificar cada tabla según su relación con la tabla base: uno a uno, uno a muchos o granularidad más detallada.
5. Agrupar fuentes por dominio: solicitud actual, solicitudes previas, bureau, personas, depósitos, tarjetas, registros fiscales y otros.

### 2.2 Descripción y exploración

1. Leer `feature_definitions.csv` y vincular cada variable con su definición y fuente.
2. Calcular distribución del objetivo y proporción de eventos.
3. Describir rangos, cardinalidad, categorías, faltantes y valores extremos.
4. Comparar distribuciones por `WEEK_NUM` para identificar cambios de población.
5. Explorar asociaciones iniciales entre variables y objetivo sin extraer conclusiones causales.
6. Comparar columnas y tipos entre entrenamiento y prueba.

### 2.3 Calidad y riesgos

1. Detectar duplicados, claves no únicas, codificaciones inconsistentes y valores imposibles.
2. Analizar el patrón de faltantes: ausencia de dato, ausencia de producto o valor desconocido.
3. Identificar campos con disponibilidad potencialmente posterior al momento de la solicitud.
4. Revisar si las relaciones entre tablas pueden multiplicar casos al hacer *joins*.
5. Identificar segmentos con poca cobertura de datos o fuerte desbalance.

## Preguntas que esta fase debe responder

- ¿Cuál es exactamente una observación a predecir?
- ¿Qué tablas aportan varias filas por caso y requieren agregación?
- ¿Qué columnas permiten un corte temporal válido?
- ¿Qué datos son escasos, inconsistentes o potencialmente filtrados desde el futuro?
- ¿Existen diferencias estructurales entre `train` y `test`?

## Evidencias y entregable

- Inventario de tablas y diagrama de granularidades.
- Diccionario de datos operativo.
- Informe EDA con tablas y gráficos.
- Informe de calidad y lista de riesgos de fuga de información.
- Hipótesis priorizadas sobre fuentes y señales útiles.

## Criterio para avanzar

No iniciar integraciones masivas ni modelado hasta conocer la tabla base, las relaciones entre tablas, la variable temporal y los principales problemas de calidad.
