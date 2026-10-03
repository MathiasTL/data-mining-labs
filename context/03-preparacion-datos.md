# 3. Preparación de datos

## Objetivo de la fase

Transformar las fuentes originales en un conjunto de modelado reproducible, con una fila por caso, variables interpretables y un tratamiento que respete el instante temporal de predicción.

## Estrategia de construcción

La tabla base es el eje. Cada tabla con múltiples registros por caso debe transformarse en características agregadas antes de unirla. Las fuentes se incorporan progresivamente para medir su aporte y facilitar la detección de errores.

## Tareas

### 3.1 Selección y partición de datos

1. Confirmar columnas permitidas y excluir identificadores técnicos o variables que filtren el futuro.
2. Definir particiones basadas en `WEEK_NUM`: semanas antiguas para entrenamiento y semanas posteriores para validación OOT.
3. Mantener el conjunto de prueba separado; no usarlo para aprender transformaciones o elegir parámetros.
4. Documentar las fuentes y columnas descartadas junto con su justificación.

### 3.2 Limpieza

1. Corregir tipos, formatos y codificaciones incoherentes.
2. Tratar faltantes de acuerdo con su significado; no imputar automáticamente sin análisis.
3. Crear indicadores de ausencia cuando puedan representar una señal válida.
4. Identificar y tratar valores imposibles o extremos con reglas explícitas y reproducibles.
5. Resolver categorías infrecuentes o no vistas de forma consistente entre entrenamiento y prueba.

### 3.3 Construcción de variables

Para registros históricos, aplicar únicamente agregaciones coherentes con el dominio:

- conteo de registros, productos o categorías distintas;
- mínimo, máximo, media, mediana, suma y dispersión de importes;
- primera, última o más reciente observación cuando exista una fecha válida;
- antigüedad, recencia y frecuencia;
- proporciones y banderas de existencia de eventos o productos.

Cada característica debe registrar tabla de origen, columnas utilizadas, agregación aplicada y justificación temporal.

### 3.4 Integración y formato

1. Validar el número de casos antes y después de cada unión.
2. Garantizar unicidad de la clave final y ausencia de duplicación de casos.
3. Asegurar que entrenamiento y prueba tengan el mismo esquema de características, salvo el objetivo.
4. Preparar la codificación e imputación necesarias para cada familia de modelos.
5. Guardar el pipeline y la lista ordenada de columnas para poder reconstruir el dataset.

## Controles obligatorios

- El número de casos tras cada *join* no debe crecer inesperadamente.
- Ninguna transformación debe aprender parámetros con la validación OOT o prueba.
- Las variables temporales deben usar exclusivamente información disponible al momento de predicción.
- Los nombres de columnas, tipos y orden deben coincidir en entrenamiento y prueba.

## Evidencias y entregable

- Tabla maestra de entrenamiento y de prueba regenerables.
- Catálogo de características y procedencia.
- Reglas de limpieza e imputación.
- Validaciones de cardinalidad e informe de fuga temporal.
- Pipeline ejecutable de preparación.

## Criterio para avanzar

Se avanza cuando el pipeline puede generar de cero las matrices de entrenamiento y prueba, conserva una fila por caso y supera los controles de cardinalidad y temporalidad.
