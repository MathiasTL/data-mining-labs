# 3. Preparación de datos

> **Estado:** fase planificada; implementación pendiente. El reparto, las dependencias y las compuertas de trabajo para cinco personas están en el [backlog operativo del capítulo 03](../context/03-preparacion-datos-backlog.md). Solo se documentarán aquí transformaciones implementadas y validadas.

## Plan de ejecución

El backlog divide la fase entre cinco frentes con propiedad de archivos disjunta. El desarrollo y las pruebas sintéticas se realizan en paralelo; las decisiones de interfaz y temporalidad, la integración final y las ejecuciones pesadas sobre datos reales se controlan mediante compuertas secuenciales. Véase [`context/03-preparacion-datos-backlog.md`](../context/03-preparacion-datos-backlog.md).

## 3.1 Seleccionar datos

### Criterio de selección

La tabla de modelado debe conservar una fila por caso de la tabla base. Las fuentes históricas con varias filas por caso se agregan antes de integrarse.

### Fuentes y columnas incluidas

<!-- Insertar tabla: fuente, variables/agrupaciones utilizadas, justificación y disponibilidad temporal. -->

### Fuentes y columnas excluidas

<!-- Insertar tabla: elemento excluido, motivo y evidencia. -->

### Partición temporal

<!-- Especificar semanas de entrenamiento, validación OOT y cualquier corte adicional. -->

## 3.2 Datos limpios

| Problema           | Regla aplicada     | Variables afectadas | Justificación |
| ------------------ | ------------------ | ------------------- | ------------- |
| Valores faltantes  | <!-- Completar --> |                     |               |
| Valores atípicos   | <!-- Completar --> |                     |               |
| Tipos/codificación | <!-- Completar --> |                     |               |

## 3.3 Datos de construcción

### Variables agregadas y derivadas

<!-- Insertar catálogo: nombre de variable, fuente, lógica, ventana temporal si existe y significado. -->

### Prevención de fuga de información

<!-- Explicar controles implementados para asegurar que cada variable está disponible antes de la predicción. -->

## 3.4 Integrar datos

### Proceso de integración

<!-- Insertar diagrama o tabla de joins: tabla base, fuente, clave, cardinalidad antes de agregar y validación posterior. -->

### Validación de cardinalidad

<!-- Reportar número de casos antes/después de cada integración y confirmar unicidad de la clave final. -->

## 3.5 Formatear datos

<!-- Describir codificación de categóricas, imputación, esquema final, lista de variables y compatibilidad train/test. -->

### Resultado de la fase

<!-- Indicar ruta o comando para regenerar la tabla maestra y el pipeline de preparación. -->
