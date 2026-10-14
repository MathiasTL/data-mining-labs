# 4. Modelado

> **Estado:** completar tras ejecutar los experimentos bajo el mismo diseño de validación.

## 4.1 Seleccionar técnica de modelado

| Modelo | Rol | Motivo de inclusión | Tratamiento de nulos/categóricas |
|---|---|---|---|
| Predicción constante | Referencia | Establece el mínimo de desempeño | No aplica |
| Regresión logística | Baseline interpretable | Comparación simple y explicable | <!-- Completar --> |
| <!-- Modelo de boosting elegido --> | Candidato no lineal | <!-- Completar --> | <!-- Completar --> |

### Criterio de elección de técnicas

<!-- Explicar la decisión considerando datos tabulares, volumen, interpretabilidad y recursos disponibles. -->

## 4.2 Generar diseño de prueba

### Validación fuera de tiempo

<!-- Especificar con precisión: semanas usadas para entrenamiento, validación y, si aplica, cortes repetidos. -->

### Controles de validez

<!-- Confirmar ausencia de fuga entre particiones, distribución del target y uso exclusivo de train para ajustar transformaciones. -->

## 4.3 Modelo de construcción

### Registro de experimentos

| Experimento | Variables | Partición | Hiperparámetros | Semilla | Tiempo | Estado |
|---|---|---|---|---:|---:|---|
| <!-- Completar --> |  |  |  |  |  |  |

### Ajuste de hiperparámetros

<!-- Describir estrategia de búsqueda, regularización y early stopping si se aplicó. -->

## 4.4 Evaluar el modelo

| Modelo | Gini Stability (OM2) | Gini / ROC-AUC (OM1) | PR-AUC | Recall | FPR a recall fijo (OM4) | Calibración (OM4) | Observaciones |
|---|---:|---:|---:|---:|---:|---|---|
| <!-- Completar --> |  |  |  |  |  |  |  |

El punto de operación para el FPR a recall fijo (por ejemplo, TPR = 0,80) es un *supuesto por acordar*.

### Importancia de variables (OM3)

| Variable o grupo | Método (ganancia, permutación o SHAP) | Importancia | Coherencia temporal |
|---|---|---:|---|
| <!-- Completar --> |  |  |  |

### Interpretación

<!-- Insertar variables importantes, análisis de errores y advertencia de que importancia no implica causalidad. -->
