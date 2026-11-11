# 4. Modelado

## Objetivo de la fase

Construir modelos comparables y reproducibles para estimar la probabilidad de incumplimiento, usando el mismo protocolo de validación fuera de tiempo.

## Diseño mínimo de experimentos

1. **Referencia trivial:** probabilidad constante basada en la prevalencia del evento.
2. **Baseline interpretable:** regresión logística con las transformaciones necesarias.
3. **Modelo no lineal:** al menos un modelo de *gradient boosting* para datos tabulares, como LightGBM, XGBoost o CatBoost.
4. **Modelo final candidato:** la mejor configuración según desempeño y estabilidad, no solo según una métrica puntual.

La elección concreta debe considerar tamaño de datos, memoria, soporte para variables categóricas, faltantes e interpretabilidad. No es obligatorio usar todos los algoritmos listados.

## Tareas

### 4.1 Diseño de prueba

1. Definir un corte OOT con `WEEK_NUM`: entrenamiento en semanas anteriores y validación en semanas posteriores.
2. Verificar que la prevalencia y el tamaño de cada partición sean razonables.
3. Si se emplean varios cortes temporales, registrar el resultado por corte y su promedio/variabilidad.
4. Una validación estratificada aleatoria puede utilizarse como contraste, pero no para reemplazar el resultado OOT.

### 4.2 Entrenamiento

1. Usar el mismo conjunto de variables y partición para comparar modelos de forma justa.
2. Controlar semilla, tiempo, memoria y versión de librerías.
3. Aplicar regularización y `early stopping` si la técnica lo permite.
4. Ajustar hiperparámetros de forma acotada; evitar búsquedas costosas sin una hipótesis clara.
5. Registrar modelo, datos, variables, parámetros, semilla y tiempo de entrenamiento.

### 4.3 Interpretación

1. Obtener importancia de variables por método apropiado: ganancia, permutación o SHAP.
2. Revisar si las variables más influyentes son coherentes con el dominio y temporalmente válidas.
3. Analizar predicciones erróneas y casos con información escasa.
4. No interpretar importancia como causalidad.

## Métricas a registrar

- Gini y ROC-AUC globales.
- PR-AUC cuando el evento sea minoritario.
- Precisión, recall y matriz de confusión para umbrales justificados.
- Curva o medida de calibración cuando el uso de probabilidades lo requiera.
- Gini/AUC por semana o bloque temporal de validación.

## Evidencias y entregable

- Tabla de experimentos comparable.
- Configuraciones reproducibles.
- Modelos candidatos serializados o regenerables.
- Métricas globales y temporales.
- Análisis inicial de importancia y errores.

## Criterio para avanzar

Se pasa a evaluación cuando existan al menos una línea base y dos candidatos comparados con el mismo esquema OOT, resultados por periodo y trazabilidad completa de los experimentos.
