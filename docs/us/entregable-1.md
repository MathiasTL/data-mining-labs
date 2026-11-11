# Trabajo Integrador - Mineria de Datos
## Prediccion del Incumplimiento Crediticio y Estabilidad Temporal
### Home Credit - Credit Risk Model Stability - Grupo 3

---

## Indice

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [1. Entendimiento Empresarial](#1-entendimiento-empresarial)
  - [1.1 Determinar los objetivos comerciales](#11-determinar-los-objetivos-comerciales)
  - [1.2 Evaluar la situacion](#12-evaluar-la-situacion)
  - [1.3 Determinar los objetivos de la mineria de datos](#13-determinar-los-objetivos-de-la-mineria-de-datos)
  - [1.4 Producir el plan del proyecto](#14-producir-el-plan-del-proyecto)
- [2. Comprension de los Datos](#2-comprension-de-los-datos)
  - [2.1 Recopilar datos iniciales](#21-recopilar-datos-iniciales)
  - [2.2 Describir datos](#22-describir-datos)
  - [2.3 Explorar datos](#23-explorar-datos)
  - [2.4 Verificar la calidad de los datos](#24-verificar-la-calidad-de-los-datos)
- [3. Preparacion de Datos](#3-preparacion-de-datos)
  - [3.1 Seleccionar datos](#31-seleccionar-datos)
  - [3.2 Datos limpios](#32-datos-limpios)
  - [3.3 Datos de construccion](#33-datos-de-construccion)
  - [3.4 Integrar datos](#34-integrar-datos)
  - [3.5 Formatear datos](#35-formatear-datos)

---

## Resumen Ejecutivo

> Pendiente de completar al finalizar las fases de modelado y evaluacion.

---

## 1. Entendimiento Empresarial

### 1.1 Determinar los objetivos comerciales

#### Compilacion de la informacion de la empresa

Home Credit es una entidad financiera internacional especializada en credito de consumo para personas con historial crediticio limitado o inexistente, que opera principalmente en mercados emergentes de Asia y Europa del Este. Su propuesta de valor es ampliar el acceso al credito a segmentos de la poblacion que los bancos tradicionales suelen excluir, utilizando fuentes de informacion alternativas para evaluar la solvencia de los solicitantes.

El problema central que motiva este proyecto es que los modelos de scoring de riesgo crediticio pierden poder predictivo con el tiempo. A medida que cambia el perfil de los solicitantes, su comportamiento de pago y el contexto economico (fenomeno conocido como covariate/concept drift), el modelo entrenado en un periodo historico puede dejar de representar bien la poblacion actual. Esto genera dos riesgos operativos concretos: subestimar el riesgo de nuevos solicitantes y aumentar las perdidas por incumplimiento, o sobreestimarlo y rechazar clientes solventes, afectando tanto la rentabilidad como el acceso al credito. La respuesta habitual, recalibrar o reentrenar el modelo con frecuencia, es costosa y reactiva, no preventiva.

La Resolucion SBS N 00053-2023 (Reglamento de Gestion de Riesgos de Modelo) exige a las entidades financieras supervisadas validar, monitorear y realizar pruebas retrospectivas de sus modelos de riesgo de credito antes y durante su uso, para asegurar que mantengan un desempeno confiable en el tiempo.

El dataset utilizado es **Home Credit - Credit Risk Model Stability**, publicado en Kaggle. Combina informacion de solicitud actual, solicitudes previas, central de riesgo (bureau domestico e internacional), historial de pagos, tarjetas y depositos, distribuida en mas de 30 tablas con distintos niveles de granularidad, y organizada por semana de observacion (WEEK_NUM). Esta estructura temporal permite no solo entrenar un modelo predictivo, sino evaluar explicitamente su estabilidad entre periodos.

El proyecto se desarrolla en un entorno academico y no representa una solucion desplegada en la entidad financiera real. Los beneficios del analisis de costo-beneficio corresponden a un escenario hipotetico de adopcion real; el prototipo del proyecto no genera gastos monetarios reales para el equipo.

#### Definicion de los objetivos de negocio

El objetivo principal es aplicar la metodologia CRISP-DM para sentar las bases de un modelo de riesgo crediticio que sea preciso y estable en el tiempo. Este objetivo general se concreta en las cuatro metas del caso de negocio (fuente: `IT Business Case Template G3 MD.docx.pdf`):

- Identificar las variables con mayor influencia en el riesgo crediticio y documentarlas.
- Disponer de al menos un modelo predictivo de incumplimiento entrenado y documentado.
- Reducir el riesgo de perdida de validez del modelo en el tiempo, evaluandolo con validacion fuera de tiempo por WEEK_NUM.
- Evitar el rechazo indebido de clientes con historial limitado pero buen comportamiento, controlando los falsos positivos.

Fuente: `development/01-entendimiento-empresarial.md`.

#### Criterios de rendimiento del negocio

- Completar las seis fases de CRISP-DM y documentar el proceso de forma reproducible.
- Obtener al menos un modelo funcional y comparar modelos de clasificacion.
- Evaluar el desempeno predictivo y su variacion temporal por WEEK_NUM, incluyendo validacion fuera de tiempo.
- Entregar el conjunto de datos preparado para modelado, la evaluacion comparativa, el analisis de estabilidad, el informe y la presentacion final.
- Construir un prototipo funcional de demostracion en un entorno controlado, sin conexion a sistemas reales.

---

### 1.2 Evaluar la situacion

#### Inventario de recursos

**Datos disponibles:**

| Recurso | Ubicacion | Descripcion |
|---|---|---|
| Dataset Home Credit (train) | `data/parquet_files/train/` | 32 archivos Parquet en 17 familias logicas |
| Dataset Home Credit (test) | `data/parquet_files/test/` | 36 archivos Parquet; muestra local de 10 filas cada uno |
| Diccionario de variables | `data/feature_definitions.csv` | 465 entradas con descripcion semantica oficial |
| Formato de prediccion | `data/sample_submission.csv` | Estructura de la salida final |

**Software y herramientas:**

| Herramienta | Uso |
|---|---|
| Python 3.11+ | Entorno de ejecucion principal |
| polars | Procesamiento eficiente de datos tabulares grandes |
| pyarrow | Lectura de archivos Parquet |
| numpy | Operaciones numericas |
| scikit-learn | Modelado y metricas |
| TensorFlow | Modelos de aprendizaje profundo (fase de modelado) |
| uv | Gestion de entorno y dependencias; garantiza reproducibilidad |

Todas las versiones estan fijadas en `uv.lock`. Toda lectura de datos pasa por `src/dm_credit/io.py`.

**Equipo humano:**

| Integrante | Responsabilidad |
|---|---|
| Pedro Lezama Gonzales | Patrocinador; valida entregables y resuelve conflictos escalados. |
| Nick Emanuel Salcedo Alfaro | Lider del proyecto; coordina cronograma y tareas, contacto con patrocinador y consolidacion del informe. |
| Mathias James Torres Lezama | Comprension y preparacion de datos; tablas de solicitud actual y bureau. |
| Andre Alexander Cuenca Echevarria | Modelado y prototipo demostrativo (API o dashboard). |
| Gabriel Mathias Davalos Velazco | Comprension y preparacion de datos; solicitudes previas, depositos y tarjetas. |
| Giancarlo Villavicencio Davila | Evaluacion comparativa y estabilidad temporal con validacion OOT por WEEK_NUM. |

**Hardware:** equipos personales de los integrantes, sin presupuesto para infraestructura externa. Las tablas mas grandes (bureau de credito, aprox. 188 millones de filas) se procesan en streaming monohilo con `POLARS_MAX_THREADS=1`.

#### Requisitos, supuestos y restricciones

**Requisitos:**

1. Aplicar la metodologia CRISP-DM durante todo el desarrollo.
2. Construir y comparar multiples modelos de clasificacion.
3. Evaluar los modelos mediante metricas apropiadas para el problema de incumplimiento crediticio.
4. Evaluar el desempeno de forma segmentada por periodos temporales.
5. Documentar el proceso y los resultados de forma reproducible.
6. Construir un prototipo funcional del modelo con fines demostrativos.

**Supuestos:**

- El dataset se usa bajo los terminos de uso publico de Kaggle, exclusivamente con fines academicos y sin fines comerciales.
- Los datos de test disponibles localmente contienen solo 10 filas; sirven para verificar compatibilidad de esquema pero no representan la distribucion real del test de la competicion.
- El patron de drift en la tasa de incumplimiento entre semanas es representativo del problema real de estabilidad.

**Restricciones:**

- No se integrara el modelo con sistemas financieros reales ni se automatizaran decisiones crediticias en produccion.
- No se usaran datos externos a los distribuidos para la competicion.
- El presupuesto monetario es cero; solo se dispone del tiempo y computo del equipo.
- Referencia normativa: Resolucion SBS N 00053-2023 (Reglamento de Gestion de Riesgos de Modelo).

#### Riesgos y contingencias

| Riesgo | Impacto | Mitigacion acordada | Estado |
|---|---|---|---|
| Alto volumen y complejidad de datos | Procesamiento lento o fallos de memoria | Procesamiento incremental, agregaciones por fuente, streaming monohilo | Pendiente |
| Fuga de informacion | Metricas artificialmente altas en entrenamiento | Validacion OOT estricta, filtros temporales antes de agregar, filtro empirico de Gini univariado | Pendiente |
| Sobreajuste | Bajo rendimiento futuro | Regularizacion, early stopping y evaluacion semanal por WEEK_NUM | Pendiente |
| Inestabilidad temporal del modelo | Degradacion de Gini Stability en OOT | Seguimiento de Gini semanal; penalizacion explicita de la tendencia negativa en la metrica oficial | Pendiente |
| Variaciones de cobertura entre periodos | Indicadores inestables perjudican Gini Stability | Identificadas en preparacion de datos; evaluacion cuidadosa o exclusion en modelado | En seguimiento |
| Prototipo de demostracion fuera de plazo | Entregable incompleto | Inicio temprano de la arquitectura del prototipo desde la fase de modelado | Pendiente |
| Limitaciones de recursos computacionales | Imposibilidad de entrenar modelos complejos | Priorizar LightGBM y XGBoost; procesar bureau en streaming por shards | Pendiente |

Si los resultados iniciales son insatisfactorios se revisara la seleccion de variables (ampliando familias bloqueadas con evidencia temporal adicional) y se ajustaran hiperparametros con validacion cruzada temporal. Si los plazos se ven comprometidos, se priorizara tener al menos un modelo base evaluado antes de iterar.

#### Terminologia

| Termino | Definicion |
|---|---|
| case_id | Identificador unico de cada solicitud de credito; unidad de analisis del proyecto |
| WEEK_NUM | Semana de observacion de la solicitud (0 a 91); eje temporal para particion OOT y evaluacion de estabilidad |
| target | Variable objetivo binaria: 1 si el solicitante incumplio, 0 si no. Solo disponible en entrenamiento |
| date_decision | Fecha en que se tomo la decision sobre la solicitud; referencia temporal as-of principal |
| OOT | Out-of-time: conjunto de validacion con observaciones posteriores al corte temporal (WEEK_NUM >= 74) |
| Gini Stability | Metrica oficial de la competicion; combina poder discriminante global, tendencia temporal y variabilidad semanal del Gini |
| Gini | 2 x ROC_AUC - 1; mide la capacidad del modelo para separar incumplidos de no incumplidos |
| Drift | Cambio en la distribucion de los datos o en la relacion entre predictores y objetivo a lo largo del tiempo |
| As-of date | Fecha de referencia a partir de la cual solo se usa informacion disponible al momento de la decision |
| Score | Probabilidad estimada de incumplimiento que produce el modelo para cada caso |
| Bureau de credito | Entidad que centraliza el historial crediticio; representado aqui por las tablas credit_bureau_a y credit_bureau_b |
| Shards | Fragmentos fisicos de una misma tabla logica distribuidos en multiples archivos Parquet |
| num_group1, num_group2 | Indices de registros dentro de un caso; identifican la profundidad de anidamiento de las tablas secundarias |

#### Analisis de costos/beneficios

El siguiente analisis corresponde a un escenario hipotetico de adopcion real del modelo por parte de una entidad financiera (fuente: `docs/us/IT Business Case Template G3 MD.docx.pdf`). El proyecto academico en si no genera costos monetarios.

**Costo total de propiedad a 5 anos:**

| Rubro | Ano 1 | Ano 2 | Ano 3 | Ano 4 | Ano 5 | Total |
|---|---|---|---|---|---|---|
| Infraestructura / Hardware | S/ 12 000 | S/ 12 000 | S/ 12 000 | S/ 12 000 | S/ 12 000 | S/ 60 000 |
| Software | S/ 5 000 | S/ 3 000 | S/ 3 000 | S/ 3 000 | S/ 3 000 | S/ 17 000 |
| Operaciones | S/ 25 000 | - | - | - | - | S/ 25 000 |
| Implementacion | S/ 20 000 | - | - | - | - | S/ 20 000 |
| Mantenimiento | S/ 8 000 | S/ 8 000 | S/ 8 000 | S/ 8 000 | S/ 8 000 | S/ 40 000 |
| **Total** | **S/ 70 000** | **S/ 23 000** | **S/ 23 000** | **S/ 23 000** | **S/ 23 000** | **S/ 162 000** |

**Beneficios proyectados a 5 anos:**

| Rubro | Ano 1 | Ano 2 | Ano 3 | Ano 4 | Ano 5 | Total |
|---|---|---|---|---|---|---|
| Reduccion de perdidas por incumplimiento | S/ 15 000 | S/ 30 000 | S/ 30 000 | S/ 30 000 | S/ 30 000 | S/ 135 000 |
| Ahorro por evitar recalibraciones reactivas | S/ 30 000 | S/ 70 000 | S/ 80 000 | S/ 80 000 | S/ 80 000 | S/ 340 000 |
| Incremento de ingresos por menor rechazo indebido | - | S/ 12 000 | S/ 15 000 | S/ 15 000 | S/ 15 000 | S/ 57 000 |
| **Total beneficios** | **S/ 45 000** | **S/ 112 000** | **S/ 125 000** | **S/ 125 000** | **S/ 125 000** | **S/ 532 000** |

**ROI:** (S/ 532 000 - S/ 162 000) / S/ 162 000 = **228 %**

---

### 1.3 Determinar los objetivos de la mineria de datos

#### Objetivos de mineria de datos

Siguiendo la declaracion de alcance del caso de negocio (fuente: `IT Business Case Template G3 MD.docx.pdf`):

1. Integrar las tablas disponibles del dataset Home Credit - Credit Risk Model Stability respetando sus distintas granularidades.
2. Tratar los valores faltantes y atipicos de cada fuente.
3. Disenar variables de comportamiento de pago y de riesgo a partir de las fuentes integradas.
4. Entrenar y comparar clasificadores binarios sobre la etiqueta de incumplimiento.
5. Aplicar validacion fuera de tiempo por WEEK_NUM.
6. Evaluar el desempeno con Gini y su estabilidad temporal.

Fuente: `development/01-entendimiento-empresarial.md`.

#### Definicion tecnica

Se construira un modelo predictivo que estime, para cada caso identificado por case_id, un score continuo de probabilidad de incumplimiento entre 0 y 1. La variable WEEK_NUM se utilizara para evaluar la estabilidad temporal de las predicciones y del desempeno del modelo.

Fuente: `development/01-entendimiento-empresarial.md`.

#### Criterios de rendimiento de mineria de datos

**Metrica principal: Gini Stability**

La metrica de seleccion de modelos es la Gini Stability Metric de la competicion Home Credit, consultada en la pagina oficial el 2026-10-12 (fuente: `results/metric_review/definition.csv`). Evalua simultaneamente el poder discriminante global, la tendencia temporal y la variabilidad entre semanas:

Para cada semana t con ambas clases presentes:

    G_t = 2 x ROC_AUC(y_t, p_t) - 1

    Gini Stability = mean(G_t) + 88 x min(0, a) - 0,5 x std_pop(r_t)

    a, b = pendiente e intercepto de la regresion lineal de G_t sobre WEEK_NUM
    r_t = G_t - (a x WEEK_NUM_t + b)  [residuos de la regresion]
    std_pop = desviacion estandar poblacional (ddof = 0)

Interpretacion:
- mean(G_t): poder discriminante promedio semanal, sin ponderar por cantidad de casos.
- 88 x min(0, a): penaliza una tendencia negativa del Gini en el tiempo; una tendencia positiva no recibe premio.
- 0,5 x std_pop(r_t): penaliza la variabilidad del Gini alrededor de su tendencia.

Condiciones que producen error explicito: semana monoclase, nulos o valores no finitos en predicciones, etiquetas fuera de {0, 1}, o menos de dos semanas evaluables. La implementacion esta en `src/dm_credit/baseline.py`, probada en `tests/test_baseline.py`.

**Metricas diagnosticas complementarias:**

| Metrica | Particion | Papel |
|---|---|---|
| ROC AUC y Gini global | Entrenamiento y OOT | Diagnostico de discriminacion global |
| Gini por semana; pendiente y desviacion semanales | OOT | Componentes auditables de Gini Stability |
| Casos y positivos por semana | OOT | Soporte de cada estimacion semanal |
| Log loss y Brier score | OOT | Calibracion del modelo |
| PR AUC | OOT | Diagnostico por desbalance de clases |

**Criterio de seleccion del modelo final:** mayor valor de Gini Stability OOT. Como criterios secundarios: interpretabilidad, coste computacional, ausencia de senales de sobreajuste y reproducibilidad.

---

### 1.4 Producir el plan del proyecto

#### Plan de proyecto

| Fase | Fecha limite | Producto esperado |
|---|---|---|
| Comprension del negocio | 05/12/2026 | Objetivos, alcance y plan definidos |
| Comprension de datos | 05/12/2026 | Inventario, exploracion y calidad de datos |
| Preparacion de datos | 05/12/2026 | Conjunto de datos preparado para modelado |
| Modelado | 05/12/2026 | Modelos comparados con validacion temporal |
| Evaluacion | 05/12/2026 | Desempeno comparativo y estabilidad analizados |
| Implementacion y presentacion final | 05/12/2026 | Prototipo demostrativo, informe y presentacion |

El cronograma se ajusto para establecer el 05/12/2026 como fecha limite comun para todas las fases del proyecto (fuente: `development/01-entendimiento-empresarial.md`).

Las decisiones de asignacion de tareas se realizan por fase de CRISP-DM en consulta con el equipo. Los conflictos se escalan progresivamente: integrantes involucrados, luego lider de proyecto, luego patrocinador. Cada experimento registra: version de datos, variables utilizadas, particion, modelo, hiperparametros, semilla aleatoria, metricas y fecha de ejecucion.

#### Preguntas criticas

**Desde una perspectiva de negocio:**

- Que se espera obtener de este proyecto? Un modelo que prediga el incumplimiento crediticio y que mantenga su poder predictivo a lo largo del tiempo, con evidencia documentada de su estabilidad.
- Como se define la finalizacion del trabajo? Con la entrega del informe final, el prototipo de demostracion funcionando y la sustentacion ante el curso antes del 05/12/2026.
- Se dispone de los recursos necesarios? Si: el dataset esta disponible localmente, el entorno tecnico esta configurado y el equipo tiene asignados roles y responsabilidades.
- Los resultados del analisis de costos/beneficios hacen que el proyecto sea viable? Si: ROI proyectado del 228 % en un escenario hipotetico de adopcion real.

**Desde una perspectiva de mineria de datos:**

- En que forma puede ayudar la mineria de datos? Identificando patrones en el historial crediticio, demografico y de comportamiento de pago que predicen el incumplimiento con mayor precision y estabilidad que reglas manuales.
- Que tecnicas se usaran? Modelos de ensamble basados en arboles (LightGBM, XGBoost) como punto de partida; modelos lineales como referencia de interpretabilidad.
- Como se sabra si los resultados son precisos? Mediante Gini Stability en OOT. El baseline de prevalencia fija el piso minimo.
- Como se desplegaran los resultados? Mediante un prototipo funcional (API, dashboard o script) en un entorno simulado, sin conexion a sistemas reales.
- El plan incluye todas las fases de CRISP-DM? Si, segun el cronograma anterior.
- Los riesgos y dependencias estan incluidos en el plan? Si, en la tabla de riesgos de la seccion 1.2.

---

## 2. Comprension de los Datos

> Esta fase esta formalmente cerrada. Los ocho puntos estan completados y trazados a codigo y reportes reproducibles.

### 2.1 Recopilar datos iniciales

#### Origenes de datos

Los datos provienen de la competicion **Home Credit - Credit Risk Model Stability** publicada en Kaggle. Corresponden a datos reales anonimizados de una entidad financiera que otorga credito de consumo. No se utilizan fuentes externas adicionales.

El tipo de fuente son archivos historicos planos en formato Parquet que contienen registros de solicitudes de credito y su resultado de pago, enriquecidos con informacion de bureau de credito, historial de solicitudes previas y datos transaccionales del solicitante. No es una base de datos operacional en tiempo real.

#### Fuentes de datos

| Recurso | Ubicacion local | Tipo | Uso |
|---|---|---|---|
| Tabla base de entrenamiento | `data/parquet_files/train/train_base.parquet` | Parquet | Universo de casos, objetivo y particion temporal |
| Tablas de entrenamiento relacionadas | `data/parquet_files/train/` (31 archivos adicionales) | Parquet | Senal predictiva por fuente |
| Tablas de prueba | `data/parquet_files/test/` (36 archivos) | Parquet | Predicciones finales; verificacion de compatibilidad de esquema |
| Diccionario de variables | `data/feature_definitions.csv` | CSV | Interpretacion semantica de 465 campos |
| Formato de prediccion | `data/sample_submission.csv` | CSV | Estructura esperada de la salida final |

Toda lectura pasa por `src/dm_credit/io.py`. La suite de pruebas al cierre de esta fase cuenta con 81 tests verificados con `uv run --extra dev pytest -q`.

#### Informe de recopilacion de datos

Se inspeccionaron los metadatos de los 68 archivos Parquet (32 de train, 36 de test), organizados en 17 familias logicas. El inventario completo esta en `results/parquet_inventory.csv`.

| Familia logica | Archivos train / test | Filas en train | Columnas | Clave candidata | Granularidad |
|---|---|---|---|---|---|
| base | 1 / 1 | 1 526 659 | 5 / 4 | case_id | Una fila por caso |
| static_0 | 2 / 3 | 1 526 659 | 168 | case_id | Una fila por caso |
| static_cb_0 | 1 / 1 | 1 500 476 | 53 | case_id | Una fila por caso (98,28 % cobertura) |
| applprev_1 | 2 / 3 | 6 525 979 | 41 | case_id + num_group1 | Historial de solicitudes previas |
| applprev_2 | 1 / 1 | 14 075 487 | 6 | case_id + num_group1 + num_group2 | Registros anidados por solicitud previa |
| credit_bureau_a_1 | 4 / 5 | 15 940 537 | 79 | case_id + num_group1 | Bureau domestico agrupado |
| credit_bureau_a_2 | 11 / 12 | 188 298 452 | 19 | case_id + num_group1 + num_group2 | Bureau domestico anidado (pagos) |
| credit_bureau_b_1 | 1 / 1 | 85 791 | 45 | case_id + num_group1 | Bureau internacional |
| credit_bureau_b_2 | 1 / 1 | 1 286 755 | 6 | case_id + num_group1 + num_group2 | Pagos de bureau internacional |
| person_1 | 1 / 1 | 2 973 991 | 37 | case_id + num_group1 | Personas y contactos relacionados al caso |
| person_2 | 1 / 1 | 1 643 410 | 11 | case_id + num_group1 + num_group2 | Direcciones de las personas |
| debitcard_1 | 1 / 1 | 157 302 | 6 | case_id + num_group1 | Tarjetas de debito |
| deposit_1 | 1 / 1 | 145 086 | 5 | case_id + num_group1 | Depositos |
| other_1 | 1 / 1 | 51 109 | 7 | case_id + num_group1 | Otros productos financieros |
| tax_registry_a_1 | 1 / 1 | 3 275 770 | 5 | case_id + num_group1 | Registros fiscales |
| tax_registry_b_1 | 1 / 1 | 1 107 933 | 5 | case_id + num_group1 | Registros fiscales (segunda fuente) |
| tax_registry_c_1 | 1 / 1 | 3 343 800 | 5 | case_id + num_group1 | Registros fiscales (tercera fuente) |

Los 36 archivos de test contienen solo 10 filas cada uno, salvo test_tax_registry_c_1.parquet con 0 filas. Son suficientes para verificar compatibilidad de esquema pero no representan la distribucion real del test de la competicion. Los 36 archivos de test difieren en numero de shards fisicos de los 32 de train para cuatro familias; esto no constituye una diferencia de esquema logico.

---

### 2.2 Describir datos

#### Caracteristicas para describir los datos

**Cantidad y estructura de la tabla base:**

La integridad estructural de `train_base.parquet` fue verificada con `src/dm_credit/integrity.py` (resultados en `results/integrity_train_base/`, ejecucion del 2026-10-10).

| Control | Resultado |
|---|---|
| Dimensiones | 1 526 659 filas x 5 columnas (61,15 MB en memoria) |
| Unicidad de case_id | 0 nulos, 0 repetidos, 0 filas exactamente duplicadas |
| Consistencia temporal | date_decision (rango 2019-01-01 a 2020-10-05): 100 % parseable con formato YYYY-MM-DD; MONTH y WEEK_NUM derivados sin discrepancias en 0 filas |
| Integridad del objetivo | target solo toma valores en {0, 1}; 0 valores fuera del conjunto |

**Tipos de valores:**

| Columna | Tipo fisico | Tipo logico | Descripcion |
|---|---|---|---|
| case_id | Int64 | Identificador entero | Identificador unico del caso; nunca predictor |
| date_decision | String | Fecha YYYY-MM-DD | Fecha de la decision; referencia as-of principal |
| MONTH | Int64 | Entero yyyymm (ej. 201901) | Mes de decision; para control temporal |
| WEEK_NUM | Int64 | Entero 0-91 | Semana de observacion; eje de evaluacion temporal |
| target | Int64 | Binario {0,1} | Variable objetivo: 1 = incumplimiento, 0 = sin incumplimiento |

La tabla base no contiene predictores propios: toda la senal predictiva proviene de las 16 fuentes relacionadas.

**Variable objetivo: desbalance de clases** (fuente: `results/profile_train_base/numeric.csv`):

| Clase | Casos | Porcentaje |
|---|---|---|
| 0 (sin incumplimiento) | 1 478 665 | 96,86 % |
| 1 (incumplimiento) | 47 994 | 3,14 % |

Por cada caso positivo existen aproximadamente 31 negativos. La exactitud simple es una metrica enganosa en este contexto.

**Esquemas de codificacion observados en las fuentes:**

- Sufijo D: fechas en formato texto YYYY-MM-DD.
- Sufijo M: categorias de texto (valores enmascarados por la competicion, por ejemplo a55475b1).
- Sufijos A y P: valores numericos de importe o proporcion.
- Sufijo L: valores numericos ordinales o de conteo.
- Sufijo T: valores de tipo ano (entero de 4 digitos).
- Booleanos: codificados como texto true/false o como entero 0/1 segun la fuente.

**Catalogo de variables de todas las fuentes** (fuente: `results/variable_catalog.csv`):

| Indicador | Valor |
|---|---|
| Apariciones de columnas por familia | 503 |
| Nombres de variable distintos | 471 |
| Con descripcion oficial en feature_definitions.csv | 464 |
| Sin descripcion (campos estructurales) | 7: case_id, num_group1, num_group2, date_decision, MONTH, WEEK_NUM, target |

| Grupo funcional | Familias incluidas | Apariciones |
|---|---|---|
| Base y estaticas | base, static_0, static_cb_0 | 226 |
| Solicitudes previas | applprev_1, applprev_2 | 47 |
| Bureau de credito | credit_bureau_a_1/2, credit_bureau_b_1/2 | 149 |
| Personas | person_1, person_2 | 48 |
| Productos y registros fiscales | debitcard_1, deposit_1, other_1, tax_registry_a/b/c_1 | 33 |

Por categoria funcional: producto/contrato (118), importe o capacidad financiera (115), mora o incumplimiento historico (65), variables temporales (60), persona/demografia (39), claves estructurales (35), otras medidas numericas (31), empleo/ingresos (24), atributos categoricos (15), objetivo (1).

#### Consideraciones legales, regulatorias y eticas

**Uso del dataset:** Los datos se utilizan bajo los terminos de uso publico de la competicion Kaggle de Home Credit, exclusivamente con fines academicos y sin fines comerciales. Esta prohibido usar los datos para tomar decisiones crediticias reales o identificar a personas especificas.

**Privacidad y datos sensibles:** Los identificadores de personas estan enmascarados en el dataset. Los cuasi-identificadores (nombres, direcciones, codigos postales, distritos, telefonos y nombre de empleador) no se usaran como predictores directos. Los atributos sensibles gender_992L y sex_738L estan excluidos de los predictores para evitar discriminacion basada en genero (fuente: `development/03-preparacion-datos.md`, seccion G2).

**Sesgos potenciales:** Los datos provienen de una entidad que opera en mercados emergentes con una poblacion sub-bancarizada. Las fuentes con baja cobertura (por ejemplo, credit_bureau_b con 2,39 %) representan solo un subconjunto de solicitantes.

**Normativa de referencia:** La Resolucion SBS N 00053-2023 establece la obligacion de validar, monitorear y realizar pruebas retrospectivas de los modelos de riesgo de credito. Este proyecto aplica esas buenas practicas como referencia metodologica.

**Exactitud y gobernanza:** La definicion oficial de cada variable en `data/feature_definitions.csv` es la unica fuente semantica disponible. No se dispone de una fuente externa para validar la exactitud individual de los registros.

#### Informe de descripcion de datos

La tabla base de entrenamiento tiene 1 526 659 casos unicos con una tasa de incumplimiento del 3,14 %. El dataset esta compuesto por 17 familias logicas distribuidas en 68 archivos Parquet. La familia mas grande es credit_bureau_a_2 con 188 298 452 filas, procesada en streaming. Las 503 apariciones de columnas incluyen 464 predictores con descripcion oficial. Los tipos abarcan enteros, decimales, texto (fechas y categorias enmascaradas) y booleanos. Se identificaron 7 campos estructurales excluidos de los predictores. La unidad de analisis es el caso de credito identificado por case_id.

---

### 2.3 Explorar datos

#### Datos continuos

Las variables continuas provienen principalmente de static_0, credit_bureau_a_1, credit_bureau_b_1, applprev_1 y person_1. Estadisticos obtenidos de `results/profile_train_*/numeric.csv`.

**Importes y saldos (static_0):**

| Variable | Min | Media | Max | Negativos |
|---|---|---|---|---|
| credamount_770A (monto del credito) | 2 000 | 49 870 | 950 000 | 0 |
| annuity_780A (cuota) | 80,8 | 4 039 | 106 007 | 0 |
| currdebt_22A (deuda actual) | 0 | 19 682 | 1 210 629 | 0 |
| avgoutstandbalancel6m_4187114A (saldo promedio 6m) | -7 588 199 | 45 985 | 1 131 136 | 13 169 |
| maxoutstandbalancel12m_4187113A (saldo maximo 12m) | -7 588 199 | 71 364 | 1 235 313 | 4 148 |
| totalsettled_863A (total liquidado) | 0 | 92 238 | 48 035 036 | 0 |

Las distribuciones de importe son fuertemente asimetricas: la media es sistematicamente mayor que la mediana, lo que indica colas largas hacia valores altos. Los valores negativos en saldos promedios y maximos pueden representar sobregiros o movimientos contables; se conservan hasta validar su semantica.

**Variables de mora DPD (static_0):**

| Variable | Min | Media | Max | Negativos |
|---|---|---|---|---|
| avgdbddpdlast24m_3658932P (DPD promedio 24m) | -1 220 | 30,1 | 4 750 | 734 651 |
| avgdbddpdlast3m_4187120P (DPD promedio 3m) | -908 | 29,5 | 4 750 | 445 711 |
| maxdbddpdlast1m_3658939P (DPD maximo 1m) | -1 237 | 45,0 | 4 804 | 360 717 |
| maxdpdlast12m_727P (DPD maximo 12m) | 0 | 16,6 | 4 629 | 0 |
| maxdpdlast24m_143P (DPD maximo 24m) | 0 | 26,4 | 4 629 | 0 |

Los valores negativos en DPD son frecuentes (734 651 negativos en avgdbddpdlast24m), lo que sugiere que representan dias antes del vencimiento, no mora. Los maximos de 4 804 dias requieren validacion de dominio.

**Bureau de credito B - mora (credit_bureau_b_1):**

| Variable | Min | Mediana | Max | Negativos |
|---|---|---|---|---|
| dpd_550P | 0 | 0 | 207 823 776 | 0 |
| dpdmax_851P | 0 | 0 | 185 124 192 | 0 |
| interesteffectiverate_369L | -1,1 | 23,55 | 73 000 | 1 |

El maximo de dpdmax_851P (185 124 192 dias) no es interpretable como mora real: es probablemente un codigo sentinel. La tasa de interes negativa (-1,1) tampoco tiene interpretacion economica directa; ambos requieren validacion de dominio.

**Bureau de credito A - pagos (credit_bureau_a_2, perfil de shards):**

| Variable | Min | Max | Negativos |
|---|---|---|---|
| pmts_dpd_303P (DPD de pagos) | -30 | 657 458 | 12 020 |
| pmts_overdue_1152A (monto vencido) | 0 | 593 465 000 | 0 |
| collater_valueofguarantee_1124L (garantia) | 0 | 98 780 000 000 | 0 |
| pmts_year_507T (ano del pago) | 2000 | 2028 | 0 |

Las garantias alcanzan hasta 98 780 millones. Los anos de pago hasta 2028 representan fechas contractuales conocidas al momento de la solicitud.

**Depositos (deposit_1):**

| Variable | Min | Mediana | Max | Negativos |
|---|---|---|---|---|
| amount_416A (monto) | -40 000 | 223,66 | 12 213 286 | 10 |

La mediana de 223,66 frente a una media de 8 422 confirma una distribucion extremadamente asimetrica.

**Tarjetas de debito (debitcard_1):**

| Variable | Min | Mediana | Max | Negativos |
|---|---|---|---|---|
| last180dayaveragebalance_704A | -308,79 | 0 | 67 777,77 | 1 |
| last180dayturnover_1134A | -187 780 | 30 000 | 1 161 820 | 9 |
| last30dayturnover_651A | -477,51 | 0 | 390 000 | 1 |

La mediana de 0 en el saldo promedio de 180 dias indica que la mayoria de los casos con cobertura no tienen actividad reciente.

**Solicitudes previas (applprev_1):**

| Variable | Min | Media | Max | Negativos |
|---|---|---|---|---|
| credacc_actualbalance_314A (saldo actual) | -134 008 | 18 270 | 2 540 730 | 891 |
| credacc_maxhisbal_375A (saldo historico maximo) | -290 265 | -2 620 | 7 988 199 | 54 209 |
| credamount_590A (monto del credito) | 0 | 40 409 | 1 000 000 | 0 |
| mainoccupationinc_437A (ingreso principal) | 0 | 41 238 | 199 600 | 0 |
| revolvingaccount_394A | 540 342 340 | 750 542 330 | 800 608 700 | 0 |

revolvingaccount_394A tiene valores en el rango de 540 a 800 millones para todos los registros no nulos. Este rango inusual sugiere que puede ser un identificador enmascarado o una codificacion especial, no un importe monetario.

#### Datos categoricos

Las variables categoricas del dataset estan enmascaradas por la competicion: sus valores son codigos hash. Estadisticos de `results/profile_train_*/categorical.csv`.

**Solicitudes previas (applprev_1):**

| Variable | Valores unicos | Categoria dominante | % dominante |
|---|---|---|---|
| profession_152M (profesion) | 11 508 | a55475b1 | 98,88 % |
| cancelreason_3545846M (motivo cancelacion) | 76 | a55475b1 | 68,24 % |
| rejectreason_755M (motivo rechazo) | 18 | a55475b1 | 71,46 % |
| credtype_587L (tipo de credito) | 3 | COL | 51,95 % |
| familystate_726L (estado familiar) | 5 | MARRIED | 72,28 % |
| inittransactioncode_279L (canal) | 3 | POS | 55,37 % |
| status_219L (estado del credito) | 11 | D | 40,88 % |
| isbidproduct_390L | 2 | False | 94,43 % |

profession_152M es casi constante (98,88 %). familystate_726L y status_219L tienen distribucion mas variada.

**Personas (person_1):**

| Variable | Valores unicos | Categoria dominante | % dominante |
|---|---|---|---|
| incometype_1044T (tipo de ingreso) | 9 | PRIVATE_SECTOR_EMPLOYEE | 32,13 % |
| empl_industry_691L (industria del empleador) | 24 | OTHER | 74,07 % |
| empl_employedtotal_800L (antiguedad laboral) | 3 | MORE_FIVE | 70,28 % |
| sex_738L (sexo) | 2 | F | 62,41 % |
| contaddr_matchlist_1032L | 1 | False | 100,00 % |
| remitter_829L | 1 | False | 100,00 % |
| role_993L | 1 | FULL | 100,00 % |

contaddr_matchlist_1032L, remitter_829L y role_993L son constantes: no aportan informacion. incometype_1044T tiene 9 categorias y es candidato a predictor relevante.

**Bureau de credito A (credit_bureau_a_1):**

| Variable | Valores unicos | Categoria dominante | % dominante |
|---|---|---|---|
| financialinstitution_382M | 390 | a55475b1 | 57,64 % |
| purposeofcred_874M (proposito del credito) | 24 | a55475b1 | 57,64 % |
| classificationofcontr_13M | 11 | a55475b1 | 83,33 % |
| description_351M | 12 | a55475b1 | 99,61 % |

description_351M es casi constante (99,61 %): sin informacion util. financialinstitution_382M tiene 390 valores distintos y contiene "Home Credit" como segunda categoria mas frecuente (2 125 519 registros).

#### Datos ordinales

Las principales variables ordinales del dataset son:

- WEEK_NUM (0 a 91): semana de observacion; ordinal temporal que define el eje de evaluacion de estabilidad.
- num_group1 por familia: en applprev_1 va de 0 a 19 (media 4,19); en credit_bureau_a_1 va de 0 a 528 (media 6,32), reflejando mayor profundidad historica del bureau.
- num_group2 (0 a 100 en credit_bureau_a_2): sub-indice de anidamiento dentro del registro padre.
- Conteos en static_0: numinstlsallpaid_934L (0 a 385, media 27,3), numinstlswithdpd10_728L (0 a 330, media 3,6), numincomingpmts_3546848L (0 a 843, media 32,4).
- Contratos activos en static_0: numactivecreds_622L (0 a 7, media 0,47), numactiverelcontr_750L (0 a 8, media 0,24).
- pmtnum_254L en static_0 (numero de cuotas pactadas): 3 a 60, media 17,07.
- persontype_1072L en person_1: 1 a 5, representa el tipo de persona en la relacion con el caso.

Fuente: `results/profile_train_static_0/numeric.csv`, `results/profile_train_applprev_1/numeric.csv`, `results/profile_train_person_1/numeric.csv`.

#### Analisis univariado

**Variable objetivo target** (fuente: `results/profile_train_base/numeric.csv`):

| Estadistico | Valor |
|---|---|
| Min | 0 |
| Max | 1 |
| Media | 0,0314 |
| Mediana | 0 |
| Desviacion estandar | 0,1745 |
| Casos clase 0 | 1 478 665 (96,86 %) |
| Casos clase 1 | 47 994 (3,14 %) |
| Nulos | 0 |

**Variable temporal WEEK_NUM** (fuente: `results/profile_train_base/numeric.csv`):

| Estadistico | Valor |
|---|---|
| Min | 0 |
| Max | 91 |
| Media | 40,77 |
| Mediana | 40 |
| Desviacion estandar | 23,80 |
| P25 | 23 |
| P75 | 55 |
| P99 | 90 |
| Casos en WEEK_NUM = 0 | 16 735 |
| Semanas observadas | 92 (ninguna ausente) |
| Semanas monoclase | 0 |

La distribucion de WEEK_NUM es aproximadamente uniforme (media 40,77, mediana 40), lo que indica que las solicitudes se distribuyen de forma razonablemente pareja a lo largo del periodo de observacion.

**Tasa de incumplimiento semanal** (fuente: `results/target_analysis/weekly.csv`):

| Estadistico | Valor |
|---|---|
| Mediana semanal | 2,79 % |
| Minimo (semana 86) | 1,77 % |
| Maximo (semana 62) | 5,21 % |
| Minimo de positivos en una semana | 21 (semana 68) |
| Mediana de casos por semana | 17 088,5 |
| Minimo de casos (semana 68) | 825 |
| Maximo de casos (semana 51) | 35 920 |

**Cobertura de familias por caso** (fuente: `results/integration_analysis/family_cardinality.csv`):

| Familia | Cobertura | Mediana filas/caso | Maximo filas/caso |
|---|---|---|---|
| base, static_0, person_1 | 100,00 % | 1 / 1 / 1 | 1 / 1 / 10 |
| static_cb_0 | 98,28 % | 1 | 1 |
| person_2 | 94,00 % | 1 | 34 |
| credit_bureau_a_1 | 90,80 % | 10 | 529 |
| credit_bureau_a_2 | 90,74 % | 96 | 9 404 |
| applprev_1 y applprev_2 | 80,01 % | 4 / 8 | 20 / 79 |
| tax_registry_c_1 | 31,59 % | 6 | 121 |
| tax_registry_a_1 | 30,00 % | 6 | 99 |
| tax_registry_b_1 | 9,87 % | 6 | 101 |
| debitcard_1 | 7,32 % | 1 | 66 |
| deposit_1 | 6,89 % | 1 | 65 |
| other_1 | 3,35 % | 1 | 1 |
| credit_bureau_b_1 y b_2 | 2,39 % | 2 / 36 | 21 / 301 |

#### Analisis multivariado

En esta etapa no es posible realizar un analisis multivariado completo entre predictores y objetivo porque la tabla base no contiene predictores propios y las fuentes secundarias aun no han sido agregadas a nivel de caso. Las correlaciones entre caracteristicas derivadas y el objetivo se calcularan en la fase de preparacion de datos.

Lo que si es posible verificar con los datos actuales:

- **Consistencia temporal interna en train_base:** MONTH y la derivacion de date_decision coinciden en el 100 % de los casos. WEEK_NUM y su derivacion a partir de date_decision tambien coinciden en el 100 %. Confirmado por `src/dm_credit/integrity.py`.
- **Tasa de incumplimiento por bloques de semanas** (fuente: `results/target_analysis/weekly.csv`): sube de ~2,68 % en las semanas 0-9 hasta ~4,28 % en las semanas 60-69, luego desciende a ~2,03 % en las semanas 80-89. Este patron no es monotono y confirma la presencia de al menos dos regimenes distintos en el periodo de observacion.
- **Cobertura diferencial entre train y OOT:** tax_registry_a_1 pasa de 34,2 % en train a 1,1 % en OOT; tax_registry_c_1 pasa de 30,3 % a 0,0 %; tax_registry_b_1 pasa de 1,8 % a 64,7 %. Esto introduce covariacion entre los indicadores de cobertura y el periodo temporal (fuente: `results/preparation/family_results.csv`).
- **Cardinalidad interna de bureau:** en credit_bureau_a_1, el numero de registros por caso va de 1 a 529 (media 6,32), indicando gran variabilidad en la profundidad del historial crediticio entre solicitantes.

#### Analisis dimensional

El dataset tiene tres dimensiones principales:

**Dimension de caso (case_id):** 1 526 659 casos unicos en entrenamiento. Es la dimension de modelado: el modelo predice una probabilidad para cada case_id. Todas las fuentes secundarias deben reducirse a una fila por caso antes del modelado. La suma de filas de todas las fuentes supera los 260 millones de registros para 1,5 millones de casos.

**Dimension temporal (WEEK_NUM):** 92 semanas (0-91) en entrenamiento, con media 40,77 y mediana 40. Las 18 semanas del OOT (74-91) concentran 194 658 casos (12,75 % del total). El drift de la tasa de incumplimiento (rango 1,77 % a 5,21 %) es el fenomeno central del proyecto.

**Dimension de fuente (17 familias logicas):** las fuentes difieren en cobertura (2,39 % a 100 %), granularidad (1 a 9 404 filas por caso), profundidad de anidamiento (0, 1 o 2) y disponibilidad temporal. La dimension de fuente determina la estrategia de agregacion: cada familia se procesa de forma independiente antes del join con la base.

#### Informe de exploracion de datos

El analisis exploratorio identifica tres hallazgos estructurales que condicionan el modelado:

1. **Inestabilidad temporal de la variable objetivo:** la tasa de incumplimiento varia entre 1,77 % (semana 86) y 5,21 % (semana 62) a lo largo de las 92 semanas. La particion OOT (semanas 74-91, tasa 2,19 %) representa un periodo de menor prevalencia que el entrenamiento (3,28 %), simulando un cambio de regimen real.

2. **Heterogeneidad de cobertura y calidad entre fuentes:** las 17 familias tienen cobertura muy dispar (2,39 % a 100 %) y patrones de ausencia radicalmente distintos (0 % a 99,97 % de nulos por columna). Los registros fiscales muestran un cambio drastico de cobertura entre train y OOT, lo cual puede penalizar la estabilidad del modelo.

3. **Riesgo de multiplicacion en joins sin agregar:** las fuentes con mayor profundidad pueden multiplicar las filas de la base entre 1,82x y 123,43x si se unen directamente. credit_bureau_a_2 alcanzaria 188 millones de filas sin agregar.

Los predictores candidatos a explorar en la fase de preparacion son: numero y resultado de solicitudes previas, atributos de la persona principal (edad derivada de birth_259D, tipo de ingreso incometype_1044T, antiguedad laboral empl_employedtotal_800L), indicadores de cobertura en bureau, y variables de mora del historial de pagos verificadas como disponibles a la fecha de decision.

---

### 2.4 Verificar la calidad de los datos

#### Exactitud

La exactitud mide el grado en que los datos representan correctamente entidades reales. En este proyecto no es verificable directamente porque no se dispone de una fuente externa de referencia. La definicion oficial de cada variable en `data/feature_definitions.csv` es la unica fuente semantica disponible.

Hallazgos:
- date_decision esta almacenada como texto aunque el 100 % de sus valores es parseable como fecha YYYY-MM-DD. No se detectaron fechas invalidas.
- Valores extremos en variables de mora (maximos de 185 millones en dpdmax_851P) y en garantias (hasta 98 780 millones en credit_bureau_a_2) no pueden confirmarse como errores sin validacion de dominio.
- Las categorias enmascaradas (codigos hash) no permiten validar su correctitud semantica directamente.
- Fechas en applprev_1 posteriores a date_decision en ~5,4 % de las filas de dtlastpmtallstes_3545839D son inexactas como datos de entrenamiento y se excluyen en la fase de preparacion (117 977 filas en train, 22 063 en OOT).

#### Completitud

La completitud mide si todos los datos estan presentes al nivel esperado. El patron de ausencia es estructuralmente heterogeneo entre fuentes (fuente: `results/quality_catalog.csv`).

| Fuente | Columnas con >= 50 % de nulos | Columna con mayor ausencia | % nulos |
|---|---|---|---|
| static_0 | 49 de 168 | clientscnt_136L | 99,97 % |
| credit_bureau_a_1 | 65 de 79 | prolongationcount_599L | 99,76 % |
| person_1 | 15 de 37 | housingtype_772L | 99,67 % |
| credit_bureau_a_2 | 8 de 19 | collater_valueofguarantee_1124L | 98,54 % |
| static_cb_0 | 36 de 53 | varios campos | >50 % |
| credit_bureau_b_1 | varios | periodicityofpmts_997L | 97,64 % |
| debitcard_1 | 2 columnas clave | saldos de 180 dias | 92-93 % |
| deposit_1 | 1 | contractenddate_991D | 54,92 % |
| tax_registry_a/b/c_1 | 0 | - | 0 % |

La ausencia puede ser senal predictiva. Los indicadores de ausencia se preservan en la fase de preparacion.

#### Consistencia

- **Consistencia interna en train_base:** perfecta. date_decision, MONTH y WEEK_NUM son coherentes en el 100 % de las filas (0 discrepancias).
- **Consistencia de tipos entre shards:** los fragmentos fisicos de cada familia logica mantienen esquemas identicos entre si.
- **Inconsistencia potencial train/test:** cuatro columnas de fecha (validfrom_1069D, assignmentdate_238D, birthdate_574D, responsedate_1012D) son String en train y Float64 en test. No se puede confirmar incompatibilidad real porque los 10 valores de test disponibles son nulos.
- **Inconsistencia de codificacion:** los booleanos estan codificados como texto true/false en algunas fuentes y como entero 0/1 en otras. La coercion explicita al tipo logico resuelve esto.

#### Integridad

Verificado con `src/dm_credit/streaming_key_integrity.py` (resultados en `results/key_integrity_all_families.csv`):

- case_id no tiene nulos en ninguna de las 17 familias.
- Las claves candidatas tienen cero grupos duplicados y cero filas duplicadas en las 17 familias.
- Cero filas huerfanas: todo caso de una fuente secundaria tiene un caso correspondiente en train_base.
- Dos columnas globalmente constantes: num_group1 de other_1 (siempre 0) y deferredmnthsnum_166L de static_0.

#### Razonabilidad

- **Tasa de incumplimiento (3,14 %):** razonable para credito de consumo en mercados emergentes con clientes de historial limitado.
- **Cobertura de bureau (90,8 % en credit_bureau_a_1):** alta para una entidad que utiliza bureau como parte del proceso de evaluacion.
- **Numero de solicitudes previas (mediana 4 por caso, maximo 20 en applprev_1):** razonable; indica que muchos solicitantes ya tienen relacion previa con la entidad.
- **Valores extremos en mora:** un maximo de 185 millones de dias no es razonable como mora real (equivale a mas de 500 000 anos). Probablemente es un codigo sentinel.
- **Cobertura diferencial de registros fiscales:** tax_registry_a_1 pasa de 34,2 % en train a 1,1 % en OOT. Esta variacion sugiere que las tres fuentes de registros fiscales se reemplazan entre si a lo largo del tiempo (fuente: `results/preparation/family_results.csv`).

#### Oportunidad

La oportunidad es el criterio mas critico del proyecto, directamente relacionado con la prevencion de fuga de informacion (fuente: `results/integration_analysis/temporal_leakage_review.csv`, `results/preparation/temporal_leakage_report.csv`):

- date_decision es la referencia as-of: solo se acepta informacion disponible en esa fecha o antes.
- applprev_1: ~5,4 % de las filas de dtlastpmtallstes_3545839D tienen fechas posteriores a date_decision. Esas filas se excluyen en la fase de preparacion (117 977 filas en train, 22 063 en OOT).
- Bureau de credito: las filas con ancla temporal posterior a date_decision tambien se excluyen (1 583 190 filas en train para credit_bureau_a).
- Estados, mora actual, aprobaciones y rechazos: pudieron actualizarse despues de la decision; se excluyen preventivamente hasta demostrar una instantanea valida.
- Registros fiscales: se filtran por presencia a date_decision. El 100 % de las filas de tax_registry_a_1 tenia fecha posterior; la fuente queda sin cobertura util.

#### Unicidad / Deduplicacion

- train_base: case_id es unico en las 1 526 659 filas (0 duplicados exactos, 0 claves repetidas).
- Fuentes de profundidad 0 (static_0, static_cb_0): case_id es unico (una fila por caso).
- Fuentes de profundidad 1: la clave compuesta case_id + num_group1 es unica en las 12 familias de profundidad 1.
- Fuentes de profundidad 2: la clave compuesta case_id + num_group1 + num_group2 es unica en las 5 familias de profundidad 2.
- La unicidad de la clave compuesta no implica que case_id sea unico en esas fuentes: las fuentes de profundidad 1 y 2 tienen multiples registros por caso y deben agregarse antes de unirse a la tabla base.

#### Validez

| Hallazgo | Variable(s) | Estado |
|---|---|---|
| target fuera de {0,1} | target | 0 valores invalidos; valido |
| Fechas no parseables | date_decision | 0 valores invalidos; valido |
| Tasas negativas | interesteffectiverate_369L en credit_bureau_b_1 | Requiere validacion de dominio |
| Categorias fuera del vocabulario de train | 3 columnas en test | Manejables con __UNKNOWN__ |
| Valores Float64 en columnas de fecha en test | 4 columnas | Pendiente de validacion con test completo |
| Booleanos como texto no estandar | varias fuentes | Coercibles; no invalidos |
| Infinitos | ninguna columna | 0 valores infinitos detectados |

La regla del proyecto: un valor no convertible al tipo logico produce un error explicito en el pipeline; NaN se trata como ausencia; nunca se convierte un valor invalido en nulo en silencio.

---

## 3. Preparacion de Datos

> Capitulo cerrado (G4, 13/10/2026). Todas las cifras proceden de `results/preparation/` generados por la reconstruccion desde cero del cierre con el comando de la seccion 3.5. Fuente principal: `development/03-preparacion-datos.md`.

La fase produce una **matriz de modelado con una fila por caso** para train, OOT y test, sin fuga temporal y con el mismo esquema en los tres splits.

| Resultado | Valor | Evidencia |
|---|---|---|
| Casos | train 1 332 001 - OOT 194 658 - test 10 | `results/preparation/F3_final_checks.csv` |
| Caracteristicas | 124: 94 predictores y 30 de cobertura | `results/preparation/C8_feature_catalog.csv` |
| Familias integradas | 12 (con tablas hijas donde procede) | `results/preparation/family_results.csv` |
| Validacion final F3 | 29/29 correctos, incluidos tres cotejos published_matrix | `results/preparation/integration_checks.csv` |
| Vista lineal | 234 predictores, sin nulos | `results/preparation/E6_views.csv` |
| Vista de arboles | 114 predictores, con nulos nativos | `results/preparation/E6_views.csv` |

### 3.1 Seleccionar datos

#### Inclusion o exclusion de datos

La base de cada split define su universo de casos. Las fuentes con varias filas por caso se agregan a una fila por case_id antes de unirlas. Una columna entra como predictor solo si la decision G2 la aprueba:

1. No es clave, cuasi-identificador ni atributo sensible (sexo y genero quedan fuera).
2. Su disponibilidad al decidir esta justificada por una regla temporal, no por su nombre.
3. No es una fecha con aspecto de calendario o proyeccion.
4. Tiene al menos 5 % de cobertura en su tabla fuente.
5. Pasa un filtro empirico de fuga calculado solo con train para las caracteristicas numericas o booleanas: Gini univariado y del indicador de ausencia no mayores a 0,5 en valor absoluto, sin cambio de signo entre tres bloques de semanas. Las ocho de texto se aprobaron por regla, sin medicion.

**Solo solicitudes previas y personas aportan predictores** en este primer corte. El resto de familias aporta indicadores de cobertura (n_records y source_present), porque G2 no encontro evidencia temporal suficiente para sus columnas.

| Familia | Aporta | Predictores | Regla temporal | Cobertura train / OOT |
|---|---|---|---|---|
| applprev (applprev_1, applprev_2) | importes, plazos, categorias, indicadores y recencias | 61 | fila valida si creationdate_885D <= date_decision; hijo solo con padre valido | 78,2 % / 85,2 % |
| person (person_1, person_2) | atributos de la persona principal (num_group1 = 0), edad, antiguedad laboral y relacion de contactos | 33 | atributos de la solicitud actual; fechas internas <= date_decision | 100 % / 100 % |
| static_0 | cobertura | 0 | - | 100 % / 100 % |
| static_cb_0 | cobertura | 0 | - | 98,1 % / 99,9 % |
| credit_bureau_a (credit_bureau_a_1) | cobertura | 0 | ancla <= date_decision; credit_bureau_a_2 omitido sin ancla aprobada | 71,0 % / 87,6 % |
| credit_bureau_b (credit_bureau_b_1, b_2) | cobertura | 0 | ancla <= date_decision; hijo con padre valido | 0,4 % / 0,9 % |
| debitcard_1, deposit_1 | cobertura | 0 | openingdate <= date_decision; fecha nula descartada | 6,7 % / 8,1 % |
| tax_registry_b_1, tax_registry_c_1 | cobertura | 0 | deductiondate / processingdate <= date_decision | b 1,8/64,7 - c 30,3/0 % |
| tax_registry_a_1 | nada: constante en train, descartada | 0 | 100 % de filas con fecha posterior a la decision | 0 % / 0 % |
| other_1 | nada: constante en train, descartada | 0 | sin fecha: todas las filas se descartan | 0 % / 0 % |

Las 94 columnas predictoras salen de 35 columnas fuente aprobadas: 19 de applprev_1 y 16 de person_1 (fuente: `results/preparation/G2_applications_people_columns.csv`).

**Resumen de exclusiones por bloque:**

| Bloque | Columnas evaluadas | Excluidas | Bloqueadas | Aprobadas |
|---|---|---|---|---|
| B3 - profundidad 0 | 226 | 31 | 195 | 0 |
| B4 - solicitudes y personas | 95 | 34 | 21 | 35 |
| B5 - bureau | 149 | 23 | 126 | 0 |
| B6 - productos y registros | 33 | 15 | 18 | 0 |

Casos destacados de exclusion:

| Elemento | Motivo |
|---|---|
| Claves, indices y fechas de particion | Control, nunca predictores |
| Direcciones, distritos, codigos postales, empleadores, nombres fiscales | Cuasi-identificadores |
| gender_992L, sex_738L | Atributos sensibles |
| riskassesment_302T | Fuga del objetivo: describe una probabilidad de incumplimiento futura |
| dtlastpmtallstes_3545839D | 5,4 % de filas con fecha posterior a la decision |
| Estados, mora, pagos, deuda y motivos de rechazo | Pudieron actualizarse despues de la decision |
| credit_bureau_a_2 | Sin ancla temporal aprobada |

**Particion temporal:**

| Split | Semanas | Casos | Positivos | Tasa |
|---|---|---|---|---|
| train | WEEK_NUM < 74 | 1 332 001 | 43 734 | 3,28 % |
| OOT | WEEK_NUM >= 74 | 194 658 | 4 260 | 2,19 % |
| test | muestra local | 10 | - | - |

Train y OOT suman 1 526 659 casos, sin solapamiento (control train_oot_partition en `F3_final_checks.csv`).

### 3.2 Datos limpios

#### Valores faltantes

Las numericas se imputan con la mediana de train y llevan un indicador de ausencia llamado nombredelvariable__missing. En las categoricas, el nulo pasa a la categoria __missing__. La ausencia puede ser senal; los parametros solo se aprenden con train.

**Patron de datos faltantes:**

El patron de ausencia es heterogeneo entre fuentes. Las fuentes con mayor ausencia son las de bureau de credito (65 de 79 columnas de credit_bureau_a_1 con mas del 50 % de nulos) y las de datos estaticos (49 de 168 en static_0). Los registros fiscales y other_1 no presentan nulos.

El mecanismo de datos faltantes observado es principalmente MNAR (missing not at random): la ausencia en bureau de credito esta correlacionada con el historial del solicitante, no es aleatoria. Por ejemplo, campos como collater_valueofguarantee_1124L estan ausentes en el 98,54 % de los registros porque la mayoria de los creditos no tienen garantia.

#### Valores atipicos

Un valor fuera del rango de train genera un aviso, sin recorte. Ninguna columna tiene recorte aprobado. Un rango nuevo en un periodo futuro es esperable y no es un error. Los maximos extremos identificados (como dpdmax_851P = 185 124 192) se conservan hasta validar su semantica.

#### Datos duplicados

Verificado con `src/dm_credit/streaming_key_integrity.py`: cero filas duplicadas en las 17 familias. La clave compuesta (case_id, o case_id + num_group1, o case_id + num_group1 + num_group2 segun la familia) es unica en todos los casos.

#### Errores o inconsistencias

- Booleanos codificados de forma inconsistente entre fuentes (texto vs entero): resuelto mediante coercion explicita al tipo logico.
- date_decision almacenada como String aunque semanticamente es una fecha: se convierte cuando se requieren operaciones de calendario.
- Cuatro columnas de fecha son String en train y Float64 en test: pendiente de validacion con el test completo; por ahora son advertencias, no errores confirmados.

#### Datos faltantes o incompletos

Reglas aplicadas (fuente: `development/03-preparacion-datos.md`):

| Problema | Regla aplicada | Justificacion |
|---|---|---|
| Valores faltantes numericos | Imputacion con mediana de train + indicador __missing | La ausencia puede ser senal; parametros solo de train |
| Valores faltantes categoricos | Nulo pasa a __missing__ | Tratamiento uniforme de ausencia |
| Valores atipicos | Aviso sin recorte | Un rango nuevo en OOT es esperable |
| Tipos y codificacion | Coercion estricta al tipo logico; valor no convertible = error; NaN = ausencia | Nunca se convierte un valor invalido en nulo en silencio |
| Categorias raras o nuevas | Frecuencia < 1 % en train o no vistas pasan a __UNKNOWN__; vocabulario congelado en train | OOT y test no amplian el vocabulario |
| Eventos posteriores a la decision | La fila se excluye o el campo se anula; filas sin ancla o con padre invalido tambien se descartan | Prevencion de fuga temporal |
| Claves invalidas | Clave nula o repetida = error | Garantiza una fila por caso |

Modulos: `src/dm_credit/preparation/coercion.py` (D4), `outliers.py` (D5), `preprocessing.py` (D1 a D3).

### 3.3 Datos de construccion

#### Variables agregadas y derivadas

Los nombres siguen la gramatica congelada en G1: `<fuente>__<variable>__<agregacion>__<ventana>`, con ventana all (todo el historial hasta la decision).

| Tipo | Ejemplo | Familias |
|---|---|---|
| Conteo, minimo, maximo, media y suma | applprev_1__credamount_590a__mean__all | applprev |
| Proporcion de una categoria aprobada | applprev_1__credtype_587l__share_cal__all | applprev, person |
| Numero de categorias distintas | applprev_1__profession_152m__n_distinct__all | applprev |
| Recencia o antiguedad en dias | applprev_1__creationdate_885d__days_since_last__all | applprev, person |
| Atributo de la persona principal | person_1__incometype_1044t__main__all | person |
| Edad en anos cumplidos a la decision | person_1__birth_259d__age_years__all | person |
| Cobertura | fuente__n_records, fuente__source_present | todas |

El catalogo completo de las 124 caracteristicas esta en `results/preparation/C8_feature_catalog.csv`.

#### Prevencion de fuga de informacion

1. **Filtro temporal antes de agregar.** Cada fila se compara con date_decision de su caso; un hijo solo se usa si su padre es valido.
2. **Sin aprendizaje fuera de train.** El preprocesador exige WEEK_NUM < 74 al ajustarse. Transformar OOT y test no modifica parametros (control views_fit_on_train_only).
3. **Filtro empirico G2.** Las 94 caracteristicas numericas/booleanas tienen un Gini univariado maximo de 0,184 en valor absoluto; ninguna fue marcada (fuente: `results/preparation/G2_applications_people_screen.csv`). No garantiza ausencia de fugas moderadas.
4. **Exclusiones registradas.** Filas descartadas por temporalidad:

| Regla | Familia | Train | OOT |
|---|---|---|---|
| Solicitud creada despues de la decision | applprev | 117 977 | 22 063 |
| Solicitud sin fecha de creacion | applprev | 65 | 1 |
| Hijo con padre invalido | applprev | 178 626 | 34 586 |
| Fecha de ultimo pago posterior, anulada | applprev | 35 089 | 6 595 |
| Inicio de empleo posterior, anulado | person | 312 | 4 |
| Fila de bureau A con ancla posterior | credit_bureau_a | 1 583 190 | 249 695 |
| Fila de bureau A sin ancla | credit_bureau_a | 7 528 249 | 832 131 |
| Hijo de bureau B con padre invalido | credit_bureau_b | 913 190 | 245 050 |
| Registro fiscal A con fecha posterior | tax_registry_a_1 | 3 260 956 | 14 814 |
| Registro fiscal C con fecha posterior | tax_registry_c_1 | 726 355 | 0 |
| Registro fiscal B con fecha posterior | tax_registry_b_1 | 14 236 | 66 436 |
| Fila sin fecha de disponibilidad | other_1 | 25 859 | 25 250 |
| Tarjeta de debito con fecha nula | debitcard_1 | 10 823 | 1 888 |

Fuente: `results/preparation/temporal_leakage_report.csv`, `results/preparation/F7_temporal_leakage_by_family.csv`.

#### Hallazgo sobre fuga en productos y registros (resuelto)

En la primera version (products-v1), los indicadores de productos y registros contaban todas las filas sin mirar su fecha. Al medirlo, el 100 % de las filas de tax_registry_a_1 tenia fecha posterior a la decision (mediana +14 dias). Desde products-v2, la cobertura solo cuenta filas disponibles a date_decision.

Una ablacion con el mismo modelo midio Gini Stability en OOT (fuente: `results/preparation/F7_coverage_ablation.csv`):

| Indicadores de productos y registros | Gradient boosting | Regresion logistica |
|---|---|---|
| Sin filtro (products-v1, con fuga) | 0,4914 | 0,4503 |
| Filtrados por fecha (products-v2, final) | 0,4913 | 0,4523 |
| Ninguno | 0,4861 | 0,4423 |

La ganancia se conserva con el filtro y es legitima. Lo que cambio fue el ajuste en train (Gini de 0,595 a 0,590), que antes aprovechaba la fuga.

### 3.4 Integrar datos

#### Proceso de integracion

Cada familia se agrega por separado a una fila por case_id y luego se une a la base con un left join 1:1. Nunca se unen dos fuentes 1:N entre si.

| Familia | Relacion con la base | Filas fuente en train | Resultado tras agregar |
|---|---|---|---|
| applprev | 1:N (profundidades 1 y 2) | 6 525 979 + 14 075 487 | 1 fila por caso |
| person | 1:N (profundidades 1 y 2) | 2 973 991 + 1 643 410 | 1 fila por caso |
| static_0, static_cb_0 | 1:1 y 1:0..1 | 1 526 659 y 1 500 476 | 1 fila por caso |
| credit_bureau_a | 1:N | 15 940 537 (a_1) | 1 fila por caso |
| credit_bureau_b | 1:N (profundidades 1 y 2) | 85 791 + 1 286 755 | 1 fila por caso |
| productos y registros | 1:N y 1:0..1 | 51 109 a 3 343 800 | 1 fila por caso |

#### Validacion de cardinalidad (29/29 controles correctos)

Controles de la ejecucion F3 archivada en `results/preparation/F3_final_checks.csv`:

- Las 12 familias de cada split usan la misma base (una unica huella por split).
- Cada familia aporta exactamente una fila por caso de la base.
- La matriz ensamblada conserva numero, orden y unicidad de casos: 1 332 001 x 129 en train, 194 658 x 129 en OOT, 10 x 128 en test (sin target).
- Ninguna columna reservada aparece como caracteristica predictora.
- F3 anade tres cotejos published_matrix de E5 (train, OOT y test).

#### Cobertura filtrada de productos y registros (products-v2)

| Fuente | Cobertura train | Cobertura OOT |
|---|---|---|
| tax_registry_b_1 | 1,8 % | 64,7 % |
| tax_registry_c_1 | 30,3 % | 0,0 % |
| credit_bureau_a | 71,0 % | 87,6 % |

Aun filtrados, los registros fiscales se reemplazan entre si con el tiempo: la deriva no es fuga sino disponibilidad de la fuente. El capitulo de modelado debe vigilar su estabilidad semanal.

### 3.5 Formatear datos

#### Dos vistas para el modelado

La matriz ensamblada pasa por dos vistas ajustadas exclusivamente con train (fuente: `results/preparation/E6_views.csv`, `results/preparation/matrix_views_spec.csv`):

| Vista | Tratamiento numerico | Tratamiento categorico | Nulos | Predictores |
|---|---|---|---|---|
| Lineal | Imputados con mediana de train, escalados con media y desviacion de train, con indicador __missing | One-hot sobre vocabulario congelado de train; columnas __missing__ y __UNKNOWN__ | Ninguno | 234 |
| Arboles | Decimales con sus nulos nativos | pl.Enum congelado con __UNKNOWN__ | Nativos (compatibles con LightGBM/XGBoost) | 114 |

**Columnas constantes descartadas (resolucion G4):** una caracteristica con un solo valor en train sale de las dos vistas (10 columnas, entre ellas tax_registry_a_1, other_1 y persontype). En la vista lineal salen ademas 43 indicadores __missing o niveles one-hot constantes en train. Quedan registradas con role = dropped en matrix_views_spec.csv.

Las columnas de control (case_id, WEEK_NUM, date_decision, MONTH y target, este ultimo solo en train y OOT) preceden a los predictores en ambas vistas. El esquema es identico en train, OOT y test (control schema_parity en F3_final_checks.csv).

#### Comando de reproduccion

El comando E7 para reproducir toda la preparacion desde los Parquet originales es (ejecucion desde cero del cierre: 101,7 s segun E7_manifest.json):

    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.build --config configs/preparation.json --from-scratch

Desde cero reconstruyo las 36 piezas E4, las audita, ensambla E5, ajusta E6 con train y pasa 29 de 29 controles. Los artefactos quedan en `data/artifacts/preparation/` con sus manifiestos y sumas de verificacion SHA-256 en `results/preparation/E7_manifest.json`.

#### Hallazgos resueltos en el cierre G4

| Hallazgo | Resolucion |
|---|---|
| Columnas constantes en train (D1) | Se descartan y se registran en matrix_views_spec.csv |
| pl.concat(how="horizontal") obsoleto en views.py (D7) | Reemplazado por hstack, que ademas exige la misma altura |
| Cobertura de productos y registros sin filtro as-of (F7) | products-v2 filtra por fecha; la ablacion confirma que la ganancia es legitima |
| E5 publicaba las tablas antes de la validacion final de esquemas (F6) | Los tres splits se publican como unidad: un fallo tardio borra lo escrito |
