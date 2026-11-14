# Fuentes bibliográficas para justificar las técnicas del modelo

| Aspecto | Información |
|---|---|
| Alcance | Diez referencias mencionadas en la conversación que corresponden a revistas Q1 o Q2 y pueden sustentar las técnicas del proyecto de riesgo crediticio. |
| Interpretación del cuartil | El cuartil corresponde a la revista y depende del índice, categoría y año; no es una clasificación individual del artículo. |
| Año de referencia de los cuartiles | 2024. Este año no necesariamente coincide con el año de publicación del artículo. Se indica en cada fila si la clasificación corresponde a JCR o SJR. |
| Relación con el proyecto | Las fuentes aportan sustento para la predicción de probabilidad de incumplimiento (PD), comparación de clasificadores, evaluación temporal, interpretabilidad y monitoreo. Citarlas no implica que todas las técnicas descritas en ellas se hayan implementado en el proyecto. |

## Referencias Q1 y Q2

| N.º | Referencia y enlace al artículo | Revista, volumen y páginas o identificador | Cuartil de referencia | Qué justifica en nuestro proyecto |
|---|---|---|---|---|
| 1 | **Dumitrescu, E., Hué, S., Hurlin, C. y Tokpavi, S. (2022).** [*Machine learning for credit scoring: Improving logistic regression with non-linear decision-tree effects.*](https://doi.org/10.1016/j.ejor.2021.06.053) | *European Journal of Operational Research*, 297(3), 1178–1192. | **Q1, JCR 2024.** | Regresión logística como referencia interpretable y comparación con técnicas que capturan relaciones no lineales. Su propuesta concreta es una regresión logística enriquecida con efectos de árboles. |
| 2 | **Bárcena Saavedra, C. A. P., Fachini-Gomes, J. B., Gomes, E. M. de C. y Kimura, H. (2024).** [*Probability of default for lifetime credit loss for IFRS 9 using machine learning competing risks survival analysis models.*](https://doi.org/10.1016/j.eswa.2024.123607) | *Expert Systems with Applications*, 249, 123607. | **Q1, JCR 2024.** | Importancia de estimar la probabilidad de incumplimiento. Estudia supervivencia y PD de vida; su formulación no equivale directamente a nuestra clasificación binaria. |
| 3 | **Bellotti, A. y Crook, J. (2014).** [*Forecasting and stress testing credit card default using dynamic models.*](https://www.sciencedirect.com/science/article/pii/S0169207013000502) | *International Journal of Forecasting*, 30(1), 143–154. | **Q1, SJR 2024.** | Evaluación temporal del riesgo y consideración de cambios en el comportamiento de pago. No establece nuestro corte específico en la semana 74. |
| 4 | **Ayari, H., Guetari, R. y Kraïem, N. (2026).** [*Machine learning powered financial credit scoring: A systematic literature review.*](https://doi.org/10.1007/s10462-025-11416-2) | *Artificial Intelligence Review*, 59, 13. | **Q1, SJR 2024.** | Marco general para seleccionar y comparar modelos de crédito, incluidos ensambles, y discutir desempeño e interpretabilidad. |
| 5 | **Chen, Y., Calabrese, R. y Martin-Barragan, B. (2024).** [*Interpretable machine learning for imbalanced credit scoring datasets.*](https://doi.org/10.1016/j.ejor.2023.06.036) | *European Journal of Operational Research*, 312(1), 357–372. | **Q1, JCR 2024.** | Uso y evaluación de **SHAP y LIME bajo desbalance de clases**. Sustenta revisar la estabilidad de las explicaciones cuando hay pocos incumplimientos. |
| 6 | **Hlongwane, R., Ramaboa, K. y Mongwe, W. (2024).** [*A novel framework for enhancing transparency in credit scoring: Leveraging Shapley values for interpretable credit scorecards.*](https://doi.org/10.1371/journal.pone.0308718) | *PLOS ONE*, 19(8), e0308718. | **Q2, JCR 2024.** | Aplicación de valores de Shapley a **Random Forest, XGBoost, LightGBM y CatBoost** para construir scorecards interpretables. La grafía del segundo apellido se recoge de la [corrección publicada en 2025](https://doi.org/10.1371/journal.pone.0329901). |
| 7 | **Nwafor, C. N., Nwafor, O. y Brahma, S. (2024).** [*Enhancing transparency and fairness in automated credit decisions: An explainable novel hybrid machine learning approach.*](https://doi.org/10.1038/s41598-024-75026-8) | *Scientific Reports*, 14, 25174. | **Q1, JCR 2024.** | Uso de **SHAP** para explicar un modelo híbrido de **1D-CNN y XGBoost**, con análisis global e individual de variables. |
| 8 | **Han, S. y Jung, H. (2024).** [*NATE: Non-pArameTric approach for Explainable credit scoring on imbalanced class.*](https://doi.org/10.1371/journal.pone.0316454) | *PLOS ONE*, 19(12), e0316454. | **Q2, JCR 2024.** | Tratamiento del **desbalance**, clasificadores basados en árboles y explicabilidad. Sirve para discutir remuestreo; no implica que debamos aplicarlo automáticamente. |
| 9 | **Talaat, F. M., Aljadani, A., Badawy, M. y Elhosseini, M. (2024).** [*Toward interpretable credit scoring: Integrating explainable artificial intelligence with deep learning for credit card default prediction.*](https://doi.org/10.1007/s00521-023-09232-2) | *Neural Computing and Applications*, 36, 4847–4865. | **Q1, SJR 2024.** | Combinación de aprendizaje profundo y explicabilidad para predecir incumplimiento. Es complementario si incluimos una red neuronal. |
| 10 | **Barddal, J. P., Loezer, L., Enembreck, F. y Lanzuolo, R. (2020).** [*Lessons learned from data stream classification applied to credit scoring.*](https://doi.org/10.1016/j.eswa.2020.113899) | *Expert Systems with Applications*, 162, 113899. | **Q1, JCR 2024.** | Aprendizaje sobre flujos de crédito, cambios temporales y métodos que incorporan **ADWIN**, como Leveraging Bagging. |

## Fuentes para contrastar los cuartiles

| Revista | Índice y cuartil de referencia | Fuente de consulta |
|---|---|---|
| *European Journal of Operational Research* | JCR 2024, Q1. | [EURO — The Association of European Operational Research Societies: Publications](https://www.euro-online.org/web/pages/106/publications). |
| *Expert Systems with Applications* | JCR 2024, Q1. | [Pusan National University: información y clasificación de la revista](https://scholar2.pusan.ac.kr/journals/019f8e7a-11e0-758b-9637-916a776ad7c6). |
| *International Journal of Forecasting* | SJR 2024, Q1. | [Research Cloud: métricas anuales de la revista](https://researchcloud.net/journal/0169-2070). |
| *Artificial Intelligence Review* | SJR 2024, Q1 en Artificial Intelligence. | [SCImago Journal & Country Rank: clasificación de la revista](https://www.scimagojr.com/journalsearch.php?clean=0&q=24141&tip=sid). |
| *PLOS ONE* | JCR 2024, Q2 en Multidisciplinary Sciences. | [Wageningen University & Research Library: registro de la revista](https://library.wur.nl/WebQuery/wda/1847536). |
| *Scientific Reports* | JCR 2024, Q1. | [Universidad Pontificia Comillas, Instituto de Investigación Tecnológica: información de la revista](https://www.iit.comillas.edu/publicacion/info_revista/en/379/Scientific_Reports). |
| *Neural Computing and Applications* | SJR 2024, Q1. | [Universidad Pontificia Comillas, Instituto de Investigación Tecnológica: información de la revista](https://www.iit.comillas.edu/publicacion/info_revista/es/651/Neural_Computing_and_Applications). |

## Prioridad para justificar la propuesta

| Decisión del proyecto | Referencias prioritarias | Aplicación |
|---|---|---|
| Clasificación y comparación de modelos | Dumitrescu et al. (2022) y Ayari et al. (2026). | Justificar una referencia interpretable mediante regresión logística y la comparación con modelos no lineales y ensambles. |
| Interpretabilidad con SHAP | Chen et al. (2024) y Hlongwane et al. (2024). | Explicar las predicciones de los modelos y evaluar la estabilidad de las explicaciones bajo desbalance de clases. |
| Evaluación temporal y monitoreo con ADWIN | Bellotti y Crook (2014) y Barddal et al. (2020). | Sustentar la evaluación en el tiempo y el estudio de métodos de monitoreo frente a cambios en datos y desempeño. |
| Concepto de probabilidad de incumplimiento (PD) | Bárcena Saavedra et al. (2024). | Fundamentar la relevancia de la PD, aclarando la diferencia entre el enfoque de supervivencia del artículo y nuestro clasificador binario. |

## Referencias fundacionales complementarias

| Técnica | Referencia original | Tipo de publicación y condición respecto de Q1/Q2 | Uso |
|---|---|---|---|
| SHAP | **Lundberg, S. M. y Lee, S.-I. (2017).** [*A unified approach to interpreting model predictions.*](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) *Advances in Neural Information Processing Systems, 30*, 4765–4774. | Conferencia NeurIPS; no cuenta como artículo de revista Q1/Q2. | Fundamento metodológico original de SHAP; complementa las aplicaciones publicadas en revistas Q1/Q2. |
| ADWIN | **Bifet, A. y Gavaldà, R. (2007).** [*Learning from time-changing data with adaptive windowing.*](https://doi.org/10.1137/1.9781611972771.42) *Proceedings of the 2007 SIAM International Conference on Data Mining*. | Conferencia SIAM SDM; no cuenta como artículo de revista Q1/Q2. | Fundamento metodológico original de ADWIN; complementa la evidencia aplicada al dominio crediticio. |
