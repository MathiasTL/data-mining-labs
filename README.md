# data-mining-labs

Trabajos del curso de Minería de Datos. Cada tarea vive en su propia carpeta, es
autocontenida (su propio `README.md`, su `environment.yml` y su pipeline) y se
reproduce de forma independiente.

## Tareas

| Carpeta | Tema | Entregable principal |
|---|---|---|
| [`01-wdi-esperanza-vida/`](01-wdi-esperanza-vida/) | Análisis descriptivo, diagnóstico y predictivo de los World Development Indicators (Banco Mundial): qué explica las brechas de esperanza de vida entre países y dónde se ubica el Perú. | Informe DOCX + dashboard Streamlit |

## Convenciones

- Una carpeta por tarea, prefijada con el número de entrega (`NN-slug`).
- Cada tarea trae su propio entorno conda/mamba; no se comparte uno global.
- Las salidas regenerables (datos derivados, figuras, documentos) se ignoran por
  git y se reconstruyen con el pipeline de cada tarea.
