# Laboratorio 1 — World Development Indicators

Análisis descriptivo, diagnóstico y predictivo de los World Development Indicators
(Banco Mundial), organizado alrededor de una pregunta:

> **¿Qué explica las brechas de esperanza de vida entre países y dónde se ubica el Perú?**

## Entregables

| Archivo | Qué es |
|---|---|
| `informe/Laboratorio1_WDI.docx` | **Informe principal**: 8 secciones, 17 figuras, 5 tablas |
| `dashboard/app.py` | Dashboard interactivo en Streamlit |
| `figures/` | Las figuras en PNG a 150 dpi |
| `results.json` | Todas las cifras calculadas; el informe las lee de aquí |
| `data/processed/panel.csv` | Panel país-año limpio (217 países × 24 años) |

## Reproducir desde cero

```bash
mamba env create -f environment.yml
mamba activate wdi-lab

python src/01_descarga.py      # API del Banco Mundial -> data/raw/
python src/02_preparacion.py   # panel limpio          -> data/processed/panel.csv
python src/03_descriptivo.py   # figuras 1-7
python src/04_diagnostico.py   # figuras 8-11
python src/05_predictivo.py    # figuras 12-15
python src/06_informe.py       # informe/Laboratorio1_WDI.docx
python src/07_verificar.py     # 36 comprobaciones: cifras, afirmaciones e integridad

streamlit run dashboard/app.py # dashboard en http://localhost:8501
```

La primera ejecución de `03_descriptivo.py` necesita un Chrome para exportar el mapa
coroplético a PNG. Si falla, instalarlo con `plotly_get_chrome -y`.

## Cómo está construido

Cada script deja sus salidas en disco y el siguiente las consume. Las cifras que
aparecen en el texto del informe **no están escritas a mano**: los scripts de análisis
las acumulan en `results.json` y `06_informe.py` las interpola al redactar. Si cambia
el análisis, se regenera el documento y el texto se actualiza solo.

| Script | Responsabilidad |
|---|---|
| `src/config.py` | Indicadores, período, rutas, paleta validada, nombres en español |
| `src/estilo.py` | Estilo compartido de las figuras y colocación de etiquetas sin solapes |
| `src/utils_resultados.py` | Lectura/escritura acumulativa de `results.json` |

## Decisiones metodológicas que conviene conocer

- **Se excluyen los agregados regionales** del Banco Mundial (Unión Europea, «Mundo»,
  «Ingreso alto»): mezclarlos con países duplicaría la información.
- **Imputación en dos niveles**: interpolación lineal dentro de cada país y, para países
  sin ningún dato del indicador, mediana de su región en ese año. El 10,2% de las celdas
  fue imputado; el detalle por indicador está en la tabla 1 del informe.
- **La mortalidad infantil se excluye de los predictores**: es un componente aritmético de
  la esperanza de vida, no un factor externo. Incluirla daría un modelo excelente y sin
  valor explicativo.
- **Validación con `GroupKFold` agrupando por país**: una partición aleatoria dejaría al
  Perú de 2015 en entrenamiento y al de 2016 en prueba, y el modelo acertaría por
  memorizar el país. El script verifica la ausencia de fuga en cada partición.
- **Los años 2020-2021 se excluyen del ajuste de la proyección temporal**: son un shock
  puntual; tratarlos como nivel arrastraría el pronóstico por debajo del último dato real.
- **Los años de catástrofe no se eliminan, se marcan.** 15 observaciones (Haití 2010,
  Somalia 2011, Sudán del Sur 2014-2018, Gaza 2023, RCA) tienen saltos de más de 10 años.
  Son reales: la esperanza de vida *de período* colapsa el año de un desastre masivo, y
  borrarlas ocultaría los peores años de los países más frágiles. El análisis de
  sensibilidad de la sección 5.2 confirma que las conclusiones no dependen de ellas.

## Verificación

`src/07_verificar.py` contrasta el informe contra los datos: que cada cifra del DOCX
coincida con `results.json`, que cada afirmación del texto sea cierta según los números,
y que el panel y el documento mantengan su integridad. Sale con código 1 si algo falla.

Fuente de datos: <https://datacatalog.worldbank.org/search/dataset/0037712/world-development-indicators>
