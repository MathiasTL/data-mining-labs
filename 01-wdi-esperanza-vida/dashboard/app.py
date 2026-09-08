"""Dashboard interactivo complementario al informe del Laboratorio 1.

Reutiliza el mismo panel limpio y los mismos resultados que el documento Word,
de modo que ninguna cifra puede divergir entre ambos entregables.

Ejecutar con:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (COLORES, DIR_PROC, ETIQUETAS, OBJETIVO, PAIS_FOCO,
                    RAMPA_SECUENCIAL, REGION_FOCO, SLOTS, UNIDADES, es)
from utils_resultados import cargar

st.set_page_config(page_title="WDI · Esperanza de vida", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")

CORTO = {
    "Latin America & Caribbean": "América Latina y Caribe",
    "Sub-Saharan Africa": "África subsahariana",
    "Europe & Central Asia": "Europa y Asia Central",
    "East Asia & Pacific": "Asia Oriental y Pacífico",
    "South Asia": "Asia del Sur",
    "Middle East, North Africa, Afghanistan & Pakistan": "Oriente Medio, N. África, Afg. y Pak.",
    "North America": "América del Norte",
}
INGRESO_ES = {"Low income": "Ingreso bajo", "Lower middle income": "Ingreso medio-bajo",
              "Upper middle income": "Ingreso medio-alto", "High income": "Ingreso alto"}


@st.cache_data
def cargar_panel() -> pd.DataFrame:
    p = pd.read_csv(DIR_PROC / "panel.csv")
    p["region_es"] = p["region"].map(lambda r: CORTO.get(r, r))
    p["ingreso_es"] = p["grupo_ingreso"].map(lambda g: INGRESO_ES.get(g, g))
    p["pais_es"] = p["pais"].map(es)
    return p


@st.cache_data
def cargar_resultados() -> dict:
    return cargar()


panel = cargar_panel()
R = cargar_resultados()
ANIO_MAX = int(panel["anio"].max())

# El color sigue a la región, no al orden en que plotly la dibuja: sin este mapa
# fijo, filtrar regiones repintaría a las que quedan y cada gráfico usaría una
# asignación distinta.
COLOR_REGION = {r: SLOTS[i % len(SLOTS)]
                for i, r in enumerate(sorted(panel["region_es"].unique()))}

# ------------------------------------------------------------------ filtros
st.sidebar.title("Filtros")
anio = st.sidebar.slider("Año", int(panel["anio"].min()), ANIO_MAX, ANIO_MAX)
regiones = st.sidebar.multiselect(
    "Regiones", sorted(panel["region_es"].unique()),
    default=sorted(panel["region_es"].unique()))
ingresos = st.sidebar.multiselect(
    "Grupos de ingreso", list(INGRESO_ES.values()),
    default=[v for v in INGRESO_ES.values() if v in set(panel["ingreso_es"])])
st.sidebar.divider()
st.sidebar.caption(
    "Fuente: World Development Indicators, Banco Mundial. "
    f"Panel de {panel['iso3'].nunique()} países, {panel['anio'].min()}-{ANIO_MAX}.")

f = panel[panel["region_es"].isin(regiones) & panel["ingreso_es"].isin(ingresos)]
fa = f[f["anio"] == anio]

st.title("Esperanza de vida en el mundo")
st.caption("Dashboard complementario al informe del Laboratorio 1 · "
           "los datos son los mismos que respaldan el documento Word")

if fa.empty:
    st.warning("Ningún país cumple los filtros seleccionados. Amplía la selección.")
    st.stop()

t1, t2, t3, t4 = st.tabs(["Panorama", "Comparar países", "Modelo", "Datos"])

# ------------------------------------------------------------------ panorama
with t1:
    c = st.columns(4)
    c[0].metric("Países en la selección", len(fa))
    c[1].metric("Mediana", f"{fa[OBJETIVO].median():.1f} años")
    peor, mejor = fa.nsmallest(1, OBJETIVO).iloc[0], fa.nlargest(1, OBJETIVO).iloc[0]
    c[2].metric(f"Máximo · {mejor['pais_es']}", f"{mejor[OBJETIVO]:.1f} años")
    c[3].metric(f"Mínimo · {peor['pais_es']}", f"{peor[OBJETIVO]:.1f} años")

    izq, der = st.columns([1.5, 1])
    with izq:
        st.subheader("Mapa mundial")
        mapa = px.choropleth(
            fa, locations="iso3", color=OBJETIVO, hover_name="pais_es",
            hover_data={"iso3": False, OBJETIVO: ":.1f", "region_es": True},
            color_continuous_scale=RAMPA_SECUENCIAL,
            labels={OBJETIVO: "Años", "region_es": "Región"})
        mapa.update_traces(marker_line_color="#fcfcfb", marker_line_width=0.5)
        mapa.update_geos(showframe=False, showcoastlines=False,
                         projection_type="natural earth", lataxis_range=[-56, 84],
                         bgcolor="rgba(0,0,0,0)", landcolor="#eeede9")
        mapa.update_layout(margin=dict(l=0, r=0, t=8, b=0), height=420,
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(mapa, use_container_width=True)
    with der:
        st.subheader("Por grupo de ingreso")
        orden = [v for v in INGRESO_ES.values() if v in set(fa["ingreso_es"])]
        # Mapa explícito: el tono oscurece con el ingreso, igual que en el informe.
        # Con color_discrete_sequence el orden dependería del orden de dibujo.
        tonos = dict(zip(list(INGRESO_ES.values()),
                         [RAMPA_SECUENCIAL[i] for i in (1, 3, 5, 7)]))
        caja = px.box(fa, x=OBJETIVO, y="ingreso_es", color="ingreso_es",
                      category_orders={"ingreso_es": orden[::-1]},
                      color_discrete_map=tonos,
                      labels={OBJETIVO: "Años", "ingreso_es": ""},
                      hover_name="pais_es")
        caja.update_layout(showlegend=False, height=420, margin=dict(l=0, r=0, t=8, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(caja, use_container_width=True)

    st.subheader("Evolución por región")
    serie = (f.groupby(["region_es", "anio"])
             .apply(lambda g: np.average(g[OBJETIVO], weights=g["poblacion"]),
                    include_groups=False)
             .rename(OBJETIVO).reset_index())
    lineas = px.line(serie, x="anio", y=OBJETIVO, color="region_es",
                     color_discrete_map=COLOR_REGION,
                     labels={"anio": "Año", OBJETIVO: "Esperanza de vida (años)",
                             "region_es": "Región"})
    lineas.add_vrect(x0=2019.5, x1=2021.5, fillcolor="#e6e5e1", opacity=0.55,
                     line_width=0, annotation_text="COVID-19",
                     annotation_position="bottom")
    lineas.update_traces(line_width=2.4)
    lineas.update_layout(height=430, hovermode="x unified", margin=dict(l=70, r=10, t=10, b=40),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         legend_title_text="")
    st.plotly_chart(lineas, use_container_width=True)

# ------------------------------------------------------------------ comparar
with t2:
    disponibles = sorted(f["pais_es"].unique())
    por_defecto = [p for p in ["Perú", "Chile", "Colombia", "México"] if p in disponibles]
    elegidos = st.multiselect("Países a comparar", disponibles,
                              default=por_defecto or disponibles[:3])
    indicador = st.selectbox(
        "Indicador", [OBJETIVO] + [c for c in ETIQUETAS if c in f.columns and c != OBJETIVO],
        format_func=lambda c: ETIQUETAS[c])

    if elegidos:
        sub = f[f["pais_es"].isin(elegidos)]
        fig = px.line(sub, x="anio", y=indicador, color="pais_es",
                      color_discrete_sequence=SLOTS,
                      labels={"anio": "Año", indicador: f"{ETIQUETAS[indicador]} ({UNIDADES[indicador]})",
                              "pais_es": "País"})
        fig.update_traces(line_width=2.4)
        fig.update_layout(height=430, hovermode="x unified", legend_title_text="",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona al menos un país.")

    st.subheader("Curva de Preston")
    st.caption("Cada punto es un país. La escala logarítmica del eje horizontal revela "
               "los rendimientos decrecientes del ingreso sobre la longevidad.")
    disp = px.scatter(fa, x="pib_pc", y=OBJETIVO, log_x=True, hover_name="pais_es",
                      size="poblacion", size_max=42, color="region_es",
                      color_discrete_map=COLOR_REGION,
                      labels={"pib_pc": "PIB per cápita, PPA (USD 2021, escala log)",
                              OBJETIVO: "Esperanza de vida (años)", "region_es": "Región"})
    d = fa.dropna(subset=["log_pib_pc", OBJETIVO])
    b, a = np.polyfit(d["log_pib_pc"], d[OBJETIVO], 1)
    malla = np.linspace(d["pib_pc"].min(), d["pib_pc"].max(), 200)
    disp.add_trace(go.Scatter(x=malla, y=a + b * np.log10(malla), mode="lines",
                              line=dict(color=COLORES["tinta"], width=2),
                              name="Ajuste logarítmico"))
    disp.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(disp, use_container_width=True)

# ------------------------------------------------------------------ modelo
with t3:
    pr = R.get("predictivo", {})
    if not pr:
        st.warning("Ejecuta src/05_predictivo.py para poblar esta pestaña.")
    else:
        c = st.columns(4)
        mejor = pr["mejor_modelo"]
        m = pr["metricas"][mejor]
        c[0].metric("Mejor modelo", mejor)
        c[1].metric("R² (validación por país)", f"{m['r2']:.3f}")
        c[2].metric("RMSE", f"{m['rmse']:.2f} años")
        c[3].metric("Mejora sobre la línea base", f"{pr['mejora_vs_baseline_rmse']:.0f}%")

        izq, der = st.columns(2)
        with izq:
            st.subheader("Desempeño de los modelos")
            met = pd.DataFrame(pr["metricas"]).T.reset_index(names="Modelo")
            st.dataframe(met.rename(columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²"}),
                         hide_index=True, use_container_width=True)
            st.caption("Validación cruzada GroupKFold(5) agrupada por país: ningún país "
                       "aparece a la vez en entrenamiento y prueba.")
        with der:
            st.subheader("Importancia de las variables")
            imp = pd.DataFrame(pr["importancias"])
            fig = px.bar(imp.sort_values("importancia"), x="importancia", y="variable",
                         orientation="h", color_discrete_sequence=[COLORES["principal"]],
                         labels={"importancia": "Pérdida de R² al permutar", "variable": ""})
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=8, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Proyección a 2030")
        cols = st.columns(len(pr["proyeccion"]))
        for col, (nombre, v) in zip(cols, pr["proyeccion"].items()):
            col.metric(f"{nombre} en 2030", f"{v['proyeccion_2030']:.1f} años",
                       f"{v['ganancia_2023_2030']:+.1f} años vs. {ANIO_MAX}",
                       delta_color="off")
            col.caption(f"Sin la pandemia habría llegado a {v['contrafactual_2030']:.1f} años "
                        f"({v['brecha_vs_sin_covid']:+.1f}).")

# ------------------------------------------------------------------ datos
with t4:
    st.subheader(f"Panel filtrado · {anio}")
    cols_ver = ["pais_es", "iso3", "region_es", "ingreso_es", OBJETIVO,
                "pib_pc", "gasto_salud_pc", "medicos_1000", "saneamiento", "fertilidad"]
    tabla = fa[cols_ver].rename(columns={
        "pais_es": "País", "iso3": "ISO3", "region_es": "Región", "ingreso_es": "Ingreso",
        OBJETIVO: "Esperanza de vida", "pib_pc": "PIB p/c (PPA)",
        "gasto_salud_pc": "Gasto salud p/c", "medicos_1000": "Médicos /1.000",
        "saneamiento": "Saneamiento %", "fertilidad": "Fertilidad",
    }).sort_values("Esperanza de vida", ascending=False)
    st.dataframe(tabla.round(2), hide_index=True, use_container_width=True, height=460)
    st.download_button("Descargar esta selección en CSV",
                       tabla.to_csv(index=False).encode("utf-8"),
                       file_name=f"wdi_{anio}.csv", mime="text/csv")
    with st.expander("Nota sobre la calidad de los datos"):
        d = R.get("datos", {})
        st.write(
            f"El {d.get('pct_celdas_imputadas', '?')}% de las celdas del panel fue imputado: "
            "primero por interpolación lineal dentro de cada país y, para los países sin ningún "
            "dato del indicador, por la mediana de su región en ese año. Los indicadores más "
            "afectados son médicos por 1.000 habitantes y matrícula secundaria.")
