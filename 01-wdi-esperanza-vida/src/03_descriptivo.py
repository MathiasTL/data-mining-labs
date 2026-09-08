"""03 — Análisis descriptivo: ¿cómo se distribuye hoy la esperanza de vida?

Produce las figuras 1 a 7 y las cifras descriptivas que cita el informe.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (COLORES, DIR_PROC, ETIQUETAS, INDICADORES, OBJETIVO,
                    PAIS_FOCO, RAMPA_SECUENCIAL, REGION_FOCO, UNIDADES, es)
from estilo import (aplicar_estilo, colores_por_region, etiquetas_finales,
                    guardar_fig, pie_fuente, titular)
from utils_resultados import guardar, limpiar_seccion

SEC = "descriptivo"
ALIASES = [v[0] for v in INDICADORES.values()]

# Nombres cortos de región para que quepan en ejes y leyendas
CORTO = {
    "Latin America & Caribbean": "América Latina y Caribe",
    "Sub-Saharan Africa": "África subsahariana",
    "Europe & Central Asia": "Europa y Asia Central",
    "East Asia & Pacific": "Asia Oriental y Pacífico",
    "South Asia": "Asia del Sur",
    "Middle East, North Africa, Afghanistan & Pakistan": "Oriente Medio, N. África, Afg. y Pak.",
    "North America": "América del Norte",
}


def fig1_distribucion(ult: pd.DataFrame, anio: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    datos = ult[OBJETIVO]
    ax.hist(datos, bins=24, color=COLORES["principal"], alpha=0.85,
            edgecolor=COLORES["superficie"], linewidth=1.2)
    media, mediana = datos.mean(), datos.median()
    ax.axvline(mediana, color=COLORES["acento"], linewidth=2, linestyle="--")
    tope = ax.get_ylim()[1]
    ax.text(mediana - 0.7, tope * 0.94, f"Mediana mundial\n{mediana:.1f} años",
            color=COLORES["tinta_sec"], fontsize=9, ha="right", va="top")
    valor_peru = float(ult.loc[ult["iso3"] == PAIS_FOCO, OBJETIVO].iloc[0])
    ax.axvline(valor_peru, color=COLORES["tinta"], linewidth=1.5)
    ax.text(valor_peru + 0.7, tope * 0.94, f"Perú\n{valor_peru:.1f} años",
            color=COLORES["tinta"], fontsize=9, ha="left", va="top")
    p25, p75 = datos.quantile(.25), datos.quantile(.75)
    titular(ax, f"La mitad de los países vive entre {p25:.0f} y {p75:.0f} años",
            f"Distribución de la esperanza de vida en {len(ult)} países, {anio}")
    ax.set_xlabel("Esperanza de vida al nacer (años)")
    ax.set_ylabel("Número de países")
    pie_fuente(fig)
    guardar_fig(fig, "fig01_distribucion")
    guardar(SEC, "distribucion", {
        "media": media, "mediana": mediana, "desv": datos.std(),
        "minimo": datos.min(), "maximo": datos.max(),
        "rango": datos.max() - datos.min(),
        "p25": datos.quantile(.25), "p75": datos.quantile(.75),
        "peru": valor_peru,
        "pais_min": ult.loc[datos.idxmin(), "pais"],
        "pais_max": ult.loc[datos.idxmax(), "pais"],
    })


def _boxplot(ax, ult, columna, orden, colores, etiquetas):
    grupos = [ult.loc[ult[columna] == g, OBJETIVO].dropna().values for g in orden]
    bp = ax.boxplot(grupos, vert=False, patch_artist=True, widths=0.62,
                    medianprops=dict(color=COLORES["superficie"], linewidth=2),
                    whiskerprops=dict(color=COLORES["tinta_tenue"], linewidth=1),
                    capprops=dict(color=COLORES["tinta_tenue"], linewidth=1),
                    flierprops=dict(marker="o", markersize=3.5,
                                    markerfacecolor=COLORES["tinta_tenue"],
                                    markeredgecolor="none", alpha=0.6))
    for parche, g in zip(bp["boxes"], orden):
        parche.set_facecolor(colores[g])
        parche.set_edgecolor(COLORES["superficie"])
        parche.set_linewidth(1.5)
    ax.set_yticks(range(1, len(orden) + 1))
    ax.set_yticklabels([etiquetas.get(g, g) for g in orden])
    return grupos


def fig2_regiones(ult: pd.DataFrame, anio: int) -> None:
    orden = (ult.groupby("region")[OBJETIVO].median().sort_values().index.tolist())
    colores = colores_por_region(ult["region"].unique())
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    grupos = _boxplot(ax, ult, "region", orden, colores, CORTO)
    for i, vals in enumerate(grupos, start=1):
        ax.text(np.median(vals), i + 0.42, f"{np.median(vals):.1f}",
                ha="center", fontsize=8.5, color=COLORES["tinta_sec"])
    medianas = ult.groupby("region")[OBJETIVO].median()
    brecha = medianas.max() - medianas.min()
    titular(ax, f"Entre la mejor y la peor región median {brecha:.1f} años de vida",
            f"Distribución por región, {anio}. La línea blanca marca la mediana regional")
    ax.set_xlabel("Esperanza de vida al nacer (años)")
    pie_fuente(fig)
    guardar_fig(fig, "fig02_regiones")
    resumen = (ult.groupby("region")[OBJETIVO]
               .agg(["count", "median", "min", "max"]).round(1)
               .sort_values("median", ascending=False))
    guardar(SEC, "por_region", {CORTO.get(r, r): d for r, d in
                                resumen.to_dict(orient="index").items()})
    guardar(SEC, "brecha_regional",
            resumen["median"].max() - resumen["median"].min())


def fig3_ingreso(ult: pd.DataFrame, anio: int) -> None:
    orden_ing = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    etiquetas = {"Low income": "Ingreso bajo", "Lower middle income": "Ingreso medio-bajo",
                 "Upper middle income": "Ingreso medio-alto", "High income": "Ingreso alto"}
    orden = [g for g in orden_ing if g in set(ult["grupo_ingreso"])]
    # Rampa secuencial: el grupo de ingreso es una magnitud ordenada, no identidades
    pasos = [RAMPA_SECUENCIAL[1], RAMPA_SECUENCIAL[3], RAMPA_SECUENCIAL[5], RAMPA_SECUENCIAL[7]]
    colores = dict(zip(orden, pasos))
    fig, ax = plt.subplots(figsize=(8.6, 4))
    grupos = _boxplot(ax, ult, "grupo_ingreso", orden, colores, etiquetas)
    for i, vals in enumerate(grupos, start=1):
        ax.text(np.median(vals), i + 0.40, f"{np.median(vals):.1f}",
                ha="center", fontsize=8.5, color=COLORES["tinta_sec"])
    titular(ax, "El ingreso ordena la esperanza de vida de forma casi perfecta",
            f"Distribución por grupo de ingreso del Banco Mundial, {anio}")
    ax.set_xlabel("Esperanza de vida al nacer (años)")
    pie_fuente(fig)
    guardar_fig(fig, "fig03_ingreso")
    res = (ult.groupby("grupo_ingreso")[OBJETIVO].agg(["count", "median"]).round(1))
    guardar(SEC, "por_ingreso", {etiquetas.get(k, k): v for k, v in
                                 res.to_dict(orient="index").items()})
    if "Low income" in res.index and "High income" in res.index:
        guardar(SEC, "brecha_ingreso",
                res.loc["High income", "median"] - res.loc["Low income", "median"])


def fig4_series(panel: pd.DataFrame) -> None:
    serie = (panel.groupby(["region", "anio"])
             .apply(lambda g: np.average(g[OBJETIVO], weights=g["poblacion"]),
                    include_groups=False)
             .rename(OBJETIVO).reset_index())
    colores = colores_por_region(serie["region"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    marcas = []
    for region, g in serie.groupby("region"):
        g = g.sort_values("anio")
        ax.plot(g["anio"], g[OBJETIVO], color=colores[region], linewidth=2)
        marcas.append((g["anio"].iloc[-1], float(g[OBJETIVO].iloc[-1]),
                       CORTO.get(region, region), False))
    # Etiqueta directa al final de cada línea: la identidad no depende solo del color
    etiquetas_finales(ax, marcas)
    ax.axvspan(2019.5, 2021.5, color=COLORES["reja"], alpha=0.55, zorder=0)
    ax.text(2020.5, ax.get_ylim()[0] + 1, "COVID-19", ha="center", fontsize=8.5,
            color=COLORES["tinta_tenue"])
    ax.set_xlim(2000, 2031)
    titular(ax, "Veinte años de avance, interrumpidos en 2020-2021",
            "Esperanza de vida por región, ponderada por población")
    ax.set_xlabel("Año")
    ax.set_ylabel("Esperanza de vida (años)")
    pie_fuente(fig)
    guardar_fig(fig, "fig04_series_regiones")

    mundo = (panel.groupby("anio")
             .apply(lambda g: np.average(g[OBJETIVO], weights=g["poblacion"]),
                    include_groups=False))
    guardar(SEC, "mundo_serie", {int(a): round(float(v), 2) for a, v in mundo.items()})
    guardar(SEC, "ganancia_2000_2019", float(mundo.loc[2019] - mundo.loc[2000]))
    guardar(SEC, "caida_covid_mundo", float(mundo.loc[2021] - mundo.loc[2019]))
    guardar(SEC, "mundo_2023", float(mundo.loc[2023]))
    guardar(SEC, "recuperado_2023", bool(mundo.loc[2023] >= mundo.loc[2019]))
    caidas = {CORTO.get(r, r): float(g.set_index("anio")[OBJETIVO].loc[2021]
                                     - g.set_index("anio")[OBJETIVO].loc[2019])
              for r, g in serie.groupby("region")}
    guardar(SEC, "caida_covid_por_region", caidas)


def fig5_mapa(ult: pd.DataFrame, anio: int) -> None:
    import plotly.express as px
    fig = px.choropleth(
        ult, locations="iso3", color=OBJETIVO,
        color_continuous_scale=RAMPA_SECUENCIAL,   # una sola tinta, claro→oscuro
        range_color=(ult[OBJETIVO].min(), ult[OBJETIVO].max()),
        labels={OBJETIVO: "Años"},
    )
    fig.update_traces(marker_line_color=COLORES["superficie"], marker_line_width=0.5)
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    bgcolor=COLORES["superficie"], landcolor="#eeede9",
                    lataxis_range=[-56, 84], lonaxis_range=[-170, 180],
                    showcountries=False)
    fig.update_layout(
        title=dict(text=f"<b>El mapa de la esperanza de vida, {anio}</b><br>"
                        f"<span style='font-size:12px;color:{COLORES['tinta_sec']}'>"
                        f"Tonos más oscuros = mayor esperanza de vida. Gris = sin dato</span>",
                   x=0.01, xanchor="left", font=dict(size=17)),
        paper_bgcolor=COLORES["superficie"], font=dict(family="Helvetica, Arial", size=12),
        margin=dict(l=8, r=8, t=78, b=42), height=470, width=980,
        coloraxis_colorbar=dict(title="Años", thickness=14, len=0.68),
    )
    fig.add_annotation(text="Fuente: World Development Indicators, Banco Mundial.",
                       showarrow=False, x=0, y=-0.05, xref="paper", yref="paper", xanchor="left",
                       font=dict(size=10, color=COLORES["tinta_tenue"]))
    from config import DIR_FIG
    ruta = DIR_FIG / "fig05_mapa.png"
    fig.write_image(ruta, scale=2)
    print(f"  → {ruta.name} ({ruta.stat().st_size/1024:.0f} KB)")


def fig6_ranking(ult: pd.DataFrame, anio: int) -> None:
    orden = ult.sort_values(OBJETIVO)
    bajos, altos = orden.head(10), orden.tail(10)
    sel = pd.concat([bajos, altos])
    mediana = ult[OBJETIVO].median()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    colores = [COLORES["acento"] if v < mediana else COLORES["principal"]
               for v in sel[OBJETIVO]]
    y = np.arange(len(sel))
    ax.barh(y, sel[OBJETIVO] - mediana, left=mediana, color=colores, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([es(n) for n in sel["pais"]], fontsize=9)
    ax.axvline(mediana, color=COLORES["tinta_sec"], linewidth=1.2)
    for yi, v in zip(y, sel[OBJETIVO]):
        despl = 0.4 if v >= mediana else -0.4
        ax.text(v + despl, yi, f"{v:.1f}", va="center",
                ha="left" if v >= mediana else "right",
                fontsize=8.5, color=COLORES["tinta_sec"])
    ax.text(mediana, len(sel) - 0.2, f"Mediana mundial {mediana:.1f}",
            ha="center", fontsize=9, color=COLORES["tinta_sec"])
    ax.set_xlim(ult[OBJETIVO].min() - 4, ult[OBJETIVO].max() + 3)
    amplitud = ult[OBJETIVO].max() - ult[OBJETIVO].min()
    titular(ax, f"Del país que menos vive al que más hay {amplitud:.0f} años de diferencia",
            f"Los 10 países con menor y mayor esperanza de vida, {anio}")
    ax.set_xlabel("Esperanza de vida al nacer (años)")
    pie_fuente(fig)
    guardar_fig(fig, "fig06_ranking")
    for nombre, df in (("top10", altos.iloc[::-1]), ("bottom10", bajos)):
        salida = df[["pais", OBJETIVO]].rename(columns={OBJETIVO: "valor"}).round(1)
        salida["pais"] = salida["pais"].map(es)
        guardar(SEC, nombre, salida)
    guardar(SEC, "amplitud", float(amplitud))


def fig7_peru(panel: pd.DataFrame) -> None:
    peru = panel[panel["iso3"] == PAIS_FOCO].sort_values("anio")
    lac = (panel[panel["region"] == REGION_FOCO].groupby("anio")
           .apply(lambda g: np.average(g[OBJETIVO], weights=g["poblacion"]),
                  include_groups=False))
    mundo = (panel.groupby("anio")
             .apply(lambda g: np.average(g[OBJETIVO], weights=g["poblacion"]),
                    include_groups=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(mundo.index, mundo.values, color=COLORES["tinta_tenue"], linewidth=2)
    ax.plot(lac.index, lac.values, color=COLORES["apoyo"], linewidth=2)
    ax.plot(peru["anio"], peru[OBJETIVO], color=COLORES["acento"], linewidth=2.6)
    etiquetas_finales(ax, [
        (2023, float(mundo.loc[2023]), "Mundo", False),
        (2023, float(lac.loc[2023]), "América Latina", False),
        (2023, float(peru.set_index("anio").loc[2023, OBJETIVO]), "Perú", True),
    ])
    ax.set_xlim(2000, 2028)
    titular(ax, "Perú: una caída abrupta y una recuperación igual de brusca",
            "Esperanza de vida al nacer")
    ax.set_ylabel("Años")
    ax.set_xlabel("Año")

    ax = axes[1]
    ax.plot(peru["anio"], peru["mort_infantil"], color=COLORES["acento"], linewidth=2.6)
    lac_mi = (panel[panel["region"] == REGION_FOCO].groupby("anio")
              .apply(lambda g: np.average(g["mort_infantil"], weights=g["poblacion"]),
                     include_groups=False))
    ax.plot(lac_mi.index, lac_mi.values, color=COLORES["apoyo"], linewidth=2)
    etiquetas_finales(ax, [
        (2023, float(peru.set_index("anio").loc[2023, "mort_infantil"]), "Perú", True),
        (2023, float(lac_mi.loc[2023]), "América Latina", False),
    ])
    ax.set_xlim(2000, 2028)
    titular(ax, "…pero su mortalidad infantil siguió bajando",
            "Muertes de menores de 1 año por cada 1.000 nacidos vivos")
    ax.set_ylabel("Por 1.000 n.v.")
    ax.set_xlabel("Año")
    fig.subplots_adjust(wspace=0.28)
    pie_fuente(fig)
    guardar_fig(fig, "fig07_peru")

    pi = peru.set_index("anio")
    guardar(SEC, "peru", {
        "ev_2000": float(pi.loc[2000, OBJETIVO]), "ev_2019": float(pi.loc[2019, OBJETIVO]),
        "ev_2021": float(pi.loc[2021, OBJETIVO]), "ev_2023": float(pi.loc[2023, OBJETIVO]),
        "caida_covid": float(pi.loc[2021, OBJETIVO] - pi.loc[2019, OBJETIVO]),
        "mi_2000": float(pi.loc[2000, "mort_infantil"]),
        "mi_2023": float(pi.loc[2023, "mort_infantil"]),
        "pib_pc_2023": float(pi.loc[2023, "pib_pc"]),
        "gasto_salud_pib_2023": float(pi.loc[2023, "gasto_salud_pib"]),
        "medicos_2023": float(pi.loc[2023, "medicos_1000"]),
        "lac_2023": float(lac.loc[2023]), "mundo_2023": float(mundo.loc[2023]),
        "brecha_vs_lac_2023": float(pi.loc[2023, OBJETIVO] - lac.loc[2023]),
    })


def tabla_descriptiva(panel: pd.DataFrame, ult: pd.DataFrame) -> None:
    filas = []
    for alias in ALIASES:
        if alias == "poblacion":
            continue
        s = ult[alias].dropna()
        filas.append({
            "indicador": ETIQUETAS[alias], "unidad": UNIDADES[alias],
            "media": round(float(s.mean()), 2), "mediana": round(float(s.median()), 2),
            "desv": round(float(s.std()), 2), "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
        })
    guardar(SEC, "tabla_estadisticos", filas)


def main() -> None:
    aplicar_estilo()
    limpiar_seccion(SEC)
    panel = pd.read_csv(DIR_PROC / "panel.csv")
    anio = int(panel["anio"].max())
    ult = panel[panel["anio"] == anio].copy()
    print(f"Análisis descriptivo — año de corte {anio}, {len(ult)} países\n")

    fig1_distribucion(ult, anio)
    fig2_regiones(ult, anio)
    fig3_ingreso(ult, anio)
    fig4_series(panel)
    fig5_mapa(ult, anio)
    fig6_ranking(ult, anio)
    fig7_peru(panel)
    tabla_descriptiva(panel, ult)
    print("\nDescriptivo completo.")


if __name__ == "__main__":
    main()
