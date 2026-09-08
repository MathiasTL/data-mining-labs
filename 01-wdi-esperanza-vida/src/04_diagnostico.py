"""04 — Análisis diagnóstico: ¿por qué existen esas brechas?

Correlaciones, curva de Preston, colinealidad (VIF), anatomía del shock COVID
y segmentación de países por perfil de desarrollo-salud.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import (COLORES, DIR_PROC, ETIQUETAS, OBJETIVO, PAIS_FOCO,
                    PALETA_SCATTER, PREDICTOR_CIRCULAR, PREDICTORES,
                    RAMPA_DIVERGENTE, REGION_FOCO, es)
from estilo import aplicar_estilo, guardar_fig, pie_fuente, titular
from utils_resultados import guardar, limpiar_seccion

SEC = "diagnostico"
CORTO = {
    "Latin America & Caribbean": "América Latina y Caribe",
    "Sub-Saharan Africa": "África subsahariana",
    "Europe & Central Asia": "Europa y Asia Central",
    "East Asia & Pacific": "Asia Oriental y Pacífico",
    "South Asia": "Asia del Sur",
    "Middle East, North Africa, Afghanistan & Pakistan": "Oriente Medio, N. África, Afg. y Pak.",
    "North America": "América del Norte",
}
CORTO_VAR = {
    "esperanza_vida": "Esperanza de vida", "pib_pc": "PIB per cápita",
    "gasto_salud_pc": "Gasto salud p/c", "gasto_salud_pib": "Gasto salud %PIB",
    "medicos_1000": "Médicos /1.000", "mort_infantil": "Mortalidad infantil",
    "fertilidad": "Fertilidad", "matricula_secundaria": "Matríc. secundaria",
    "inmunizacion": "Inmunización", "agua_potable": "Agua potable",
    "saneamiento": "Saneamiento", "urbanizacion": "Urbanización",
    "poblacion_65": "Población 65+",
}


def fig8_correlaciones(ult: pd.DataFrame) -> None:
    cols = [OBJETIVO, PREDICTOR_CIRCULAR] + PREDICTORES
    corr = ult[cols].corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list("div", RAMPA_DIVERGENTE[::-1])
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    etiquetas = [CORTO_VAR[c] for c in cols]
    ax.set_xticks(range(len(cols)), etiquetas, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols)), etiquetas, fontsize=8.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            # El texto va en tinta, no en el color de la serie
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if abs(v) > 0.62 else COLORES["tinta_sec"])
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.grid(which="minor", color=COLORES["superficie"], linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Correlación de Spearman", fontsize=9, color=COLORES["tinta_sec"])
    cb.outline.set_visible(False)
    titular(ax, "Todo lo que acompaña a una vida más larga está correlacionado entre sí",
            "Correlación de rangos entre indicadores, último año disponible")
    pie_fuente(fig)
    guardar_fig(fig, "fig08_correlaciones")

    con_obj = corr[OBJETIVO].drop(OBJETIVO).sort_values(key=abs, ascending=False)
    guardar(SEC, "corr_con_objetivo",
            {CORTO_VAR[k]: round(float(v), 3) for k, v in con_obj.items()})
    guardar(SEC, "top3_correlaciones",
            [{"variable": CORTO_VAR[k], "rho": round(float(v), 3)}
             for k, v in con_obj.head(3).items()])


def fig9_preston(ult: pd.DataFrame) -> None:
    d = ult.dropna(subset=["log_pib_pc", OBJETIVO]).copy()
    x, y = d["log_pib_pc"].values, d[OBJETIVO].values
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # Comparación con el ajuste lineal en niveles, para mostrar por qué se usa log
    b2, a2 = np.polyfit(d["pib_pc"], y, 1)
    pred2 = a2 + b2 * d["pib_pc"]
    r2_lineal = 1 - ((y - pred2) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.scatter(d["pib_pc"], y, s=26, color=COLORES["principal"], alpha=0.55,
               edgecolor=COLORES["superficie"], linewidth=0.6, zorder=2)
    malla = np.linspace(d["pib_pc"].min(), d["pib_pc"].max(), 300)
    ax.plot(malla, a + b * np.log10(malla), color=COLORES["tinta"], linewidth=2, zorder=3)
    ax.set_xscale("log")

    fila = d[d["iso3"] == PAIS_FOCO].iloc[0]
    ax.scatter([fila["pib_pc"]], [fila[OBJETIVO]], s=130, color=COLORES["acento"],
               edgecolor=COLORES["superficie"], linewidth=2, zorder=4)
    residuo_peru = float(fila[OBJETIVO] - (a + b * fila["log_pib_pc"]))
    ax.annotate(f"Perú\n{residuo_peru:+.1f} años vs. lo esperado",
                xy=(fila["pib_pc"], fila[OBJETIVO]), xytext=(18, -34),
                textcoords="offset points", fontsize=9, color=COLORES["tinta"],
                arrowprops=dict(arrowstyle="-", color=COLORES["tinta_tenue"], lw=1),
                bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORES["superficie"],
                          edgecolor="none", alpha=0.88))
    ax.text(0.98, 0.06, f"Ajuste logarítmico: R² = {r2:.2f}\n"
                        f"(ajuste lineal en niveles: R² = {r2_lineal:.2f})",
            transform=ax.transAxes, ha="right", fontsize=9, color=COLORES["tinta_sec"])
    titular(ax, "Duplicar el ingreso ya no compra los mismos años de vida",
            "Curva de Preston: PIB per cápita (escala logarítmica) y esperanza de vida")
    ax.set_xlabel("PIB per cápita, PPA (USD 2021, escala log)")
    ax.set_ylabel("Esperanza de vida (años)")
    pie_fuente(fig)
    guardar_fig(fig, "fig09_preston")

    guardar(SEC, "preston", {
        "pendiente_por_decuplicar": float(b), "intercepto": float(a),
        "r2_log": float(r2), "r2_lineal": float(r2_lineal),
        "anios_por_duplicar_pib": float(b * np.log10(2)),
        "residuo_peru": residuo_peru,
    })
    d["residuo_preston"] = y - pred
    mejores = d.nlargest(5, "residuo_preston")[["pais", "residuo_preston"]]
    peores = d.nsmallest(5, "residuo_preston")[["pais", "residuo_preston"]]
    for nombre, df in (("preston_sobre", mejores), ("preston_bajo", peores)):
        df = df.copy(); df["pais"] = df["pais"].map(es)
        guardar(SEC, nombre, df.round(1))


def tabla_vif(ult: pd.DataFrame) -> None:
    X = ult[PREDICTORES].dropna()
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=PREDICTORES)
    Xs.insert(0, "const", 1.0)
    filas = [{"variable": CORTO_VAR[c],
              "vif": round(float(variance_inflation_factor(Xs.values, i)), 2)}
             for i, c in enumerate(Xs.columns) if c != "const"]
    filas.sort(key=lambda f: -f["vif"])
    guardar(SEC, "vif", filas)
    altos = [f for f in filas if f["vif"] > 5]
    guardar(SEC, "vif_altos", altos)
    print("  VIF (>5 indica colinealidad problemática):")
    for f in filas:
        marca = "  <-- alto" if f["vif"] > 5 else ""
        print(f"    {f['variable']:22s} {f['vif']:6.2f}{marca}")


def fig10_covid(panel: pd.DataFrame) -> None:
    ancho = panel.pivot_table(index=["iso3", "pais", "region"], columns="anio",
                              values=OBJETIVO).reset_index()
    ancho["delta"] = ancho[2021] - ancho[2019]
    ancho["recuperacion"] = ancho[2023] - ancho[2019]

    por_region = (panel[panel["anio"].isin([2019, 2021])]
                  .pivot_table(index="region", columns="anio", values=OBJETIVO,
                               aggfunc=lambda s: np.average(s)))
    por_region["delta"] = por_region[2021] - por_region[2019]
    por_region = por_region.sort_values("delta")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                            gridspec_kw={"width_ratios": [1, 1.05]})
    # Panel izquierdo: polaridad (pérdida vs ganancia) -> paleta divergente
    ax = axes[0]
    colores = [RAMPA_DIVERGENTE[-2] if v < 0 else RAMPA_DIVERGENTE[1]
               for v in por_region["delta"]]
    y = np.arange(len(por_region))
    ax.barh(y, por_region["delta"], color=colores, height=0.66)
    ax.set_yticks(y, [CORTO.get(r, r) for r in por_region.index], fontsize=9)
    ax.axvline(0, color=COLORES["tinta_sec"], linewidth=1)
    for yi, v in zip(y, por_region["delta"]):
        ax.text(v + (-0.08 if v < 0 else 0.08), yi, f"{v:+.1f}", va="center",
                ha="right" if v < 0 else "left", fontsize=8.5, color=COLORES["tinta_sec"])
    ax.set_xlim(por_region["delta"].min() - 0.9, 0.6)
    titular(ax, "Ninguna región salió indemne", "Cambio en años, 2019 a 2021 (promedio simple de países)")
    ax.set_xlabel("Años de esperanza de vida")

    ax = axes[1]
    peores = ancho.nsmallest(15, "delta").sort_values("delta", ascending=False)
    y = np.arange(len(peores))
    colores = [COLORES["acento"] if i == PAIS_FOCO else RAMPA_DIVERGENTE[-2]
               for i in peores["iso3"]]
    ax.barh(y, peores["delta"], color=colores, height=0.66)
    ax.set_yticks(y, [es(p) for p in peores["pais"]], fontsize=9)
    ax.axvline(0, color=COLORES["tinta_sec"], linewidth=1)
    for yi, (v, iso) in enumerate(zip(peores["delta"], peores["iso3"])):
        ax.text(v - 0.15, yi, f"{v:+.1f}", va="center", ha="right", fontsize=8.5,
                color=COLORES["tinta"] if iso == PAIS_FOCO else COLORES["tinta_sec"],
                fontweight="semibold" if iso == PAIS_FOCO else "normal")
    ax.set_xlim(peores["delta"].min() - 1.6, 0.3)
    rank_peru = int((ancho["delta"] < float(ancho.loc[ancho["iso3"] == PAIS_FOCO, "delta"].iloc[0])).sum()) + 1
    titular(ax, f"Perú fue el caso nº {rank_peru} del mundo",
            "Los 15 países con mayor pérdida de esperanza de vida, 2019 a 2021")
    ax.set_xlabel("Años de esperanza de vida")
    fig.subplots_adjust(wspace=0.42)
    pie_fuente(fig)
    guardar_fig(fig, "fig10_covid")

    fila_peru = ancho[ancho["iso3"] == PAIS_FOCO].iloc[0]
    guardar(SEC, "covid", {
        "delta_peru": float(fila_peru["delta"]),
        "ranking_peru": rank_peru,
        "n_paises": int(len(ancho)),
        "recuperacion_peru_2023": float(fila_peru["recuperacion"]),
        "n_paises_con_caida": int((ancho["delta"] < 0).sum()),
        "delta_mediano": float(ancho["delta"].median()),
        "peores5": [{"pais": es(r["pais"]), "delta": round(float(r["delta"]), 1)}
                    for _, r in ancho.nsmallest(5, "delta").iterrows()],
        "por_region": {CORTO.get(r, r): round(float(v), 2)
                       for r, v in por_region["delta"].items()},
        "n_sin_recuperar_2023": int((ancho["recuperacion"] < 0).sum()),
    })


def fig11_clusters(ult: pd.DataFrame) -> None:
    cols = PREDICTORES + [OBJETIVO]
    d = ult.dropna(subset=cols).copy()
    X = StandardScaler().fit_transform(d[cols])

    siluetas = {}
    for k in range(2, 9):
        etiqueta = KMeans(n_clusters=k, n_init=25, random_state=42).fit_predict(X)
        siluetas[k] = float(silhouette_score(X, etiqueta))
    # k=2 sólo separa "ricos vs pobres": se elige la mejor solución no binaria
    K = max((k for k in siluetas if k >= 3), key=lambda k: siluetas[k])
    modelo = KMeans(n_clusters=K, n_init=50, random_state=42).fit(X)
    d["cluster"] = modelo.labels_

    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(X)
    d["pc1"], d["pc2"] = comp[:, 0], comp[:, 1]

    # Los clústeres se ordenan y nombran por su esperanza de vida media
    orden = d.groupby("cluster")[OBJETIVO].mean().sort_values(ascending=False).index
    nombres = ["Desarrollo alto", "Transición intermedia", "Rezago sanitario",
               "Grupo 4", "Grupo 5"][:K]
    mapa_nombre = {c: nombres[i] for i, c in enumerate(orden)}
    d["perfil"] = d["cluster"].map(mapa_nombre)
    colores = {nombres[i]: PALETA_SCATTER[i] for i in range(K)}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for perfil, g in d.groupby("perfil"):
        ax.scatter(g["pc1"], g["pc2"], s=34, color=colores[perfil], alpha=0.75,
                   edgecolor=COLORES["superficie"], linewidth=0.7, label=perfil)
    fila = d[d["iso3"] == PAIS_FOCO].iloc[0]
    ax.scatter([fila["pc1"]], [fila["pc2"]], s=150, facecolor="none",
               edgecolor=COLORES["tinta"], linewidth=2, zorder=5)
    ax.annotate("Perú", xy=(fila["pc1"], fila["pc2"]), xytext=(12, 12),
                textcoords="offset points", fontsize=9.5, color=COLORES["tinta"],
                fontweight="semibold")
    manejadores, etiquetas_leyenda = ax.get_legend_handles_labels()
    por_nombre = dict(zip(etiquetas_leyenda, manejadores))
    ax.legend([por_nombre[n] for n in nombres], nombres, loc="upper left", fontsize=9)
    var = pca.explained_variance_ratio_
    titular(ax, f"Los países se agrupan en {K} perfiles de desarrollo-salud",
            f"Proyección PCA ({100*var.sum():.0f}% de la varianza explicada)")
    ax.set_xlabel(f"Componente 1 ({100*var[0]:.0f}%)")
    ax.set_ylabel(f"Componente 2 ({100*var[1]:.0f}%)")

    ax = axes[1]
    ks, vals = list(siluetas), list(siluetas.values())
    ax.plot(ks, vals, color=COLORES["neutro"] if "neutro" in COLORES else COLORES["tinta_tenue"],
            linewidth=2, marker="o", markersize=6)
    ax.scatter([K], [siluetas[K]], s=140, color=COLORES["acento"], zorder=5)
    ax.annotate(f"k = {K} elegido\nsilueta {siluetas[K]:.2f}", xy=(K, siluetas[K]),
                xytext=(14, 10), textcoords="offset points", fontsize=9,
                color=COLORES["tinta"])
    ax.annotate("k = 2 sólo separa\nricos de pobres", xy=(2, siluetas[2]),
                xytext=(16, -6), textcoords="offset points", fontsize=8.5,
                color=COLORES["tinta_sec"])
    ax.set_xlabel("Número de clústeres (k)")
    ax.set_ylabel("Coeficiente de silueta")
    titular(ax, "Elección de k", "Silueta media por número de clústeres")
    fig.subplots_adjust(wspace=0.26)
    pie_fuente(fig)
    guardar_fig(fig, "fig11_clusters")

    perfiles = []
    for perfil, g in d.groupby("perfil"):
        perfiles.append({
            "perfil": perfil, "n_paises": int(len(g)),
            "esperanza_vida": round(float(g[OBJETIVO].mean()), 1),
            "pib_pc": int(g["pib_pc"].mean()),
            "gasto_salud_pc": int(g["gasto_salud_pc"].mean()),
            "medicos_1000": round(float(g["medicos_1000"].mean()), 2),
            "saneamiento": round(float(g["saneamiento"].mean()), 1),
            "fertilidad": round(float(g["fertilidad"].mean()), 2),
            "region_dominante": CORTO.get(g["region"].mode().iloc[0], g["region"].mode().iloc[0]),
            "ejemplos": ", ".join(es(p) for p in g.nlargest(3, "poblacion")["pais"]),
        })
    perfiles.sort(key=lambda p: -p["esperanza_vida"])
    guardar(SEC, "clusters", perfiles)
    guardar(SEC, "k_elegido", K)
    guardar(SEC, "siluetas", {str(k): round(v, 3) for k, v in siluetas.items()})
    guardar(SEC, "varianza_pca", round(float(var.sum() * 100), 1))
    guardar(SEC, "cluster_peru", fila_perfil := str(d.loc[d["iso3"] == PAIS_FOCO, "perfil"].iloc[0]))
    lac = d[d["region"] == REGION_FOCO]["perfil"].value_counts()
    guardar(SEC, "clusters_lac", {str(k): int(v) for k, v in lac.items()})
    print(f"  k elegido = {K} (silueta {siluetas[K]:.3f}); Perú cae en '{fila_perfil}'")


def main() -> None:
    aplicar_estilo()
    limpiar_seccion(SEC)
    panel = pd.read_csv(DIR_PROC / "panel.csv")
    anio = int(panel["anio"].max())
    ult = panel[panel["anio"] == anio].copy()
    print(f"Análisis diagnóstico — corte {anio}\n")
    fig8_correlaciones(ult)
    fig9_preston(ult)
    tabla_vif(ult)
    fig10_covid(panel)
    fig11_clusters(ult)
    print("\nDiagnóstico completo.")


if __name__ == "__main__":
    main()
