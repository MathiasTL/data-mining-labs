"""05 — Análisis predictivo.

(a) Modelo transversal: predecir la esperanza de vida de un país-año a partir de
    sus indicadores socioeconómicos, validado con GroupKFold por país para que
    ningún país aparezca simultáneamente en entrenamiento y prueba.
(b) Modelo temporal: proyección de la esperanza de vida de Perú y de América
    Latina hasta 2030, comparando la tendencia previa a la pandemia con la
    trayectoria observada después.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import (COLORES, DIR_PROC, OBJETIVO, PAIS_FOCO, PREDICTORES,
                    RAMPA_DIVERGENTE, REGION_FOCO, es)
from estilo import aplicar_estilo, etiquetas_finales, guardar_fig, pie_fuente, titular
from utils_resultados import guardar, limpiar_seccion

SEC = "predictivo"
CORTO_VAR = {
    "pib_pc": "PIB per cápita", "gasto_salud_pc": "Gasto salud p/c",
    "gasto_salud_pib": "Gasto salud %PIB", "medicos_1000": "Médicos /1.000",
    "fertilidad": "Fertilidad", "matricula_secundaria": "Matríc. secundaria",
    "inmunizacion": "Inmunización", "agua_potable": "Agua potable",
    "saneamiento": "Saneamiento", "urbanizacion": "Urbanización",
    "poblacion_65": "Población 65+",
}


def definir_modelos() -> dict:
    return {
        "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42),
    }


def evaluar(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame, str]:
    X = panel[PREDICTORES].values
    y = panel[OBJETIVO].values
    grupos = panel["iso3"].values
    cv = GroupKFold(n_splits=5)

    # Verificación explícita: ningún país cruza entre entrenamiento y prueba
    for tr, te in cv.split(X, y, grupos):
        assert not (set(grupos[tr]) & set(grupos[te])), "Fuga de países entre folds"

    resultados = {}
    predicciones = {}
    baseline = []
    for tr, te in cv.split(X, y, grupos):
        baseline.append(np.full(len(te), y[tr].mean()))
    idx_test = np.concatenate([te for _, te in cv.split(X, y, grupos)])
    y_test = y[idx_test]
    base = np.concatenate(baseline)
    resultados["Media global (baseline)"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, base))),
        "mae": float(mean_absolute_error(y_test, base)),
        "r2": float(r2_score(y_test, base)),
    }

    for nombre, modelo in definir_modelos().items():
        preds = np.zeros(len(y))
        for tr, te in cv.split(X, y, grupos):
            m = clone(modelo)   # instancia nueva por fold: nada se entrena dos veces
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        resultados[nombre] = {
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
        predicciones[nombre] = preds
        print(f"  {nombre:26s} RMSE {resultados[nombre]['rmse']:5.2f} | "
              f"MAE {resultados[nombre]['mae']:5.2f} | R² {resultados[nombre]['r2']:.3f}")

    mejor = min(predicciones, key=lambda n: resultados[n]["rmse"])
    salida = panel[["iso3", "pais", "region", "anio", OBJETIVO]].copy()
    salida["prediccion"] = predicciones[mejor]
    salida["residuo"] = salida[OBJETIVO] - salida["prediccion"]
    return resultados, salida, mejor


def fig12_ajuste(salida: pd.DataFrame, mejor: str, metricas: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(salida["prediccion"], salida[OBJETIVO], s=9, alpha=0.28,
               color=COLORES["principal"], edgecolor="none")
    lim = [min(salida["prediccion"].min(), salida[OBJETIVO].min()) - 1,
           max(salida["prediccion"].max(), salida[OBJETIVO].max()) + 1]
    ax.plot(lim, lim, color=COLORES["tinta"], linewidth=1.6, linestyle="--")
    ax.text(lim[1] - 0.5, lim[1] - 2.4, "Predicción perfecta", ha="right",
            fontsize=9, color=COLORES["tinta_sec"], rotation=0)
    peru = salida[salida["iso3"] == PAIS_FOCO]
    ax.scatter(peru["prediccion"], peru[OBJETIVO], s=30, color=COLORES["acento"],
               edgecolor=COLORES["superficie"], linewidth=0.5, zorder=4, label="Perú")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    m = metricas[mejor]
    ax.text(0.97, 0.06, f"{mejor}\nR² = {m['r2']:.3f}   RMSE = {m['rmse']:.2f} años",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=COLORES["tinta_sec"])
    titular(ax, "El modelo explica la mayor parte de la variación entre países",
            "Valores observados frente a predichos, validación cruzada por país")
    ax.set_xlabel("Esperanza de vida predicha (años)")
    ax.set_ylabel("Esperanza de vida observada (años)")
    pie_fuente(fig)
    guardar_fig(fig, "fig12_ajuste")


def fig13_importancias(panel: pd.DataFrame) -> None:
    X, y = panel[PREDICTORES].values, panel[OBJETIVO].values
    grupos = panel["iso3"].values
    tr, te = next(iter(GroupKFold(n_splits=5).split(X, y, grupos)))
    modelo = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                       max_depth=3, random_state=42).fit(X[tr], y[tr])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = permutation_importance(modelo, X[te], y[te], n_repeats=15,
                                     random_state=42, scoring="r2")
    d = (pd.DataFrame({"variable": [CORTO_VAR[c] for c in PREDICTORES],
                       "importancia": imp.importances_mean,
                       "sd": imp.importances_std})
         .sort_values("importancia"))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    y_pos = np.arange(len(d))
    ax.barh(y_pos, d["importancia"], xerr=d["sd"], height=0.66,
            color=COLORES["principal"],
            error_kw=dict(ecolor=COLORES["tinta_tenue"], lw=1, capsize=2))
    ax.set_yticks(y_pos, d["variable"], fontsize=9.5)
    margen = d["importancia"].max() * 0.035
    for yi, (v, sd) in enumerate(zip(d["importancia"], d["sd"])):
        ax.text(v + sd + margen, yi, f"{v:.3f}", va="center", fontsize=8.5,
                color=COLORES["tinta_sec"])
    ax.set_xlim(min(0, d["importancia"].min() - 0.02), d["importancia"].max() * 1.24)
    orden = d.sort_values("importancia", ascending=False)
    primera, segunda = orden.iloc[0], orden.iloc[1]
    veces = primera["importancia"] / segunda["importancia"]
    titular(ax, f"{primera['variable']} domina: {veces:.0f} veces el peso del siguiente factor",
            "Importancia por permutación: caída del R² al desordenar cada variable")
    ax.set_xlabel("Pérdida de R² al permutar la variable")
    pie_fuente(fig)
    guardar_fig(fig, "fig13_importancias")

    d = d.sort_values("importancia", ascending=False)
    guardar(SEC, "importancias", [{"variable": r["variable"],
                                   "importancia": round(float(r["importancia"]), 4)}
                                  for _, r in d.iterrows()])
    guardar(SEC, "top3_importancia", [r["variable"] for _, r in d.head(3).iterrows()])


def fig14_residuos(salida: pd.DataFrame, anio: int) -> None:
    ult = salida[salida["anio"] == anio].copy()
    ult["pais_es"] = ult["pais"].map(es)
    sobre = ult.nlargest(10, "residuo")
    bajo = ult.nsmallest(10, "residuo")
    sel = pd.concat([bajo.sort_values("residuo"), sobre.sort_values("residuo")])

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    y = np.arange(len(sel))
    colores = [RAMPA_DIVERGENTE[-2] if v < 0 else RAMPA_DIVERGENTE[1] for v in sel["residuo"]]
    ax.barh(y, sel["residuo"], color=colores, height=0.66)
    ax.set_yticks(y, sel["pais_es"], fontsize=9)
    ax.axvline(0, color=COLORES["tinta_sec"], linewidth=1)
    for yi, v in zip(y, sel["residuo"]):
        ax.text(v + (0.12 if v > 0 else -0.12), yi, f"{v:+.1f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8.5, color=COLORES["tinta_sec"])
    peru = ult[ult["iso3"] == PAIS_FOCO]
    r_peru = float(peru["residuo"].iloc[0])
    ax.text(0.98, 0.03,
            f"Perú: {r_peru:+.1f} años respecto a lo predicho",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=COLORES["tinta"])
    ax.set_xlim(sel["residuo"].min() - 1.4, sel["residuo"].max() + 1.4)
    titular(ax, "Quiénes viven más —y menos— de lo que sus recursos predicen",
            f"Residuos del modelo en {anio}: observado menos predicho")
    ax.set_xlabel("Años de diferencia respecto a la predicción")
    pie_fuente(fig)
    guardar_fig(fig, "fig14_residuos")

    guardar(SEC, "residuo_peru", r_peru)
    for nombre, df in (("residuos_positivos", sobre), ("residuos_negativos", bajo)):
        guardar(SEC, nombre, [{"pais": r["pais_es"], "residuo": round(float(r["residuo"]), 1)}
                              for _, r in df.iterrows()])


def fig15_proyeccion(panel: pd.DataFrame) -> None:
    def serie_de(df):
        return (df.groupby("anio")
                .apply(lambda g: np.average(g[OBJETIVO], weights=g["poblacion"]),
                       include_groups=False))

    peru = panel[panel["iso3"] == PAIS_FOCO].set_index("anio")[OBJETIVO]
    lac = serie_de(panel[panel["region"] == REGION_FOCO])
    horizonte = np.arange(2024, 2031)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5), sharey=False)
    resumen = {}
    for ax, (nombre, serie) in zip(axes, [("Perú", peru), ("América Latina y el Caribe", lac)]):
        # (i) Tendencia pre-pandemia: qué habría pasado sin COVID
        pre = serie.loc[2000:2019]
        b, a = np.polyfit(pre.index.values, pre.values, 1)
        contrafactual = a + b * np.arange(2000, 2031)

        # (ii) Suavizado exponencial de Holt. Los años 2020-2021 son un shock
        #      puntual documentado: ajustarlos como si fueran nivel deprimiría la
        #      proyección por debajo del último dato observado. Se interpolan para
        #      el ajuste y se declara en el gráfico.
        serie_ajuste = serie.copy()
        serie_ajuste.loc[2020:2021] = np.nan
        serie_ajuste = serie_ajuste.interpolate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holt = ExponentialSmoothing(serie_ajuste.values, trend="add",
                                        damped_trend=True,
                                        initialization_method="estimated").fit()
        proy = holt.forecast(len(horizonte))
        residuo_sd = float(np.std(serie_ajuste.values - holt.fittedvalues, ddof=1))
        banda = 1.96 * residuo_sd * np.sqrt(np.arange(1, len(horizonte) + 1))

        ax.plot(serie.index, serie.values, color=COLORES["principal"], linewidth=2.4,
                label="Observado")
        ax.plot(np.arange(2000, 2031), contrafactual, color=COLORES["tinta_tenue"],
                linewidth=1.8, linestyle="--", label="Tendencia 2000-2019 prolongada")
        ax.plot(np.r_[serie.index[-1], horizonte], np.r_[serie.values[-1], proy],
                color=COLORES["acento"], linewidth=2.4, label="Proyección Holt (IC 95%)")
        ax.fill_between(horizonte, proy - banda, proy + banda,
                        color=COLORES["acento"], alpha=0.14, linewidth=0)
        ax.axvspan(2019.5, 2021.5, color="#e6e5e1", alpha=0.6, zorder=0)
        ax.text(2020.5, ax.get_ylim()[0] + 0.25, "shock\nexcluido", ha="center",
                fontsize=7.5, color=COLORES["tinta_tenue"], linespacing=1.15)
        ax.legend(loc="upper left", fontsize=8.5)

        # Valores de 2030 rotulados directamente, sin que se pisen entre sí
        etiquetas_finales(ax, [
            (2030, float(proy[-1]), f"{proy[-1]:.1f} proyectado", True),
            (2030, float(contrafactual[-1]), f"{contrafactual[-1]:.1f} sin COVID", False),
        ], dx=0.5)
        ax.set_xlim(2000, 2037)
        titular(ax, nombre, "Observado, tendencia pre-pandemia y proyección a 2030")
        ax.set_xlabel("Año")
        ax.set_ylabel("Esperanza de vida (años)")

        resumen[nombre] = {
            "tendencia_anual_2000_2019": round(float(b), 3),
            "obs_2023": round(float(serie.loc[2023]), 2),
            "proyeccion_2030": round(float(proy[-1]), 2),
            "contrafactual_2030": round(float(contrafactual[-1]), 2),
            "brecha_vs_sin_covid": round(float(proy[-1] - contrafactual[-1]), 2),
            "banda_2030": round(float(banda[-1]), 2),
            "ganancia_2023_2030": round(float(proy[-1] - serie.loc[2023]), 2),
        }
    fig.subplots_adjust(wspace=0.24)
    pie_fuente(fig)
    guardar_fig(fig, "fig15_proyeccion")
    guardar(SEC, "proyeccion", resumen)


def main() -> None:
    aplicar_estilo()
    limpiar_seccion(SEC)
    panel = pd.read_csv(DIR_PROC / "panel.csv").dropna(subset=PREDICTORES + [OBJETIVO])
    anio = int(panel["anio"].max())
    print(f"Modelo transversal — {len(panel)} observaciones país-año, "
          f"{panel['iso3'].nunique()} países, {len(PREDICTORES)} predictores")
    print("Validación: GroupKFold(5) agrupando por país\n")

    metricas, salida, mejor = evaluar(panel)
    base = metricas["Media global (baseline)"]
    print(f"\n  Mejor modelo: {mejor} "
          f"(RMSE {metricas[mejor]['rmse']:.2f} vs baseline {base['rmse']:.2f})\n")

    guardar(SEC, "n_observaciones", int(len(panel)))
    guardar(SEC, "n_paises", int(panel["iso3"].nunique()))
    guardar(SEC, "n_predictores", len(PREDICTORES))
    guardar(SEC, "metricas", {k: {m: round(v, 3) for m, v in d.items()}
                              for k, d in metricas.items()})
    guardar(SEC, "mejor_modelo", mejor)
    guardar(SEC, "mejora_vs_baseline_rmse",
            round(100 * (1 - metricas[mejor]["rmse"] / base["rmse"]), 1))

    # Análisis de sensibilidad: ¿dependen las conclusiones de los 15 años de
    # catástrofe (terremoto de Haití, hambruna somalí, guerras)? Se reentrena el
    # mismo modelo sin ellos y se compara. No se eliminan del análisis principal:
    # son valores reales y borrarlos sería ocultar los peores años de esos países.
    if "anio_catastrofe" in panel.columns and panel["anio_catastrofe"].any():
        sin_cat = panel[~panel["anio_catastrofe"]]
        met_sc, _, mejor_sc = evaluar(sin_cat)
        m_con, m_sin = metricas[mejor], met_sc[mejor_sc]
        print(f"\n  Sensibilidad — excluyendo {int(panel['anio_catastrofe'].sum())} años "
              f"de catástrofe:")
        print(f"    con ellos:  RMSE {m_con['rmse']:.2f} | R² {m_con['r2']:.3f}")
        print(f"    sin ellos:  RMSE {m_sin['rmse']:.2f} | R² {m_sin['r2']:.3f}")
        guardar(SEC, "sensibilidad", {
            "n_excluidos": int(panel["anio_catastrofe"].sum()),
            "mejor_modelo_sin": mejor_sc,
            "rmse_con": round(m_con["rmse"], 3), "rmse_sin": round(m_sin["rmse"], 3),
            "r2_con": round(m_con["r2"], 3), "r2_sin": round(m_sin["r2"], 3),
            "cambia_conclusion": bool(abs(m_con["r2"] - m_sin["r2"]) > 0.05),
        })

    fig12_ajuste(salida, mejor, metricas)
    fig13_importancias(panel)
    fig14_residuos(salida, anio)
    fig15_proyeccion(pd.read_csv(DIR_PROC / "panel.csv"))
    print("\nPredictivo completo.")


if __name__ == "__main__":
    main()
