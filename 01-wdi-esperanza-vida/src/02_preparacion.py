"""02 — Construcción del panel país-año limpio.

Pasos: unir indicadores en formato ancho, adjuntar región y grupo de ingreso,
excluir los agregados regionales, filtrar países con poca cobertura del
objetivo, e imputar los huecos restantes documentando cuántos fueron.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import (ANIO_MAX, ANIO_MIN, COBERTURA_MINIMA, DIR_PROC, DIR_RAW,
                    ETIQUETAS, INDICADORES, OBJETIVO)
from utils_resultados import guardar

ALIASES = [v[0] for v in INDICADORES.values()]


def construir_panel_crudo() -> pd.DataFrame:
    panel = None
    for alias in ALIASES:
        df = pd.read_csv(DIR_RAW / f"{alias}.csv")[["iso3", "anio", "valor"]]
        df = df.rename(columns={"valor": alias})
        panel = df if panel is None else panel.merge(df, on=["iso3", "anio"], how="outer")

    paises = pd.read_csv(DIR_RAW / "paises.csv")
    panel = panel.merge(
        paises[["iso3", "pais", "region", "grupo_ingreso", "es_agregado"]],
        on="iso3", how="left",
    )
    # La API devuelve algunos nombres con espacios sobrantes ("Latin America & Caribbean ")
    for col in ("pais", "region", "grupo_ingreso"):
        panel[col] = panel[col].str.strip()
    return panel


def main() -> None:
    panel = construir_panel_crudo()
    print(f"Panel crudo (incluye agregados): {panel.shape[0]} filas × {panel.shape[1]} columnas")

    # 1. Quedarse solo con países soberanos y el período de análisis
    panel = panel[~panel["es_agregado"].fillna(True)].copy()
    panel = panel[panel["anio"].between(ANIO_MIN, ANIO_MAX)]
    panel = panel.drop(columns=["es_agregado"])
    print(f"Tras excluir agregados: {panel.shape[0]} filas, "
          f"{panel['iso3'].nunique()} países")

    # 2. Cobertura por indicador ANTES de imputar
    cobertura_antes = (panel[ALIASES].notna().mean() * 100).round(1)
    print("\nCobertura por indicador (% de celdas con dato):")
    for alias, pct in cobertura_antes.sort_values().items():
        print(f"  {pct:5.1f}%  {ETIQUETAS[alias]}")

    # 3. Filtrar países con poca cobertura del objetivo
    cob_pais = panel.groupby("iso3")[OBJETIVO].apply(lambda s: s.notna().mean())
    validos = cob_pais[cob_pais >= COBERTURA_MINIMA].index
    excluidos = sorted(set(cob_pais.index) - set(validos))
    panel = panel[panel["iso3"].isin(validos)].copy()
    print(f"\nPaíses excluidos por cobertura <{COBERTURA_MINIMA:.0%} del objetivo: "
          f"{len(excluidos)} ({', '.join(excluidos) if excluidos else 'ninguno'})")

    # 4. Imputación en dos niveles, contando cada valor rellenado
    faltantes_antes = panel[ALIASES].isna().sum()

    panel = panel.sort_values(["iso3", "anio"])
    # 4a. Interpolación lineal dentro de cada país (series lentas y suaves)
    panel[ALIASES] = (panel.groupby("iso3")[ALIASES]
                      .transform(lambda g: g.interpolate(limit_direction="both")))
    faltantes_tras_interp = panel[ALIASES].isna().sum()

    # 4b. Lo que sigue vacío (país sin ningún dato del indicador) toma la
    #     mediana de su región en ese año; como último recurso, la global.
    for alias in ALIASES:
        med_reg = panel.groupby(["region", "anio"])[alias].transform("median")
        panel[alias] = panel[alias].fillna(med_reg)
        panel[alias] = panel[alias].fillna(panel.groupby("anio")[alias].transform("median"))
    faltantes_despues = panel[ALIASES].isna().sum()

    detalle_imputacion = {
        alias: {
            "faltantes_originales": int(faltantes_antes[alias]),
            "pct_original": round(100 * faltantes_antes[alias] / len(panel), 1),
            "imputados_interpolacion": int(faltantes_antes[alias] - faltantes_tras_interp[alias]),
            "imputados_mediana_regional": int(faltantes_tras_interp[alias] - faltantes_despues[alias]),
            "sin_dato_final": int(faltantes_despues[alias]),
        }
        for alias in ALIASES
    }
    print("\nImputación (valores rellenados por método):")
    for alias, d in sorted(detalle_imputacion.items(),
                           key=lambda kv: -kv[1]["faltantes_originales"]):
        if d["faltantes_originales"]:
            print(f"  {alias:22s} faltaban {d['faltantes_originales']:5d} "
                  f"({d['pct_original']:4.1f}%) → interp {d['imputados_interpolacion']:5d}, "
                  f"mediana reg. {d['imputados_mediana_regional']:5d}, "
                  f"quedan {d['sin_dato_final']}")

    # 4c. Marcado de años de catástrofe. La esperanza de vida "de período" se
    #     desploma en el año de un desastre masivo y se recupera al siguiente
    #     (Haití 2010, Somalia 2011, Gaza 2023). Son valores reales, no errores,
    #     y no se eliminan: se marcan para poder hacer análisis de sensibilidad.
    #     El umbral de 10 años preserva caídas menores pero reales como la del
    #     COVID (máximo -6,4) o Siria 2012 (-6,1).
    UMBRAL_SALTO = 10
    panel = panel.sort_values(["iso3", "anio"])
    salto = panel.groupby("iso3")[OBJETIVO].diff().abs()
    panel["anio_catastrofe"] = (salto > UMBRAL_SALTO).fillna(False)
    catastrofes = (panel[panel["anio_catastrofe"]][["pais", "anio", OBJETIVO]]
                   .sort_values("anio"))
    print(f"\nAños de catástrofe marcados (salto interanual > {UMBRAL_SALTO} años): "
          f"{len(catastrofes)} en {panel[panel['anio_catastrofe']]['iso3'].nunique()} países")
    for _, r in catastrofes.iterrows():
        print(f"  {r['pais']:28s} {int(r['anio'])}  ->  {r[OBJETIVO]:.1f} años")
    guardar("datos", "n_anios_catastrofe", int(panel["anio_catastrofe"].sum()))
    guardar("datos", "paises_catastrofe",
            sorted(panel[panel["anio_catastrofe"]]["pais"].unique().tolist()))
    guardar("datos", "detalle_catastrofes",
            [{"pais": r["pais"], "anio": int(r["anio"]),
              "esperanza_vida": round(float(r[OBJETIVO]), 1)}
             for _, r in catastrofes.iterrows()])

    # 5. Variables derivadas para el análisis
    panel["log_pib_pc"] = np.log10(panel["pib_pc"].where(panel["pib_pc"] > 0))
    panel["log_gasto_salud_pc"] = np.log10(panel["gasto_salud_pc"].where(panel["gasto_salud_pc"] > 0))

    panel = panel[["iso3", "pais", "region", "grupo_ingreso", "anio"]
                  + ALIASES + ["log_pib_pc", "log_gasto_salud_pc", "anio_catastrofe"]]
    panel.to_csv(DIR_PROC / "panel.csv", index=False)

    # 6. Verificaciones
    assert panel[OBJETIVO].isna().sum() == 0, "Quedan nulos en la variable objetivo"
    assert panel["iso3"].nunique() >= 180, f"Muy pocos países: {panel['iso3'].nunique()}"
    assert len(panel) >= 4000, f"Muy pocas filas: {len(panel)}"

    anio_ultimo = int(panel.loc[panel[OBJETIVO].notna(), "anio"].max())
    guardar("datos", "n_paises_panel", int(panel["iso3"].nunique()))
    guardar("datos", "n_filas_panel", int(len(panel)))
    guardar("datos", "n_regiones", int(panel["region"].nunique()))
    guardar("datos", "anio_ultimo", anio_ultimo)
    guardar("datos", "paises_excluidos", excluidos)
    guardar("datos", "cobertura_antes", cobertura_antes.to_dict())
    guardar("datos", "imputacion", detalle_imputacion)
    guardar("datos", "pct_celdas_imputadas",
            round(100 * faltantes_antes.sum() / (len(panel) * len(ALIASES)), 1))

    print(f"\npanel.csv guardado: {panel.shape[0]} filas × {panel.shape[1]} columnas | "
          f"{panel['iso3'].nunique()} países | {panel['region'].nunique()} regiones | "
          f"último año con dato: {anio_ultimo}")


if __name__ == "__main__":
    main()
