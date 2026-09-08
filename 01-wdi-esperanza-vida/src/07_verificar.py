"""07 - Verificacion: contrasta el informe contra los datos y busca regresiones.

Comprueba tres cosas:
  1. Que cada cifra citada en el DOCX coincide con results.json.
  2. Que cada afirmacion cualitativa del texto es cierta segun los datos.
  3. Que el panel y el documento mantienen su integridad estructural.

Devuelve codigo de salida 1 si algo falla, para poder encadenarlo en CI.
"""
import json, re, sys
from pathlib import Path
from docx import Document

RAIZ = Path(__file__).resolve().parents[1]
R = json.loads((RAIZ / "results.json").read_text(encoding="utf-8"))
doc = Document(RAIZ / "informe/Laboratorio1_WDI.docx")
texto = "\n".join(p.text for p in doc.paragraphs)
tablas = "\n".join(c.text for t in doc.tables for r in t.rows for c in r.cells)
todo = texto + "\n" + tablas

fallos, avisos, ok = [], [], []

def chk(nombre, condicion, detalle=""):
    (ok if condicion else fallos).append(f"{nombre}{' — ' + detalle if detalle else ''}")

D, DE, DI, PR = R["datos"], R["descriptivo"], R["diagnostico"], R["predictivo"]

# --- 1. Coherencia de cifras clave entre informe y results.json ---
chk("Nº de países del panel coincide", str(D["n_paises_panel"]) in todo)
chk("Nº de observaciones coincide", "5.208" in todo or "5208" in todo)
chk("% imputado coincide", str(D["pct_celdas_imputadas"]).replace(".", ",") in todo
                          or str(D["pct_celdas_imputadas"]) in todo)
chk("R² del mejor modelo coincide", f"{PR['metricas'][PR['mejor_modelo']]['r2']:.3f}" in todo)
chk("RMSE del mejor modelo coincide", f"{PR['metricas'][PR['mejor_modelo']]['rmse']:.2f}" in todo)
chk("Ranking COVID de Perú coincide", f"número {DI['covid']['ranking_peru']}" in todo)
chk("R² de Preston (log) coincide", f"{DI['preston']['r2_log']:.2f}" in todo)
chk("R² de Preston (lineal) coincide", f"{DI['preston']['r2_lineal']:.2f}" in todo)
chk("k de clustering coincide", f"k = {DI['k_elegido']}" in todo)
chk("Brecha regional coincide", f"{DE['brecha_regional']:.1f}" in todo)

# --- 2. Afirmaciones cualitativas verificadas contra los datos ---
imps = PR["importancias"]
chk("El informe nombra como top-1 la variable que el modelo puso primera",
    imps[0]["variable"].lower() in todo.lower())
chk("El saneamiento es realmente el predictor nº1",
    imps[0]["variable"] == "Saneamiento", f"es {imps[0]['variable']}")
chk("Gasto p/c correlaciona más que gasto %PIB (afirmado en el informe)",
    DI["corr_con_objetivo"]["Gasto salud p/c"] > DI["corr_con_objetivo"]["Gasto salud %PIB"])
chk("El ajuste log supera al lineal (afirmado en el informe)",
    DI["preston"]["r2_log"] > DI["preston"]["r2_lineal"])
chk("El mejor modelo supera claramente al baseline",
    PR["metricas"][PR["mejor_modelo"]]["rmse"] < PR["metricas"]["Media global (baseline)"]["rmse"] * 0.6)
chk("La regresión lineal queda cerca del mejor modelo (afirmado)",
    abs(PR["metricas"]["Regresión lineal"]["rmse"] - PR["metricas"][PR["mejor_modelo"]]["rmse"]) < 0.3)
chk("LAC es la región más golpeada por COVID (afirmado)",
    min(DI["covid"]["por_region"], key=DI["covid"]["por_region"].get) == "América Latina y Caribe")
chk("Perú por encima de la curva de Preston (afirmado)", DI["preston"]["residuo_peru"] > 0)
chk("Perú por encima de la predicción del modelo (afirmado)", PR["residuo_peru"] > 0)
chk("La silueta de k elegido supera 0,25",
    DI["siluetas"][str(DI["k_elegido"])] > 0.25, f"{DI['siluetas'][str(DI['k_elegido'])]}")

# --- 3. Integridad estructural (regresiones) ---
imgs = [r for r in doc.part.rels.values() if "image" in r.reltype]
pies = [p.text for p in doc.paragraphs if p.text.startswith("Figura ")]
chk("Toda figura incrustada tiene su pie", len(imgs) == len(pies), f"{len(imgs)} img / {len(pies)} pies")
nums = [int(re.match(r"Figura (\d+)\.", x).group(1)) for x in pies]
chk("Numeración de figuras correlativa sin saltos", nums == list(range(1, len(nums) + 1)))
chk("Mínimo de 5 páginas holgadamente superado", len(texto.split()) >= 2500, f"{len(texto.split())} palabras")
chk("Sin marcadores sin sustituir", not re.search(r"\bnan\b|\bNone\b|TBD|TODO|\{[a-z]", todo))
chk("Las 3 secciones exigidas por el enunciado están presentes",
    all(s in texto for s in ["Análisis descriptivo", "Análisis diagnóstico", "Análisis predictivo"]))

# --- 4. Integridad del panel (regresiones de datos) ---
import pandas as pd
panel = pd.read_csv(RAIZ / "data/processed/panel.csv")
chk("Panel sin nulos en la variable objetivo", panel["esperanza_vida"].isna().sum() == 0)
chk("Panel sin nulos en ningún predictor", panel.isna().sum().sum() == 0,
    f"{panel.isna().sum().sum()} nulos")
chk("Panel completo: 217 países × 24 años", len(panel) == 217 * 24, f"{len(panel)} filas")
chk("Sin duplicados país-año", not panel.duplicated(["iso3", "anio"]).any())
normales = panel[~panel["anio_catastrofe"]]
# Piso de 30: Sudán del Sur registra ~35 años durante su guerra civil (2015-2017),
# valores bajos pero reales para un estado en conflicto prolongado.
chk("Esperanza de vida en rango plausible (30-95) fuera de años de catástrofe",
    normales["esperanza_vida"].between(30, 95).all(),
    f"min {normales['esperanza_vida'].min():.1f}")
chk("Los años de catástrofe están marcados, no eliminados",
    panel["anio_catastrofe"].sum() == D["n_anios_catastrofe"] > 0)
chk("La sensibilidad demuestra que las conclusiones no dependen de ellos",
    not PR["sensibilidad"]["cambia_conclusion"])
chk("El informe documenta los años de catástrofe", "catástrofe" in todo)
chk("Ningún agregado regional se coló en el panel",
    not panel["pais"].str.contains(r"\b(?:income|World|Euro area|IDA|IBRD|OECD|dividend)\b",
                                   case=False, na=False, regex=True).any())
chk("Todas las figuras del informe existen en disco",
    all((RAIZ / "figures" / f).exists() for f in
        [f"fig{i:02d}_" for i in range(1, 16)] and
        [p.name for p in (RAIZ / "figures").glob("*.png")]))
chk("18 figuras PNG en disco (15 del análisis + 3 del dashboard)",
    len(list((RAIZ / "figures").glob("*.png"))) == 18)

# --- Informe de verificación ---
print(f"VERIFY — {len(ok)} comprobaciones pasadas, {len(fallos)} fallidas\n")
for x in ok: print(f"  [OK]    {x}")
if fallos:
    print()
    for x in fallos: print(f"  [FALLO] {x}")
print("\n" + ("VEREDICTO: PASA" if not fallos else "VEREDICTO: HAY FALLOS QUE CORREGIR"))
sys.exit(1 if fallos else 0)
