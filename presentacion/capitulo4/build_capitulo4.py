#!/usr/bin/env python3
"""Build Capitulo4_Modelamiento.pptx (slide-maker deckkit).

Reproduce:  python3 presentacion/capitulo4/build_capitulo4.py
Then render: see presentacion/capitulo4/qa/ (soffice -> PDF -> pdftoppm).

Sources of truth: development/04-modelado.md, development/05-evaluacion.md,
docs/referencias-tecnicas-modelado.md, development/01-entendimiento-empresarial.md,
results/modeling/v2-run1/ (weekly_gini.csv is read directly for the weekly chart).
Language: neutral professional Spanish (no voseo). Numbers use decimal comma.
"""
import csv
import os
import sys
from pathlib import Path

SKILL = os.environ.get(
    "SLIDE_MAKER_DIR",
    "/Users/mathiastl/Library/Application Support/orca/claude-profiles/"
    "a125f7e0-2f5e-4a6c-b29c-eb142e87c93a/home/skills/slide-maker",
)
sys.path.insert(0, str(Path(SKILL) / "scripts"))
import deckkit as dk  # noqa: E402
from deckkit import RGBColor, PP_ALIGN, MSO_ANCHOR  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUN = REPO / "results" / "modeling" / "v2-run1"
OUT = ROOT / "Capitulo4_Modelamiento.pptx"

# ----------------------------------------------------------------- style
HEAD = "Times New Roman"  # metric-compatible with Liberation Serif, so the PDF matches PowerPoint
BODY = "Arial"
dk.FONT = BODY
BG = "F4F8F7"      # page
PANEL = "E1EEED"   # light teal panel
INK = "0A3640"     # deep teal (dark ground, titles)
TEAL = "13707A"    # LightGBM / primary
TEAL_M = "7FB0B4"  # logistic regression
GREY_C = "C3CFD1"  # constant / neutral
AMBER = "A15F00"   # validation windows / partial
AMBER_T = "8A5A00"  # amber text on light ground (>=4.5:1)
BRICK = "A9402A"   # not met
TEXT = "1F2D31"
MUTED = "56686D"
WHITE = "FFFFFF"
NBSP = " "


def C(h):
    return RGBColor.from_string(h)


def T3(h):
    return RGBColor.from_string(h)


def n(x, d=3):
    """Spanish number format: decimal comma."""
    return f"{x:.{d}f}".replace(".", ",")


def th(x):
    """Thousands with a non-breaking space."""
    return f"{x:,}".replace(",", NBSP)


W, H = 13.333, 7.5
ML = 0.6
CW = W - 2 * ML

SECTIONS = {
    "4.1": "SELECCIÓN DE TÉCNICAS",
    "4.2": "PLAN DE PRUEBAS",
    "4.3": "CONSTRUCCIÓN DE MODELOS",
    "4.4": "EVALUACIÓN",
}


def R(t, size, color=TEXT, bold=False, italic=False, font=BODY):
    t = t.replace(" %", NBSP + "%")
    return (t, size, C(color), bold, italic, font)


def tx(s, x, y, w, h, t, size=14, color=TEXT, bold=False, font=BODY, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, italic=False, ls=None, after=0):
    t = t.replace(" %", NBSP + "%")
    if font == HEAD and any(ch.isdigit() for ch in t) and len(t) <= 10:
        font = BODY  # Georgia has old-style figures: numerals go to a lining face
    dk.text(s, x, y, w, h, [[R(t, size, color, bold, italic, font)]], align=align, anchor=anchor,
            space_after=after, line_spacing=ls)


def paras(s, x, y, w, h, ps, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, after=6, ls=None):
    dk.text(s, x, y, w, h, ps, align=align, anchor=anchor, space_after=after, line_spacing=ls)


def new_slide(prs, ground=BG):
    s = dk.add_slide(prs)
    dk.slide_background(s, C(ground))
    return s


def rail(s, sec, dark=False):
    """Rubric rail: 4.1-4.4 tracker, the current section filled. Present on every content slide."""
    order = list(SECTIONS)
    segw, gap, y = 1.05, 0.12, 0.5
    x0 = W - ML - (segw * 4 + gap * 3)
    for i, k in enumerate(order):
        x = x0 + i * (segw + gap)
        cur = k == sec
        seg = dk.box(s, x, y, segw, 0.09, fill=C(TEAL if cur else "C9D8D8"))
        if not cur:
            dk.decorative(seg, "tracker segment; the section label above it carries the meaning")
        tx(s, x, y - 0.3, segw, 0.26, k, 11, TEAL if cur else MUTED, cur, align=PP_ALIGN.CENTER)


def chrome(s, prs_idx, sec, title, total=None):
    rail(s, sec)
    tx(s, ML, 0.26, 7.5, 0.3, f"{sec}  ·  {SECTIONS[sec]}", 12, TEAL, True)
    tx(s, ML, 0.72, CW, 1.0, title, 25, INK, True, font=HEAD, anchor=MSO_ANCHOR.TOP, ls=1.05)
    foot = "Segundo avance · Capítulo 4: Modelamiento · UNMSM · FISI"
    if sec in ("4.2", "4.3", "4.4"):
        foot = "Fuente: results/modeling/v2-run1 y capítulos 4 y 5  ·  " + foot
    tx(s, ML, H - 0.42, 11, 0.28, foot, 10, MUTED)
    tx(s, W - ML - 1.0, H - 0.42, 1.0, 0.28, str(prs_idx), 10, MUTED, align=PP_ALIGN.RIGHT)
    dk.a11y_title(s, title)


def card(s, x, y, w, h, fill=PANEL, line=None):
    return dk.box(s, x, y, w, h, fill=C(fill), line=C(line) if line else None)


def chip_status(s, x, y, w, label, color):
    dk.box(s, x, y, w, 0.34, fill=C(color), round=True)
    tx(s, x, y, w, 0.34, label, 12, WHITE, True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def table(s, x, y, w, rows, col_w, size=12, row_h=0.4, highlight=None, numeric=None, colors=None,
          bold_cells=None):
    """dk.table + per-cell colouring. colors = {(r, c): hex} using row indices incl. header."""
    rows = [[str(c).replace(" %", NBSP + "%") for c in r] for r in rows]
    bottom = dk.table(s, x, y, w, rows, col_w=col_w, size=size, row_h=row_h, highlight=highlight,
                      numeric_cols=numeric, head_c=T3(INK), body_c=T3(TEXT), rule_c=T3("9FB4B6"),
                      hi_fill=T3(PANEL), hi_c=T3(TEAL), font=BODY)
    shp = s.shapes[-1]
    tbl = shp.table
    for (r, c), col in (colors or {}).items():
        for p in tbl.cell(r, c).text_frame.paragraphs:
            for run in p.runs:
                run.font.color.rgb = C(col)
                run.font.bold = True
    for (r, c) in (bold_cells or []):
        for p in tbl.cell(r, c).text_frame.paragraphs:
            for run in p.runs:
                run.font.bold = True
    return bottom


def style_bars(ch, colors, labels, size=12, gap=55, point_colors=None, label_color=TEXT):
    from pptx.enum.chart import XL_LABEL_POSITION
    from pptx.util import Pt
    plot = ch.plots[0]
    plot.gap_width = gap
    plot.vary_by_categories = False
    for i, ser in enumerate(ch.series):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = C(colors[i])
        ser.format.line.fill.background()
        if point_colors and i in point_colors:
            for j, col in point_colors[i].items():
                pt = ser.points[j]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = C(col)
                pt.format.line.fill.background()
        for j, lab in enumerate(labels[i]):
            dl = ser.points[j].data_label
            tf = dl.text_frame
            tf.text = lab
            r = tf.paragraphs[0].runs[0]
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.name = BODY
            r.font.color.rgb = C(label_color)
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    ch.category_axis.tick_labels.font.size = Pt(12)
    ch.category_axis.tick_labels.font.name = BODY
    ch.category_axis.format.line.color.rgb = C("9FB4B6")


def legend_manual(s, x, y, items, size=12, color=TEXT):
    """items = [(hex, label)]; returns right x."""
    cx = x
    for col, lab in items:
        dk.box(s, cx, y + 0.07, 0.18, 0.18, fill=C(col))
        wlab = 0.12 * len(lab) * size / 12 + 0.2
        tx(s, cx + 0.26, y, wlab, 0.3, lab, size, color)
        cx += 0.26 + wlab + 0.2
    return cx


def notes(s, t):
    dk.speaker_notes(s, t)


# ----------------------------------------------------------------- data
def weekly_oot():
    out = {}
    for r in csv.DictReader(open(RUN / "weekly_gini.csv")):
        if r["stage"] == "final" and r["period"] == "OOT":
            out.setdefault(r["model"], {})[int(r["WEEK_NUM"])] = float(r["gini"])
    return out


WK = weekly_oot()
WEEKS = sorted(WK["lightgbm"])
assert WEEKS == list(range(74, 92)), WEEKS
assert all(WK["lightgbm"][w] > WK["logistic"][w] for w in WEEKS), "LightGBM must win 18/18"
LGB_MEAN = sum(WK["lightgbm"].values()) / 18
LR_MEAN = sum(WK["logistic"].values()) / 18
assert n(LGB_MEAN) == "0,503" and n(LR_MEAN) == "0,460", (LGB_MEAN, LR_MEAN)

prs = dk.blank_deck(W, H)
IDX = [0]


def slide(sec=None, title=None, ground=BG):
    IDX[0] += 1
    s = new_slide(prs, ground)
    if sec:
        chrome(s, IDX[0], sec, title)
    return s


# ================================================================= 1. COVER
def s_cover():
    s = slide(ground=INK)
    dk.box(s, 0, 0, 0.22, H, fill=C(TEAL))
    tx(s, 0.9, 0.55, 8.2, 0.34, "UNIVERSIDAD NACIONAL MAYOR DE SAN MARCOS", 15, "9FD0D3", True)
    tx(s, 0.9, 0.92, 8.2, 0.34, "Facultad de Ingeniería de Sistemas e Informática", 14, "D5E8E9")
    ttl = "Riesgo crediticio y estabilidad temporal de un modelo de predicción de incumplimiento"
    th_ = dk.measure_text([(ttl, True)], 7.3, 32, font=HEAD, line_spacing=1.08) + 0.3
    tx(s, 0.9, 1.6, 7.3, th_, ttl, 32, WHITE, True, font=HEAD, ls=1.08)
    ey = 1.6 + th_ + 0.75
    tx(s, 0.9, ey, 7.7, 0.35, "Datos de Home Credit  ·  Credit Risk Model Stability", 15, "9FD0D3")
    tx(s, 0.9, ey + 0.45, 7.7, 0.4, "Segundo avance del proyecto integrador  ·  Capítulo 4: Modelamiento",
       16, WHITE, True)
    tx(s, 0.9, ey + 0.88, 7.7, 0.32, "CRISP-DM, fase 4", 14, "D5E8E9")
    paras(s, 0.9, 5.65, 4.3, 1.1, [
        [R("DOCENTE", 11, "9FD0D3", True)],
        [R("Pedro Martin Lezama Gonzales", 14, WHITE)],
        [R("Lima, Perú  ·  2026", 13, "D5E8E9")],
    ], after=3)
    paras(s, 5.3, 5.65, 4.4, 1.5, [
        [R("INTEGRANTES", 11, "9FD0D3", True)],
        [R("Nick Salcedo  ·  Gabriel Davalos", 13, WHITE)],
        [R("Andre Cuenca  ·  Giancarlo Villavicencio", 13, WHITE)],
        [R("Mathias Torres", 13, WHITE)],
    ], after=3)
    # Signature motif: the expanding-window validation scheme (P1, P2, P3, OOT).
    x0, wtot, y0 = 8.95, 3.75, 1.35
    tx(s, x0, y0 - 0.5, wtot, 0.3, "Ventanas de validación fuera de tiempo", 12, "9FD0D3", True)
    sc = wtot / 92
    rows = [(40, 10, 0), (50, 10, 1), (60, 14, 2), (60, 18, 3)]  # fit end, val len, row
    gap_extra = [0, 0, 0, 14]
    for fit_end, vlen, i in rows:
        y = y0 + i * 0.62
        dk.box(s, x0, y, fit_end * sc, 0.38, fill=C(TEAL))
        vs = fit_end + gap_extra[i]
        dk.box(s, x0 + vs * sc, y, vlen * sc, 0.38, fill=C(AMBER if i < 3 else "F2D58A"))
    labs = ["P1", "P2", "P3", "OOT"]
    for i, l in enumerate(labs):
        tx(s, x0 - 0.55, y0 + i * 0.62, 0.5, 0.38, l, 12, "D5E8E9", True, anchor=MSO_ANCHOR.MIDDLE,
           align=PP_ALIGN.RIGHT)
    tx(s, x0, y0 + 2.2, 1.2, 0.26, "semana 0", 11, "9FD0D3")
    tx(s, x0 + wtot - 1.2, y0 + 2.2, 1.2, 0.26, "semana 91", 11, "9FD0D3", align=PP_ALIGN.RIGHT)
    legend_manual(s, x0, y0 + 2.6, [(TEAL, "ajuste"), (AMBER, "validación")], 12, color="D5E8E9")
    notes(s, "Buenas tardes. Presentamos el segundo avance de nuestro proyecto integrador, "
             "correspondiente al capítulo 4, Modelamiento, la cuarta fase de CRISP-DM. "
             "Trabajamos con los datos de Home Credit y estudiamos cómo se mantiene en el tiempo la "
             "capacidad de un modelo para predecir el incumplimiento de pago. "
             "Seguiremos el orden de la rúbrica: seleccionar técnicas, diseñar las pruebas, construir "
             "los modelos y evaluarlos.")


# ================================================================= 2. CRISP-DM + AGENDA
def s_crisp():
    s = slide()
    rail(s, None)  # neutral on the agenda: nothing selected yet
    tx(s, ML, 0.26, 7.5, 0.3, "UBICACIÓN EN CRISP-DM", 12, TEAL, True)
    tx(s, ML, 0.72, CW, 0.9,
       "El capítulo 4 es la fase de Modelado y se organiza en cuatro tareas",
       25, INK, True, font=HEAD, ls=1.05)
    dk.a11y_title(s, "El capítulo 4 corresponde a la fase de Modelado")
    phases = ["Comprensión del negocio", "Comprensión de los datos", "Preparación de los datos",
              "Modelado", "Evaluación", "Despliegue"]
    gap = 0.12
    pw = (CW - gap * 5) / 6
    y = 2.0
    for i, p in enumerate(phases):
        x = ML + i * (pw + gap)
        cur = i == 3
        dk.box(s, x, y, pw, 1.05, fill=C(INK if cur else "DCE8E7"))
        tx(s, x + 0.1, y + 0.08, pw - 0.2, 0.3, f"FASE {i + 1}", 11, "9FD0D3" if cur else MUTED, True)
        tx(s, x + 0.1, y + 0.4, pw - 0.2, 0.6, p, 13, WHITE if cur else TEXT, cur)
    tx(s, ML + 3 * (pw + gap), y + 1.1, pw, 0.3, "estamos aquí", 12, TEAL, True, align=PP_ALIGN.CENTER)
    items = [
        ("4.1", "Selección de técnicas", "Casos reales, artículos Q1/Q2 y modelos ordenados por complejidad"),
        ("4.2", "Plan de pruebas", "Particiones, reglas de selección, métricas y controles de validez"),
        ("4.3", "Construcción de modelos", "Registro de experimentos, hiperparámetros y quitar variables"),
        ("4.4", "Evaluación de modelos", "Resultados fuera de tiempo, objetivos y próximos pasos"),
    ]
    tx(s, ML, 3.5, 6, 0.3, "LAS CUATRO TAREAS DEL CAPÍTULO", 12, TEAL, True)
    cy, ch_ = 3.85, 2.55
    cw_ = (CW - gap * 3 * 2) / 4
    for i, (k, t, d) in enumerate(items):
        x = ML + i * (cw_ + gap * 2)
        card(s, x, cy, cw_, ch_)
        dk.box(s, x, cy, cw_, 0.09, fill=C(TEAL))
        tx(s, x + 0.2, cy + 0.25, cw_ - 0.4, 0.6, k, 34, TEAL, True, font=HEAD)
        tx(s, x + 0.2, cy + 0.95, cw_ - 0.4, 0.6, t, 16, INK, True)
        tx(s, x + 0.2, cy + 1.55, cw_ - 0.4, 0.9, d, 13, MUTED)
    tx(s, ML, 6.55, CW, 0.3, "El indicador 4.1 a 4.4, arriba a la derecha de cada diapositiva, marca la tarea en curso.",
       12, MUTED, italic=True)
    notes(s, "CRISP-DM tiene seis fases. Este capítulo corresponde a la cuarta, Modelado. "
             "Lo desarrollamos en cuatro tareas que coinciden con la rúbrica de la sesión: "
             "4.1 selección de técnicas, 4.2 plan de pruebas, 4.3 construcción de los modelos y "
             "4.4 evaluación. En la parte superior de cada diapositiva verán un indicador que muestra "
             "en qué tarea estamos.")


# ================================================================= 3. PROBLEM + OBJECTIVES
def s_objetivos():
    s = slide()
    rail(s, None)
    tx(s, ML, 0.26, 7.5, 0.3, "CONTEXTO  ·  OBJETIVOS DEL PROYECTO", 12, TEAL, True)
    tx(s, ML, 0.72, CW, 1.0,
       "Buscamos una evaluación de riesgo precisa, explicable y que avise cuándo deja de ser confiable",
       25, INK, True, font=HEAD, ls=1.05)
    dk.a11y_title(s, "Objetivos del proyecto")
    lw = 3.9
    card(s, ML, 1.95, lw, 4.9)
    tx(s, ML + 0.25, 2.12, lw - 0.5, 0.3, "EL PROBLEMA", 12, TEAL, True)
    paras(s, ML + 0.25, 2.5, lw - 0.5, 4.2, [
        [R("Un modelo de riesgo puede perder capacidad predictiva con el tiempo si cambian el perfil de "
           "los solicitantes, su comportamiento de pago o el contexto.", 15, TEXT)],
        [R("Estudiamos el problema con el dataset Home Credit – Credit Risk Model Stability (Kaggle), "
           "que permite medir el desempeño por semana.", 15, TEXT)],
        [R("Es un proyecto académico: no se toman decisiones crediticias reales.", 15, MUTED, italic=True)],
    ], after=12)
    rx = ML + lw + 0.35
    rw = W - ML - rx
    tx(s, rx, 1.95, rw, 0.3, "OBJETIVOS DE NEGOCIO (ON)", 12, TEAL, True)
    table(s, rx, 2.28, rw, [
        ["", "Objetivo"],
        ["ON1", "Identificar mejor las solicitudes de mayor riesgo"],
        ["ON2", "Reducir el rechazo indebido de solventes, incluidos los de historial limitado"],
        ["ON3", "Mantener confiable la evaluación y saber cuándo reentrenar"],
    ], [0.7, rw - 0.7], size=12, row_h=0.4, bold_cells=[(1, 0), (2, 0), (3, 0)])
    tx(s, rx, 4.2, rw, 0.3, "OBJETIVOS DE MINERÍA DE DATOS (OM)", 12, TEAL, True)
    table(s, rx, 4.53, rw, [
        ["", "Objetivo", "Sirve a"],
        ["OM1", "Clasificación binaria de incumplimiento, validada fuera de tiempo", "ON1, ON3"],
        ["OM2", "Variables que más contribuyen a las predicciones, sin asumir causalidad", "ON3, ON1"],
        ["OM3", "Relación entre detectar incumplimientos y rechazar a quienes sí pagan", "ON2, ON1"],
        ["OM4", "Detectar cuándo cambian el desempeño o la población de solicitantes", "ON3"],
    ], [0.7, rw - 1.9, 1.2], size=12, row_h=0.4, bold_cells=[(1, 0), (2, 0), (3, 0), (4, 0)])
    notes(s, "El problema es que un modelo de riesgo puede perder capacidad predictiva con el tiempo. "
             "Nuestro objetivo general es contar con una evaluación precisa, explicable y que avise "
             "cuándo deja de ser confiable. Lo traducimos en tres objetivos de negocio y cuatro "
             "objetivos de minería de datos. Al final del capítulo revisaremos cuáles se cumplieron.")


# ================================================================= 4. DATA CHARACTERISTICS
def s_datos():
    s = slide("4.1", "Los datos son tabulares, grandes y con faltantes estructurales: eso orienta la elección")
    dk.stat_row(s, ML, 2.0, CW, [
        (f"1,33{NBSP}M", "", "solicitudes en desarrollo"),
        ("112–231", "", "predictores según la vista"),
        ("≈ 3,4 %", "", "tasa de incumplimiento"),
        (f"16{NBSP}GB", "", "memoria de la máquina"),
    ], ink=C(INK), accent=C(TEAL), fig_size=34, label_c=T3(MUTED))
    cy, chh = 4.25, 2.6
    gap = 0.3
    cw_ = (CW - 2 * gap) / 3
    cards = [
        ("Faltantes estructurales", "Muchos datos faltan por la propia fuente. Los árboles tratan nulos y "
         "categóricas de forma nativa; la regresión logística exige imputar y codificar."),
        ("Clase minoritaria", "Menos de 4 de cada 100 solicitudes incumplen. Por eso se reportan Gini, "
         "PR-AUC y calibración, no solo la tasa de aciertos."),
        ("Cómputo acotado", "Una máquina de 16 GB. LightGBM captura efectos no lineales con bajo costo "
         "de cómputo, y se explica con SHAP [6]."),
    ]
    for i, (t, d) in enumerate(cards):
        x = ML + i * (cw_ + gap)
        card(s, x, cy, cw_, chh)
        dk.box(s, x, cy, 0.09, chh, fill=C(TEAL))
        tx(s, x + 0.3, cy + 0.25, cw_ - 0.5, 0.4, t, 18, INK, True)
        tx(s, x + 0.3, cy + 0.8, cw_ - 0.5, 1.7, d, 15, TEXT)
    card(s, ML, 3.5, CW, 0.55, fill="F0E6D3")
    dk.box(s, ML, 3.5, 0.09, 0.55, fill=C(AMBER))
    paras(s, ML + 0.3, 3.5, CW - 0.5, 0.55, [[
        R("Caso de estudio: ", 12.5, AMBER_T, True),
        R("dataset público Home Credit – Credit Risk Model Stability (Kaggle), con más de 30 tablas de distinta "
          "granularidad.", 12.5, TEXT)]], anchor=MSO_ANCHOR.MIDDLE, after=0)
    notes(s, "Antes de elegir técnicas miramos los datos. Tenemos 1,33 millones de solicitudes en desarrollo, "
             "entre 112 y 231 predictores según la vista del modelo, muchos faltantes estructurales y una "
             "tasa de incumplimiento cercana al 3,4 por ciento. Esto implica tres cosas: "
             "necesitamos manejar nulos y categóricas, la clase de interés es minoritaria y el cómputo es "
             "limitado. La literatura motiva comparar una regresión logística interpretable con un modelo "
             "que capture efectos no lineales.")


# ================================================================= 5. MODELS BY COMPLEXITY
def s_modelos():
    s = slide("4.1", "Tres modelos, de menor a mayor complejidad: constante, regresión logística y LightGBM")
    gap = 0.3
    cw_ = (CW - 2 * gap) / 3
    base_y = 5.15
    heights = [2.2, 2.75, 3.3]
    data = [
        ("1", "Predicción constante", "Referencia mínima", GREY_C, INK,
         "Predice la tasa de incumplimiento del bloque de ajuste. Fija el piso: Gini 0.",
         "Nulos y categóricas: no aplica."),
        ("2", "Regresión logística", "Referencia interpretable", TEAL_M, INK,
         "Estándar en scoring de crédito, con coeficientes legibles [1].",
         "Nulos: mediana más indicador de faltante. Categóricas: one-hot con vocabulario congelado."),
        ("3", "LightGBM", "Candidato no lineal", TEAL, WHITE,
         "Captura interacciones y no linealidades en datos tabulares grandes; se explica con SHAP [6].",
         "Nulos y categóricas nativos, con vocabulario congelado."),
    ]
    for i, (num, name, role, fill, ink, why, nulls) in enumerate(data):
        x = ML + i * (cw_ + gap)
        h = heights[i]
        y = base_y - h
        dk.box(s, x, y, cw_, h, fill=C(fill))
        tx(s, x + 0.25, y + 0.12, 0.6, 0.55, num, 28, ink, True, font=HEAD)
        tx(s, x + 0.85, y + 0.17, cw_ - 1.0, 0.35, name, 17, ink, True)
        tx(s, x + 0.85, y + 0.5, cw_ - 1.0, 0.3, role, 12, ink)
        paras(s, x + 0.25, y + 0.95, cw_ - 0.5, h - 1.0, [
            [R(why, 12.5, ink)],
            [R(nulls, 12, ink, italic=True)],
        ], after=5)
    tx(s, ML, 5.25, 4.5, 0.3, "complejidad creciente  →", 12, MUTED, True)
    card(s, ML, 5.6, CW, 0.7)
    dk.box(s, ML, 5.6, 0.09, 0.7, fill=C(TEAL))
    paras(s, ML + 0.3, 5.6, CW - 0.5, 0.7, [[
        R("Criterio de progresión: ", 13, INK, True),
        R("se parte de lo más simple; lo más complejo reemplaza solo si mejora el Gini Stability "
          "fuera de partición y gana en al menos 2 de 3 particiones.", 13, TEXT)]],
          anchor=MSO_ANCHOR.MIDDLE, after=0)
    card(s, ML, 6.4, CW, 0.5, fill="F0E6D3")
    dk.box(s, ML, 6.4, 0.09, 0.5, fill=C(AMBER))
    paras(s, ML + 0.3, 6.4, CW - 0.5, 0.5, [[
        R("Fuera de esta versión: ", 12.5, AMBER_T, True),
        R("una red neuronal (MLP), por alcance y recursos. Queda como trabajo futuro, sin resultados.",
          12.5, TEXT)]], anchor=MSO_ANCHOR.MIDDLE, after=0)
    notes(s, "Ordenamos los modelos de menor a mayor complejidad. Primero la predicción constante, que fija "
             "el piso: Gini cero. Segundo, la regresión logística, estándar en scoring y fácil de interpretar. "
             "Tercero, LightGBM, que captura relaciones no lineales y maneja nulos y categóricas de forma "
             "nativa. Partimos de lo más simple y un modelo más complejo reemplaza al anterior solo si mejora "
             "el Gini Stability fuera de partición y gana en al menos dos de tres particiones. "
             "Una red neuronal quedó fuera de esta versión por alcance y recursos; es trabajo futuro y no "
             "presentamos ningún resultado de ella.")


# ================================================================= 6. PAPERS
def s_papers():
    s = slide("4.1", "Cinco artículos de revistas Q1 y Q2 respaldan la comparación, la explicabilidad y la evaluación temporal")
    rows = [
        ["Referencia", "Revista · cuartil", "Qué justifica y caso que estudia", "Alcance de la cita"],
        ["Dumitrescu, Hué, Hurlin y Tokpavi (2022) [1]",
         "European Journal of Operational Research · Q1, JCR 2024",
         "Regresión logística como referencia interpretable frente a métodos que capturan no linealidades",
         "Propone PLTR; nuestro modelo no lineal es LightGBM"],
        ["Hlongwane, Ramaboa y Mongwe (2024) [6]",
         "PLOS ONE · Q2, JCR 2024",
         "Valores de Shapley en modelos de árboles, incluido LightGBM, para scorecards interpretables",
         "Construye scorecards; aún no implementamos una"],
        ["Chen, Calabrese y Martin-Barragan (2024) [5]",
         "European Journal of Operational Research · Q1, JCR 2024",
         "Estabilidad de SHAP y LIME con datasets de crédito desbalanceados",
         "Repetimos muestras del ranking SHAP; no variamos el desbalance"],
        ["Barddal, Loezer, Enembreck y Lanzuolo (2020) [10]",
         "Expert Systems with Applications · Q1, JCR 2024",
         "Cambios temporales en flujos de datos de crédito y detección con ADWIN",
         "Incluye aprendizaje adaptativo, que no implementamos"],
        ["Bellotti y Crook (2013) [3]",
         "International Journal of Forecasting · Q1, SJR 2024",
         "Tasas de incumplimiento de tarjetas en periodos posteriores y cambios de comportamiento",
         "Usa supervivencia; no es nuestro clasificador ni el protocolo OOT"],
    ]
    table(s, ML, 1.95, CW, rows, [2.7, 2.75, 3.85, 2.8], size=11, row_h=0.62,
          bold_cells=[(r, 0) for r in range(1, 6)])
    card(s, ML, 6.0, CW, 0.78, fill="F0E6D3")
    dk.box(s, ML, 6.0, 0.09, 0.78, fill=C(AMBER))
    paras(s, ML + 0.3, 6.0, CW - 0.5, 0.78, [[
        R("Nota: ", 12, AMBER_T, True),
        R("el cuartil es de la revista, no del artículo; año de referencia 2024. SHAP (Lundberg y Lee, 2017, NeurIPS) "
          "y ADWIN (Bifet y Gavaldà, 2007, SIAM SDM) son fuentes fundacionales de conferencia: "
          "no cuentan como Q1 ni Q2.", 12, TEXT)]], anchor=MSO_ANCHOR.MIDDLE, after=0)
    notes(s, "Sustentamos las técnicas con cinco artículos de revistas Q1 y Q2. Dumitrescu y colegas justifican "
             "comparar la regresión logística con métodos no lineales. Hlongwane y colegas aplican valores de "
             "Shapley a modelos de árboles como LightGBM. Chen y colegas estudian la estabilidad de las "
             "explicaciones con clases desbalanceadas. Barddal y colegas tratan los cambios temporales en "
             "crédito, y Bellotti y Crook evalúan tasas de incumplimiento en periodos posteriores. "
             "La última columna aclara qué no hicimos de cada trabajo. El cuartil corresponde a la revista, "
             "con año de referencia 2024. SHAP y ADWIN son fuentes originales de conferencias, por lo que no "
             "las contamos como Q1 o Q2.")


# ================================================================= 7. PARTITIONS
def s_particiones():
    s = slide("4.2", "Cuatro ventanas fuera de tiempo: tres para elegir y una, el OOT, para evaluar una sola vez")
    ax0, axw = 1.55, W - ML - 1.55
    sc = axw / 92
    ty = 2.0
    for wk in (0, 40, 50, 60, 74):
        tx(s, ax0 + wk * sc - 0.3, ty, 0.6, 0.26, str(wk), 11, MUTED, align=PP_ALIGN.CENTER)
    tx(s, ax0 + 91 * sc - 0.55, ty, 0.9, 0.26, "91", 11, MUTED, align=PP_ALIGN.CENTER)
    tx(s, ML, ty, 1.0, 0.26, "semana", 11, MUTED)
    spec = [
        ("P1", 0, 40, 40, 50, "ajuste 0–39", "valid. 40–49"),
        ("P2", 0, 50, 50, 60, "ajuste 0–49", "valid. 50–59"),
        ("P3", 0, 60, 60, 74, "ajuste 0–59", "valid. 60–73"),
        ("OOT", 0, 60, 74, 92, "ajuste 0–59 (modelo de P3)", "OOT 74–91"),
    ]
    for i, (name, a0, a1, v0, v1, la, lv) in enumerate(spec):
        y = 2.35 + i * 0.62
        tx(s, ML, y, 0.9, 0.42, name, 15, INK, True, anchor=MSO_ANCHOR.MIDDLE)
        dk.box(s, ax0 + a0 * sc, y, (a1 - a0) * sc, 0.42, fill=C(TEAL))
        tx(s, ax0 + a0 * sc + 0.1, y, (a1 - a0) * sc - 0.2, 0.42, la, 11.5, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
        if name == "OOT":
            dk.box(s, ax0 + 60 * sc, y, 14 * sc, 0.42, fill=C("B7D3D5"))
            tx(s, ax0 + 60 * sc, y, 14 * sc, 0.42, "calibración", 10.5, INK, True, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE)
            fill = INK
        else:
            fill = AMBER
        dk.box(s, ax0 + v0 * sc, y, (v1 - v0) * sc, 0.42, fill=C(fill))
        tx(s, ax0 + v0 * sc, y, (v1 - v0) * sc, 0.42, lv, 11 if name != "OOT" else 12, WHITE, True,
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    table(s, ML, 4.95, CW, [
        ["Partición", "Ajuste (semanas)", "Validación (semanas)", "Casos de validación", "Positivos"],
        ["P1", "0 a 39", "40 a 49", th(248961), th(9012)],
        ["P2", "0 a 49", "50 a 59", th(215392), th(8771)],
        ["P3", "0 a 59", "60 a 73", th(112852), th(4230)],
        ["OOT", "modelo de P3", "74 a 91", th(194658), th(4260)],
    ], [1.6, 2.6, 2.8, 2.9, 2.23], size=12, row_h=0.37, numeric=[3, 4], highlight=3,
          bold_cells=[(r, 0) for r in range(1, 5)])
    notes(s, "Usamos validación fuera de tiempo con ventanas que se expanden. En P1 ajustamos con las semanas 0 a 39 "
             "y validamos en 40 a 49; en P2 ajustamos con 0 a 49 y validamos en 50 a 59; en P3 ajustamos con "
             "0 a 59 y validamos en 60 a 73. El modelo de P3 se evalúa una única vez en el OOT, las semanas 74 a 91, "
             "con 194 658 casos y 4 260 positivos. La validación de P3, semanas 60 a 73, se usa también para "
             "calibrar y fijar umbrales.")


# ================================================================= 8. RULES
def s_reglas():
    s = slide("4.2", "Reglas fijadas antes de mirar el OOT: elegir con P1 a P3, calibrar con P3 y consultar el OOT una vez")
    gap = 0.2
    cw_ = (CW - 3 * gap) / 4
    steps = [
        ("1", "Seleccionar", "Hiperparámetros y grupos de variables según el Gini Stability fuera de "
         "partición en las semanas 40 a 73 (34 semanas). La mejora debe repetirse en 2 de 3 particiones."),
        ("2", "Calibrar y fijar umbrales", "Calibración isotónica y umbrales se aprenden con la "
         "validación de P3."),
        ("3", "Modelo final", "El de P3, sin reajuste, para que el calibrador corresponda al modelo "
         "evaluado. El reajuste con semanas 0 a 73 solo se reporta como contraste."),
        ("4", "Consultar el OOT", "Semanas 74 a 91, una sola vez. Un marcador final.done impide repetirlo "
         "sin dejar registro."),
    ]
    y, hh = 1.95, 2.75
    for i, (k, t, d) in enumerate(steps):
        x = ML + i * (cw_ + gap)
        card(s, x, y, cw_, hh, fill=PANEL if i < 3 else INK)
        c_ink = INK if i < 3 else WHITE
        c_txt = TEXT if i < 3 else "E6F1F2"
        tx(s, x + 0.2, y + 0.12, 0.6, 0.6, k, 30, TEAL if i < 3 else "9FD0D3", True, font=HEAD)
        tx(s, x + 0.2, y + 0.75, cw_ - 0.4, 0.35, t, 15, c_ink, True)
        tx(s, x + 0.2, y + 1.15, cw_ - 0.4, hh - 1.2, d, 13, c_txt)
    cy = 4.95
    cw2 = (CW - 0.3) / 2
    card(s, ML, cy, cw2, 1.95)
    dk.box(s, ML, cy, 0.09, 1.95, fill=C(TEAL))
    tx(s, ML + 0.3, cy + 0.15, cw2 - 0.5, 0.3, "CORRECCIÓN ANTES DE CONSULTAR EL OOT", 11.5, TEAL, True)
    tx(s, ML + 0.3, cy + 0.5, cw2 - 0.5, 1.4,
       "El Gini Stability por partición (10 a 14 semanas) quedaba dominado por la penalización de pendiente "
       "(−1,8 frente a un Gini medio de 0,40, en una prueba con el 5 % de los datos). "
       "Se reemplazó por el cálculo fuera de partición.", 13.5, TEXT)
    x2 = ML + cw2 + 0.3
    card(s, x2, cy, cw2, 1.95, fill="F0E6D3")
    dk.box(s, x2, cy, 0.09, 1.95, fill=C(AMBER))
    tx(s, x2 + 0.3, cy + 0.15, cw2 - 0.5, 0.3, "LIMITACIÓN DECLARADA", 11.5, AMBER_T, True)
    tx(s, x2 + 0.3, cy + 0.5, cw2 - 0.5, 1.4,
       "El OOT ya había sido examinado en el avance 2. Es la única evaluación fuera de tiempo disponible, "
       "porque el test local tiene 10 casos.", 13.5, TEXT)
    notes(s, "Estas reglas se fijaron antes de consultar el OOT. Uno: seleccionamos hiperparámetros y grupos de "
             "variables con el Gini Stability calculado fuera de partición sobre las semanas 40 a 73, y una "
             "mejora debe repetirse en dos de las tres particiones. Dos: la calibración isotónica y los umbrales "
             "se aprenden con la validación de P3. Tres: el modelo final es el de P3, sin reajuste. "
             "Cuatro: el OOT se consulta una sola vez. Antes de consultarlo corregimos el cálculo del Gini "
             "Stability porque la penalización de pendiente dominaba con pocas semanas. "
             "Como limitación, el OOT ya se había examinado en el avance 2 y no hay otro periodo intacto.")


# ================================================================= 9. METRICS
def s_metricas():
    s = slide("4.2", "Cada objetivo tiene su métrica y su criterio de éxito definidos de antemano")
    rows = [
        ["Métrica", "Qué mide", "Criterio de éxito"],
        ["Gini Stability fuera de tiempo", "Discriminación semanal, penalizada por su variabilidad", "Superior a la referencia (OM1)"],
        ["ECE y razón predicho / observado", "Si la probabilidad calibrada coincide con la tasa real", "ECE < 0,01 y desvío < 20 % (OM1)"],
        ["Brier y log loss", "Calidad de la probabilidad calibrada", "Comparar con el Brier de la tasa base"],
        ["ROC-AUC y PR-AUC", "Orden global del riesgo con clase minoritaria", "Complementarias, se reportan"],
        ["Falsos positivos a recall 70, 80 y 90 %", "Buenos pagadores rechazados a igual detección", "Menos que la referencia (OM3, ON2)"],
        ["Estabilidad del ranking SHAP", "Si las variables importantes se repiten entre muestras y periodos", "Ranking reproducible (OM2)"],
        ["PSI, ADWIN y calibración semanal", "Cambios del desempeño o de la población", "Alertas ubicadas en el tiempo (OM4)"],
    ]
    table(s, ML, 1.95, CW, rows, [3.6, 4.9, 3.63], size=12, row_h=0.56,
          bold_cells=[(r, 0) for r in range(1, 8)])
    notes(s, "Definimos las métricas y los criterios antes de evaluar. La métrica principal es el Gini Stability "
             "fuera de tiempo, que resume el Gini semanal penalizado por su variabilidad. Para calibración usamos "
             "ECE y la razón entre riesgo predicho y observado, con el criterio de ECE menor que 0,01 y desvío "
             "menor que 20 por ciento. Para OM3 medimos falsos positivos a igual recall. Para OM2 revisamos la "
             "estabilidad del ranking SHAP, y para OM4 los detectores de cambio.")


# ================================================================= 10. CONTROLS
def s_controles():
    s = slide("4.2", "Seis controles evitan fugas de información temporal en la validación")
    items = [
        ("Transformaciones por partición", "Imputación, escalado, vocabulario y descarte de constantes se reajustan en "
         "cada bloque de ajuste."),
        ("Columnas de control fuera", "case_id, WEEK_NUM, date_decision, MONTH y target nunca son predictores; "
         "hay una aserción en el código."),
        ("Selección sin OOT", "Un test ejecuta la selección y las pruebas de quitar variables con el archivo "
         "OOT ausente."),
        ("Etapa final sin efecto del OOT", "Un test altera etiquetas y scores del OOT: calibrador, umbrales, "
         "selección y exclusiones no cambian."),
        ("Disponibilidad temporal", "Cada variable fue revisada en la auditoría de variables predictoras."),
        ("Revisión independiente", "Del código antes de la corrida: sin caminos de fuga; los siete hallazgos "
         "se corrigieron."),
    ]
    gx, gy = 0.3, 0.3
    cw_ = (CW - 2 * gx) / 3
    ch_ = 2.3
    for i, (t, d) in enumerate(items):
        x = ML + (i % 3) * (cw_ + gx)
        y = 1.95 + (i // 3) * (ch_ + gy)
        card(s, x, y, cw_, ch_)
        tx(s, x + 0.25, y + 0.15, 0.7, 0.6, str(i + 1), 30, TEAL, True, font=HEAD)
        tx(s, x + 0.95, y + 0.25, cw_ - 1.15, 0.7, t, 15, INK, True, anchor=MSO_ANCHOR.TOP)
        tx(s, x + 0.25, y + 1.0, cw_ - 0.5, 1.25, d, 14, TEXT)
    notes(s, "Para que la evaluación sea válida aplicamos seis controles. Las transformaciones se reajustan en cada "
             "partición solo con su bloque de ajuste. Las columnas de control, como case_id o la semana, nunca "
             "entran como predictores. Hay tests que ejecutan la selección sin el archivo OOT y que alteran las "
             "etiquetas del OOT para comprobar que la etapa final no cambia. La disponibilidad temporal de las "
             "variables se revisó en una auditoría, y una revisión independiente del código corrigió siete hallazgos.")


# ================================================================= 11. EXPERIMENT LOG
def s_registro():
    s = slide("4.3", "Cinco experimentos reproducibles con una semilla fija y una corrida de 11 minutos")
    dk.stat_row(s, ML, 1.9, CW, [
        ("122", "", "características (matriz E5-v2)"),
        ("42", "", "semilla"),
        ("11 min", "", "duración total"),
        (f"6,6{NBSP}GB", "", "pico de memoria"),
    ], ink=C(INK), accent=C(TEAL), fig_size=30, label_c=T3(MUTED))
    tx(s, ML, 3.32, CW, 0.3, "Máquina: 10 núcleos y 16 GB.", 12, MUTED)
    rows = [
        ["Experimento", "Variables", "Partición", "Hiperparámetros", "Tiempo de ajuste", "Estado"],
        ["Constante", "Ninguna", "P1 a P3, OOT", "Tasa base del ajuste", "0 s", "Referencia"],
        ["Regresión logística", "Vista lineal (hasta 228)", "P1 a P3", "C ∈ {0,1; 1}", "11 s (6 ajustes)", "Seleccionada: C = 0,1"],
        ["LightGBM", "Vista de árboles (hasta 110)", "P1 a P3", "hojas ∈ {31; 63}, árboles ∈ {300; 600}, tasa 0,05", "184 s", "Seleccionada: 63 hojas, 300 árboles"],
        ["Quitar variables", "4 grupos", "P1 a P3", "Configuración seleccionada", "311 s (etapa)", "Ningún grupo retirado"],
        ["Final", "Todas", "P3 y OOT", "Seleccionada", "Logística 2,1 s; LightGBM 17,4 s", "Ejecutado"],
    ]
    table(s, ML, 3.7, CW, rows, [1.9, 2.2, 1.45, 2.85, 1.75, 1.98], size=11, row_h=0.5,
          bold_cells=[(r, 0) for r in range(1, 6)])
    notes(s, "Este es el registro de experimentos. Todos usan la matriz E5-v2 con 122 características, semilla 42 "
             "y una máquina de 10 núcleos y 16 gigabytes. La corrida completa tomó 11 minutos con un pico de "
             "memoria de 6,6 gigabytes. Hay cinco experimentos: el constante, la regresión logística, LightGBM, "
             "las pruebas de quitar variables y la etapa final. Las siglas RL corresponden a regresión logística.")


# ================================================================= 12. HYPERPARAMETERS
def s_hiper():
    s = slide("4.3", "Una grilla pequeña y comparable: LightGBM con 63 hojas y 300 árboles gana la selección")
    rows = [
        ["Modelo", "Configuración", "Gini Stability fuera de partición", "Gini semanal medio", "AUC"],
        ["Constante", "—", "0,000", "0,000", "0,508"],
        ["Regresión logística", "C = 0,1 (elegida)", "0,288", "0,397", "0,705"],
        ["Regresión logística", "C = 1", "0,286", "0,397", "0,705"],
        ["LightGBM", "31 hojas, 300 árboles", "0,426", "0,451", "0,727"],
        ["LightGBM", "31 hojas, 600 árboles", "0,427", "0,452", "0,728"],
        ["LightGBM", "63 hojas, 300 árboles (elegida)", "0,428", "0,453", "0,728"],
        ["LightGBM", "63 hojas, 600 árboles", "0,425", "0,451", "0,727"],
    ]
    tw = 7.45
    table(s, ML, 1.95, tw, rows, [1.7, 2.3, 1.6, 1.05, 0.8], size=11, row_h=0.46, numeric=[2, 3, 4],
          colors={(2, 1): TEAL, (6, 1): TEAL}, bold_cells=[(2, 2), (6, 2), (2, 1), (6, 1)])
    cx = ML + tw + 0.35
    cwid = W - ML - cx
    tx(s, cx, 1.95, cwid, 0.5, "Gini Stability fuera de partición (eje desde 0)", 13, INK, True)
    cats = ["RL\nC=0,1", "RL\nC=1", "LGB\n31/300", "LGB\n31/600", "LGB\n63/300", "LGB\n63/600"]
    vals = [0.288, 0.286, 0.426, 0.427, 0.428, 0.425]
    ch = dk.native_chart(s, cx, 2.35, cwid, 3.0, cats, [("Gini Stability", vals)], kind="column",
                         legend=False, font=BODY)
    style_bars(ch, [GREY_C], [[n(v) for v in vals]], size=11, gap=40,
               point_colors={0: {0: TEAL_M, 1: GREY_C, 2: GREY_C, 3: GREY_C, 4: TEAL, 5: GREY_C}})
    tx(s, cx, 5.4, cwid, 0.5, "Resaltadas: configuración elegida de cada modelo. RL: regresión logística; "
       "LGB: LightGBM (hojas/árboles).", 10.5, MUTED)
    card(s, ML, 6.15, CW, 0.7)
    dk.box(s, ML, 6.15, 0.09, 0.7, fill=C(TEAL))
    paras(s, ML + 0.3, 6.15, CW - 0.5, 0.7, [[
        R("Sin early stopping, ", 12.5, INK, True),
        R("para no ajustar el número de árboles con la validación. Las cuatro configuraciones de LightGBM "
          "difieren en 0,003 como máximo: la elección es poco sensible a la grilla.", 12.5, TEXT)]],
          anchor=MSO_ANCHOR.MIDDLE, after=0)
    notes(s, "Hicimos una búsqueda en grilla pequeña y comparable entre modelos, sin early stopping para no ajustar "
             "el número de árboles con la validación. La regresión logística se queda con C igual a 0,1, con "
             "Gini Stability fuera de partición de 0,288. En LightGBM la mejor configuración es 63 hojas y 300 "
             "árboles, con 0,428, aunque las cuatro configuraciones difieren en 0,003 como máximo. "
             "La constante sirve de piso con Gini cero.")


# ================================================================= 13. ABLATION
def s_ablacion():
    s = slide("4.3", "Quitar variables por grupos: ningún grupo se retira y la fuente fiscal no aporta en el ajuste")
    rows = [
        ["Grupo de variables", "Columnas que llegan al modelo", "Resultado"],
        ["Cobertura (*__n_records, *__source_present)", "19 de 28; 9 son constantes", "Se conserva en ambos modelos"],
        ["person_2__*", "2", "Se conserva"],
        ["Contactos (person_1__n_contact_persons__all)", "1", "Se conserva"],
        ["tax_registry_b_1__*", "0", "Sin efecto"],
    ]
    table(s, ML, 1.95, CW, rows, [4.7, 3.2, 4.23], size=12.5, row_h=0.55,
          colors={(4, 2): AMBER_T}, bold_cells=[(r, 0) for r in range(1, 5)] + [(4, 2)])
    cy = 4.95
    cw2 = (CW - 0.3) / 2
    card(s, ML, cy, cw2, 1.9)
    dk.box(s, ML, cy, 0.09, 1.9, fill=C(TEAL))
    tx(s, ML + 0.3, cy + 0.15, cw2 - 0.5, 0.3, "REGLA DE RETIRO", 11.5, TEAL, True)
    tx(s, ML + 0.3, cy + 0.5, cw2 - 0.5, 1.35,
       "Un grupo se retira si, sin él, no empeora el Gini Stability fuera de partición, no empeora el Gini "
       "semanal medio en al menos 2 de 3 particiones y no aumenta la desviación semanal.", 13.5, TEXT)
    x2 = ML + cw2 + 0.3
    card(s, x2, cy, cw2, 1.9)
    dk.box(s, x2, cy, 0.09, 1.9, fill=C(AMBER))
    tx(s, x2 + 0.3, cy + 0.15, cw2 - 0.5, 0.3, "POR QUÉ LA FUENTE FISCAL NO TIENE EFECTO", 11.5, AMBER_T, True)
    tx(s, x2 + 0.3, cy + 0.5, cw2 - 0.5, 1.35,
       "La fuente tax_registry_b_1 vale cero en todas las semanas 0 a 59 y aparece recién desde la semana 60: "
       "el modelo de P3 no puede usarla.", 13.5, TEXT)
    notes(s, "Probamos retirar cuatro grupos de variables con una regla fija: un grupo se retira si no empeora el "
             "Gini Stability fuera de partición, no empeora el Gini semanal medio en dos de tres particiones y "
             "no aumenta la desviación semanal. Ningún grupo se retiró. El grupo de la fuente fiscal "
             "tax_registry_b_1 no llega al modelo, porque vale cero en las semanas 0 a 59 y aparece desde "
             "la semana 60.")


# ================================================================= 14. OOT COMPARISON
def s_oot():
    s = slide("4.4", "En el OOT, LightGBM supera a la regresión logística: Gini Stability de 0,484 frente a 0,438")
    cx, cwid = ML, 4.5
    tx(s, cx, 1.95, cwid, 0.3, "Gini Stability en el OOT, semanas 74 a 91 (eje desde 0)", 13, INK, True)
    cats = ["Constante", "Regresión\nlogística", "LightGBM"]
    vals = [0.0, 0.438, 0.484]
    ch = dk.native_chart(s, cx, 2.3, cwid, 3.3, cats, [("Gini Stability", vals)], kind="column",
                         legend=False, font=BODY)
    style_bars(ch, [GREY_C], [[n(v) for v in vals]], size=14, gap=45,
               point_colors={0: {0: GREY_C, 1: TEAL_M, 2: TEAL}})
    rx = cx + cwid + 0.4
    rw = W - ML - rx
    rows = [
        ["Modelo", "Gini Stab.", "Gini medio", "ROC-AUC", "PR-AUC", "Recall al umbral de P3"],
        ["Constante", "0,000", "0,000", "0,500", "0,022", "—"],
        ["Regresión logística", "0,438", "0,460", "0,731", "0,062", "81,0 %"],
        ["LightGBM", "0,484", "0,503", "0,752", "0,075", "80,7 %"],
        ["LightGBM, reajuste 0 a 73", "0,493", "—", "0,757", "0,076", "72,8 %"],
    ]
    table(s, rx, 2.0, rw, rows, [2.2, 0.9, 0.95, 0.9, 0.85, rw - 5.8], size=11, row_h=0.55, numeric=[1, 2, 3, 4, 5],
          highlight=2, bold_cells=[(r, 0) for r in range(1, 5)])
    tx(s, rx, 5.0, rw, 0.6, "El reajuste 0 a 73 es solo un contraste: su umbral ya no corresponde a ese modelo "
       "(recall de 72,8 %).", 11.5, MUTED, italic=True)
    card(s, ML, 5.85, CW, 1.0)
    dk.box(s, ML, 5.85, 0.09, 1.0, fill=C(TEAL))
    paras(s, ML + 0.3, 5.85, CW - 0.5, 1.0, [[
        R("+0,046 de Gini Stability ", 15, TEAL, True),
        R(f"sobre {th(194658)} casos y {th(4260)} incumplimientos (2,19 %). Se cumple OM1 en discriminación y, "
          "por extensión, ON1: el modelo ordena el riesgo mejor que una referencia simple.", 13, TEXT)]],
          anchor=MSO_ANCHOR.MIDDLE, after=0)
    notes(s, "Estos son los resultados en el OOT, semanas 74 a 91, con 194 658 casos y 4 260 incumplimientos. "
             "LightGBM alcanza un Gini Stability de 0,484 frente a 0,438 de la regresión logística, una "
             "diferencia de 0,046; la predicción constante, como esperábamos, queda en cero. También mejora el "
             "ROC-AUC, de 0,731 a 0,752, y el PR-AUC, de 0,062 a 0,075. El reajuste con semanas 0 a 73 llega a "
             "0,493, pero lo reportamos solo como contraste porque su umbral de P3 ya no corresponde. "
             "Con esto se cumple OM1 en discriminación.")


# ================================================================= 15. WEEKLY GINI
def s_semanal():
    s = slide("4.4", "LightGBM gana en las 18 semanas del OOT y no muestra tendencia descendente")
    cats = [str(w) for w in WEEKS]
    ser = [("Regresión logística", [round(WK["logistic"][w], 3) for w in WEEKS]),
           ("LightGBM", [round(WK["lightgbm"][w], 3) for w in WEEKS])]
    cw_ = 8.4
    tx(s, ML, 1.95, cw_, 0.3, "Gini semanal en el OOT (eje vertical desde 0,3)", 13, INK, True)
    ch = dk.native_chart(s, ML, 2.3, cw_, 4.45, cats, ser, kind="line_markers", legend=False, font=BODY,
                         palette=[TEAL_M, TEAL], smooth=False, value_fmt="[$-40A]0.0")
    from pptx.util import Pt
    from pptx.enum.chart import XL_MARKER_STYLE
    for ser_, col, wd in zip(ch.series, [TEAL_M, TEAL], [2.25, 3.0]):
        ser_.format.line.color.rgb = C(col)
        ser_.format.line.width = Pt(wd)
        ser_.smooth = False
        ser_.marker.style = XL_MARKER_STYLE.CIRCLE
        ser_.marker.size = 6
        ser_.marker.format.fill.solid()
        ser_.marker.format.fill.fore_color.rgb = C(col)
        ser_.marker.format.line.color.rgb = C(col)
    ch.value_axis.minimum_scale = 0.3
    ch.value_axis.maximum_scale = 0.6
    ch.value_axis.major_unit = 0.1
    ch.category_axis.tick_labels.font.size = Pt(11)
    # Value-axis labels are drawn as text (decimal comma): PowerPoint/LibreOffice would use the viewer's
    # locale. A fixed inner plot rectangle makes their position exact.
    from lxml import etree
    from pptx.enum.chart import XL_TICK_LABEL_POSITION
    ns = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    pa = ch._chartSpace.chart.plotArea
    for old in pa.findall("{%s}layout" % ns):
        pa.remove(old)
    PX, PY, PW, PH = 0.07, 0.04, 0.91, 0.83
    lay = etree.fromstring(
        '<c:layout xmlns:c="%s"><c:manualLayout><c:layoutTarget val="inner"/><c:xMode val="edge"/>'
        '<c:yMode val="edge"/><c:x val="%s"/><c:y val="%s"/><c:w val="%s"/><c:h val="%s"/>'
        '</c:manualLayout></c:layout>' % (ns, PX, PY, PW, PH))
    pa.insert(0, lay)
    ch.value_axis.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    CY, CH_ = 2.3, 4.45
    for v in (0.3, 0.4, 0.5, 0.6):
        yy = CY + CH_ * (PY + PH * (1 - (v - 0.3) / 0.3))
        tx(s, ML, yy - 0.14, PX * cw_ - 0.08, 0.28, n(v, 1), 11, MUTED, align=PP_ALIGN.RIGHT,
           anchor=MSO_ANCHOR.MIDDLE)
    legend_manual(s, ML + 5.2, 1.93, [(TEAL, "LightGBM"), (TEAL_M, "Regresión logística")], 12)
    rx = ML + cw_ + 0.35
    rw = W - ML - rx
    cards = [
        ("18 de 18", "semanas en que LightGBM supera a la regresión logística", TEAL),
        (n(LGB_MEAN), f"Gini semanal medio de LightGBM (regresión logística: {n(LR_MEAN)})", TEAL),
        ("0,437", "peor semana de LightGBM (semana 77). La regresión logística llega a 0,374 y 0,377 en las semanas 81 y 82", INK),
    ]
    y = 1.95
    for big, small, col in cards:
        card(s, rx, y, rw, 1.5)
        tx(s, rx + 0.25, y + 0.1, rw - 0.5, 0.65, big, 30, col, True, font=HEAD)
        tx(s, rx + 0.25, y + 0.75, rw - 0.5, 0.75, small, 11.5, TEXT)
        y += 1.68
    notes(s, "Aquí vemos el Gini semana por semana. LightGBM, en teal oscuro, queda por encima de la regresión "
             "logística en las 18 semanas. Su Gini semanal medio es 0,503 y su peor semana es la 77, con 0,437. "
             "La regresión logística cae más en sus peores semanas, la 81 y la 82, con 0,374 y 0,377. "
             "No hay tendencia descendente en el periodo. El eje vertical empieza en 0,3 para distinguir las "
             "dos líneas; las cifras exactas están en el capítulo 5.")


# ================================================================= 16. FALSE POSITIVES
def s_fp():
    s = slide("4.4", "A igual detección, LightGBM rechaza a unos 7 700 buenos pagadores menos con recall de 80 %")
    cw_ = 7.6
    tx(s, ML, 1.95, cw_, 0.3, "Tasa de falsos positivos en el OOT, por recall objetivo (eje desde 0)", 13, INK, True)
    cats = ["Recall 70 %", "Recall 80 %", "Recall 90 %"]
    lr = [37.1, 49.5, 66.0]
    lg = [34.0, 45.4, 63.1]
    ch = dk.native_chart(s, ML, 2.3, cw_, 3.7, cats, [("Regresión logística", lr), ("LightGBM", lg)],
                         kind="column", legend=False, font=BODY)
    style_bars(ch, [TEAL_M, TEAL], [[n(v, 1) + " %" for v in lr], [n(v, 1) + " %" for v in lg]], size=12, gap=60)
    legend_manual(s, ML + 0.1, 6.1, [(TEAL_M, "Regresión logística"), (TEAL, "LightGBM")], 12)
    rx = ML + cw_ + 0.4
    rw = W - ML - rx
    card(s, rx, 1.95, rw, 2.3, fill=INK)
    tx(s, rx + 0.3, 2.1, rw - 0.6, 0.9, "~7 700", 44, WHITE, True, font=HEAD)
    tx(s, rx + 0.3, 3.1, rw - 0.6, 1.1,
       f"buenos pagadores rechazados de menos con recall de 80 %, de {th(190398)} en el OOT", 13, "E6F1F2")
    card(s, rx, 4.4, rw, 2.45)
    tx(s, rx + 0.3, 4.55, rw - 0.6, 0.3, "LECTURA", 11.5, TEAL, True)
    paras(s, rx + 0.3, 4.9, rw - 0.6, 1.9, [
        [R("−4,1 puntos de FPR a recall de 80 % (49,5 % a 45,4 %).", 12.5, TEXT)],
        [R("Con el umbral fijo, ambos modelos alcanzan el recall pedido; la caída de la tasa base no lo afecta.", 12.5, TEXT)],
        [R("Se cumple OM3 y el criterio general de ON2.", 12.5, TEAL, True)],
    ], after=6)
    notes(s, "Para OM3 fijamos los umbrales en la validación de P3 y los aplicamos sin cambios al OOT. Con un recall "
             "objetivo de 80 por ciento, la regresión logística rechaza al 49,5 por ciento de los buenos "
             "pagadores y LightGBM al 45,4, es decir 4,1 puntos menos. En el OOT hay 190 398 buenos pagadores, "
             "lo que equivale a unos 7 700 rechazos de menos. LightGBM rechaza menos buenos pagadores en los "
             "tres niveles de recall. Con esto se cumple OM3 y el criterio general de ON2.")


# ================================================================= 17. CALIBRATION (NOT MET)
def s_calibracion():
    s = slide("4.4", "Calibración no cumplida: el modelo sobreestima el riesgo 1,62 veces en el OOT")
    cw_ = 7.7
    tx(s, ML, 1.95, cw_, 0.3, "Tasa observada y riesgo predicho con LightGBM calibrado (eje desde 0)", 13, INK, True)
    cats = ["Semanas\n60 a 63", "Semanas\n64 a 65", "Semanas\n66 a 73", "OOT\n74 a 91"]
    obs = [4.8, 2.9, 2.2, 2.2]
    pred = [4.0, 3.6, 3.4, 3.5]
    ratio = ["0,83", "1,25", "1,52", "1,62"]
    ch = dk.native_chart(s, ML, 2.3, cw_, 3.6, cats, [("Tasa observada", obs), ("Riesgo predicho", pred)],
                         kind="column", legend=False, font=BODY)
    style_bars(ch, [GREY_C, BRICK], [[n(v, 1) + " %" for v in obs], [n(v, 1) + " %" for v in pred]], size=12, gap=60)
    legend_manual(s, ML + 0.1, 5.95, [(GREY_C, "Tasa observada"), (BRICK, "Riesgo predicho")], 12)
    tx(s, ML + 0.1, 6.35, cw_, 0.3, "Predicho / observado:  " + "   ·   ".join(ratio), 12.5, INK, True)
    rx = ML + cw_ + 0.4
    rw = W - ML - rx
    chip_status(s, rx, 1.95, 1.9, "NO CUMPLIDO", BRICK)
    dk.stat_row(s, rx, 2.45, rw, [
        ("0,0136", "", "ECE (criterio < 0,01)"),
        ("1,62", "×", "predicho / observado (criterio: desvío < 20 %)"),
    ], ink=C(BRICK), accent=C(BRICK), fig_size=28, label_c=T3(MUTED))
    paras(s, rx, 4.2, rw, 2.7, [
        [R("El orden del riesgo se mantiene; la escala no. ", 12.5, INK, True),
         R("Sirve para umbrales relativos (OM3), no como probabilidad absoluta, por ejemplo en provisiones, "
           "sin recalibrar.", 12.5, TEXT)],
        [R("Causa temporal: ", 12.5, INK, True),
         R("la tasa observada cae de 4,8 % a 2,2 %, mientras el calibrador se ajustó con las semanas 60 a 73 "
           "mezcladas.", 12.5, TEXT)],
        [R("Hipótesis, no verificada: madurez de la etiqueta; el horizonte de target no está publicado.", 12, MUTED, italic=True)],
    ], after=8)
    notes(s, "Este es un resultado negativo que queremos mostrar con claridad. La calibración no se cumple: el "
             "modelo predice un riesgo de 3,5 por ciento frente a una tasa observada de 2,2 en el OOT, una razón "
             "de 1,62, y el ECE es 0,0136 cuando el criterio era menor que 0,01. La causa es temporal: la tasa "
             "observada cae de 4,8 a 2,2 por ciento mientras el calibrador se ajustó con las semanas 60 a 73 "
             "mezcladas. El orden del riesgo se mantiene, por eso OM3 sí se cumple, pero la probabilidad no es "
             "utilizable como valor absoluto sin recalibrar. Una hipótesis es la madurez de la etiqueta, pero no "
             "podemos verificarla porque el horizonte del target no está publicado.")


# ================================================================= 18. SEGMENT (NOT MET)
def s_segmento():
    s = slide("4.4", "En el segmento sin historial observado, LightGBM no supera a la regresión logística")
    chip_status(s, ML, 1.95, 1.9, "NO CUMPLIDO", BRICK)
    tx(s, ML + 2.1, 1.95, CW - 2.1, 0.34, "ON2: ventaja en solicitantes con historial limitado", 13, INK, True,
       anchor=MSO_ANCHOR.MIDDLE)
    rows = [
        ["Segmento OOT", "Casos", "Incumplimientos", "AUC logística", "AUC LightGBM", "FPR logística (recall 80 %)", "FPR LightGBM (recall 80 %)"],
        ["Sin historial observado", th(11223), "234", "0,709", "0,707", "51,5 %", "53,6 %"],
        ["Resto", th(183435), th(4026), "0,735", "0,756", "47,8 %", "43,5 %"],
    ]
    table(s, ML, 2.5, CW, rows, [2.3, 1.0, 1.8, 1.35, 1.45, 2.1, 2.13], size=12, row_h=0.6, numeric=[1, 2, 3, 4, 5, 6],
          colors={(1, 3): TEAL, (1, 5): TEAL, (2, 4): TEAL, (2, 6): TEAL},
          bold_cells=[(1, 0), (2, 0)])
    gx = 0.3
    cw_ = (CW - 2 * gx) / 3
    cy, chh = 4.6, 2.25
    cards = [
        ("Toda la ventaja viene de quienes tienen historial", "En el segmento que ON2 menciona no hay evidencia de mejora; con 234 incumplimientos la diferencia es pequeña frente a la incertidumbre."),
        ("El segmento se reduce", "Pasa de 11,6 % de los casos en desarrollo a 5,8 % en el OOT."),
        ("Definición operativa", "Sin solicitud previa válida y sin registros de bureau A1 ni B1. Indica historial no observado por estas fuentes, no ausencia de historial crediticio."),
    ]
    for i, (t, d) in enumerate(cards):
        x = ML + i * (cw_ + gx)
        card(s, x, cy, cw_, chh)
        dk.box(s, x, cy, 0.09, chh, fill=C(BRICK if i == 0 else TEAL))
        tx(s, x + 0.3, cy + 0.15, cw_ - 0.5, 0.65, t, 14, INK, True)
        tx(s, x + 0.3, cy + 0.85, cw_ - 0.5, 1.35, d, 12, TEXT)
    notes(s, "Un segundo resultado negativo. ON2 menciona explícitamente a los solicitantes con historial limitado. "
             "Definimos de forma operativa el segmento sin historial observado: sin solicitud previa válida y sin "
             "registros de bureau A1 ni B1. En ese segmento LightGBM no supera a la regresión logística: el AUC es "
             "0,707 frente a 0,709 y la tasa de falsos positivos es mayor. Toda la ventaja de LightGBM proviene "
             "de los clientes con historial. Con 234 incumplimientos la diferencia es pequeña frente a la "
             "incertidumbre, y además el segmento baja de 11,6 a 5,8 por ciento de los casos. "
             "Recordemos que esta definición no prueba ausencia de historial crediticio.")


# ================================================================= 19. SHAP
def s_shap():
    s = slide("4.4", "Las cinco variables principales son las mismas en P3 y en el OOT, y su ranking es estable")
    tw = 6.6
    tx(s, ML, 1.95, tw, 0.3, "Ranking SHAP de LightGBM (20 000 casos por muestra)", 13, INK, True)
    rows = [
        ["Puesto", "Variable", "En P3", "En OOT"],
        ["1", "Tipo de ingreso del solicitante", "1", "1"],
        ["2", "Edad del solicitante", "2", "2"],
        ["3", "Plazo medio de solicitudes previas", "3", "3"],
        ["4", "Nivel educativo en solicitudes previas", "4", "4"],
        ["5", "Registros del registro fiscal C", "5", "5"],
    ]
    table(s, ML, 2.35, tw, rows, [0.9, 3.8, 0.95, 0.95], size=12.5, row_h=0.52, numeric=[2, 3],
          bold_cells=[(r, 1) for r in range(1, 6)])
    rx = ML + tw + 0.4
    rw = W - ML - rx
    dk.stat_row(s, rx, 1.95, rw, [
        ("≥ 0,9997", "", "correlación de Spearman mínima entre 5 muestras"),
        ("20 de 20", "", "variables principales coinciden en todas las muestras"),
    ], ink=C(TEAL), accent=C(TEAL), fig_size=26, label_c=T3(MUTED))
    card(s, rx, 3.65, rw, 1.75)
    tx(s, rx + 0.25, 3.78, rw - 0.5, 0.3, "CAMBIOS ENTRE P3 Y OOT", 11.5, TEAL, True)
    tx(s, rx + 0.25, 4.1, rw - 0.5, 1.3,
       "person_2__n_records sube del puesto 28 al 11 y person_2__source_present del 73 al 50, en línea con la "
       "caída de presencia de person_2 (de 99 % a 70 %).", 12, TEXT)
    dk.box(s, ML, 5.75, CW, 0.9, fill=C(INK))
    tx(s, ML + 0.3, 5.75, CW - 0.6, 0.9,
       "La importancia describe asociación con la predicción, no causalidad.", 17, WHITE, True, font=HEAD,
       anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Para OM2 medimos la importancia de variables con SHAP sobre 20 000 casos, y repetimos el ranking con "
             "cinco muestras independientes. La correlación de Spearman mínima entre rankings fue de al menos "
             "0,9997 y las veinte variables principales coincidieron en todas las muestras, lo que responde a "
             "la preocupación de Chen y colegas sobre explicaciones inestables. Las cinco primeras variables "
             "son las mismas en P3 y en el OOT: tipo de ingreso, edad, plazo medio de solicitudes previas, "
             "nivel educativo previo y registros del registro fiscal C. Insistimos: la importancia describe "
             "asociación con la predicción, no causalidad.")


# ================================================================= 20. OM4
def s_om4():
    s = slide("4.4", "Detección de cambios: solo el detector de calibración, agregado después, ubica el cambio")
    rows = [
        ["Detector", "Resultado", "Lectura"],
        ["PSI semanal del score", "Máximo 0,04, sin alertas", "La distribución del score no cambia"],
        ["ADWIN sobre log loss semanal", "Sin alertas", "Con 32 puntos semanales tiene poco poder [10]"],
        ["PSI de variables", "Idioma y educación de person_1: PSI de 1,6 a 2,9 desde la semana 79", "Cambio fuerte de población sin pérdida de Gini"],
        ["Fuente tax_registry_b_1", "Ausente en las semanas 0 a 59, presente desde la 60", "Cambio de fuentes fechado; el modelo de P3 no puede usarla"],
        ["Calibración semanal (agregado en la evaluación)", "Primera alerta en la semana 66; 26 de 32 semanas en alerta", "Detecta el cambio de tasa base que los otros no ven"],
    ]
    table(s, ML, 1.95, CW, rows, [3.2, 4.8, 4.13], size=11.5, row_h=0.56,
          bold_cells=[(r, 0) for r in range(1, 6)], highlight=4)
    cy = 5.55
    cw2 = (CW - 0.3) / 2
    chip_status(s, ML, cy, 1.2, "PARCIAL", AMBER_T)
    tx(s, ML + 1.4, cy, cw2 + 3, 0.34,
       "OM4: los detectores definidos antes de la corrida no alertaron.", 13, INK, True, anchor=MSO_ANCHOR.MIDDLE)
    paras(s, ML, cy + 0.45, CW, 0.9, [[
        R("Regla de la alerta de calibración: ", 12, INK, True),
        R("la razón predicho / observado sale de [0,8; 1,25] dos semanas seguidas. Se calculó sobre predicciones ya "
          "congeladas y requiere etiquetas. Ventana sugerida: recalibrar con semanas posteriores a la 65, no "
          "reentrenar el ranking.", 12, TEXT)]], after=0)
    notes(s, "Para OM4 definimos de antemano dos detectores: el PSI del score y ADWIN sobre el log loss. Ninguno "
             "produjo alertas, aunque sí hubo un cambio real: la tasa de incumplimiento cayó. Después de ver "
             "este resultado agregamos un detector de calibración semanal, que lo ubica en la semana 66. "
             "Lo declaramos como un agregado posterior; por eso OM4 queda parcial. La ventana de "
             "reentrenamiento que sugiere la evidencia es recalibrar con semanas posteriores a la 65, "
             "no reentrenar el ranking. También vimos cambios fuertes de población, como el idioma y la "
             "educación, sin pérdida de Gini.")


# ================================================================= 21. OBJECTIVES COMPLIANCE
def s_cumplimiento():
    s = slide("4.4", "Cumplimos la discriminación, la explicabilidad y los falsos positivos; calibración y segmento quedan abiertos")
    rows = [
        ["Objetivo", "Estado", "Evidencia"],
        ["OM1, discriminación", "Cumplido", "Gini Stability 0,484 frente a 0,438; 18 de 18 semanas"],
        ["OM1, calibración", "No cumplido", "Predicho / observado 1,62; ECE 0,0136"],
        ["OM2", "Cumplido", "SHAP estable; cambios por periodo identificados"],
        ["OM3", "Cumplido", "−4,1 puntos de FPR a recall de 80 %"],
        ["OM4", "Parcial", "Solo el detector de calibración, agregado después, ubica el cambio"],
        ["ON1", "Cumplido", "En orden de riesgo, vía OM1"],
        ["ON2", "Parcial", "Cumple en la población total; no en el segmento sin historial"],
        ["ON3", "Parcial", "Se detecta cuándo y por qué cambia la confiabilidad (escala, no orden), con un detector posterior"],
    ]
    col = {"Cumplido": TEAL, "No cumplido": BRICK, "Parcial": AMBER_T}
    colors = {(i, 1): col[rows[i][1]] for i in range(1, 9)}
    table(s, ML, 1.95, CW, rows, [2.6, 1.8, 7.73], size=12.5, row_h=0.5, colors=colors, highlight=1,
          bold_cells=[(r, 0) for r in range(1, 9)])
    notes(s, "Este es el balance de objetivos. Cumplidos: OM1 en discriminación, OM2 y OM3, y ON1 en orden de "
             "riesgo. No cumplido: la calibración de OM1. Parciales: OM4, porque el detector que ubica el "
             "cambio se agregó después; ON2, porque no se cumple en el segmento sin historial; y ON3, "
             "porque detectamos cuándo y por qué cambia la confiabilidad, pero con un detector posterior. "
             "Preferimos mostrar estos resultados abiertos antes que ocultarlos.")


# ================================================================= 22. SELECTED MODEL + LIMITS
def s_modelo_final():
    s = slide("4.4", "Modelo seleccionado: LightGBM, útil para ordenar el riesgo y no para estimar probabilidades absolutas")
    card(s, ML, 1.95, CW, 1.6, fill=INK)
    tx(s, ML + 0.3, 2.05, 5, 0.3, "MODELO SELECCIONADO", 11.5, "9FD0D3", True)
    tx(s, ML + 0.3, 2.38, CW - 0.6, 0.45, "LightGBM · 63 hojas · 300 árboles · tasa de aprendizaje 0,05 · semilla 42", 16, WHITE, True)
    tx(s, ML + 0.3, 2.85, CW - 0.6, 0.5,
       "Ajustado con las semanas 0 a 59 sobre la vista de árboles de E5-v2 (110 predictores efectivos), con "
       "calibración isotónica y umbrales de las semanas 60 a 73.", 12.5, "E6F1F2")
    gx = 0.3
    cw2 = (CW - gx) / 2
    cy = 3.75
    card(s, ML, cy, cw2, 3.1)
    dk.box(s, ML, cy, 0.09, 3.1, fill=C(TEAL))
    tx(s, ML + 0.3, cy + 0.12, cw2 - 0.5, 0.3, "POR QUÉ ESTE MODELO", 11.5, TEAL, True)
    paras(s, ML + 0.3, cy + 0.48, cw2 - 0.5, 2.7, [
        [R("Mayor Gini Stability en el OOT (criterio principal) y ventaja en todas las semanas.", 13, TEXT)],
        [R("Ajuste en 17 segundos.", 13, TEXT)],
        [R("Explicaciones estables y reproducción con un comando por etapa.", 13, TEXT)],
        [R("Su probabilidad no debe usarse como valor absoluto hasta recalibrarla.", 13, BRICK, True)],
    ], after=8)
    x2 = ML + cw2 + gx
    card(s, x2, cy, cw2, 3.1, fill="F0E6D3")
    dk.box(s, x2, cy, 0.09, 3.1, fill=C(AMBER))
    tx(s, x2 + 0.3, cy + 0.12, cw2 - 0.5, 0.3, "LIMITACIONES", 11.5, AMBER_T, True)
    paras(s, x2 + 0.3, cy + 0.48, cw2 - 0.5, 2.7, [
        [R("OOT ya examinado en el avance 2; no existe un periodo intacto.", 12, TEXT)],
        [R("Horizonte de target no publicado: la madurez de etiqueta no se puede verificar.", 12, TEXT)],
        [R("Segmento sin historial pequeño (234 incumplimientos) y definido por ausencia de fuentes.", 12, TEXT)],
        [R("Una sola semilla de entrenamiento; la estabilidad se midió para SHAP, no para el modelo.", 12, TEXT)],
        [R("Detector de calibración diseñado después de ver el resultado.", 12, TEXT)],
    ], after=6)
    notes(s, "El modelo seleccionado es LightGBM con 63 hojas y 300 árboles, ajustado con las semanas 0 a 59 sobre "
             "110 predictores efectivos, con calibración isotónica y umbrales de las semanas 60 a 73. Lo "
             "elegimos por su mayor Gini Stability en el OOT, su ventaja en todas las semanas, su ajuste en 17 "
             "segundos y sus explicaciones estables. Su probabilidad no debe usarse como valor absoluto hasta "
             "recalibrarla. Entre las limitaciones: el OOT ya se examinó, el horizonte del target no es público, "
             "el segmento sin historial es pequeño, usamos una sola semilla y el detector de calibración se "
             "diseñó después de ver el resultado.")


# ================================================================= 23. NEXT STEPS
def s_proximos():
    s = slide("4.4", "Próximos pasos: recalibrar, vigilar la calibración y atender al segmento sin historial")
    steps = [
        ("Recalibrar", "con las semanas posteriores a la 65 y validar el resultado en un periodo nuevo."),
        ("Incorporar la alerta de calibración semanal", "al monitoreo, con umbral fijado de antemano."),
        ("Modelos específicos o variables alternativas", "para el segmento sin historial (ON2)."),
        ("Verificar con el proveedor el horizonte de target", "para confirmar o descartar la madurez de etiqueta."),
        ("Repetir con varias semillas", "y estimar intervalos de las diferencias entre modelos."),
    ]
    y = 1.95
    for i, (t, d) in enumerate(steps):
        card(s, ML, y, CW, 0.82, fill=INK if i == 0 else PANEL)
        dk.box(s, ML, y, 0.75, 0.82, fill=C(TEAL))
        tx(s, ML, y, 0.75, 0.82, str(i + 1), 24, WHITE, True, font=HEAD, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        paras(s, ML + 1.0, y, CW - 1.2, 0.82, [[R(t + " ", 15, WHITE if i == 0 else INK, True),
                                                  R(d, 15, "E6F1F2" if i == 0 else TEXT)]
                                                 + ([R("   PRIORIDAD INMEDIATA", 11, "F2D58A", True)] if i == 0 else [])],
              anchor=MSO_ANCHOR.MIDDLE, after=0)
        y += 0.95
    notes(s, "Los próximos pasos son cinco. Primero, recalibrar con las semanas posteriores a la 65 y validar en un "
             "periodo nuevo. Segundo, incorporar la alerta de calibración semanal al monitoreo con un umbral "
             "fijado de antemano. Tercero, explorar modelos específicos o variables alternativas para el "
             "segmento sin historial. Cuarto, verificar con el proveedor el horizonte del target. Y quinto, "
             "repetir con varias semillas y estimar intervalos de las diferencias entre modelos.")


# ================================================================= 24. REFERENCES
def s_refs():
    s = slide()
    tx(s, ML, 0.26, 7.5, 0.3, "REFERENCIAS", 12, TEAL, True)
    tx(s, ML, 0.72, CW, 0.6, "Referencias de los artículos y fuentes fundacionales citados", 25, INK, True, font=HEAD)
    dk.a11y_title(s, "Referencias")
    refs = [
        "[1] Dumitrescu, E., Hué, S., Hurlin, C. y Tokpavi, S. (2022). Machine learning for credit scoring: Improving "
        "logistic regression with non-linear decision-tree effects. European Journal of Operational Research, "
        "297(3), 1178–1192. https://doi.org/10.1016/j.ejor.2021.06.053",
        "[3] Bellotti, A. y Crook, J. (2013). Forecasting and stress testing credit card default using dynamic models. "
        "International Journal of Forecasting, 29(4), 563–574. https://doi.org/10.1016/j.ijforecast.2013.04.003",
        "[5] Chen, Y., Calabrese, R. y Martin-Barragan, B. (2024). Interpretable machine learning for imbalanced credit "
        "scoring datasets. European Journal of Operational Research, 312(1), 357–372. "
        "https://doi.org/10.1016/j.ejor.2023.06.036",
        "[6] Hlongwane, R., Ramaboa, K. y Mongwe, W. (2024). A novel framework for enhancing transparency in credit "
        "scoring: Leveraging Shapley values for interpretable credit scorecards. PLOS ONE, 19(8), e0308718. "
        "https://doi.org/10.1371/journal.pone.0308718",
        "[10] Barddal, J. P., Loezer, L., Enembreck, F. y Lanzuolo, R. (2020). Lessons learned from data stream "
        "classification applied to credit scoring. Expert Systems with Applications, 162, 113899. "
        "https://doi.org/10.1016/j.eswa.2020.113899",
    ]
    paras(s, ML, 1.5, CW, 3.2, [[R(r, 13, TEXT)] for r in refs], after=7)
    tx(s, ML, 4.85, CW, 0.3, "FUENTES FUNDACIONALES (CONFERENCIAS, NO Q1 NI Q2)", 11.5, TEAL, True)
    found = [
        "Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in "
        "Neural Information Processing Systems, 30, 4765–4774.",
        "Bifet, A. y Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. Proceedings of the "
        "2007 SIAM International Conference on Data Mining. https://doi.org/10.1137/1.9781611972771.42",
    ]
    paras(s, ML, 5.2, CW, 1.3, [[R(r, 12, TEXT)] for r in found], after=7)
    tx(s, ML, 6.55, CW, 0.3, "Datos: Home Credit – Credit Risk Model Stability (Kaggle). Cuartiles: JCR o SJR 2024, de la revista.",
       11, MUTED)
    tx(s, ML, H - 0.42, 9, 0.28, "Segundo avance · Capítulo 4: Modelamiento · UNMSM · FISI", 10, MUTED)
    tx(s, W - ML - 1.0, H - 0.42, 1.0, 0.28, str(IDX[0]), 10, MUTED, align=PP_ALIGN.RIGHT)
    notes(s, "Estas son las referencias completas de los cinco artículos que sustentan las técnicas, con la "
             "numeración usada en las diapositivas, y las dos fuentes fundacionales de SHAP y ADWIN, que son de "
             "conferencia y por eso no se cuentan como Q1 o Q2.")


# ================================================================= 25. CLOSING
def s_cierre():
    s = slide(ground=INK)
    dk.box(s, 0, 0, 0.22, H, fill=C(TEAL))
    tx(s, 0.9, 0.7, 8, 0.3, "CONCLUSIONES DEL CAPÍTULO 4", 13, "9FD0D3", True)
    tx(s, 0.9, 1.2, 8.5, 1.5, "Un modelo que ordena mejor el riesgo, y una escala que aún debe recalibrarse",
       32, WHITE, True, font=HEAD, ls=1.08)
    pts = [
        ("0,484", "Gini Stability de LightGBM en el OOT, y gana en 18 de 18 semanas"),
        ("1,62×", "sobreestimación del riesgo: la calibración no se cumple"),
        ("> 65", "semanas posteriores con las que se propone recalibrar"),
    ]
    y = 3.35
    for big, small in pts:
        tx(s, 0.9, y, 1.9, 0.7, big, 30, "9FD0D3", True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
        tx(s, 2.9, y, 6.4, 0.7, small, 15, "E6F1F2", anchor=MSO_ANCHOR.MIDDLE)
        y += 0.85
    tx(s, 0.9, 5.95, 8, 0.5, "Gracias. ¿Preguntas?", 20, WHITE, True, font=HEAD)
    tx(s, 0.9, 6.5, 11, 0.3, "Nick Salcedo · Gabriel Davalos · Andre Cuenca · Giancarlo Villavicencio · Mathias Torres",
       11, "D5E8E9")
    # motif echo
    x0, wtot, y0 = 9.9, 3.0, 1.0
    sc = wtot / 92
    for i, (fit_end, vlen, gap) in enumerate([(40, 10, 0), (50, 10, 0), (60, 14, 0), (60, 18, 14)]):
        y = y0 + i * 0.5
        tx(s, x0 - 0.5, y, 0.45, 0.3, ["P1", "P2", "P3", "OOT"][i], 11, "D5E8E9", True,
           align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        dk.box(s, x0, y, fit_end * sc, 0.3, fill=C(TEAL))
        dk.box(s, x0 + (fit_end + gap) * sc, y, vlen * sc, 0.3, fill=C(AMBER if i < 3 else "F2D58A"))
    tx(s, x0, y0 + 2.1, wtot, 0.3, "Ventanas de validación fuera de tiempo", 11, "9FD0D3")
    dk.a11y_title(s, "Conclusiones y cierre")
    notes(s, "Para cerrar: LightGBM ordena mejor el riesgo que la regresión logística, con un Gini Stability de "
             "0,484 y ventaja en las 18 semanas del OOT. Pero la escala de su probabilidad aún no es confiable, "
             "con una sobreestimación de 1,62 veces, y por eso el siguiente paso es recalibrar con las "
             "semanas posteriores a la 65. Agradecemos su atención y quedamos atentos a sus preguntas.")


for f in [s_cover, s_crisp, s_objetivos, s_datos, s_modelos, s_papers, s_particiones, s_reglas, s_metricas,
          s_controles, s_registro, s_hiper, s_ablacion, s_oot, s_semanal, s_fp, s_calibracion, s_segmento,
          s_shap, s_om4, s_cumplimiento, s_modelo_final, s_proximos, s_refs, s_cierre]:
    f()

dk.declare_delivery(str(OUT), "presented", builds="static")
dk.lint_layout(prs, strict=os.environ.get("STRICT", "1") == "1")
prs.save(str(OUT))
print("saved", OUT, "slides:", len(prs.slides))
