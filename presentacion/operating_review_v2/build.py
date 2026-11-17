"""Build the DataMine operating review from the supplied PowerPoint template."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from xml.dom import minidom as DOM
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "reference.pptx"
OUTPUT = HERE / "Revision_Modelo_Crediticio_Operating_Review.pptx"
ORDER = [1, 2, 3, 4, 12, 17, 5, 10, 18, 23, 24, 28, 29, 30, 31]


def direct_children(node, tag):
    return [c for c in node.childNodes if c.nodeType == c.ELEMENT_NODE and c.tagName == tag]


def first(node, tag):
    nodes = node.getElementsByTagName(tag)
    return nodes[0] if nodes else None


def set_text(node, value):
    if node.firstChild:
        node.firstChild.nodeValue = value
    else:
        node.appendChild(node.ownerDocument.createTextNode(value))


def replace_shape(doc, shape_id, paragraphs):
    shapes = doc.getElementsByTagName("p:sp")
    target = None
    for shape in shapes:
        nv = first(shape, "p:cNvPr")
        if nv and nv.getAttribute("id") == str(shape_id):
            target = shape
            break
    if target is None:
        raise KeyError(f"Missing shape {shape_id}")
    body = first(target, "p:txBody")
    ps = direct_children(body, "a:p")
    if len(paragraphs) > len(ps):
        raise ValueError(f"Shape {shape_id} has only {len(ps)} paragraphs")
    for i, p in enumerate(ps):
        value = paragraphs[i] if i < len(paragraphs) else ""
        runs = p.getElementsByTagName("a:t")
        if not runs and value:
            r = doc.createElement("a:r")
            t = doc.createElement("a:t")
            r.appendChild(t)
            p.appendChild(r)
            runs = [t]
        for j, t in enumerate(runs):
            set_text(t, value if j == 0 else "")


TEXT = {
    1: {
        4: ["Riesgo crediticio\nModelo y evaluación"],
        5: ["Informe de avance", "16 nov 2026"],
        2: ["DataMine", "UNMSM · FISI"],
        3: ["2026"], 7: ["DataMine"],
    },
    2: {3: ["Evaluación del modelo v2"], 5: ["Contenido"], 6: ["2"]},
    3: {
        533: ["Síntesis: mejor ordenamiento, calibración pendiente"],
        8: ["RESULTADOS"], 10: ["LÍMITES"],
        534: ["LightGBM supera a la regresión logística en el OOT." ,
              "Gini Stability: 0,484 frente a 0,438.",
              "AUC: 0,752 frente a 0,731; PR-AUC: 0,075 frente a 0,062.",
              "Gana en las 18 semanas OOT.",
              "FPR al 80 % de recall: 45,4 % frente a 49,5 %."],
        9: ["El riesgo absoluto requiere recalibración.",
            "Predicción / observado: 1,62; ECE: 0,0136.",
            "Sin ventaja en historial limitado: AUC 0,707 vs 0,709.",
            "PSI y ADWIN no alertaron el cambio de prevalencia.",
            "Prototipo y modelo serializado: pendientes."], 5: ["3"],
    },
    4: {
        533: ["Qué resuelve el proyecto"],
        5: ["Clasificar solicitudes según riesgo de impago y sostener la decisión en el tiempo con métricas verificables."],
        7: ["TRES OBJETIVOS"],
        20: ["01  Ordenar el riesgo", "Comparar discriminación con una línea base constante y regresión logística."],
        21: ["02  Reducir rechazos", "Medir falsos positivos a un recall cercano al 80 % de impagos."],
        22: ["03  Vigilar el tiempo", "Evaluar estabilidad semanal, calibración y calidad de las explicaciones."],
        6: ["4"],
    },
    12: {
        8: ["Fuentes seleccionadas · 1 de 2"],
        2: ["Tres aportes", "Los papers orientan diseño y evaluación; no implican que todos sus métodos estén implementados."],
        9: ["Barddal et al. · 2020", "Credit scoring temporal y aprendizaje adaptativo. Motiva el corte por semanas y la vigilancia del drift. DOI: 10.1016/j.eswa.2020.113899"],
        10: ["Chen et al. · 2024", "SHAP bajo desbalance de clases. Motiva comprobar estabilidad de rankings explicativos. DOI: 10.1016/j.ejor.2023.06.036"],
        11: ["Dumitrescu et al. · 2022", "Comparación entre modelos lineales y no lineales. Sustenta el benchmark logístico; PLTR no se implementó. DOI: 10.1016/j.ejor.2021.06.053"],
        12: ["5"],
    },
    17: {
        533: ["Fuentes seleccionadas · 2 de 2"],
        14: ["Dos aportes", "LightGBM y la dimensión temporal completan el marco del experimento.", "Ambos se usan como referencia conceptual."],
        3: ["Evidencia de modelos de árboles y explicabilidad para scoring crediticio."],
        4: ["Hlongwane et al. · 2024", "LightGBM + Shapley", "DOI: 10.1371/journal.pone.0308718"],
        11: ["Pronóstico de default por período; inspira seguimiento temporal del riesgo."],
        13: ["Bellotti y Crook · 2013", "Supervivencia discreta: no implementada", "DOI: 10.1016/j.ijforecast.2013.04.003"],
        26: ["La selección combina capacidad predictiva, explicación y validación temporal; el modelo entrenado usa LightGBM."],
        6: ["6"],
    },
    5: {
        533: ["Diseño temporal y datos"],
        5: ["E5-v2 reúne las variables de solicitudes y antecedentes; pasó 29 de 29 controles de integración."],
        7: ["PROTOCOLO SIN MEZCLAR SEMANAS"],
        534: ["Entrenamiento · 0–59", "Ajuste de candidatos; 122 características candidatas y 110 predictores efectivos para árboles."],
        2: ["Validación · 60–73", "P3 fija calibración isotónica y umbrales antes del período de prueba."],
        3: ["OOT · 74–91", "194 658 casos; 4 260 impagos (2,19 %). Evaluación final en 18 semanas."],
        6: ["7"],
    },
    10: {
        533: ["Modelos entrenados y criterio de selección"],
        6: ["Comparación controlada", "Mismos datos y período OOT; la elección principal usa Gini Stability, con chequeos de AUC, PR-AUC y errores."],
        8: ["Constante", "Referencia mínima: AUC 0,500 y Gini Stability 0."],
        10: ["Regresión logística", "C = 0,1; base interpretable y competitiva."],
        3: ["LightGBM seleccionado", "63 hojas, 300 árboles, tasa 0,05; semilla 42."],
        5: ["Refit 0–73", "Solo contraste posterior; no sustituye al modelo P3 ni sus umbrales."],
        12: ["8"],
    },
    18: {
        533: ["Métricas OOT del modelo seleccionado"],
        14: ["Comparación P3", "194 658 casos de las semanas 74–91. ↓ indica que un valor menor es preferible."],
        4: ["9"],
    },
    23: {
        11: ["Discriminación fuera de tiempo"],
        38: ["0,484"], 39: ["Gini Stability · LightGBM"],
        41: ["0,752"], 42: ["AUC · LightGBM"],
        6: ["Mejor ranking", "LightGBM supera a logística en Gini Stability (0,484 vs 0,438) y AUC (0,752 vs 0,731). Barras en porcentaje."],
        5: ["10"],
    },
    24: {
        11: ["Estabilidad por semana OOT"],
        38: ["18/18"], 39: ["Semanas gana LightGBM"],
        41: ["0,437"], 42: ["Peor Gini semanal · semana 77"],
        43: ["Gini semanal en porcentaje, semanas 74–91. El promedio de LightGBM es 0,503; supera a logística en cada semana."],
        2: ["11"],
    },
    28: {
        11: ["Umbral y calibración"],
        12: ["45,4 %"], 16: ["FPR · LightGBM"],
        35: ["49,5 %"], 36: ["FPR · logística"],
        38: ["1,62×"], 39: ["Predicho / observado"],
        41: ["0,0136"], 42: ["ECE · LightGBM"],
        43: ["Al recall cercano al 80 %, la reducción de FPR equivale a unos 7 700 buenos pagadores menos rechazados en OOT."],
        5: ["La calibración falla el objetivo ECE < 0,01. No usar probabilidades absolutas sin recalibración y nueva validación."],
        2: ["12"],
    },
    29: {
        3: ["Explicabilidad y estabilidad de SHAP"],
        14: ["Ranking reproducible", "Cinco muestras independientes en P3 y OOT; se comparó el orden de importancia de variables."],
        13: ["0,9997"], 10: ["Spearman mínimo entre rankings"],
        4: ["20/20"], 11: ["Variables principales coinciden"],
        5: ["5"], 12: ["Muestras por período"],
        9: ["13"],
    },
    30: {
        3: ["Decisión y próximos pasos"],
        28: ["Balance", "La discriminación y la estabilidad temporal respaldan LightGBM para ordenar casos.",
             "La calibración OOT, el segmento con poco historial y la detección de drift requieren trabajo adicional.",
             "Siguiente hito: recalibrar, validar por segmentos y construir el prototipo."],
        30: ["OM1"], 31: ["Ranking logrado"],
        32: ["OM2"], 33: ["FPR mejoró"],
        35: ["OM3"], 36: ["SHAP estable"],
        8: ["OM4"], 9: ["Drift parcial"],
        10: ["1,62×"], 11: ["Recalibrar"],
        12: ["0"], 13: ["Modelo serializado"],
        14: ["14"],
    },
    31: {
        4: ["Siguiente hito"],
        5: ["Recalibrar · validar · prototipar", "Fuentes y resultados en el repositorio DataMine"],
        6: ["DataMine · 2026"],
    },
}


TABLE = [
    ["Métrica", "Constante", "Logística", "LightGBM", "Lectura"],
    ["Gini Stability", "0,000", "0,438", "0,484", "↑"],
    ["AUC", "0,500", "0,731", "0,752", "↑"],
    ["PR-AUC", "0,022", "0,062", "0,075", "↑"],
    ["Brier", "0,02152", "0,02136", "0,02113", "↓"],
    ["Log loss", "0,1073", "0,1012", "0,0991", "↓"],
    ["ECE", "0,0105", "0,0142", "0,0136", "< 0,01"],
    ["Pred. / obs.", "1,48", "1,65", "1,62", "≈ 1"],
    ["FPR @ 80 %", "—", "49,5 %", "45,4 %", "↓"],
]


def replace_table(doc):
    table = first(doc, "a:tbl")
    rows = direct_children(table, "a:tr")
    for row, values in zip(rows, TABLE):
        cells = direct_children(row, "a:tc")
        for cell, value in zip(cells, values):
            ts = cell.getElementsByTagName("a:t")
            for j, t in enumerate(ts):
                set_text(t, value if j == 0 else "")


def replace_agenda(doc):
    table = first(doc, "a:tbl")
    topics = [
        "Síntesis del experimento", "Objetivos y fuentes", "Datos y diseño temporal",
        "Modelo y métricas OOT", "Explicabilidad y límites", "Decisión y próximos pasos",
    ]
    rows = direct_children(table, "a:tr")
    for i, row in enumerate(rows[:6]):
        cells = direct_children(row, "a:tc")
        vals = [f"{i+1:02d}", topics[i]]
        for cell, val in zip(cells, vals):
            ts = cell.getElementsByTagName("a:t")
            if ts:
                set_text(ts[0], val)
                for t in ts[1:]: set_text(t, "")


def set_chart_cache(ref, labels, values, formula):
    f = first(ref, "c:f")
    set_text(f, formula)
    cache = first(ref, "c:strCache") or first(ref, "c:numCache")
    pt_count = first(cache, "c:ptCount")
    pt_count.setAttribute("val", str(len(values)))
    for pt in list(cache.getElementsByTagName("c:pt")):
        cache.removeChild(pt)
    for i, value in enumerate(values):
        pt = cache.ownerDocument.createElement("c:pt")
        pt.setAttribute("idx", str(i))
        v = cache.ownerDocument.createElement("c:v")
        v.appendChild(cache.ownerDocument.createTextNode(str(value)))
        pt.appendChild(v)
        cache.appendChild(pt)


def replace_chart(data, labels, series_names, series_values):
    d = DOM.parseString(data)
    for j, ser in enumerate(d.getElementsByTagName("c:ser")):
        set_chart_cache(first(ser, "c:tx"), labels, [series_names[j]], f"Sheet1!${'BC'[j]}$1")
        set_chart_cache(first(ser, "c:cat"), labels, labels, f"Sheet1!$A$2:$A${len(labels)+1}")
        set_chart_cache(first(ser, "c:val"), labels, series_values[j], f"Sheet1!${'BC'[j]}$2:${'BC'[j]}${len(labels)+1}")
    chart = first(d, "c:chart")
    for title in direct_children(chart, "c:title"):
        chart.removeChild(title)
    auto = first(chart, "c:autoTitleDeleted")
    if auto is None:
        auto = d.createElement("c:autoTitleDeleted")
        chart.insertBefore(auto, chart.firstChild)
    auto.setAttribute("val", "1")
    for tag in d.getElementsByTagName("c:showVal"):
        tag.setAttribute("val", "0")
    if len(labels) == 3:
        legend = first(chart, "c:legend")
        layout = first(legend, "c:manualLayout")
        for tag, value in [("c:x", "0.18"), ("c:w", "0.70")]:
            first(layout, tag).setAttribute("val", value)
    return d.toxml(encoding="utf-8")


def replace_workbook(data, labels, names, series):
    with ZipFile(io.BytesIO(data)) as inp:
        pieces = {name: inp.read(name) for name in inp.namelist()}
    sheet = DOM.parseString(pieces["xl/worksheets/sheet1.xml"])
    strings = DOM.parseString(pieces["xl/sharedStrings.xml"])
    table = DOM.parseString(pieces["xl/tables/table1.xml"])
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    values = [" ", *names, *labels]
    sst = strings.documentElement
    for c in list(sst.childNodes):
        if c.nodeType == c.ELEMENT_NODE and c.localName == "si": sst.removeChild(c)
    for v in values:
        si = strings.createElementNS(ns, "si")
        t = strings.createElementNS(ns, "t")
        t.appendChild(strings.createTextNode(v))
        si.appendChild(t)
        sst.appendChild(si)
    sst.setAttribute("count", str(len(values)))
    sst.setAttribute("uniqueCount", str(len(values)))
    dim = first(sheet, "dimension")
    dim.setAttribute("ref", f"A1:C{len(labels)+1}")
    body = first(sheet, "sheetData")
    for c in list(body.childNodes): body.removeChild(c)
    for i in range(len(labels)+1):
        row = sheet.createElementNS(ns, "row")
        row.setAttribute("r", str(i+1))
        row.setAttribute("spans", "1:3")
        for j, letter in enumerate("ABC"):
            cell = sheet.createElementNS(ns, "c")
            cell.setAttribute("r", f"{letter}{i+1}")
            v = sheet.createElementNS(ns, "v")
            if i == 0:
                cell.setAttribute("t", "s")
                v.appendChild(sheet.createTextNode(str(j)))
            elif j == 0:
                cell.setAttribute("t", "s")
                v.appendChild(sheet.createTextNode(str(2+i)))
            else:
                v.appendChild(sheet.createTextNode(str(series[j-1][i-1])))
            cell.appendChild(v)
            row.appendChild(cell)
        body.appendChild(row)
    table.documentElement.setAttribute("ref", f"A1:C{len(labels)+1}")
    cols = first(table, "tableColumns").getElementsByTagName("tableColumn")
    for col, name in zip(cols, [" ", *names]): col.setAttribute("name", name)
    pieces["xl/worksheets/sheet1.xml"] = sheet.toxml(encoding="utf-8")
    pieces["xl/sharedStrings.xml"] = strings.toxml(encoding="utf-8")
    pieces["xl/tables/table1.xml"] = table.toxml(encoding="utf-8")
    out = io.BytesIO()
    with ZipFile(out, "w") as z:
        for name, content in pieces.items(): z.writestr(name, content)
    return out.getvalue()


def build():
    with ZipFile(SOURCE) as z:
        pieces = {name: z.read(name) for name in z.namelist()}
    for number in ORDER:
        key = f"ppt/slides/slide{number}.xml"
        doc = DOM.parseString(pieces[key])
        for shape_id, paras in TEXT[number].items(): replace_shape(doc, shape_id, paras)
        if number == 2: replace_agenda(doc)
        if number == 18: replace_table(doc)
        pieces[key] = doc.toxml(encoding="utf-8")
    presentation = DOM.parseString(pieces["ppt/presentation.xml"])
    slide_list = first(presentation, "p:sldIdLst")
    old = direct_children(slide_list, "p:sldId")
    for node in old: slide_list.removeChild(node)
    for number in ORDER: slide_list.appendChild(old[number-1])
    pieces["ppt/presentation.xml"] = presentation.toxml(encoding="utf-8")

    bars_labels = ["Constante", "Logística", "LightGBM"]
    bars_names = ["AUC (%)", "Gini Stability (%)"]
    bars = [[50.0, 73.062, 75.184], [0.0, 43.836, 48.408]]
    pieces["ppt/charts/chart1.xml"] = replace_chart(pieces["ppt/charts/chart1.xml"], bars_labels, bars_names, bars)
    wk1 = "ppt/embeddings/Microsoft_Excel_Worksheet.xlsx"
    pieces[wk1] = replace_workbook(pieces[wk1], bars_labels, bars_names, bars)

    with (ROOT / "results/modeling/v2-run1/weekly_gini.csv").open() as f:
        rows = list(csv.DictReader(f))
    filtered = [r for r in rows if r["stage"] == "final" and r["period"] == "OOT" and r["variant"] == "p3_model"]
    labels = [str(i) for i in range(74, 92)]
    lines = []
    for model in ["logistic", "lightgbm"]:
        by_week = {int(r["WEEK_NUM"]): float(r["gini"]) * 100 for r in filtered if r["model"] == model}
        lines.append([round(by_week[i], 2) for i in range(74, 92)])
    line_names = ["Logística", "LightGBM"]
    pieces["ppt/charts/chart2.xml"] = replace_chart(pieces["ppt/charts/chart2.xml"], labels, line_names, lines)
    wk2 = "ppt/embeddings/Microsoft_Excel_Worksheet1.xlsx"
    pieces[wk2] = replace_workbook(pieces[wk2], labels, line_names, lines)

    with ZipFile(OUTPUT, "w") as z:
        for name, content in pieces.items(): z.writestr(name, content)
    print(OUTPUT)


if __name__ == "__main__":
    build()
