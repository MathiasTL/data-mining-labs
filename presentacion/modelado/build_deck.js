/**
 * Builds the modeling-stage deck: "Riesgo crediticio · Modelo y evaluación".
 *
 * Dependencies: npm install pptxgenjs react react-dom react-icons sharp
 * Run:          node presentacion/modelado/build_deck.js
 * PDF export:   soffice --headless --convert-to pdf --outdir presentacion/modelado <pptx>
 */
"use strict";

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..", "..");
const OUT = path.join(HERE, "Riesgo_Crediticio_Modelo_y_Evaluacion.pptx");
const WEEKLY_CSV = path.join(ROOT, "results", "modeling", "v2-run1", "weekly_gini.csv");

// Palette: ink green dominates, teal marks achieved results, amber marks pending work.
const P = {
  ink: "0E2A2B",
  teal: "0F766E",
  tealSoft: "DDF0EC",
  mint: "9FE0D3",
  amber: "B45309",
  amberSoft: "FBEBD3",
  tint: "F2F6F5",
  muted: "5B6B6A",
  gray: "94A3A1",
  line: "D9E2E0",
  white: "FFFFFF",
};
const HEAD = "Georgia";
const BODY = "Arial";

const W = 13.333;
const MX = 0.6;
const CW = W - 2 * MX; // content width

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Riesgo crediticio: modelo y evaluación";
pres.author = "DataMine";
pres.company = "UNMSM · FISI";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: "DARK",
  background: { color: P.ink },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: P.white },
  objects: [
    {
      text: {
        text: "DataMine · Riesgo crediticio: modelo y evaluación",
        options: { x: MX, y: 6.98, w: 7, h: 0.3, fontSize: 10, color: P.muted, margin: 0, fontFace: BODY },
      },
    },
    {
      placeholder: {
        options: {
          name: "title", type: "title", x: MX, y: 0.78, w: CW, h: 0.75,
          fontFace: HEAD, fontSize: 30, bold: true, color: P.ink, align: "left", valign: "middle", margin: 0,
        },
        text: "",
      },
    },
  ],
  slideNumber: { x: W - MX - 0.6, y: 6.98, w: 0.6, h: 0.3, fontSize: 10, color: P.muted, align: "right", fontFace: BODY },
});

// ---------- helpers ----------

function text(slide, value, opts) {
  slide.addText(value, { isTextBox: true, margin: 0, fontFace: BODY, color: P.ink, valign: "top", ...opts });
}

function card(slide, x, y, w, h, fill = P.tint, name) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, rectRadius: 0.12, objectName: name,
  });
}

function eyebrow(slide, value, color = P.teal, y = 0.44, x = MX) {
  text(slide, value.toUpperCase(), { x, y, w: 9, h: 0.3, fontSize: 11, bold: true, color, charSpacing: 2, valign: "middle" });
}

function label(slide, value, x, y, w, color = P.teal) {
  text(slide, value.toUpperCase(), { x, y, w, h: 0.3, fontSize: 12, bold: true, color, charSpacing: 1.5, valign: "middle" });
}

async function icon(name, color) {
  const Icon = fa[name];
  if (!Icon) throw new Error(`Unknown icon: ${name}`);
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: `#${color}`, size: "256" }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function badge(slide, name, x, y, d, fill, color) {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill } });
  slide.addImage({ data: await icon(name, color), x: x + d * 0.27, y: y + d * 0.27, w: d * 0.46, h: d * 0.46 });
}

function bullets(items, fontSize = 15) {
  return items.map((t, i) => ({
    text: t,
    options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: 9, fontSize },
  }));
}

function stat(slide, x, y, w, h, value, caption, o = {}) {
  const { fill = P.tint, color = P.teal, size = 44, captionSize = 13 } = o;
  card(slide, x, y, w, h, fill);
  text(slide, value, { x: x + 0.3, y: y + 0.25, w: w - 0.5, h: h * 0.5, fontSize: size, bold: true, color, valign: "middle" });
  text(slide, caption, { x: x + 0.3, y: y + h * 0.5 + 0.3, w: w - 0.5, h: h * 0.5 - 0.45, fontSize: captionSize, color: P.muted });
}

// Decorative ascending bars: risk ordering motif for the dark slides.
function riskBars(slide) {
  const heights = [1.3, 2.1, 2.9, 3.8, 4.8];
  heights.forEach((h, i) => {
    const last = i === heights.length - 1;
    slide.addShape(pres.ShapeType.roundRect, {
      x: 9.15 + i * 0.74, y: 6.5 - h, w: 0.56, h, rectRadius: 0.1,
      fill: { color: last ? P.mint : P.teal, transparency: last ? 0 : 55 - i * 12 },
    });
  });
}

function content(section, eyebrowText, title) {
  const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  eyebrow(slide, eyebrowText);
  slide.addText(title, { placeholder: "title" });
  return slide;
}

function parseCsv(raw) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (quoted) {
      if (ch === '"' && raw[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (ch !== "\r") field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift();
  return rows.filter((r) => r.length === header.length).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

function weeklyGini() {
  const rows = parseCsv(fs.readFileSync(WEEKLY_CSV, "utf8")).filter(
    (r) => r.stage === "final" && r.period === "OOT" && r.variant === "p3_model",
  );
  const weeks = Array.from({ length: 18 }, (_, i) => 74 + i);
  const series = (model) => {
    const byWeek = Object.fromEntries(rows.filter((r) => r.model === model).map((r) => [Number(r.WEEK_NUM), Number(r.gini) * 100]));
    return weeks.map((w) => {
      if (!(w in byWeek)) throw new Error(`Missing week ${w} for ${model}`);
      return Math.round(byWeek[w] * 100) / 100;
    });
  };
  return { weeks: weeks.map(String), logistic: series("logistic"), lightgbm: series("lightgbm") };
}

const chartText = {
  catAxisLabelColor: P.muted, valAxisLabelColor: P.muted,
  catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY, legendFontFace: BODY, titleFontFace: BODY,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, legendFontSize: 12, legendColor: P.ink,
  showTitle: true, titleFontSize: 12, titleColor: P.muted,
  valGridLine: { color: P.line, size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "t",
};

// ---------- slides ----------

const S = {
  intro: "Inicio",
  s1: "01 · Síntesis del experimento",
  s2: "02 · Objetivos y fuentes",
  s3: "03 · Datos y diseño temporal",
  s4: "04 · Modelo y métricas OOT",
  s5: "05 · Explicabilidad y límites",
  s6: "06 · Decisión y próximos pasos",
};

async function build() {
  // 1 · Cover
  pres.addSection({ title: S.intro });
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: S.intro });
    text(s, "DataMine", { x: MX, y: 0.5, w: 4, h: 0.3, fontSize: 12, color: P.mint, bold: true, charSpacing: 2 });
    text(s, "2026", { x: W - MX - 2, y: 0.5, w: 2, h: 0.3, fontSize: 12, color: P.mint, align: "right" });
    s.addText(
      [
        { text: "Riesgo crediticio", options: { color: P.white, breakLine: true } },
        { text: "Modelo y evaluación", options: { color: P.mint } },
      ],
      { isTextBox: true, margin: 0, x: MX, y: 2.0, w: 8.3, h: 2.1, fontFace: HEAD, fontSize: 48, bold: true, valign: "top" },
    );
    text(s, "INFORME DE AVANCE", { x: MX, y: 5.75, w: 3.5, h: 0.3, fontSize: 11, bold: true, color: P.mint, charSpacing: 2 });
    text(s, "16 nov 2026", { x: MX, y: 6.1, w: 3.5, h: 0.4, fontSize: 18, color: P.white });
    text(s, "DATAMINE", { x: 4.4, y: 5.75, w: 3.5, h: 0.3, fontSize: 11, bold: true, color: P.mint, charSpacing: 2 });
    text(s, "UNMSM · FISI", { x: 4.4, y: 6.1, w: 3.5, h: 0.4, fontSize: 18, color: P.white });
    riskBars(s);
  }

  // 2 · Agenda
  {
    const s = content(S.intro, "Contenido", "Evaluación del modelo");
    const topics = [
      "Síntesis del experimento", "Objetivos y fuentes", "Datos y diseño temporal",
      "Modelo y métricas OOT", "Explicabilidad y límites", "Decisión y próximos pasos",
    ];
    const cw = (CW - 2 * 0.3) / 3;
    topics.forEach((t, i) => {
      const x = MX + (i % 3) * (cw + 0.3);
      const y = 1.95 + Math.floor(i / 3) * 2.45;
      card(s, x, y, cw, 2.15);
      text(s, String(i + 1).padStart(2, "0"), { x: x + 0.35, y: y + 0.3, w: 2, h: 0.7, fontSize: 34, bold: true, color: P.teal, valign: "middle" });
      text(s, t, { x: x + 0.35, y: y + 1.2, w: cw - 0.7, h: 0.7, fontSize: 19, bold: true });
    });
  }

  // 3 · Synthesis
  pres.addSection({ title: S.s1 });
  {
    const s = content(S.s1, S.s1, "Síntesis: mejor ordenamiento, calibración pendiente");
    const pw = (CW - 0.3) / 2;
    const panels = [
      {
        x: MX, fill: P.tealSoft, color: P.teal, icon: "FaCheck", head: "Resultados",
        lead: "LightGBM supera a la regresión logística en el OOT.",
        items: [
          "Gini Stability: 0,484 frente a 0,438.",
          "AUC: 0,752 frente a 0,731; PR-AUC: 0,075 frente a 0,062.",
          "Gana en las 18 semanas OOT.",
          "FPR al 80 % de recall: 45,4 % frente a 49,5 %.",
        ],
      },
      {
        x: MX + pw + 0.3, fill: P.amberSoft, color: P.amber, icon: "FaTriangleExclamation", head: "Límites",
        lead: "El riesgo absoluto requiere recalibración.",
        items: [
          "Predicción / observado: 1,62; ECE: 0,0136.",
          "Sin ventaja en historial limitado: AUC 0,707 vs 0,709.",
          "PSI y ADWIN no alertaron el cambio de prevalencia.",
          "Prototipo y modelo serializado: pendientes.",
        ],
      },
    ];
    for (const p of panels) {
      const y = 1.95;
      card(s, p.x, y, pw, 4.7, p.fill);
      await badge(s, p.icon, p.x + 0.4, y + 0.38, 0.5, p.color, P.white);
      label(s, p.head, p.x + 1.05, y + 0.48, 3, p.color);
      text(s, p.lead, { x: p.x + 0.4, y: y + 1.15, w: pw - 0.8, h: 0.8, fontSize: 19, bold: true });
      text(s, bullets(p.items), { x: p.x + 0.4, y: y + 2.1, w: pw - 0.8, h: 2.4 });
    }
  }

  // 4 · Objectives
  pres.addSection({ title: S.s2 });
  {
    const s = content(S.s2, S.s2, "Qué resuelve el proyecto");
    text(s, "Clasificar solicitudes según riesgo de impago y sostener la decisión en el tiempo con métricas verificables.", {
      x: MX, y: 1.8, w: 10.6, h: 0.85, fontSize: 20, color: P.ink,
    });
    label(s, "Tres objetivos", MX, 2.95, 4);
    const goals = [
      ["FaArrowDownWideShort", "01", "Ordenar el riesgo", "Comparar discriminación con una línea base constante y regresión logística."],
      ["FaUserCheck", "02", "Reducir rechazos", "Medir falsos positivos a un recall cercano al 80 % de impagos."],
      ["FaChartLine", "03", "Vigilar el tiempo", "Evaluar estabilidad semanal, calibración y calidad de las explicaciones."],
    ];
    const cw = (CW - 2 * 0.3) / 3;
    for (let i = 0; i < goals.length; i++) {
      const [ic, n, head, desc] = goals[i];
      const x = MX + i * (cw + 0.3), y = 3.4;
      card(s, x, y, cw, 3.2);
      await badge(s, ic, x + 0.35, y + 0.35, 0.62, P.teal, P.white);
      s.addText(
        [{ text: `${n}  `, options: { color: P.teal } }, { text: head, options: { color: P.ink } }],
        { isTextBox: true, margin: 0, x: x + 0.35, y: y + 1.2, w: cw - 0.7, h: 0.45, fontFace: BODY, fontSize: 20, bold: true, valign: "middle" },
      );
      text(s, desc, { x: x + 0.35, y: y + 1.8, w: cw - 0.7, h: 1.1, fontSize: 15, color: P.muted });
    }
  }

  // 5 · Sources 1/2
  {
    const s = content(S.s2, S.s2, "Fuentes seleccionadas · 1 de 2");
    label(s, "Tres aportes", MX, 1.75, 4);
    text(s, "Los papers orientan diseño y evaluación; no implican que todos sus métodos estén implementados.", {
      x: MX, y: 2.12, w: CW, h: 0.45, fontSize: 16,
    });
    const papers = [
      ["Barddal et al. · 2020", "Credit scoring temporal y aprendizaje adaptativo. Motiva el corte por semanas y la vigilancia del drift.", "DOI: 10.1016/j.eswa.2020.113899"],
      ["Chen et al. · 2024", "SHAP bajo desbalance de clases. Motiva comprobar estabilidad de rankings explicativos.", "DOI: 10.1016/j.ejor.2023.06.036"],
      ["Dumitrescu et al. · 2022", "Comparación entre modelos lineales y no lineales. Sustenta el benchmark logístico; PLTR no se implementó.", "DOI: 10.1016/j.ejor.2021.06.053"],
    ];
    const cw = (CW - 2 * 0.3) / 3;
    for (let i = 0; i < papers.length; i++) {
      const [who, what, doi] = papers[i];
      const x = MX + i * (cw + 0.3), y = 3.0;
      card(s, x, y, cw, 3.6);
      await badge(s, "FaBookOpen", x + 0.35, y + 0.35, 0.5, P.tealSoft, P.teal);
      text(s, who, { x: x + 0.35, y: y + 1.05, w: cw - 0.7, h: 0.4, fontSize: 18, bold: true, valign: "middle" });
      text(s, what, { x: x + 0.35, y: y + 1.6, w: cw - 0.7, h: 1.25, fontSize: 15 });
      text(s, doi, { x: x + 0.35, y: y + 2.95, w: cw - 0.7, h: 0.3, fontSize: 11, color: P.muted });
    }
  }

  // 6 · Sources 2/2
  {
    const s = content(S.s2, S.s2, "Fuentes seleccionadas · 2 de 2");
    label(s, "Dos aportes", MX, 1.75, 4);
    text(s, "LightGBM y la dimensión temporal completan el marco del experimento. Ambos se usan como referencia conceptual.", {
      x: MX, y: 2.12, w: CW, h: 0.45, fontSize: 16,
    });
    const papers = [
      ["Hlongwane et al. · 2024", "Evidencia de modelos de árboles y explicabilidad para scoring crediticio.", "LightGBM + Shapley", "DOI: 10.1371/journal.pone.0308718"],
      ["Bellotti y Crook · 2013", "Pronóstico de default por período; inspira seguimiento temporal del riesgo.", "Supervivencia discreta: no implementada", "DOI: 10.1016/j.ijforecast.2013.04.003"],
    ];
    const pw = (CW - 0.3) / 2;
    for (let i = 0; i < papers.length; i++) {
      const [who, what, method, doi] = papers[i];
      const x = MX + i * (pw + 0.3), y = 2.95;
      card(s, x, y, pw, 2.7);
      await badge(s, "FaBookOpen", x + 0.35, y + 0.32, 0.5, P.tealSoft, P.teal);
      text(s, who, { x: x + 1.0, y: y + 0.32, w: pw - 1.3, h: 0.5, fontSize: 17, bold: true, valign: "middle" });
      text(s, what, { x: x + 0.35, y: y + 1.0, w: pw - 0.7, h: 0.75, fontSize: 16 });
      text(s, method, { x: x + 0.35, y: y + 1.82, w: pw - 0.7, h: 0.3, fontSize: 13, bold: true, color: P.teal });
      text(s, doi, { x: x + 0.35, y: y + 2.17, w: pw - 0.7, h: 0.3, fontSize: 11, color: P.muted });
    }
    card(s, MX, 5.9, CW, 0.8, P.ink);
    text(s, "La selección combina capacidad predictiva, explicación y validación temporal; el modelo entrenado usa LightGBM.", {
      x: MX + 0.35, y: 5.9, w: CW - 0.7, h: 0.8, fontSize: 14, color: P.white, valign: "middle",
    });
  }

  // 7 · Temporal design
  pres.addSection({ title: S.s3 });
  {
    const s = content(S.s3, S.s3, "Diseño temporal y datos");
    label(s, "Protocolo sin mezclar semanas", MX, 1.75, 6);
    text(s, "E5 reúne las variables de solicitudes y antecedentes; pasó 29 de 29 controles de integración.", {
      x: MX, y: 2.12, w: CW, h: 0.45, fontSize: 16,
    });
    const stages = [
      { weeks: 60, fill: P.tealSoft, fg: P.ink, range: "Semanas 0–59", head: "Entrenamiento · 0–59", desc: "Ajuste de candidatos; 122 características candidatas y 110 predictores efectivos para árboles." },
      { weeks: 14, fill: P.teal, fg: P.white, range: "60–73", head: "Validación · 60–73", desc: "P3 fija calibración isotónica y umbrales antes del período de prueba." },
      { weeks: 18, fill: P.ink, fg: P.white, range: "74–91", head: "OOT · 74–91", desc: "194 658 casos; 4 260 impagos (2,19 %). Evaluación final en 18 semanas." },
    ];
    const gap = 0.05;
    const unit = (CW - 2 * gap) / 92;
    let tx = MX;
    const cw = (CW - 2 * 0.3) / 3;
    stages.forEach((st, i) => {
      const w = st.weeks * unit;
      s.addShape(pres.ShapeType.roundRect, { x: tx, y: 3.0, w, h: 0.6, rectRadius: 0.08, fill: { color: st.fill } });
      text(s, st.range, { x: tx, y: 3.0, w, h: 0.6, fontSize: 13, bold: true, color: st.fg, align: "center", valign: "middle" });
      tx += w + gap;
      const x = MX + i * (cw + 0.3), y = 4.05;
      card(s, x, y, cw, 2.55);
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.35, y: y + 0.46, w: 0.2, h: 0.2, fill: { color: st.fill }, line: { color: P.teal, width: 1 } });
      text(s, st.head, { x: x + 0.7, y: y + 0.33, w: cw - 1.0, h: 0.45, fontSize: 18, bold: true, valign: "middle" });
      text(s, st.desc, { x: x + 0.35, y: y + 1.05, w: cw - 0.7, h: 1.3, fontSize: 15, color: P.muted });
    });
  }

  // 8 · Models
  pres.addSection({ title: S.s4 });
  {
    const s = content(S.s4, S.s4, "Modelos entrenados y criterio de selección");
    label(s, "Comparación controlada", MX, 1.75, 5);
    text(s, "Mismos datos y período OOT; la elección principal usa Gini Stability, con chequeos de AUC, PR-AUC y errores.", {
      x: MX, y: 2.12, w: CW, h: 0.45, fontSize: 16,
    });
    const y0 = 3.0, h0 = 3.6;
    card(s, MX, y0, 4.6, h0, P.ink);
    s.addShape(pres.ShapeType.roundRect, { x: MX + 0.4, y: y0 + 0.42, w: 1.75, h: 0.34, rectRadius: 0.17, fill: { color: P.teal } });
    text(s, "SELECCIONADO", { x: MX + 0.4, y: y0 + 0.42, w: 1.75, h: 0.34, fontSize: 10, bold: true, color: P.white, align: "center", valign: "middle", charSpacing: 1.5 });
    text(s, "LightGBM", { x: MX + 0.4, y: y0 + 1.1, w: 3.8, h: 0.8, fontFace: HEAD, fontSize: 36, bold: true, color: P.white, valign: "middle" });
    text(s, "63 hojas, 300 árboles, tasa 0,05; semilla 42.", { x: MX + 0.4, y: y0 + 2.15, w: 3.8, h: 0.9, fontSize: 16, color: P.mint });
    const others = [
      ["Regresión logística", "C = 0,1; base interpretable y competitiva."],
      ["Constante", "Referencia mínima: AUC 0,500 y Gini Stability 0."],
      ["Refit 0–73", "Solo contraste posterior; no sustituye al modelo P3 ni sus umbrales."],
    ];
    const rx = MX + 4.9, rw = CW - 4.9, rh = (h0 - 2 * 0.2) / 3;
    others.forEach(([name, desc], i) => {
      const y = y0 + i * (rh + 0.2);
      card(s, rx, y, rw, rh);
      text(s, name, { x: rx + 0.35, y, w: 2.5, h: rh, fontSize: 17, bold: true, valign: "middle" });
      text(s, desc, { x: rx + 3.0, y, w: rw - 3.35, h: rh, fontSize: 15, color: P.muted, valign: "middle" });
    });
  }

  // 9 · Metrics table
  {
    const s = content(S.s4, S.s4, "Métricas OOT del modelo seleccionado");
    label(s, "Comparación P3", MX, 1.75, 5);
    text(s, "194 658 casos de las semanas 74–91. ↓ indica que un valor menor es preferible.", {
      x: MX, y: 2.12, w: CW, h: 0.4, fontSize: 16,
    });
    const rows = [
      ["Métrica", "Constante", "Logística", "LightGBM", "Lectura"],
      ["Gini Stability", "0,000", "0,438", "0,484", "↑"],
      ["AUC", "0,500", "0,731", "0,752", "↑"],
      ["PR-AUC", "0,022", "0,062", "0,075", "↑"],
      ["Brier", "0,02152", "0,02136", "0,02113", "↓"],
      ["Log loss", "0,1073", "0,1012", "0,0991", "↓"],
      ["ECE", "0,0105", "0,0142", "0,0136", "< 0,01"],
      ["Pred. / obs.", "1,48", "1,65", "1,62", "≈ 1"],
      ["FPR @ 80 %", "—", "49,5 %", "45,4 %", "↓"],
    ];
    const none = { type: "none" };
    const table = rows.map((r, ri) =>
      r.map((v, ci) => {
        const head = ri === 0;
        const best = ci === 3;
        return {
          text: v,
          options: {
            fontFace: BODY, fontSize: head ? 13 : 14, bold: head || best || ci === 0,
            color: head ? P.white : best ? P.teal : ci === 0 ? P.ink : P.muted,
            fill: { color: head ? P.ink : best ? P.tealSoft : P.white },
            align: ci === 0 ? "left" : "center", valign: "middle",
            margin: [0, 0.15, 0, 0.2],
            border: [none, none, { type: "solid", pt: 0.75, color: P.line }, none],
          },
        };
      }),
    );
    s.addTable(table, { x: MX, y: 2.75, w: CW, colW: [3.73, 2.1, 2.1, 2.1, 2.1], rowH: 0.44, objectName: "Métricas OOT" });
  }

  // 10 · Discrimination
  {
    const s = content(S.s4, S.s4, "Discriminación fuera de tiempo");
    const labels = ["Constante", "Logística", "LightGBM"];
    s.addChart(
      pres.charts.BAR,
      [
        { name: "AUC (%)", labels, values: [50.0, 73.062, 75.184] },
        { name: "Gini Stability (%)", labels, values: [0.0, 43.836, 48.408] },
      ],
      {
        x: MX, y: 1.85, w: 6.9, h: 4.85, barDir: "col", barGapWidthPct: 70, chartColors: [P.teal, P.ink],
        valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, title: "AUC y Gini Stability en OOT (%)", ...chartText,
      },
    );
    const rx = 7.95, rw = W - MX - rx;
    text(s, "Mejor ranking", { x: rx, y: 2.0, w: rw, h: 0.5, fontFace: HEAD, fontSize: 24, bold: true, valign: "middle" });
    text(s, "LightGBM supera a logística en Gini Stability (0,484 vs 0,438) y AUC (0,752 vs 0,731). Barras en porcentaje.", {
      x: rx, y: 2.7, w: rw, h: 1.2, fontSize: 16,
    });
    const sw = (rw - 0.25) / 2;
    stat(s, rx, 4.3, sw, 2.3, "0,484", "Gini Stability · LightGBM", { size: 40, fill: P.tealSoft });
    stat(s, rx + sw + 0.25, 4.3, sw, 2.3, "0,752", "AUC · LightGBM", { size: 40, fill: P.tealSoft });
  }

  // 11 · Weekly stability
  {
    const s = content(S.s4, S.s4, "Estabilidad por semana OOT");
    const wk = weeklyGini();
    s.addChart(
      pres.charts.LINE,
      [
        { name: "Logística", labels: wk.weeks, values: wk.logistic },
        { name: "LightGBM", labels: wk.weeks, values: wk.lightgbm },
      ],
      {
        x: MX, y: 1.85, w: 6.9, h: 4.85, chartColors: [P.gray, P.teal], lineSize: 2.5, lineDataSymbolSize: 6,
        valAxisMinVal: 30, valAxisMaxVal: 60, valAxisMajorUnit: 10, title: "Gini semanal (%) · semanas 74–91", ...chartText,
      },
    );
    const rx = 7.95, rw = W - MX - rx;
    text(s, "Gini semanal en porcentaje, semanas 74–91. El promedio de LightGBM es 0,503; supera a logística en cada semana.", {
      x: rx, y: 2.0, w: rw, h: 1.3, fontSize: 16,
    });
    const sw = (rw - 0.25) / 2;
    stat(s, rx, 4.0, sw, 2.6, "18/18", "Semanas gana LightGBM", { size: 40, fill: P.tealSoft });
    stat(s, rx + sw + 0.25, 4.0, sw, 2.6, "0,437", "Peor Gini semanal · semana 77", { size: 40 });
  }

  // 12 · Threshold and calibration
  {
    const s = content(S.s4, S.s4, "Umbral y calibración");
    const pw = (CW - 0.3) / 2;
    const panels = [
      {
        x: MX, fill: P.tealSoft, color: P.teal, icon: "FaUserCheck", head: "Umbral",
        body: "Al recall cercano al 80 %, la reducción de FPR equivale a unos 7 700 buenos pagadores menos rechazados en OOT.",
        stats: [["45,4 %", "FPR · LightGBM", P.teal], ["49,5 %", "FPR · logística", P.muted]],
      },
      {
        x: MX + pw + 0.3, fill: P.amberSoft, color: P.amber, icon: "FaTriangleExclamation", head: "Calibración",
        body: "La calibración falla el objetivo ECE < 0,01. No usar probabilidades absolutas sin recalibración y nueva validación.",
        stats: [["1,62×", "Predicho / observado", P.amber], ["0,0136", "ECE · LightGBM", P.amber]],
      },
    ];
    for (const p of panels) {
      const y = 1.95;
      card(s, p.x, y, pw, 4.7, p.fill);
      await badge(s, p.icon, p.x + 0.4, y + 0.38, 0.5, p.color, P.white);
      label(s, p.head, p.x + 1.05, y + 0.48, 3, p.color);
      text(s, p.body, { x: p.x + 0.4, y: y + 1.15, w: pw - 0.8, h: 1.1, fontSize: 16 });
      const sw = (pw - 0.8 - 0.25) / 2;
      p.stats.forEach(([v, c, col], i) => {
        stat(s, p.x + 0.4 + i * (sw + 0.25), y + 2.45, sw, 1.85, v, c, { fill: P.white, color: col, size: 36, captionSize: 12 });
      });
    }
  }

  // 13 · SHAP
  pres.addSection({ title: S.s5 });
  {
    const s = content(S.s5, S.s5, "Explicabilidad y estabilidad de SHAP");
    label(s, "Ranking reproducible", MX, 1.75, 5);
    text(s, "Cinco muestras independientes en P3 y OOT; se comparó el orden de importancia de variables.", {
      x: MX, y: 2.12, w: CW, h: 0.45, fontSize: 16,
    });
    const stats = [
      ["0,9997", "Spearman mínimo entre rankings"],
      ["20/20", "Variables principales coinciden"],
      ["5", "Muestras por período"],
    ];
    const cw = (CW - 2 * 0.3) / 3;
    stats.forEach(([v, c], i) => {
      const x = MX + i * (cw + 0.3), y = 3.0;
      card(s, x, y, cw, 3.6, i === 0 ? P.ink : P.tint);
      text(s, v, { x: x + 0.4, y: y + 0.55, w: cw - 0.8, h: 1.3, fontSize: 64, bold: true, color: i === 0 ? P.mint : P.teal, valign: "middle" });
      text(s, c, { x: x + 0.4, y: y + 2.35, w: cw - 0.8, h: 0.8, fontSize: 16, color: i === 0 ? P.white : P.muted });
    });
  }

  // 14 · Decision
  pres.addSection({ title: S.s6 });
  {
    const s = content(S.s6, S.s6, "Decisión y próximos pasos");
    const tiles = [
      ["OM1", "Ranking logrado", true], ["OM2", "FPR mejoró", true], ["OM3", "SHAP estable", true],
      ["OM4", "Drift parcial", false], ["1,62×", "Recalibrar", false], ["0", "Modelo serializado", false],
    ];
    const tw = 2.45, th = 2.25, g = 0.2;
    for (let i = 0; i < tiles.length; i++) {
      const [v, c, ok] = tiles[i];
      const x = MX + (i % 3) * (tw + g), y = 1.95 + Math.floor(i / 3) * (th + g);
      card(s, x, y, tw, th, ok ? P.tealSoft : P.amberSoft);
      await badge(s, ok ? "FaCheck" : "FaTriangleExclamation", x + tw - 0.72, y + 0.3, 0.42, ok ? P.teal : P.amber, P.white);
      text(s, v, { x: x + 0.3, y: y + 0.75, w: tw - 0.5, h: 0.75, fontSize: 36, bold: true, color: ok ? P.teal : P.amber, valign: "middle" });
      text(s, c, { x: x + 0.3, y: y + 1.6, w: tw - 0.5, h: 0.4, fontSize: 14 });
    }
    const bx = MX + 3 * tw + 2 * g + 0.3, bw = W - MX - bx, bh = 2 * th + g;
    card(s, bx, 1.95, bw, bh, P.ink);
    text(s, "Balance", { x: bx + 0.4, y: 2.3, w: bw - 0.8, h: 0.5, fontFace: HEAD, fontSize: 24, bold: true, color: P.white, valign: "middle" });
    const paras = [
      "La discriminación y la estabilidad temporal respaldan LightGBM para ordenar casos.",
      "La calibración OOT, el segmento con poco historial y la detección de drift requieren trabajo adicional.",
      "Siguiente hito: recalibrar, validar por segmentos y construir el prototipo.",
    ];
    text(
      s,
      paras.map((t, i) => ({ text: t, options: { breakLine: i < paras.length - 1, paraSpaceAfter: 16, color: i === 2 ? P.mint : P.white, bold: i === 2 } })),
      { x: bx + 0.4, y: 3.05, w: bw - 0.8, h: 3.4, fontSize: 15.5 },
    );
  }

  // 15 · Closing
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: S.s6 });
    text(s, "DataMine · 2026", { x: MX, y: 0.5, w: 5, h: 0.3, fontSize: 12, color: P.mint, bold: true, charSpacing: 2 });
    text(s, "Siguiente hito", { x: MX, y: 2.2, w: 10, h: 1.2, fontFace: HEAD, fontSize: 56, bold: true, color: P.white, valign: "middle" });
    const steps = ["Recalibrar", "Validar", "Prototipar"];
    steps.forEach((t, i) => {
      const x = MX + i * 2.75;
      s.addShape(pres.ShapeType.roundRect, { x, y: 3.95, w: 2.5, h: 0.8, rectRadius: 0.4, fill: { color: i === 2 ? P.mint : P.teal } });
      s.addText(
        [{ text: `${i + 1}   `, options: { bold: true } }, { text: t, options: { bold: true } }],
        { isTextBox: true, margin: 0, x, y: 3.95, w: 2.5, h: 0.8, fontFace: BODY, fontSize: 18, color: i === 2 ? P.ink : P.white, align: "center", valign: "middle" },
      );
    });
    text(s, "Fuentes y resultados en el repositorio DataMine", { x: MX, y: 6.1, w: 8, h: 0.4, fontSize: 16, color: P.white });
    riskBars(s);
  }

  await pres.writeFile({ fileName: OUT });
  // Drop directory entries from the package: some cloud viewers reject them.
  try {
    require("child_process").execFileSync("zip", ["-dq", OUT, "*/"]);
  } catch (err) {
    console.warn("Could not strip folder entries from the package:", err.message);
  }
  console.log(OUT);
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
