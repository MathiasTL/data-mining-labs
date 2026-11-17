// Chapter 4 (Modeling) deck builder. Reads results from ../insumos and writes capitulo4_modelado.pptx.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const IN = path.join(__dirname, "..", "insumos");
const FIG = path.join(IN, "figures");
const OUT = path.join(__dirname, "capitulo4_modelado.pptx");

// ---------------------------------------------------------------- data
function csv(name) {
  const [head, ...rows] = fs.readFileSync(path.join(IN, name), "utf8").trim().split(/\r?\n/);
  const split = (line) => {
    const out = []; let cur = "", q = false;
    for (const ch of line) {
      if (ch === '"') q = !q;
      else if (ch === "," && !q) { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur); return out;
  };
  const cols = split(head);
  return rows.map((r) => Object.fromEntries(split(r).map((v, i) => [cols[i], isNaN(+v) || v === "" ? v : +v])));
}
const cmp = Object.fromEntries(csv("model_comparison.csv").map((r) => [r.model, r]));
const weekly = csv("weekly_gini.csv");
const sweep = csv("threshold_sweep.csv");
const shap = csv("shap_importance.csv");
const example = csv("shap_example_case.csv");
const drift = csv("drift_weekly.csv");
const adwin = csv("adwin_detections.csv");
const log = fs.readFileSync(path.join(IN, "experiment_log.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
const shapLog = log.find((r) => r.stage === "shap");

const LR = cmp["Logística L2"], RF = cmp["Random Forest"], GB = cmp["LightGBM"];
const n = (v, d = 3) => v.toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: false });
const pct = (v, d = 1) => n(v * 100, d) + " %";
const int = (v) => String(Math.round(v)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

// ---------------------------------------------------------------- theme
const THEME = {
  name: "Riesgo Estable",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14232E", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EDF3F4",
    accent1: "0E7C7B", accent2: "E39B2D", accent3: "2A78D6", accent4: "EB6834",
    accent5: "1BAF7A", accent6: "8A3B5C", hlink: "0E7C7B", folHlink: "0B3C49",
  },
};
const K = { ink: "14232E", deep: "0B3C49", teal: "0E7C7B", amber: "E39B2D", tint: "EDF3F4", mute: "5B6B73",
  line: "D5E0E3", white: "FFFFFF", lr: "2A78D6", rf: "EB6834", gb: "1BAF7A", soft: "9FC6C8" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Capítulo 4: Modelamiento";

pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: K.white },
  objects: [
    { text: { text: "Capítulo 4 · Modelamiento", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: K.mute, isTextBox: true } } },
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.35, w: 9, h: 0.35, fontSize: 12, bold: true, color: K.teal, charSpacing: 2 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.68, w: 12.1, h: 1.0, fontFace: "Cambria", fontSize: 25, bold: true, color: K.ink, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: K.mute, align: "right" },
});
pres.defineSlideMaster({ title: "OSCURO", background: { color: K.deep }, objects: [] });

let sectionName = "Inicio";
function section(t) { sectionName = t; pres.addSection({ title: t }); }
function content(kicker, title) {
  const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sectionName });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function tx(s, text, o) { s.addText(text, { isTextBox: true, fontFace: "Calibri", color: K.ink, fontSize: 14, valign: "top", margin: 0, ...o }); }
function card(s, x, y, w, h, fill = K.tint) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}
function stat(s, x, y, w, value, label, color = K.teal) {
  card(s, x, y, w, 1.45);
  tx(s, value, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.65, fontFace: "Cambria", fontSize: 32, bold: true, color });
  tx(s, label, { x: x + 0.25, y: y + 0.85, w: w - 0.5, h: 0.5, fontSize: 12, color: K.mute });
}
function chip(s, x, y, color) { s.addShape(pres.shapes.OVAL, { x, y, w: 0.16, h: 0.16, fill: { color }, line: { color } }); }
const TH = (t) => ({ text: t, options: { bold: true, color: K.white, fill: { color: K.deep }, fontSize: 13 } });
function table(s, rows, o) {
  s.addTable(rows, { fontFace: "Calibri", fontSize: 13, color: K.ink, border: { type: "solid", pt: 0.75, color: K.line },
    valign: "middle", margin: [4, 8, 4, 8], ...o });
}
const AX = { catAxisLabelColor: K.mute, valAxisLabelColor: K.mute, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, valGridLine: { color: "E3E9EB", size: 0.75 }, catGridLine: { style: "none" },
  legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: K.ink, titleFontFace: "+mn-lt" };

// ================================================================ 1. cover
section("Inicio");
{
  const s = pres.addSlide({ masterName: "OSCURO", sectionTitle: sectionName });
  tx(s, "UNIVERSIDAD NACIONAL MAYOR DE SAN MARCOS", { x: 0.8, y: 0.6, w: 11, h: 0.4, fontSize: 14, bold: true, color: K.white, charSpacing: 2 });
  tx(s, "Facultad de Ingeniería de Sistemas e Informática", { x: 0.8, y: 1.0, w: 11, h: 0.35, fontSize: 13, color: K.soft });
  tx(s, "Estabilidad del riesgo crediticio en la originación", { x: 0.8, y: 2.0, w: 8.2, h: 1.6, fontFace: "Cambria", fontSize: 40, bold: true, color: K.white });
  tx(s, "Segundo avance del proyecto integrador · Capítulo 4: Modelamiento", { x: 0.8, y: 3.75, w: 8.2, h: 0.4, fontSize: 18, color: K.amber });
  tx(s, "CRISP-DM · Fase 4", { x: 0.8, y: 4.25, w: 8, h: 0.35, fontSize: 14, color: K.soft });
  card(s, 9.4, 1.9, 3.3, 4.6, "12495A");
  tx(s, "INTEGRANTES", { x: 9.7, y: 2.15, w: 2.8, h: 0.3, fontSize: 11, bold: true, color: K.amber, charSpacing: 2 });
  tx(s, ["Nick Salcedo", "Gabriel Davalos", "Andre Cuenca", "Giancarlo Villavicencio", "Mathias Torres"].map((t, i, a) =>
    ({ text: t, options: { breakLine: i < a.length - 1 } })), { x: 9.7, y: 2.5, w: 2.9, h: 2.2, fontSize: 14, color: K.white, paraSpaceAfter: 6 });
  tx(s, "DOCENTE", { x: 9.7, y: 4.95, w: 2.8, h: 0.3, fontSize: 11, bold: true, color: K.amber, charSpacing: 2 });
  tx(s, "Pedro Martin Lezama Gonzales", { x: 9.7, y: 5.28, w: 2.9, h: 0.6, fontSize: 14, color: K.white });
  tx(s, "Lima, Perú 2026", { x: 0.8, y: 6.6, w: 5, h: 0.35, fontSize: 13, color: K.soft });
}

// ================================================================ 2. CRISP-DM
section("Contexto");
{
  const s = content("Ubicación metodológica", "El modelado convierte los datos preparados en un score de riesgo");
  const phases = [["1", "Comprensión del negocio", "Objetivos ON/OM"], ["2", "Comprensión de los datos", "Más de 30 tablas"],
    ["3", "Preparación de los datos", "Matriz por caso"], ["4", "Modelado", "Este avance"], ["5", "Evaluación", "Siguiente fase"], ["6", "Despliegue", "Prototipo"]];
  phases.forEach(([num, name, note], i) => {
    const x = 0.6 + i * 2.05, on = num === "4";
    card(s, x, 2.0, 1.85, 2.6, on ? K.deep : K.tint);
    tx(s, num, { x: x + 0.2, y: 2.15, w: 1.4, h: 0.7, fontFace: "Cambria", fontSize: 36, bold: true, color: on ? K.amber : K.teal });
    tx(s, name, { x: x + 0.2, y: 2.95, w: 1.5, h: 0.9, fontSize: 15, bold: true, color: on ? K.white : K.ink });
    tx(s, note, { x: x + 0.2, y: 3.95, w: 1.5, h: 0.5, fontSize: 12, color: on ? K.soft : K.mute });
  });
  card(s, 0.6, 5.05, 12.1, 1.5);
  tx(s, "Foco del capítulo", { x: 0.9, y: 5.25, w: 4, h: 0.35, fontSize: 14, bold: true, color: K.teal });
  tx(s, "Estimar la probabilidad de incumplimiento de cada solicitud, comprobar que se mantiene en semanas posteriores al entrenamiento, explicarla y detectar cuándo reentrenar.",
    { x: 0.9, y: 5.62, w: 6.6, h: 0.8, fontSize: 14 });
  tx(s, "4.1 Selección de técnicas  ·  4.2 Plan de pruebas  ·  4.3 Construcción  ·  4.4 Evaluación", { x: 7.8, y: 5.62, w: 4.7, h: 0.8, fontSize: 14, color: K.deep, bold: true });
}

// ================================================================ 3. business objectives
{
  const s = content("Objetivos de negocio", "Una evaluación de riesgo precisa, explicable y que avise cuándo deja de ser confiable");
  const on = [
    ["ON1", "Reducir las pérdidas asociadas al incumplimiento, mejorando la identificación de las solicitudes con mayor riesgo.", "El modelo ordena el riesgo mejor que una referencia simple, medido en el periodo posterior al entrenamiento."],
    ["ON2", "Favorecer el acceso al crédito de los solicitantes solventes, incluidos quienes tienen historial limitado, reduciendo el riesgo de rechazarlos indebidamente.", "Menos solventes rechazados que con la referencia, para el mismo nivel de detección de incumplimientos."],
    ["ON3", "Mantener confiable la evaluación de riesgo a lo largo del tiempo y saber cuándo hay que reentrenarla.", "Cada pérdida de confiabilidad se detecta indicando su semana y se explica con las variables que cambiaron."],
  ];
  on.forEach(([id, obj, crit], i) => {
    const y = 1.85 + i * 1.6;
    card(s, 0.6, y, 12.1, 1.42);
    s.addShape(pres.shapes.OVAL, { x: 0.85, y: y + 0.36, w: 0.7, h: 0.7, fill: { color: K.deep }, line: { color: K.deep } });
    tx(s, id, { x: 0.85, y: y + 0.36, w: 0.7, h: 0.7, fontSize: 14, bold: true, color: K.white, align: "center", valign: "middle" });
    tx(s, obj, { x: 1.8, y: y + 0.22, w: 5.6, h: 1.05, fontSize: 14, valign: "middle" });
    tx(s, [{ text: "Criterio de éxito: ", options: { bold: true, color: K.teal } }, { text: crit }], { x: 7.7, y: y + 0.22, w: 4.8, h: 1.05, fontSize: 13, color: K.ink, valign: "middle" });
  });
}

// ================================================================ 4. mining objectives
{
  const s = content("Objetivos de minería de datos", "Cuatro resultados analíticos que aportan evidencia a los tres fines de negocio");
  const rows = [
    [TH("OM"), TH("Objetivo de minería"), TH("Sirve a"), TH("Criterio de éxito")],
    ["OM1", "Clasificación binaria supervisada que estime la probabilidad de incumplimiento de cada solicitud; entrenada con WEEK_NUM < 74 y validada fuera de tiempo (OOT) en las 18 semanas ≥ 74.", "ON1, ON3", "Gini Stability OOT mayor que la del modelo base; calibración revisada con Brier y curva de confiabilidad."],
    ["OM2", "Identificar las variables que más contribuyen a las predicciones, global y por caso, sin asumir causalidad, comparando entrenamiento con OOT.", "ON3, ON1", "Ranking de factores reproducible, con los cambios entre periodos identificados."],
    ["OM3", "Evaluar, para varios umbrales, el balance entre detectar incumplimientos y clasificar como riesgosos a clientes que pagan.", "ON2, ON1", "Para el mismo nivel de detección, menos falsos positivos que el modelo base."],
    ["OM4", "Detectar los momentos en que cambian el desempeño o la población de solicitantes, para definir las ventanas de reentrenamiento.", "ON3", "Alertas con su semana identificada, contrastadas con el Gini semanal."],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, color: j === 0 ? K.teal : K.ink, fill: { color: i % 2 ? K.white : K.tint } } })));
  table(s, rows, { x: 0.6, y: 1.8, w: 12.1, colW: [0.8, 5.7, 1.2, 4.4], fontSize: 12.5 });
  tx(s, "La relación es N:M: cada objetivo de negocio queda cubierto por al menos dos objetivos de minería.", { x: 0.6, y: 6.45, w: 12, h: 0.4, fontSize: 13, color: K.mute, italic: true });
}

// ================================================================ 5. techniques
section("4.1 Selección");
{
  const s = content("4.1 Selección de técnicas", "Del modelo base interpretable al ensamble que capta relaciones no lineales");
  const lv = [
    ["NIVEL 1 · MODELO BASE", "Regresión logística L2", "Combinación lineal de variables imputadas, escaladas y codificadas. Da la probabilidad directamente y sirve de referencia explicable.", K.lr],
    ["NIVEL 2 · ENSAMBLE POR BAGGING", "Random Forest", "Muchos árboles profundos sobre muestras y subconjuntos de variables, promediados para reducir la varianza.", K.rf],
    ["NIVEL 3 · ENSAMBLE POR BOOSTING", "LightGBM", "Árboles sucesivos que corrigen los errores previos; trata nulos y categorías de forma nativa.", K.gb],
  ];
  lv.forEach(([k, name, d, c], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.85, 3.85, 2.75);
    chip(s, x + 0.3, 2.13, c);
    tx(s, k, { x: x + 0.6, y: 2.05, w: 3.1, h: 0.3, fontSize: 11, bold: true, color: K.mute, charSpacing: 1 });
    tx(s, name, { x: x + 0.3, y: 2.5, w: 3.3, h: 0.5, fontFace: "Cambria", fontSize: 22, bold: true });
    tx(s, d, { x: x + 0.3, y: 3.1, w: 3.3, h: 1.4, fontSize: 13, color: K.ink });
  });
  const rows = [[TH("Objetivo"), TH("Técnica")],
    ["OM2 · Explicación", "Contribuciones TreeSHAP de LightGBM, globales y por caso"],
    ["OM3 · Umbral", "Curva FPR–recall y curva de confiabilidad (calibración)"],
    ["OM4 · Ventanas", "PSI semanal del score y ADWIN sobre el score y sobre la pérdida por caso"]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, fill: { color: i % 2 ? K.white : K.tint } } })));
  table(s, rows, { x: 0.6, y: 4.9, w: 12.1, colW: [3.0, 9.1], fontSize: 13 });
}

// ================================================================ 6. test plan
section("4.2 Plan de pruebas");
{
  const s = content("4.2 Plan de pruebas", "Separación temporal por semana: el modelo nunca ve el periodo en que se evalúa");
  const blocks = [["Ajuste", "Semanas 0–63", "1 282 082", "3,32 %", 64], ["Selección", "Semanas 64–73", "49 919", "2,42 %", 10], ["Prueba OOT", "Semanas 74–91", "194 658", "2,19 %", 18]];
  const total = 92, W = 12.1;
  let x = 0.6;
  blocks.forEach(([name, weeks, cases, rate, w], i) => {
    const raw = blocks.map((b) => Math.max(b[4] / total, 0.2)), sum = raw.reduce((a, b) => a + b, 0);
    const bw = W * raw[i] / sum, col = [K.soft, K.amber, K.deep][i];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.9, w: bw - 0.1, h: 0.55, fill: { color: col }, line: { color: col }, rectRadius: 0.06 });
    tx(s, `${name} · ${weeks}`, { x: x + 0.15, y: 1.9, w: bw - 0.3, h: 0.55, fontSize: 13, bold: true, color: i === 2 ? K.white : K.ink, valign: "middle" });
    x += bw;
  });
  const rows = [[TH("Bloque"), TH("Semanas"), TH("Casos"), TH("Incumplimientos"), TH("Tasa"), TH("Uso")],
    ["Ajuste", "0 a 63", "1 282 082", "42 524", "3,32 %", "Entrenar candidatos"],
    ["Selección", "64 a 73", "49 919", "1 210", "2,42 %", "Elegir hiperparámetros, early stopping y umbral"],
    ["Reajuste final", "0 a 73", "1 332 001", "43 734", "3,28 %", "Modelo definitivo con los parámetros elegidos"],
    ["Prueba OOT", "74 a 91 (18)", "194 658", "4 260", "2,19 %", "Evaluación ciega y estabilidad"]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, align: j >= 2 && j <= 4 ? "right" : "left", fill: { color: i === 4 ? "DCEBEC" : i % 2 ? K.white : K.tint } } })));
  table(s, rows, { x: 0.6, y: 2.85, w: 12.1, colW: [1.9, 1.7, 1.6, 1.9, 1.2, 3.8] });
  card(s, 0.6, 5.55, 5.9, 1.15); card(s, 6.8, 5.55, 5.9, 1.15);
  tx(s, [{ text: "Sin fuga temporal. ", options: { bold: true, color: K.teal } }, { text: "Imputación, escalado y vocabulario de categorías se ajustan solo con semanas < 74." }], { x: 0.85, y: 5.7, w: 5.4, h: 0.9, fontSize: 13 });
  tx(s, [{ text: "Cambio de prevalencia. ", options: { bold: true, color: K.teal } }, { text: "La tasa baja de 3,28 % a 2,19 % en OOT: por eso se evalúa también la calibración." }], { x: 7.05, y: 5.7, w: 5.4, h: 0.9, fontSize: 13 });
}

// ================================================================ 7. metrics
{
  const s = content("Métricas y criterio de selección", "La Gini Stability premia el poder predictivo y castiga su caída en el tiempo");
  card(s, 0.6, 1.85, 6.3, 2.35, K.deep);
  tx(s, "MÉTRICA PRINCIPAL", { x: 0.9, y: 2.05, w: 5, h: 0.3, fontSize: 11, bold: true, color: K.amber, charSpacing: 2 });
  tx(s, "Gini Stability", { x: 0.9, y: 2.4, w: 5.8, h: 0.6, fontFace: "Cambria", fontSize: 28, bold: true, color: K.white });
  tx(s, "media(Gini semanal) + 88 · min(0, a) − 0,5 · desv(residuos)", { x: 0.9, y: 3.05, w: 5.8, h: 0.4, fontFace: "Courier New", fontSize: 13, color: K.white });
  tx(s, "Gini semanal = 2 · AUC − 1; a = pendiente de la recta ajustada sobre las semanas OOT.", { x: 0.9, y: 3.5, w: 5.8, h: 0.6, fontSize: 12, color: K.soft });
  const comp = [["Gini / AUC global", "OM1"], ["PR-AUC (clase minoritaria)", "OM1"], ["Brier y curva de confiabilidad", "OM1"], ["FPR a recall 0,80", "OM3"], ["Ranking TreeSHAP train vs. OOT", "OM2"], ["PSI del score (0,10 / 0,25) y ADWIN (δ = 0,002)", "OM4"]];
  comp.forEach(([m, om], i) => {
    const y = 1.85 + i * 0.4;
    tx(s, om, { x: 7.3, y, w: 0.8, h: 0.35, fontSize: 13, bold: true, color: K.teal });
    tx(s, m, { x: 8.1, y, w: 4.6, h: 0.35, fontSize: 14 });
  });
  const rules = [["1", "Selección dentro de cada familia por AUC en semanas 64–73."], ["2", "Comparación entre familias con la Gini Stability OOT."],
    ["3", "A igual desempeño, se prefiere el modelo más simple y mejor calibrado."], ["4", "OOT se abre una sola vez, con los modelos ya fijados."]];
  rules.forEach(([k, t], i) => {
    const x = 0.6 + i * 3.05;
    card(s, x, 4.6, 2.85, 1.95);
    tx(s, k, { x: x + 0.25, y: 4.75, w: 1, h: 0.6, fontFace: "Cambria", fontSize: 28, bold: true, color: K.amber });
    tx(s, t, { x: x + 0.25, y: 5.4, w: 2.4, h: 1.05, fontSize: 13 });
  });
}

// ================================================================ 8. variables
{
  const s = content("Variables de entrada", "94 predictores de solicitudes previas y personas, más indicadores de cobertura por fuente");
  stat(s, 0.6, 1.85, 2.9, "94", "predictores aprobados en preparación");
  stat(s, 3.7, 1.85, 2.9, "30", "indicadores de cobertura (registros y presencia por fuente)");
  stat(s, 6.8, 1.85, 2.9, "234", "columnas en la vista lineal", K.lr);
  stat(s, 9.9, 1.85, 2.8, "114", "columnas en la vista de árboles", K.gb);
  const rows = [[TH("Vista"), TH("Modelos"), TH("Tratamiento")],
    ["Lineal", "Logística L2", "Imputación con estadísticos de train, indicadores de nulo, escalado y one-hot de categorías (raras → __UNKNOWN__)."],
    ["Árboles", "Random Forest, LightGBM", "Nulos nativos y categorías como códigos enteros con vocabulario congelado en train."]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, fill: { color: i % 2 ? K.white : K.tint } } })));
  table(s, rows, { x: 0.6, y: 3.65, w: 12.1, colW: [1.6, 2.8, 7.7] });
  tx(s, "Fuentes con fecha filtradas contra date_decision: solo entra información disponible al momento de decidir.", { x: 0.6, y: 6.2, w: 12, h: 0.4, fontSize: 13, color: K.mute, italic: true });
}

// ================================================================ 9. configurations
section("4.3 Construcción");
{
  const s = content("4.3 Construcción", "Seis configuraciones evaluadas y reajuste final en menos de un minuto por modelo");
  const sec = (stage, model) => log.find((r) => r.stage === stage && r.model === model).seconds;
  const rows = [[TH("Familia"), TH("Configuraciones evaluadas"), TH("Cant."), TH("Elegida")],
    ["Logística L2", "C = 0,01; 0,1; 1 · solver lbfgs · 300 iteraciones máx.", "3", "C = 1"],
    ["Random Forest", "255 hojas · 300 árboles · 63,2 % de filas y 50 % de variables por árbol", "1", "Única"],
    ["LightGBM", "31 y 63 hojas · lr 0,05 · early stopping (100) · min_child 200 · L2 = 1", "2", "63 hojas, 173 árboles"]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, color: j === 0 ? [K.lr, K.rf, K.gb][i - 1] : K.ink, align: j === 2 ? "center" : "left", fill: { color: i % 2 ? K.white : K.tint } } })));
  table(s, rows, { x: 0.6, y: 1.85, w: 12.1, colW: [2.0, 6.6, 0.9, 2.6] });
  stat(s, 0.6, 4.4, 3.8, "1 332 001", "filas en el reajuste final (semanas 0–73)", K.ink);
  stat(s, 4.75, 4.4, 3.8, n(sec("refit", "LightGBM"), 1).replace(",", ",") + " s", "reajuste de LightGBM (173 árboles)", K.gb);
  stat(s, 8.9, 4.4, 3.8, n(sec("refit", "Logística L2"), 1) + " s", "reajuste de la logística L2", K.lr);
  tx(s, "Semilla 42 en todos los modelos; el registro de cada experimento queda en experiment_log.jsonl.", { x: 0.6, y: 6.2, w: 12, h: 0.4, fontSize: 13, color: K.mute, italic: true });
}

// ================================================================ 10. validation
{
  const s = content("Comparación en selección (semanas 64–73)", "LightGBM lidera desde la selección, antes de abrir el periodo de prueba");
  const sel = log.filter((r) => r.stage === "select");
  const label = (r) => `${r.model} · ${r.config.replace("leaves=", "").replace(/^(\d+)$/, "$1 hojas")}`;
  const ordered = [...sel].sort((a, b) => b.val_auc - a.val_auc);
  s.addChart(pres.charts.BAR, [{ name: "AUC", labels: ordered.map((r) => label(r).replace("C=", "C = ").replace(/· (\d+)$/, "· $1 hojas")), values: ordered.map((r) => +r.val_auc.toFixed(4)) }], {
    x: 0.6, y: 1.8, w: 7.6, h: 4.9, barDir: "bar", catAxisOrientation: "maxMin", valAxisMinVal: 0.68, valAxisMaxVal: 0.75,
    chartColors: ordered.map((r) => (r.model === "LightGBM" ? K.gb : r.model === "Random Forest" ? K.rf : K.lr)),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: K.ink, dataLabelFormatCode: "0.000", dataLabelFontFace: "+mn-lt",
    showLegend: false, barGapWidthPct: 45, valAxisLabelFormatCode: "0.00", ...AX });
  stat(s, 8.6, 1.85, 4.1, "+" + n((GB.val_auc - LR.val_auc) * 100, 1) + " pts", "de AUC de LightGBM sobre la logística en selección", K.gb);
  stat(s, 8.6, 3.5, 4.1, "+" + n((GB.val_auc - RF.val_auc) * 100, 1) + " pts", "de AUC de LightGBM sobre Random Forest", K.gb);
  card(s, 8.6, 5.15, 4.1, 1.55);
  tx(s, "Con 63 hojas, el early stopping se detuvo en 173 árboles; con 31 hojas necesitó 479 y quedó por debajo.", { x: 8.85, y: 5.3, w: 3.6, h: 1.3, fontSize: 13 });
}

// ================================================================ 11. OOT results
section("4.4 Evaluación");
{
  const s = content("Resultados fuera de tiempo (18 semanas)", `LightGBM alcanza una Gini Stability de ${n(GB.gini_stability)} y supera al modelo base`);
  const rows = [[TH("Modelo"), TH("AUC"), TH("Gini"), TH("Gini Stab."), TH("PR-AUC"), TH("Brier")],
    ...[GB, LR, RF].map((r, i) => [r.model, n(r.oot_auc), n(r.oot_gini), n(r.gini_stability), n(r.oot_pr_auc), n(r.oot_brier, 4)]
      .map((c, j) => ({ text: c, options: { bold: j === 0 || i === 0, color: j === 0 ? [K.gb, K.lr, K.rf][i] : K.ink, align: j ? "right" : "left", fill: { color: i === 0 ? "DDF1EA" : i % 2 ? K.white : K.tint } } })))];
  table(s, rows, { x: 0.6, y: 1.85, w: 7.0, colW: [1.9, 1.0, 1.0, 1.2, 1.0, 0.9] });
  s.addImage({ path: path.join(FIG, "04_roc_oot.png"), x: 8.0, y: 1.75, w: 4.7, h: 3.67, altText: "Curva ROC en OOT de los tres modelos" });
  stat(s, 0.6, 3.75, 3.35, "+" + n(GB.gini_stability - LR.gini_stability), "Gini Stability sobre la logística (criterio OM1 cumplido)", K.gb);
  stat(s, 4.25, 3.75, 3.35, n(GB.oot_auc), "AUC OOT de LightGBM (val. 0,741)", K.gb);
  card(s, 0.6, 5.45, 12.1, 1.25);
  tx(s, [{ text: "Lectura: ", options: { bold: true, color: K.teal } }, { text: `el AUC sube de selección a OOT en los tres modelos, sin señales de sobreajuste temporal. Random Forest ordena casi como la logística (${n(RF.oot_auc)} vs ${n(LR.oot_auc)}), pero su Gini Stability cae a ${n(RF.gini_stability)} por mayor variación semanal relativa a su media.` }],
    { x: 0.85, y: 5.6, w: 11.6, h: 1.0, fontSize: 13.5 });
}

// ================================================================ 12. weekly stability
{
  const s = content("Estabilidad semanal (OM1 · ON3)", "El Gini semanal de LightGBM se mantiene sobre el modelo base, sin tendencia a la baja");
  const weeks = [...new Set(weekly.map((r) => r.WEEK_NUM))];
  const ser = (m) => ({ name: m, labels: weeks.map(String), values: weekly.filter((r) => r.model === m).map((r) => +r.gini.toFixed(3)) });
  s.addChart(pres.charts.LINE, [ser("LightGBM"), ser("Logística L2"), ser("Random Forest")], {
    x: 0.6, y: 1.75, w: 8.2, h: 4.95, chartColors: [K.gb, K.lr, K.rf], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    valAxisMinVal: 0.3, valAxisMaxVal: 0.65, valAxisLabelFormatCode: "0.00", showLegend: true, legendPos: "b",
    showValAxisTitle: true, valAxisTitle: "Gini semanal", valAxisTitleColor: K.mute, valAxisTitleFontSize: 11, valAxisTitleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "Semana (WEEK_NUM)", catAxisTitleColor: K.mute, catAxisTitleFontSize: 11, catAxisTitleFontFace: "+mn-lt", ...AX });
  const wins = weeks.filter((w) => weekly.find((r) => r.WEEK_NUM === w && r.model === "LightGBM").gini > weekly.find((r) => r.WEEK_NUM === w && r.model === "Logística L2").gini).length;
  stat(s, 9.2, 1.85, 3.5, `${wins} de 18`, "semanas en que LightGBM supera a la logística", K.gb);
  stat(s, 9.2, 3.45, 3.5, `${n(GB.weekly_gini_min, 2)} – ${n(GB.weekly_gini_max, 2)}`, "rango del Gini semanal de LightGBM", K.ink);
  card(s, 9.2, 5.05, 3.5, 1.65);
  tx(s, `Pendiente casi nula (${n(GB.weekly_gini_slope, 4)} por semana): la penalización por tendencia es cero; el castigo viene solo de la variación (desv. ${n(GB.weekly_gini_resid_std)}).`, { x: 9.4, y: 5.18, w: 3.1, h: 1.45, fontSize: 12.5 });
}

// ================================================================ 13. threshold and calibration
{
  const s = content("Umbral y calibración (OM3 · ON2)", "A igual detección, LightGBM marca como riesgosos a menos clientes que sí pagan");
  const rec = [0.5, 0.6, 0.7, 0.8, 0.9];
  const ser = (m) => ({ name: m, labels: rec.map((r) => `recall ${n(r, 1)}`), values: rec.map((r) => +(sweep.find((x) => x.model === m && Math.abs(x.recall - r) < 1e-9).fpr * 100).toFixed(1)) });
  s.addChart(pres.charts.BAR, [ser("LightGBM"), ser("Logística L2"), ser("Random Forest")], {
    x: 0.6, y: 1.75, w: 6.2, h: 3.4, barDir: "col", chartColors: [K.gb, K.lr, K.rf], barGapWidthPct: 60, showLegend: true, legendPos: "b",
    showTitle: true, title: "FPR (%) para cada nivel de detección", titleColor: K.ink, titleFontSize: 12, valAxisLabelFormatCode: "0", ...AX });
  s.addImage({ path: path.join(FIG, "04_calibration_oot.png"), x: 7.2, y: 1.85, w: 3.9, h: 3.05, altText: "Curva de confiabilidad en OOT" });
  const saved = LR.fp_at_recall80 - GB.fp_at_recall80;
  stat(s, 0.6, 5.3, 3.0, int(saved), "solventes menos marcados como riesgosos a recall 0,80 (18 semanas)", K.gb);
  stat(s, 3.8, 5.3, 3.0, `${pct(GB.fpr_at_recall80)}`, `FPR a recall 0,80 (logística ${pct(LR.fpr_at_recall80)})`, K.gb);
  card(s, 7.0, 5.15, 5.7, 1.6);
  tx(s, [{ text: "Historial limitado (sin solicitudes previas, " + int(GB.thin_cases) + " casos): ", options: { bold: true, color: K.teal } },
    { text: `AUC ${n(GB.thin_auc)} y FPR ${pct(GB.thin_fpr_at_recall80)} frente a ${pct(LR.thin_fpr_at_recall80)} de la logística. ` },
    { text: `Calibración: score medio ${pct(GB.oot_mean_score, 2)} vs. tasa real ${pct(GB.oot_prevalence, 2)}; conviene recalibrar al nivel vigente.` }],
    { x: 7.2, y: 5.25, w: 5.35, h: 1.45, fontSize: 12 });
  tx(s, "Curva de confiabilidad por deciles de score", { x: 11.3, y: 2.1, w: 1.3, h: 1.2, fontSize: 11, color: K.mute, italic: true });
}

// ================================================================ 14. explanation
{
  const s = content("Explicación del modelo (OM2 · ON3)", "Los factores principales se mantienen, pero una fuente gana peso fuera de tiempo");
  const top = shap.slice(0, 10);
  const pretty = (f) => ({
    "person_1__incometype_1044t__main__all": "Tipo de ingreso", "person_1__birth_259d__age_years__all": "Edad del solicitante",
    "applprev_1__tenor_203l__mean__all": "Plazo medio de solicitudes previas", "tax_registry_b_1__n_records": "Registros tributarios B (cantidad)",
    "applprev_1__education_1138m__share_p33_146_175__all": "Educación en solicitudes previas", "applprev_2__children_per_parent_mean__all": "Detalles por solicitud previa",
    "applprev_1__employedfrom_700d__days_since_last__all": "Antigüedad laboral declarada", "credit_bureau_a_1__n_records": "Registros de buró A (cantidad)",
    "tax_registry_c_1__n_records": "Registros tributarios C (cantidad)", "applprev_1__approvaldate_319d__days_since_last__all": "Días desde la última aprobación",
  }[f] || f);
  s.addChart(pres.charts.BAR, [
    { name: "Entrenamiento", labels: top.map((r) => pretty(r.feature)), values: top.map((r) => +r.mean_abs_train.toFixed(3)) },
    { name: "OOT", labels: top.map((r) => pretty(r.feature)), values: top.map((r) => +r.mean_abs_oot.toFixed(3)) }], {
    x: 0.6, y: 1.75, w: 7.4, h: 5.0, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [K.soft, K.teal], barGapWidthPct: 40,
    showLegend: true, legendPos: "b", showTitle: true, title: "Contribución media absoluta (log-odds), LightGBM", titleColor: K.ink, titleFontSize: 12,
    valAxisLabelFormatCode: "0.00", catAxisLabelFontSize: 10, ...AX, catAxisLabelFontSize: 10 });
  stat(s, 8.4, 1.85, 2.05, `${shapLog.top10_overlap}/10`, "factores del top-10 compartidos", K.teal);
  stat(s, 10.65, 1.85, 2.05, n(shapLog.spearman_rank, 2), "correlación de rankings", K.teal);
  const tax = shap.find((r) => r.feature === "tax_registry_b_1__n_records");
  card(s, 8.4, 3.5, 4.3, 1.5);
  tx(s, [{ text: "Cambio detectado: ", options: { bold: true, color: K.amber } }, { text: `los registros tributarios B pasan del puesto ${tax.rank_train} al ${tax.rank_oot} y su contribución se multiplica por ${n(tax.mean_abs_oot / tax.mean_abs_train, 0)}: cambió la cobertura de esa fuente.` }],
    { x: 8.6, y: 3.6, w: 3.9, h: 1.35, fontSize: 12.5 });
  card(s, 8.4, 5.2, 4.3, 1.55);
  tx(s, [{ text: "Explicación por caso: ", options: { bold: true, color: K.teal } }, { text: `para un incumplidor con score ${pct(shapLog.example_score, 1)}, el mayor aporte al riesgo es el plazo medio previo (+${n(example[0].contribution_logodds, 2)} log-odds). La contribución no implica causalidad.` }],
    { x: 8.6, y: 5.3, w: 3.9, h: 1.4, fontSize: 12.5 });
}

// ================================================================ 15. retraining windows
{
  const s = content("Ventanas de reentrenamiento (OM4 · ON3)", "La población se mantiene estable; ADWIN marca semanas puntuales para revisar");
  s.addChart(pres.charts.BAR, [{ name: "PSI del score", labels: drift.map((r) => String(r.WEEK_NUM)), values: drift.map((r) => +r.score_psi.toFixed(3)) }], {
    x: 0.6, y: 1.75, w: 7.6, h: 3.6, barDir: "col", chartColors: [K.teal], valAxisMinVal: 0, valAxisMaxVal: 0.12, valAxisMajorUnit: 0.02,
    valAxisLabelFormatCode: "0.00", showLegend: false, showTitle: true, title: "PSI semanal del score frente a entrenamiento (umbral de vigilancia 0,10)", titleColor: K.ink, titleFontSize: 12,
    barGapWidthPct: 35, ...AX });
  const wk = (st) => adwin.filter((r) => r.stream === st && r.WEEK_NUM >= 74).map((r) => r.WEEK_NUM).join(", ");
  const rows = [[TH("Señal"), TH("Necesita etiqueta"), TH("Semanas detectadas")],
    ["PSI del score > 0,10", "No", "Ninguna (máx. " + n(Math.max(...drift.map((r) => r.score_psi)), 3) + ")"],
    ["ADWIN sobre el score", "No", wk("score")],
    ["ADWIN sobre la pérdida por caso", "Sí", wk("logloss")]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, fill: { color: i % 2 ? K.white : K.tint } } })));
  table(s, rows, { x: 0.6, y: 5.55, w: 7.6, colW: [3.2, 1.8, 2.6], fontSize: 12.5 });
  const lowest = [...weekly.filter((r) => r.model === "LightGBM")].sort((a, b) => a.gini - b.gini).slice(0, 3).map((r) => r.WEEK_NUM).sort((a, b) => a - b);
  stat(s, 8.6, 1.85, 4.1, lowest.join(" · "), "semanas con el Gini semanal más bajo de LightGBM", K.amber);
  card(s, 8.6, 3.5, 4.1, 3.25);
  tx(s, [{ text: "Contraste con el desempeño\n", options: { bold: true, color: K.teal, breakLine: true } },
    { text: "ADWIN sobre el score avisa en la semana 82, justo tras la caída de las semanas 81–82, sin esperar a conocer el incumplimiento. La señal con etiqueta llega después (87 y 90), porque el pago se observa con rezago.", options: { breakLine: true } },
    { text: "\nPropuesta: revisar el modelo ante una alerta ADWIN del score y reentrenar si se confirma con el Gini semanal." }],
    { x: 8.8, y: 3.65, w: 3.7, h: 3.0, fontSize: 12.5 });
}

// ================================================================ 16. closing decision
section("Cierre");
{
  const s = content("Verificaciones, limitaciones y decisión", "LightGBM queda seleccionado para la fase de evaluación");
  const cols = [
    ["VERIFICACIONES", K.teal, ["Transformaciones ajustadas solo con semanas < 74.", "Hiperparámetros fijados antes de abrir OOT.", "Mismos casos y etiquetas para los tres modelos.", "Semilla y registro de cada experimento."]],
    ["LIMITACIONES", K.amber, ["Scores sobreestiman la tasa vigente: falta recalibrar.", "Solo dos familias aportan predictores; el resto, cobertura.", "Búsqueda de hiperparámetros acotada.", "Datos de competición, sin validación externa."]],
    ["DECISIÓN", K.gb, [`LightGBM, 63 hojas y 173 árboles.`, `Gini Stability ${n(GB.gini_stability)} vs ${n(LR.gini_stability)} del modelo base.`, `${int(LR.fp_at_recall80 - GB.fp_at_recall80)} falsos positivos menos a recall 0,80.`, "Siguiente fase: recalibración y regla de alerta."]],
  ];
  cols.forEach(([h, c, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.85, 3.85, 4.8);
    tx(s, h, { x: x + 0.3, y: 2.05, w: 3.3, h: 0.35, fontSize: 13, bold: true, color: c, charSpacing: 2 });
    tx(s, items.map((t, j) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: j < items.length - 1 } })), { x: x + 0.3, y: 2.55, w: 3.3, h: 3.9, fontSize: 14, paraSpaceAfter: 10 });
  });
}

// ================================================================ 17. thanks
{
  const s = pres.addSlide({ masterName: "OSCURO", sectionTitle: sectionName });
  tx(s, "¡Muchas gracias!", { x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontFace: "Cambria", fontSize: 54, bold: true, color: K.white });
  tx(s, "Segundo avance del proyecto integrador · Capítulo 4: Modelamiento", { x: 0.8, y: 3.9, w: 11.7, h: 0.5, fontSize: 18, color: K.amber });
  tx(s, "Nick Salcedo · Gabriel Davalos · Andre Cuenca · Giancarlo Villavicencio · Mathias Torres", { x: 0.8, y: 4.6, w: 11.7, h: 0.4, fontSize: 14, color: K.soft });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
