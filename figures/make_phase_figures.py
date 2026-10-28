"""Generate phase 02/03 figures from the committed result CSVs.

Run from the repo root:
    uv run --with matplotlib python figures/make_phase_figures.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
CUTOFF_WEEK = 74

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "font.size": 10,
    }
)
BLUE, LIGHT, RED = "#1f5f8b", "#9cc3de", "#c0392b"


def read(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def barh(ax, labels, values, color=BLUE, fmt="{:,.0f}"):
    ypos = range(len(labels))
    ax.barh(ypos, values, color=color)
    ax.set_yticks(list(ypos), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    top = max(values) if values else 1
    es = str.maketrans({",": ".", ".": ","})
    for y, v in zip(ypos, values):
        ax.text(v + top * 0.01, y, fmt.format(v).translate(es), va="center", fontsize=9)
    ax.set_xlim(0, top * 1.15)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))
    )


# --- 02: weekly target ------------------------------------------------------
weekly = read(RES / "target_analysis" / "weekly.csv")
weeks = [int(r["WEEK_NUM"]) for r in weekly]
rate = [float(r["default_rate_pct"]) for r in weekly]
cases = [int(r["cases"]) for r in weekly]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])
a1.plot(weeks, rate, color=BLUE, lw=2)
a1.axvline(CUTOFF_WEEK, color=RED, ls="--")
a1.text(CUTOFF_WEEK + 1, max(rate), "Corte OOT (semana 74)", color=RED, ha="left", va="top")
a1.set_ylabel("Tasa de default (%)")
a1.set_title("Tasa de default semanal y volumen de casos (base de entrenamiento)")
a2.bar(weeks, cases, color=LIGHT)
a2.axvline(CUTOFF_WEEK, color=RED, ls="--")
a2.set_ylabel("Casos por semana")
a2.set_xlabel("Semana (WEEK_NUM)")
save(fig, "02_weekly_default_rate.png")

# --- 03: per-family results -------------------------------------------------
fam = [r for r in read(RES / "preparation" / "family_results.csv") if r["split"] == "train"]
leak = {r["family"]: r for r in read(RES / "preparation" / "F7_temporal_leakage_by_family.csv")}
runs = {
    r["family"]: float(r["build_seconds"])
    for r in read(RES / "preparation" / "E4_runs.csv")
    if r["split"] == "train"
}

by_cov = sorted(fam, key=lambda r: -float(r["coverage_pct"]))
fig, ax = plt.subplots(figsize=(8, 5))
barh(ax, [r["family"] for r in by_cov], [float(r["coverage_pct"]) for r in by_cov], fmt="{:.1f}%")
ax.set_xlim(0, 115)
ax.set_xlabel("% de casos de entrenamiento con al menos una fila en la fuente")
ax.set_title("Cobertura por familia de fuentes (train, 1.332.001 casos)")
save(fig, "03_coverage_by_family.png")

by_feat = sorted(fam, key=lambda r: -(int(r["predictor_features"]) + int(r["coverage_features"])))
labels = [r["family"] for r in by_feat]
pred = [int(r["predictor_features"]) for r in by_feat]
covf = [int(r["coverage_features"]) for r in by_feat]
fig, ax = plt.subplots(figsize=(8, 5))
ypos = range(len(labels))
ax.barh(ypos, pred, color=BLUE, label="Predictores")
ax.barh(ypos, covf, left=pred, color=LIGHT, label="Columnas de cobertura")
ax.set_yticks(list(ypos), labels)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Columnas en la matriz final")
ax.set_title("Características por familia")
ax.legend(frameon=False)
save(fig, "03_features_by_family.png")

leak_rows = sorted(
    ((f, int(r["rows_excluded_train"])) for f, r in leak.items() if int(r["rows_excluded_train"]) > 0),
    key=lambda x: -x[1],
)
fig, ax = plt.subplots(figsize=(8, 3))
barh(ax, [f for f, _ in leak_rows], [v for _, v in leak_rows], color=RED)
ax.set_xlabel("Filas de la fuente excluidas por la regla de disponibilidad temporal (train)")
ax.set_title("Filtro de fuga temporal")
save(fig, "03_temporal_exclusions.png")

by_sec = sorted(runs.items(), key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(8, 5))
barh(ax, [f for f, _ in by_sec], [s for _, s in by_sec], fmt="{:.1f} s")
ax.set_xlabel("Tiempo de construcción (segundos, train)")
ax.set_title("Tiempo de construcción de agregados por familia")
save(fig, "03_build_seconds.png")

# --- 03: final matrices -----------------------------------------------------
views = read(RES / "preparation" / "E6_views.csv")
split_rows = {"train": 1332001, "oot": 194658}
positives = {"train": 43734, "oot": 4260}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
names = list(split_rows)
a1.bar(names, [split_rows[s] for s in names], color=[BLUE, LIGHT])
a1.set_title("Filas por partición")
a1.set_ylabel("Casos")
a2.bar(names, [100 * positives[s] / split_rows[s] for s in names], color=[BLUE, LIGHT])
a2.set_title("Tasa de positivos")
a2.set_ylabel("%")
for ax, vals, f in (
    (a1, [split_rows[s] for s in names], "{:,}"),
    (a2, [100 * positives[s] / split_rows[s] for s in names], "{:.2f}%"),
):
    for i, v in enumerate(vals):
        label = f.format(v).translate(str.maketrans({",": ".", ".": ","}))
        ax.text(i, v, label, ha="center", va="bottom")
    ax.set_ylim(0, max(vals) * 1.12)
a1.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))
)
a2.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))
)
fig.suptitle("Matrices finales: entrenamiento frente a OOT")
save(fig, "03_split_sizes.png")

lin = next(r for r in views if r["view"] == "linear" and r["split"] == "train")
tree = next(r for r in views if r["view"] == "tree" and r["split"] == "train")
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(["lineal", "árboles"], [int(lin["columns"]), int(tree["columns"])], color=[BLUE, LIGHT])
for i, r in enumerate((lin, tree)):
    ax.text(i, int(r["columns"]), r["columns"], ha="center", va="bottom")
ax.set_ylabel("Columnas")
ax.set_title("Vistas de la matriz final (mismas filas)")
save(fig, "03_matrix_views.png")

# --- 02: parquet files per family (2.1) -------------------------------------
import re
from collections import Counter

files = Counter()
for r in read(RES / "parquet_inventory.csv"):
    name = re.sub(r"^(train|test)_", "", r["file"].removesuffix(".parquet"))
    if re.search(r"_\d+_\d+$", name):
        name = re.sub(r"_\d+$", "", name)
    files[(name, r["split"])] += 1
fams = sorted({f for f, _ in files}, key=lambda f: -(files[(f, "train")] + files[(f, "test")]))
n_train = sum(v for (f, s), v in files.items() if s == "train")
n_test = sum(v for (f, s), v in files.items() if s == "test")
fig, ax = plt.subplots(figsize=(8, 5.5))
ypos = list(range(len(fams)))
h = 0.38
ax.barh([y - h / 2 for y in ypos], [files[(f, "train")] for f in fams], height=h, color=BLUE, label="Train")
ax.barh([y + h / 2 for y in ypos], [files[(f, "test")] for f in fams], height=h, color=LIGHT, label="Test")
ax.set_yticks(ypos, fams)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Archivos Parquet")
ax.legend(frameon=False, loc="lower right")
ax.set_title(f"{n_train} archivos de train y {n_test} de test; cuatro familias en varios fragmentos")
save(fig, "02_parquet_files_by_family.png")

# --- 02: null rate by family (2.4) ------------------------------------------
qs = read(RES / "quality_catalog_summary.csv")
rows = sorted(
    ((r["family"], 100 * int(r["columns_50pct_null_or_more"]) / int(r["columns"]), r) for r in qs),
    key=lambda x: -x[1],
)
fig, ax = plt.subplots(figsize=(8, 5.5))
vals = [v for _, v, _ in rows]
barh(ax, [f for f, _, _ in rows], vals, fmt="{:.1f}%")
for y, (_, _, r) in enumerate(rows):
    ax.text(
        1.01,
        y,
        f'{r["columns_50pct_null_or_more"]} de {r["columns"]}',
        va="center",
        ha="left",
        fontsize=8,
        color="#555",
        transform=ax.get_yaxis_transform(),
    )
ax.set_xlabel("% de columnas con al menos 50 % de nulos (columnas afectadas de columnas totales, a la derecha)")
ax.set_title("credit_bureau_a_1 tiene 65 de 79 columnas con al menos 50 % de nulos")
save(fig, "02_null_columns_by_family.png")

# --- 02: join multiplication factor (2.4) -----------------------------------
jr = [r for r in read(RES / "integration_analysis" / "join_risk.csv") if float(r["mean_expansion_all_base"]) > 1]
jr.sort(key=lambda r: -float(r["mean_expansion_all_base"]))
fig, ax = plt.subplots(figsize=(8, 5))
barh(ax, [r["family"] for r in jr], [float(r["mean_expansion_all_base"]) for r in jr], color=RED, fmt="{:.2f}×")
ax.set_xlabel("Filas tras un left join directo / filas de la base (factor medio)")
ax.set_title("Un join directo de credit_bureau_a_2 multiplica la base por 123,43")
save(fig, "02_join_multiplication.png")

# --- 03: rows excluded by temporal rule, train vs OOT (3.2) -----------------
leak_all = [
    r
    for r in read(RES / "preparation" / "F7_temporal_leakage_by_family.csv")
    if int(r["rows_excluded_train"]) + int(r["rows_excluded_oot"]) > 0
]
leak_all.sort(key=lambda r: -int(r["rows_excluded_train"]))
fig, ax = plt.subplots(figsize=(8, 4.5))
ypos = list(range(len(leak_all)))
ax.barh([y - h / 2 for y in ypos], [int(r["rows_excluded_train"]) for r in leak_all], height=h, color=BLUE, label="Train")
ax.barh([y + h / 2 for y in ypos], [int(r["rows_excluded_oot"]) for r in leak_all], height=h, color=LIGHT, label="OOT")
ax.set_xscale("symlog", linthresh=1)
ax.set_yticks(ypos, [r["family"] for r in leak_all])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
es = str.maketrans({",": ".", ".": ","})
for y, r in zip(ypos, leak_all):
    for dy, k in ((-h / 2, "rows_excluded_train"), (h / 2, "rows_excluded_oot")):
        v = int(r[k])
        ax.text(max(v, 1) * 1.15, y + dy, f"{v:,}".translate(es), va="center", fontsize=8)
ax.set_xlim(1, 1e8)
ax.set_xlabel("Filas excluidas por la regla de disponibilidad temporal (escala logarítmica)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("El filtro temporal excluye 9,1 millones de filas de bureau A en train")
save(fig, "03_exclusions_train_vs_oot.png")
