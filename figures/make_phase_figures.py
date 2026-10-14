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
ax.set_title("Filtro de fuga temporal (F7)")
save(fig, "03_temporal_exclusions.png")

by_sec = sorted(runs.items(), key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(8, 5))
barh(ax, [f for f, _ in by_sec], [s for _, s in by_sec], fmt="{:.1f} s")
ax.set_xlabel("Tiempo de construcción (segundos, train)")
ax.set_title("Tiempo de construcción de agregados por familia (E4)")
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
