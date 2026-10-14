"""Generate the complementary EDA figures from results/eda/*.csv (written by `uv run -m dm_credit.eda`).

Run from the repo root:
    uv run --with matplotlib python figures/make_eda_figures.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "results" / "eda"
OUT = ROOT / "figures"

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
SHORT = {
    "person_1__birth_259d__age_years__all": "Edad del solicitante (años)",
    "applprev_1__credamount_590a__mean__all": "Monto medio de solicitudes previas",
}


def read(name: str) -> list[dict]:
    with open(EDA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def es(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


# --- Variables continuas: histogramas -------------------------------------------------
hist = read("histograms.csv")
features = list(dict.fromkeys(r["feature"] for r in hist))
fig, axes = plt.subplots(1, len(features), figsize=(10, 3.4))
for ax, feat in zip(axes, features):
    rows = [r for r in hist if r["feature"] == feat]
    left = [float(r["bin_left"]) for r in rows]
    width = [float(r["bin_right"]) - float(r["bin_left"]) for r in rows]
    ax.bar(left, [int(r["count"]) for r in rows], width=width, align="edge", color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_title(SHORT.get(feat, feat), fontsize=10)
    ax.set_ylabel("Casos")
    thousands = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    ax.yaxis.set_major_formatter(thousands)
    ax.xaxis.set_major_formatter(thousands)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
fig.suptitle("Distribución de variables continuas (entrenamiento)", fontsize=11)
save(fig, "02_eda_continuous.png")

# --- Variables categóricas y ordinal: tasa de incumplimiento por categoría -------------
rates = read("category_rates.csv")
panels = [
    ("person_1__incometype_1044t__main__all", "Tipo de ingreso (nominal)", None),
    ("person_1__empl_employedtotal_800l__main__all", "Antigüedad laboral total (ordinal)",
     ["LESS_ONE", "MORE_ONE", "MORE_FIVE", "__missing__"]),
]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, (feat, title, order) in zip(axes, panels):
    rows = [r for r in rates if r["feature"] == feat and float(r["share"]) >= 0.3]
    if order:
        rows.sort(key=lambda r: order.index(r["category"]))
    labels = [r["category"].replace("__missing__", "(sin dato)") for r in rows]
    values = [float(r["target_rate"]) for r in rows]
    ax.barh(range(len(rows)), values, color=[LIGHT if r["category"] == "__missing__" else BLUE for r in rows])
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(values):
        ax.text(v + 0.05, y, es(v, 2) + " %", va="center", fontsize=8)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Tasa de incumplimiento (%)")
fig.suptitle("Tasa de incumplimiento por categoría (entrenamiento)", fontsize=11)
save(fig, "02_eda_categorical.png")

# --- Multivariado: mapa de calor de correlaciones ---------------------------------------
corr = read("correlation_top15.csv")
names = list(dict.fromkeys(r["feature_1"] for r in corr))
idx = {n: i for i, n in enumerate(names)}
matrix = [[0.0] * len(names) for _ in names]
for r in corr:
    matrix[idx[r["feature_1"]]][idx[r["feature_2"]]] = float(r["pearson_r"]) if r["pearson_r"] else 0.0
labels = [n.split("__", 1)[1].rsplit("__", 1)[0] if "__" in n else n for n in names]
fig, ax = plt.subplots(figsize=(8.6, 7.2))
im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(names)), labels, rotation=90, fontsize=7)
ax.set_yticks(range(len(names)), labels, fontsize=7)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="r de Pearson")
ax.set_title("Correlación entre los 15 predictores con mayor |Gini| univariado", fontsize=10)
save(fig, "02_eda_correlation.png")

# --- Dimensional: varianza explicada por componentes principales -----------------------
pca = read("pca_variance.csv")
comp = [int(r["components"]) for r in pca]
cum = [100 * float(r["cumulative_variance"]) for r in pca]
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(comp, cum, color=BLUE, linewidth=2)
for level in (80, 90, 95):
    k = next(c for c, v in zip(comp, cum) if v >= level)
    ax.axhline(level, color=LIGHT, linestyle="--", linewidth=1)
    ax.annotate(f"{level} %: {k} componentes", (k, level), xytext=(k + 8, level - 7), fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#666"})
ax.set_xlabel("Número de componentes principales")
ax.set_ylabel("Varianza acumulada (%)")
ax.set_ylim(0, 102)
ax.set_title(f"Análisis de componentes principales de la vista lineal ({len(comp)} predictores)", fontsize=10)
save(fig, "02_eda_pca.png")
