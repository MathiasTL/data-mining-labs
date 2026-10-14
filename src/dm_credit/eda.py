"""EDA complementario sobre la matriz de entrenamiento (WEEK_NUM < 74).

Completa la exploración del capítulo 02 con lo que pide la rúbrica del examen parcial:
resumen de variables continuas y atípicos por rango intercuartílico, asociación de las
categóricas con el objetivo (chi-cuadrado y V de Cramér), correlación de Spearman para la
única variable ordinal, patrón de datos faltantes por tabla fuente, pares de predictores muy
correlacionados (Pearson) y varianza explicada por componentes principales.

Solo usa entrenamiento: OOT y test no intervienen. Correlaciones y PCA usan una muestra
aleatoria fija (semilla 0) para acotar memoria; el resto usa todas las filas.

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.eda
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA

from dm_credit import io

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TARGET = "target"
MISSING = "__missing__"
CONTROLS = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target")
# única característica con orden natural documentado (antigüedad laboral total)
ORDINAL = {"person_1__empl_employedtotal_800l__main__all": ["LESS_ONE", "MORE_ONE", "MORE_FIVE"]}


def continuous_summary(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Estadísticos por columna y porcentaje de atípicos fuera de [Q1 − 1,5·IQR, Q3 + 1,5·IQR]."""
    rows = []
    for c in columns:
        s = df[c].cast(pl.Float64)
        v = s.drop_nulls()
        q1, q3 = v.quantile(0.25, "linear"), v.quantile(0.75, "linear")
        iqr = (q3 - q1) if v.len() else None
        out = ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum() if v.len() else 0
        rows.append({
            "feature": c,
            "n_nonnull": v.len(),
            "pct_null": 100 * s.null_count() / s.len() if s.len() else 0.0,
            "mean": v.mean(), "std": v.std(), "min": v.min(), "q1": q1, "median": v.median(), "q3": q3, "max": v.max(),
            "skewness": float(stats.skew(v.to_numpy())) if v.len() > 2 else None,
            "pct_iqr_outliers": 100 * out / v.len() if v.len() else 0.0,
        })
    return pl.DataFrame(rows)


def histogram(df: pl.DataFrame, col: str, bins: int = 30) -> pl.DataFrame:
    """Conteos por intervalo de igual ancho (sin nulos) para graficar sin releer los datos."""
    values = df[col].drop_nulls().cast(pl.Float64).to_numpy()
    counts, edges = np.histogram(values, bins=bins)
    return pl.DataFrame({"feature": [col] * bins, "bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})


def _with_missing(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].cast(pl.String).fill_null(MISSING)


def category_rates(df: pl.DataFrame, col: str, target: str = TARGET) -> pl.DataFrame:
    """Casos, participación y tasa de positivos por categoría (el nulo es su propio nivel)."""
    frame = pl.DataFrame({"category": _with_missing(df, col), "y": df[target]})
    return (
        frame.group_by("category")
        .agg(n=pl.len(), target_rate=100 * pl.col("y").mean())
        .with_columns(feature=pl.lit(col), share=100 * pl.col("n") / frame.height)
        .select("feature", "category", "n", "share", "target_rate")
        .sort("n", descending=True)
    )


def categorical_association(df: pl.DataFrame, columns: Sequence[str], target: str = TARGET) -> pl.DataFrame:
    """Chi-cuadrado de independencia con el objetivo y V de Cramér (sin corrección de Yates)."""
    rows = []
    for c in columns:
        table = (
            pl.DataFrame({"k": _with_missing(df, c), "y": df[target]})
            .group_by("k", "y").len()
            .pivot(on="y", index="k", values="len").fill_null(0)
        )
        counts = table.drop("k").to_numpy()
        chi2, p, dof, _ = stats.chi2_contingency(counts, correction=False)
        n = counts.sum()
        v = float(np.sqrt(chi2 / (n * (min(counts.shape) - 1)))) if min(counts.shape) > 1 else 0.0
        top = table.with_columns(t=pl.sum_horizontal(pl.exclude("k"))).sort("t", descending=True).row(0, named=True)
        rows.append({"feature": c, "n_categories": table.height, "top_category": top["k"],
                     "top_share": 100 * top["t"] / n, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v})
    return pl.DataFrame(rows)


def ordinal_spearman(df: pl.DataFrame, col: str, order: Sequence[str], target: str = TARGET) -> dict:
    """Correlación de rangos de Spearman entre una variable ordinal (orden declarado) y el objetivo."""
    rank = {k: i for i, k in enumerate(order)}
    frame = df.select(pl.col(col).replace_strict(rank, default=None).alias("r"), pl.col(target)).drop_nulls()
    rho, p = stats.spearmanr(frame["r"].to_numpy(), frame[target].to_numpy())
    return {"feature": col, "order": " < ".join(order), "n": frame.height, "spearman_rho": float(rho), "p_value": float(p)}


def missing_pattern(df: pl.DataFrame, groups: Mapping[str, Sequence[str]]) -> pl.DataFrame:
    """Por grupo: filas con algún nulo y filas con todos nulos (ausencia estructural de la fuente)."""
    rows = []
    for name, cols in groups.items():
        nulls = df.select([pl.col(c).is_null() for c in cols])
        any_null = nulls.select(pl.any_horizontal(pl.all())).to_series()
        all_null = nulls.select(pl.all_horizontal(pl.all())).to_series()
        rows.append({"group": name, "n_features": len(cols),
                     "pct_rows_any_null": 100 * any_null.sum() / df.height,
                     "pct_rows_all_null": 100 * all_null.sum() / df.height})
    return pl.DataFrame(rows)


def high_correlations(df: pl.DataFrame, columns: Sequence[str], threshold: float = 0.9) -> pl.DataFrame:
    """Pares con |r de Pearson| ≥ umbral, calculado con las filas donde ambos valores existen."""
    rows = []
    for a, b in itertools.combinations(columns, 2):
        pair = df.select(pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)).drop_nulls()
        if pair.height < 3 or pair[a].std() == 0 or pair[b].std() == 0:
            continue
        r = float(np.corrcoef(pair[a].to_numpy(), pair[b].to_numpy())[0, 1])
        if abs(r) >= threshold:
            rows.append({"feature_1": a, "feature_2": b, "pearson_r": r, "n": pair.height})
    schema = {"feature_1": pl.String, "feature_2": pl.String, "pearson_r": pl.Float64, "n": pl.Int64}
    return pl.DataFrame(rows, schema=schema).sort(pl.col("pearson_r").abs(), descending=True)


def correlation_matrix(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Matriz de Pearson por pares (formato largo) para el mapa de calor."""
    rows = []
    for a in columns:
        for b in columns:
            if a == b:
                rows.append({"feature_1": a, "feature_2": b, "pearson_r": 1.0})
                continue
            pair = df.select(pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)).drop_nulls()
            r = float(np.corrcoef(pair[a].to_numpy(), pair[b].to_numpy())[0, 1]) if pair.height > 2 else None
            rows.append({"feature_1": a, "feature_2": b, "pearson_r": r})
    return pl.DataFrame(rows)


def pca_variance(x: np.ndarray) -> pl.DataFrame:
    """Varianza explicada por componente sobre columnas estandarizadas (las constantes se omiten)."""
    std = x.std(axis=0)
    keep = std > 0
    z = (x[:, keep] - x[:, keep].mean(axis=0)) / std[keep]
    ratio = PCA(svd_solver="full").fit(z).explained_variance_ratio_
    return pl.DataFrame({"components": np.arange(1, ratio.size + 1), "explained_variance": ratio,
                         "cumulative_variance": np.cumsum(ratio)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    prep = io.data_dir() / "artifacts" / "preparation"
    ap.add_argument("--matrix", type=Path, default=prep / "E5-v1" / "feature_matrix__train.parquet")
    ap.add_argument("--linear", type=Path, default=prep / "E6-v1" / "linear__train.parquet")
    ap.add_argument("--catalog", type=Path, default=RESULTS / "preparation" / "C8_feature_catalog.csv")
    ap.add_argument("--screen", type=Path, default=RESULTS / "preparation" / "G2_applications_people_screen.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "eda")
    ap.add_argument("--sample", type=int, default=200_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.matrix)
    if df["WEEK_NUM"].max() >= 74:
        raise ValueError("EDA must only use the train partition (WEEK_NUM < 74)")
    catalog = pl.read_csv(args.catalog).filter(pl.col("role") == "predictor")
    numeric = catalog.filter(pl.col("logical_type").is_in(["int", "float"]))["name"].to_list()
    categorical = catalog.filter(pl.col("logical_type") == "string")["name"].to_list()
    sample = df.sample(n=min(args.sample, df.height), seed=0)

    continuous_summary(df, numeric).write_csv(args.out / "continuous_summary.csv")
    categorical_association(df, categorical).sort("cramers_v", descending=True).write_csv(args.out / "categorical_association.csv")
    pl.concat([category_rates(df, c) for c in categorical]).write_csv(args.out / "category_rates.csv")
    pl.DataFrame([ordinal_spearman(df, c, o) for c, o in ORDINAL.items()]).write_csv(args.out / "ordinal_spearman.csv")
    hist = ["person_1__birth_259d__age_years__all", "applprev_1__credamount_590a__mean__all"]
    age = df[hist[0]].drop_nulls()
    bins = {hist[0]: int(age.max() - age.min()), hist[1]: 40}  # edad entera: un intervalo por año
    pl.concat([histogram(df, c, bins=bins[c]) for c in hist]).write_csv(args.out / "histograms.csv")
    groups = {t: g["name"].to_list() for (t,), g in catalog.group_by("source_table", maintain_order=True)}
    missing_pattern(df, groups).write_csv(args.out / "missing_pattern.csv")
    high_correlations(sample, numeric).write_csv(args.out / "high_correlations.csv")
    screen = pl.read_csv(args.screen).filter(pl.col("gini_all").is_not_null())
    top = screen.sort(pl.col("gini_all").abs(), descending=True).head(15)["feature"].to_list()
    correlation_matrix(sample, top).write_csv(args.out / "correlation_top15.csv")
    lin = pl.read_parquet(args.linear).sample(n=min(args.sample, df.height), seed=0)
    pca_variance(lin.drop([c for c in CONTROLS if c in lin.columns]).to_numpy().astype(float)).write_csv(args.out / "pca_variance.csv")
    pl.DataFrame({
        "key": ["rows_train", "sample_rows", "seed", "matrix", "linear_view", "numeric_predictors", "categorical_predictors"],
        "value": [str(df.height), str(sample.height), "0", "E5-v1/feature_matrix__train.parquet", "E6-v1/linear__train.parquet",
                  str(len(numeric)), str(len(categorical))],
    }).write_csv(args.out / "run_metadata.csv")
    print(f"EDA escrito en {args.out}")


if __name__ == "__main__":
    main()
