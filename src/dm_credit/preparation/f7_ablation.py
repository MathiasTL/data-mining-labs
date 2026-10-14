"""F7 — Ablación de los indicadores de cobertura de productos y registros.

Entrena el mismo modelo sobre la misma vista de E6 con y sin las columnas de
``debitcard_1``, ``deposit_1``, ``other_1`` y ``tax_registry_{a,b,c}_1`` y compara
Gini Stability en OOT (métrica oficial, ``dm_credit.baseline``). Sirvió para
decidir el filtro as-of de ``products-v2``: sin filtro, la ganancia venía de
filas posteriores a la decisión; con filtro, la ganancia se mantiene y es legítima.

Gradient boosting usa la vista de árboles; la regresión logística, la lineal.
Los hiperparámetros son fijos y sin búsqueda: es una comparación, no un modelo.

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.f7_ablation
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from dm_credit import io
from dm_credit.baseline import gini, gini_stability_components, weekly_gini

FAMILIES = ("debitcard_1", "deposit_1", "other_1", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1")
CONTROLS = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target")
MODELS = {"hist_gbm": "tree", "logistic": "linear"}
RESULTS = Path(__file__).resolve().parents[3] / "results" / "preparation"


def flag_columns(columns) -> list[str]:
    """Columnas (de origen o de salida lineal) de las familias de productos y registros."""
    return [c for c in columns if c.startswith(tuple(f"{f}__" for f in FAMILIES))]


def _matrix(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values, categorical = [], []
    for c in cols:
        s = df[c]
        enum = isinstance(s.dtype, pl.Enum)
        values.append((s.to_physical() if enum else s).cast(pl.Float64).to_numpy())
        categorical.append(enum)
    return np.column_stack(values), np.array(categorical)


def _model(name: str, categorical: np.ndarray):
    if name == "hist_gbm":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, categorical_features=categorical, random_state=0,
        )
    return LogisticRegression(max_iter=300, C=0.1)


def compare(train: pl.DataFrame, oot: pl.DataFrame, *, model: str) -> pl.DataFrame:
    """Una fila por variante (con y sin indicadores) con Gini global y Gini Stability en OOT."""
    features = [c for c in train.columns if c not in CONTROLS]
    flags = set(flag_columns(features))
    rows = []
    for variant, cols in (("with_flags", features), ("without_flags", [c for c in features if c not in flags])):
        started = time.perf_counter()
        x_train, categorical = _matrix(train, cols)
        x_oot, _ = _matrix(oot, cols)
        fitted = _model(model, categorical).fit(x_train, train["target"].to_numpy())
        p_train, p_oot = fitted.predict_proba(x_train)[:, 1], fitted.predict_proba(x_oot)[:, 1]
        weekly = weekly_gini(oot["target"].to_numpy(), p_oot, oot["WEEK_NUM"].to_numpy())
        c = gini_stability_components(weekly["gini"].to_numpy(), weekly["WEEK_NUM"].to_numpy())
        rows.append({
            "model": model, "variant": variant, "n_features": len(cols),
            "gini_train": gini(train["target"].to_numpy(), p_train), "gini_oot": gini(oot["target"].to_numpy(), p_oot),
            "weekly_mean": c.mean_weekly_gini, "slope": c.slope, "slope_penalty": c.slope_penalty,
            "residual_penalty": c.variability_penalty, "gini_stability": c.score,
            "seconds": round(time.perf_counter() - started, 1),
        })
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "E6-v1")
    ap.add_argument("--out", type=Path, default=RESULTS / "F7_coverage_ablation.csv")
    args = ap.parse_args()
    parts = []
    for model, view in MODELS.items():
        train = pl.read_parquet(args.views / f"{view}__train.parquet")
        oot = pl.read_parquet(args.views / f"{view}__oot.parquet")
        parts.append(compare(train, oot, model=model))
        del train, oot
    report = pl.concat(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(args.out)
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(report.drop("seconds"))
    print(f"informe: {args.out}")


if __name__ == "__main__":
    main()
