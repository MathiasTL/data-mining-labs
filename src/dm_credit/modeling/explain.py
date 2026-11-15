"""OM2: SHAP global (TreeExplainer) con semilla fija y coeficientes estandarizados de la logística."""

from __future__ import annotations

import numpy as np
import polars as pl
import shap


def shap_importance(model, X: np.ndarray, *, n_sample: int = 20000, seed: int = 42) -> pl.DataFrame:
    """Media de |SHAP| por variable sobre una muestra de filas con semilla fija."""
    rng = np.random.default_rng(seed)
    if len(X) > n_sample:
        X = X[np.sort(rng.choice(len(X), n_sample, replace=False))]
    explainer = shap.TreeExplainer(model.obj)
    values = explainer.shap_values(X, check_additivity=False)
    if isinstance(values, list):  # versiones antiguas: una matriz por clase
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    mean_abs = np.abs(values).mean(axis=0)
    out = pl.DataFrame({"feature": model.names, "mean_abs_shap": mean_abs}).sort(
        "mean_abs_shap", descending=True, maintain_order=True
    )
    return out.with_columns(rank=pl.int_range(1, out.height + 1))


def rank_change(val: pl.DataFrame, oot: pl.DataFrame) -> pl.DataFrame:
    """Cambio de rango (OOT menos P3 val): negativo = sube en importancia."""
    t = val.select("feature", pl.col("mean_abs_shap").alias("shap_val"), pl.col("rank").alias("rank_val")).join(
        oot.select("feature", pl.col("mean_abs_shap").alias("shap_oot"), pl.col("rank").alias("rank_oot")),
        on="feature",
        how="full",
        coalesce=True,
    )
    return t.with_columns(rank_change=pl.col("rank_oot") - pl.col("rank_val")).sort("rank_val")


def logistic_std_coefs(model, X_fit: np.ndarray) -> pl.DataFrame:
    """Coeficiente por desviación estándar de la columna en el bloque de ajuste (|coef * std|)."""
    coef = model.obj.coef_[0]
    std = X_fit.std(axis=0)
    out = pl.DataFrame({"feature": model.names, "coef": coef, "std_coef": coef * std})
    return out.sort(pl.col("std_coef").abs(), descending=True, maintain_order=True)


SHAP_SEEDS = (42, 43, 44, 45, 46)


def stability_summary(tables: list[pl.DataFrame], top_k: int = 20) -> dict:
    """Estabilidad del ranking SHAP entre muestras independientes (Chen et al., 2024).

    Spearman entre rankings completos y Jaccard del top-k, por pares de muestras; media y mínimo.
    ``per_feature`` trae el rango medio y su desviación (ddof=1) entre muestras.
    """
    feats = tables[0].sort("feature")["feature"].to_list()
    ranks = np.array([t.sort("feature")["rank"].to_numpy() for t in tables], float)
    tops = [set(t.sort("rank").head(top_k)["feature"].to_list()) for t in tables]
    rho, jac = [], []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            d = ranks[i] - ranks[j]
            n = len(feats)
            rho.append(1 - 6 * float((d**2).sum()) / (n * (n**2 - 1)))
            jac.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    std = ranks.std(axis=0, ddof=1) if len(tables) > 1 else np.zeros(len(feats))
    return {
        "n_samples": len(tables),
        "spearman_mean": float(np.mean(rho)) if rho else float("nan"),
        "spearman_min": float(np.min(rho)) if rho else float("nan"),
        "jaccard_mean": float(np.mean(jac)) if jac else float("nan"),
        "jaccard_min": float(np.min(jac)) if jac else float("nan"),
        "per_feature": pl.DataFrame({"feature": feats, "rank_mean": ranks.mean(axis=0), "rank_std": std}).sort("rank_mean"),
    }


def shap_stability(model, X: np.ndarray, *, seeds=SHAP_SEEDS, n_sample: int = 20000, top_k: int = 20) -> dict:
    """SHAP global en K muestras independientes (una semilla cada una) y su resumen de estabilidad."""
    tables = [shap_importance(model, X, n_sample=n_sample, seed=s) for s in seeds]
    summary = stability_summary(tables, top_k)
    summary["exhaustive"] = bool(len(X) <= n_sample)  # si la muestra es toda la población, no hay variación
    summary["n_rows"] = int(len(X))
    return {"summary": summary, "tables": tables}
