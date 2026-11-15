"""Métricas del protocolo: discriminación, calibración y umbrales por recall."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from dm_credit.baseline import gini, gini_stability, gini_stability_components

EPS = 1e-15
RECALLS = (0.70, 0.80, 0.90)


def auc(y, p) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def pr_auc(y, p) -> float:
    """Average precision del score crudo (con score constante vale la tasa base)."""
    return float(average_precision_score(np.asarray(y), np.asarray(p, float)))


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def logloss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_ratio(y, p) -> float:
    """Tasa media predicha / tasa observada."""
    return float(np.mean(p) / np.mean(y))


def calibration_table(y, p, n_bins: int = 10) -> pl.DataFrame:
    """Bins de igual frecuencia aproximada por rango del score, sin partir empates.

    Cada valor único de score se asigna entero a un bin según el punto medio de su rango
    acumulado (``(acum_previo + n_valor/2) / N * n_bins``). Así los empates (mesetas de la
    isotónica) nunca se reparten entre bins y el resultado no depende del orden de las filas.
    Puede haber menos de ``n_bins`` bins si las mesetas son grandes; los bins vacíos se omiten.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    uniq, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    before = np.cumsum(counts) - counts
    bin_of_value = np.minimum(((before + counts / 2.0) / len(p) * n_bins).astype(int), n_bins - 1)
    b = bin_of_value[inverse]
    rows = []
    for k in np.unique(b):
        m = b == k
        rows.append(
            {
                "bin": int(k) + 1,
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()),
                "obs_rate": float(y[m].mean()),
            }
        )
    return pl.DataFrame(rows)


def ece(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    w = t["n"].to_numpy() / t["n"].sum()
    return float(np.sum(w * np.abs(t["mean_pred"].to_numpy() - t["obs_rate"].to_numpy())))


def weekly_table(y, p, weeks) -> pl.DataFrame:
    """Tabla semanal; ``gini`` es nulo si la semana tiene una sola clase."""
    y, p, weeks = np.asarray(y), np.asarray(p, float), np.asarray(weeks)
    rows = []
    for w in np.unique(weeks):
        m = weeks == w
        yw, pw = y[m], p[m]
        two = len(np.unique(yw)) == 2
        rows.append(
            {
                "WEEK_NUM": int(w),
                "n": int(m.sum()),
                "n_pos": int(yw.sum()),
                "gini": gini(yw, pw) if two else None,
                "pr_auc": pr_auc(yw, pw) if two else None,
                "mean_pred": float(pw.mean()),
                "obs_rate": float(yw.mean()),
                "logloss": logloss(yw, pw),
            }
        )
    return pl.DataFrame(rows, schema_overrides={"gini": pl.Float64, "pr_auc": pl.Float64})


def summarize(y, p, weeks, n_bins: int = 10) -> dict:
    """Métricas de un periodo; Gini Stability sobre semanas con ambas clases."""
    y, p, weeks = np.asarray(y), np.asarray(p, float), np.asarray(weeks)
    tab = weekly_table(y, p, weeks)
    ok = tab.drop_nulls("gini")
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_weeks": int(ok.height),
        "n_weeks_skipped": int(tab.height - ok.height),
        "auc": auc(y, p),
        "pr_auc": pr_auc(y, p),
        "brier": brier(y, p),
        "logloss": logloss(y, p),
        "ece": ece(y, p, n_bins),
        "calib_ratio": calibration_ratio(y, p),
        "mean_weekly_gini": math.nan,
        "std_weekly_gini": math.nan,
        "slope": math.nan,
        "gini_stability": math.nan,
    }
    if ok.height >= 2:
        g, wk = ok["gini"].to_numpy(), ok["WEEK_NUM"].to_numpy()
        comp = gini_stability_components(g, wk)
        out.update(
            mean_weekly_gini=comp.mean_weekly_gini,
            std_weekly_gini=float(np.std(g)),
            slope=comp.slope,
            gini_stability=gini_stability(g, wk),
        )
    return out


def threshold_for_recall(y, p, recall: float) -> float:
    """Mayor umbral (score >= t) con al menos ``recall`` de los positivos."""
    y, p = np.asarray(y), np.asarray(p, float)
    if not (y == 1).any():
        raise ValueError("threshold_for_recall needs at least one positive case")
    pos = np.sort(p[y == 1])[::-1]
    k = max(1, math.ceil(recall * len(pos) - 1e-9))
    return float(pos[k - 1])


def fpr_at_threshold(y, p, t: float) -> float:
    y, p = np.asarray(y), np.asarray(p, float)
    return float(np.mean(p[y == 0] >= t))


def recall_at_threshold(y, p, t: float) -> float:
    y, p = np.asarray(y), np.asarray(p, float)
    return float(np.mean(p[y == 1] >= t))


def fpr_at_recall(y, p, recall: float) -> float:
    """FPR sobre la propia curva del periodo (umbral recalculado)."""
    return fpr_at_threshold(y, p, threshold_for_recall(y, p, recall))


def oof_summary(y, p, weeks) -> dict:
    """Serie fuera de partición (semanas contiguas, cada una predicha por un modelo que no la vio).

    El Gini semanal solo usa filas de la misma semana, así que unir predicciones de particiones distintas
    equivale a unir sus tablas semanales. AUC y PR-AUC se calculan sobre todas las filas unidas.
    """
    y, p, weeks = np.asarray(y), np.asarray(p, float), np.asarray(weeks)
    tab = weekly_table(y, p, weeks).drop_nulls("gini")
    g, wk = tab["gini"].to_numpy(), tab["WEEK_NUM"].to_numpy()
    comp = gini_stability_components(g, wk)
    return {
        "oof_gini_stability": gini_stability(g, wk),
        "mean_weekly_gini": comp.mean_weekly_gini,
        "slope": comp.slope,
        "residual_std": comp.residual_std,
        "std_weekly_gini": float(np.std(g)),
        "auc": auc(y, p),
        "pr_auc": pr_auc(y, p),
        "n_weeks": int(len(g)),
    }
