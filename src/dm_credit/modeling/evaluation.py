"""Umbrales fijados en P3 (60-73), aplicación al OOT y segmento de historial limitado (ON2)."""

from __future__ import annotations

import numpy as np
import polars as pl

from dm_credit.modeling import metrics as mt

LIMITED_FLAGS = (
    "applprev_1__source_present",
    "credit_bureau_a_1__source_present",
    "credit_bureau_b_1__source_present",
)


def fit_thresholds(y, p, recalls=mt.RECALLS) -> dict:
    return {r: mt.threshold_for_recall(y, p, r) for r in recalls}


def threshold_metrics(y, p, thresholds: dict) -> list[dict]:
    """(a) umbral fijo aplicado al periodo y (b) FPR sobre la curva propia del periodo."""
    rows = []
    for r, t in thresholds.items():
        rows.append(
            {
                "recall_target": r,
                "threshold": t,
                "recall_fixed": mt.recall_at_threshold(y, p, t),
                "fpr_fixed": mt.fpr_at_threshold(y, p, t),
                "fpr_curve": mt.fpr_at_recall(y, p, r),
                "note": None,
            }
        )
    return rows


def limited_history_mask(df: pl.DataFrame) -> np.ndarray | None:
    """Sin solicitud previa y sin bureau A/B. None si falta alguna de las tres banderas."""
    if any(c not in df.columns for c in LIMITED_FLAGS):
        return None
    present = pl.any_horizontal(*[pl.col(c).fill_null(False) for c in LIMITED_FLAGS])
    return df.select((~present).alias("m"))["m"].to_numpy()


def segment_metrics(y, p, segment: np.ndarray, thresholds: dict) -> list[dict]:
    """FPR y recall por segmento a los umbrales globales de cada recall objetivo."""
    y, p = np.asarray(y), np.asarray(p, float)
    parts = {"all": np.ones(len(y), bool), "limited_history": segment, "rest": ~segment}
    rows = []
    for name, m in parts.items():
        ym, pm = y[m], p[m]
        for r, t in thresholds.items():
            has_pos, has_neg = (ym == 1).any(), (ym == 0).any()
            rows.append(
                {
                    "segment": name,
                    "recall_target": r,
                    "threshold": t,
                    "n": int(m.sum()),
                    "n_pos": int(ym.sum()),
                    "base_rate": float(ym.mean()) if len(ym) else float("nan"),
                    "recall": mt.recall_at_threshold(ym, pm, t) if has_pos else float("nan"),
                    "fpr": mt.fpr_at_threshold(ym, pm, t) if has_neg else float("nan"),
                }
            )
    return rows


def undefined_threshold_rows(recalls=mt.RECALLS) -> list[dict]:
    """Score constante: no hay curva ROC, el FPR a un recall dado no tiene sentido (se emite NaN)."""
    nan = float("nan")
    return [
        {"recall_target": r, "threshold": nan, "recall_fixed": nan, "fpr_fixed": nan, "fpr_curve": nan,
         "note": "constant score: FPR at recall is undefined"}
        for r in recalls
    ]


def fit_thresholds_constant(recalls=mt.RECALLS) -> dict:
    """Umbrales del score constante: no existen; las claves fijan los recalls a reportar como NaN."""
    return {r: float("nan") for r in recalls}
