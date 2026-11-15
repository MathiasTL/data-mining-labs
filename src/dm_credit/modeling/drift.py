"""OM4: PSI semanal y ADWIN sobre la pérdida semanal."""

from __future__ import annotations

import numpy as np
import polars as pl
from river.drift import ADWIN

EPS = 1e-4
PSI_WARN, PSI_ALERT = 0.10, 0.25


def _bins(ref: np.ndarray, cur: np.ndarray, n_bins: int):
    """Proporciones por bin: valores de la referencia (o deciles), bin nulo y bin 'otros'."""
    ref_ok = ref[~np.isnan(ref)]
    cur_ok = cur[~np.isnan(cur)]
    uniq = np.unique(ref_ok)
    if len(uniq) <= n_bins:
        idx_ref = np.searchsorted(uniq, ref_ok)
        pos = np.clip(np.searchsorted(uniq, cur_ok), 0, len(uniq) - 1)
        known = uniq[pos] == cur_ok
        idx_cur = np.where(known, pos, len(uniq))  # no visto -> 'otros'
        k = len(uniq) + 1
    else:
        edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, n_bins + 1)[1:-1]))
        idx_ref = np.searchsorted(edges, ref_ok, side="right")
        idx_cur = np.searchsorted(edges, cur_ok, side="right")
        k = len(edges) + 1
    def prop(idx, n_null, n_total):
        c = np.bincount(idx, minlength=k).tolist() + [n_null]
        return np.asarray(c, float) / max(n_total, 1)
    return (
        prop(idx_ref, int(np.isnan(ref).sum()), len(ref)),
        prop(idx_cur, int(np.isnan(cur).sum()), len(cur)),
    )


def psi(ref, cur, n_bins: int = 10) -> float:
    """Population Stability Index de ``cur`` frente a ``ref``; el nulo es un bin propio."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    if np.isnan(ref).all() or len(cur) == 0:
        return float("nan")  # sin referencia observada o sin datos actuales no hay PSI
    p, q = _bins(ref, cur, n_bins)
    keep = (p > 0) | (q > 0)
    p, q = np.clip(p[keep], EPS, None), np.clip(q[keep], EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def weekly_psi(ref, cur, weeks, n_bins: int = 10) -> pl.DataFrame:
    cur, weeks = np.asarray(cur, float), np.asarray(weeks)
    rows = [
        {"WEEK_NUM": int(w), "n": int((weeks == w).sum()), "psi": psi(ref, cur[weeks == w], n_bins)}
        for w in np.unique(weeks)
    ]
    return pl.DataFrame(rows)


def adwin_alerts(weeks, values, delta: float = 0.002) -> list[int]:
    """Semanas donde ADWIN declara cambio. ``clock=1`` evalúa en cada semana (la serie es corta)."""
    detector = ADWIN(delta=delta, clock=1)
    alerts = []
    for w, v in zip(weeks, values):
        detector.update(float(v))
        if detector.drift_detected:
            alerts.append(int(w))
    return alerts
