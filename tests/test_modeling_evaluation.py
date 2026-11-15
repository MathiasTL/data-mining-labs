"""Pruebas de umbrales fijados en P3, segmento de historial limitado y deriva por semana."""

import numpy as np
import polars as pl
import pytest

from dm_credit.modeling import evaluation as ev

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.5, 0.4, 0.3, 0.2, 0.1])


def test_fit_thresholds_on_val_then_apply_to_other_period():
    th = ev.fit_thresholds(Y, S, recalls=(0.5, 1.0))
    assert th == {0.5: pytest.approx(0.8), 1.0: pytest.approx(0.6)}
    # periodo nuevo: scores desplazados -0.3 -> el umbral fijo cae en otro punto de la curva
    rows = ev.threshold_metrics(Y, S - 0.3, th)
    r = {x["recall_target"]: x for x in rows}
    assert r[0.5]["recall_fixed"] == pytest.approx(0.0)  # 0.8 > todos los scores nuevos
    assert r[0.5]["fpr_fixed"] == pytest.approx(0.0)
    assert r[0.5]["fpr_curve"] == pytest.approx(1 / 6)  # curva propia del periodo
    assert r[0.5]["threshold"] == pytest.approx(0.8)


def test_limited_history_mask_requires_three_flags_and_nulls_are_absent():
    df = pl.DataFrame(
        {
            "applprev_1__source_present": [False, True, None],
            "credit_bureau_a_1__source_present": [False, False, False],
            "credit_bureau_b_1__source_present": [False, False, None],
        }
    )
    assert ev.limited_history_mask(df).tolist() == [True, False, True]
    assert ev.limited_history_mask(df.drop("credit_bureau_b_1__source_present")) is None


def test_segment_metrics_at_global_thresholds():
    seg = np.array([False, False, False, False, True, True, True, True, True, True])
    th = {0.5: 0.45}
    rows = ev.segment_metrics(Y, S, seg, th)
    by = {r["segment"]: r for r in rows}
    # segmento: 6 negativos (0 positivos): fpr = (0.85, 0.5)>=0.45 -> 2/6, recall indefinido
    assert by["limited_history"]["fpr"] == pytest.approx(2 / 6)
    assert np.isnan(by["limited_history"]["recall"])
    assert by["rest"]["recall"] == pytest.approx(1.0)
    assert by["all"]["n"] == 10


def test_undefined_threshold_rows_for_constant_score():
    rows = ev.undefined_threshold_rows((0.7, 0.8))
    assert [r["recall_target"] for r in rows] == [0.7, 0.8]
    for r in rows:
        assert np.isnan(r["fpr_fixed"]) and np.isnan(r["fpr_curve"]) and np.isnan(r["recall_fixed"])
        assert "constant" in r["note"]
    assert ev.threshold_metrics(Y, S, {0.5: 0.8})[0]["note"] is None
