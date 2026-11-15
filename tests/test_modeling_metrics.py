"""Pruebas de métricas con ejemplos pequeños calculados a mano."""

import math

import numpy as np
import polars as pl
import pytest

from dm_credit.modeling import metrics as mt

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.5, 0.4, 0.3, 0.2, 0.1])


def test_brier_and_logloss():
    y, p = np.array([1, 0]), np.array([0.8, 0.1])
    assert mt.brier(y, p) == pytest.approx((0.04 + 0.01) / 2)
    assert mt.logloss(y, p) == pytest.approx(-(math.log(0.8) + math.log(0.9)) / 2)


def test_decile_table_ece_and_ratio():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    tab = mt.calibration_table(y, p, n_bins=2)
    assert tab["n"].to_list() == [2, 2]
    assert tab["mean_pred"].to_list() == pytest.approx([0.15, 0.85])
    assert tab["obs_rate"].to_list() == [0.0, 1.0]
    assert mt.ece(y, p, n_bins=2) == pytest.approx(0.15)
    assert mt.calibration_ratio(y, p) == pytest.approx(1.0)
    assert mt.calibration_ratio(y, p * 2) == pytest.approx(2.0)


def test_threshold_for_recall_and_fpr():
    t = mt.threshold_for_recall(Y, S, 0.5)
    assert t == pytest.approx(0.8)
    assert mt.fpr_at_threshold(Y, S, t) == pytest.approx(1 / 6)
    assert mt.recall_at_threshold(Y, S, t) == pytest.approx(0.5)
    # recall 0.8 -> ceil(3.2)=4 positivos -> umbral 0.6
    assert mt.threshold_for_recall(Y, S, 0.8) == pytest.approx(0.6)


def test_fpr_at_recall_curve_and_fixed_threshold_differ():
    fpr_curve = mt.fpr_at_recall(Y, S, 0.5)
    assert fpr_curve == pytest.approx(1 / 6)
    # umbral fijado en otro periodo (0.45): recall 4/4, fpr 2/6
    assert mt.fpr_at_threshold(Y, S, 0.45) == pytest.approx(2 / 6)


def test_weekly_table_skips_single_class_weeks():
    y = np.array([1, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4])
    w = np.array([1, 1, 2, 2, 3, 3])
    tab = mt.weekly_table(y, p, w)
    assert tab["WEEK_NUM"].to_list() == [1, 2, 3]
    assert tab["gini"].to_list()[:2] == [1.0, 1.0]
    assert tab["gini"][2] is None
    assert tab["mean_pred"][0] == pytest.approx(0.5)


def test_gini_stability_summary_uses_baseline():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.6, 0.1, 0.9])
    w = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    s = mt.summarize(y, p, w)
    # ginis semanales 1, 1, -1, 1 (semana 3 invertida)
    from dm_credit.baseline import gini_stability

    assert s["gini_stability"] == pytest.approx(gini_stability([1, 1, -1, 1], [1, 2, 3, 4]))
    assert s["n_weeks"] == 4
    assert s["auc"] == pytest.approx(mt.auc(y, p))


def test_ece_and_table_do_not_split_tied_scores_by_row_order():
    rng = np.random.default_rng(0)
    # meseta isotónica: pocos valores únicos con muchos empates
    p = rng.choice([0.01, 0.02, 0.05, 0.2], size=2001, p=[0.5, 0.2, 0.2, 0.1])
    y = (rng.random(2001) < p).astype(int)
    base = mt.ece(y, p)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(p))
        assert mt.ece(y[perm], p[perm]) == pytest.approx(base, abs=1e-12)
    tab = mt.calibration_table(y, p)
    # ningún valor de score aparece en dos bins
    assert tab["n"].sum() == len(p)
    assert tab["mean_pred"].n_unique() == tab.height
    assert tab.height <= 4


def test_threshold_for_recall_without_positives_raises():
    with pytest.raises(ValueError, match="positive"):
        mt.threshold_for_recall(np.zeros(5, int), np.arange(5) / 5, 0.7)


def test_pr_auc_hand_computed_and_in_summary_and_weekly():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    assert mt.pr_auc(y, p) == pytest.approx((1.0 + 2 / 3) / 2)
    s = mt.summarize(np.array([1, 0, 1, 0, 1, 0, 0, 1]), np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.6, 0.1, 0.9]), np.array([1, 1, 2, 2, 3, 3, 4, 4]))
    assert 0 < s["pr_auc"] <= 1
    tab = mt.weekly_table(y, p, np.array([1, 1, 2, 2]))
    assert "pr_auc" in tab.columns and tab["pr_auc"][0] == pytest.approx(1.0)
    one_class = mt.weekly_table(np.array([0, 0]), np.array([0.1, 0.2]), np.array([5, 5]))
    assert one_class["pr_auc"][0] is None


def test_constant_score_has_zero_gini_and_does_not_crash():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    w = np.repeat(np.arange(4), 100)
    s = mt.summarize(y, np.full(400, 0.1), w)
    assert s["auc"] == pytest.approx(0.5) and s["gini_stability"] == pytest.approx(0.0)
    assert s["pr_auc"] == pytest.approx(y.mean())
    assert s["ece"] == pytest.approx(abs(0.1 - y.mean()))


def test_oof_summary_is_one_series_over_contiguous_weeks():
    from dm_credit.baseline import gini_stability

    rng = np.random.default_rng(3)
    w = np.repeat(np.arange(40, 48), 60)
    y = (rng.random(len(w)) < 0.2).astype(int)
    p = np.clip(0.2 + 0.3 * (y - 0.2) * rng.random(len(w)) + rng.normal(0, 0.1, len(w)), 0.001, 0.999)
    s = mt.oof_summary(y, p, w)
    tab = mt.weekly_table(y, p, w).drop_nulls("gini")
    assert s["oof_gini_stability"] == pytest.approx(gini_stability(tab["gini"].to_numpy(), tab["WEEK_NUM"].to_numpy()))
    assert s["n_weeks"] == 8 and s["std_weekly_gini"] == pytest.approx(float(np.std(tab["gini"].to_numpy())))
    assert s["mean_weekly_gini"] == pytest.approx(float(tab["gini"].mean()))
    for k in ("slope", "residual_std", "auc", "pr_auc"):
        assert np.isfinite(s[k]), k
