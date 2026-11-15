"""Pruebas de modelos, selección y calibración."""

import numpy as np
import polars as pl
import pytest

from dm_credit.modeling import calibration as cb
from dm_credit.modeling import folds as fd
from dm_credit.modeling import metrics as mt
from dm_credit.modeling import models as md
from dm_credit.modeling import pipeline as pp
from dm_credit.modeling import selection as sel

from modeling_fixtures import synthetic_matrix, synthetic_spec

pytestmark = pytest.mark.filterwarnings("ignore")


def test_grids_are_comparable_and_fixed():
    assert [c.params["C"] for c in md.grid("logistic")] == [0.1, 1.0]
    lg = md.grid("lightgbm")
    assert sorted({c.params["num_leaves"] for c in lg}) == [31, 63]
    assert sorted({c.params["n_estimators"] for c in lg}) == [300, 600]
    assert {c.params["learning_rate"] for c in lg} == {0.05}
    assert len({c.config_id for c in lg}) == len(lg)


def test_logistic_and_lightgbm_learn_signal_and_are_deterministic():
    df = synthetic_matrix(per_week=150)
    ctx = pp.prepare_fold(df, synthetic_spec(), fd.get_fold("P1"))
    for name, extra in (("logistic", {}), ("lightgbm", {"n_estimators": 20, "min_child_samples": 5})):
        cfg = md.grid(name)[0]
        cfg = md.ModelConfig(cfg.model, {**cfg.params, **extra})
        p1 = md.fit_predict(cfg, ctx, ["val"])["val"]
        p2 = md.fit_predict(cfg, ctx, ["val"])["val"]
        assert np.allclose(p1, p2)
        assert mt.auc(ctx.y_val, p1) > 0.65
        assert ((p1 >= 0) & (p1 <= 1)).all()


def test_lightgbm_prefix_predictions_equal_smaller_model():
    df = synthetic_matrix(per_week=150)
    ctx = pp.prepare_fold(df, synthetic_spec(), fd.get_fold("P1"))
    base = {"num_leaves": 7, "min_child_samples": 5}
    big = md.ModelConfig("lightgbm", {**md.grid("lightgbm")[0].params, **base, "n_estimators": 20})
    small = md.ModelConfig("lightgbm", {**big.params, "n_estimators": 10})
    multi = md.fit_predict(big, ctx, ["val"], n_estimators_eval=(10, 20))
    alone = md.fit_predict(small, ctx, ["val"])
    assert np.allclose(multi[("val", 10)], alone["val"])


def _sel_tables(oof_a, oof_b, mwg_a, mwg_b):
    oof = pl.DataFrame([
        {"model": "m", "config_id": "A", "complexity": 1, "oof_gini_stability": oof_a},
        {"model": "m", "config_id": "B", "complexity": 2, "oof_gini_stability": oof_b},
    ])
    folds = pl.DataFrame(
        [{"model": "m", "config_id": "A", "fold": f, "mean_weekly_gini": v} for f, v in zip(("P1", "P2", "P3"), mwg_a)]
        + [{"model": "m", "config_id": "B", "fold": f, "mean_weekly_gini": v} for f, v in zip(("P1", "P2", "P3"), mwg_b)]
    )
    return oof, folds


def test_selection_needs_higher_oof_and_two_of_three_folds_and_prefers_simpler():
    # B gana el OOF pero su Gini semanal medio solo es mayor en 1 de 3 particiones -> A
    oof, folds = _sel_tables(0.40, 0.50, (0.40, 0.41, 0.42), (0.60, 0.40, 0.40))
    assert sel.select_config(oof, folds, "m") == "A"
    # B gana el OOF y 2 de 3 particiones -> B
    oof, folds = _sel_tables(0.40, 0.50, (0.40, 0.41, 0.42), (0.50, 0.50, 0.40))
    assert sel.select_config(oof, folds, "m") == "B"
    # B gana 3 de 3 particiones pero su OOF es menor -> A
    oof, folds = _sel_tables(0.40, 0.30, (0.40, 0.41, 0.42), (0.50, 0.50, 0.50))
    assert sel.select_config(oof, folds, "m") == "A"
    best, trace = sel.select_config(oof, folds, "m", return_trace=True)
    assert trace[0]["challenger"] == "B" and trace[0]["replaced"] is False and trace[0]["folds_won"] == 3


def test_isotonic_calibrator_fit_on_val_and_apply():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p**2).astype(int)  # sobreestima
    cal = cb.fit_calibrator(y, p)
    q = cal.predict(p)
    assert abs(q.mean() - y.mean()) < 0.01
    assert mt.ece(y, q) < mt.ece(y, p)
    assert ((q > 0) & (q < 1)).all()
    # fuera de rango se recorta (sin extrapolar)
    assert cal.predict(np.array([-1.0, 2.0])).tolist() == [q[np.argmin(p)], q[np.argmax(p)]]


def _arm(gs, std, mwg):
    oof = {"oof_gini_stability": gs, "std_weekly_gini": std}
    folds = pl.DataFrame({"fold": ["P1", "P2", "P3"], "mean_weekly_gini": mwg})
    return oof, folds


def test_ablation_rule_oof_folds_and_std():
    base = _arm(0.40, 0.05, (0.40, 0.41, 0.42))
    ok = sel.ablation_decision(*base, *_arm(0.41, 0.04, (0.41, 0.41, 0.40)))
    assert ok["retire"] is True and ok["folds_mwg_not_worse"] == 2
    assert sel.ablation_decision(*base, *_arm(0.39, 0.04, (0.41, 0.41, 0.42)))["retire"] is False  # OOF peor
    assert sel.ablation_decision(*base, *_arm(0.41, 0.06, (0.41, 0.41, 0.42)))["retire"] is False  # std mayor
    assert sel.ablation_decision(*base, *_arm(0.41, 0.04, (0.41, 0.30, 0.30)))["retire"] is False  # 1 de 3
    tie = sel.ablation_decision(*base, *_arm(0.40, 0.05, (0.40, 0.41, 0.42)))
    assert tie["retire"] is True  # "no empeora": el empate cuenta


def test_selection_rule_hash_changes_with_rule_text():
    assert sel.RULE_HASH and len(sel.RULE_HASH) == 64
