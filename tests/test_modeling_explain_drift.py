"""Pruebas de SHAP/coeficientes (OM2) y deriva PSI/ADWIN (OM4)."""

import numpy as np
import polars as pl
import pytest

from dm_credit.modeling import drift as dr
from dm_credit.modeling import explain as ex
from dm_credit.modeling import folds as fd
from dm_credit.modeling import models as md
from dm_credit.modeling import pipeline as pp

from modeling_fixtures import synthetic_matrix, synthetic_spec

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def fitted():
    df = synthetic_matrix(per_week=200)
    ctx = pp.prepare_fold(df, synthetic_spec(), fd.get_fold("P1"))
    out = {}
    for name in ("logistic", "lightgbm"):
        cfg = md.grid(name)[0]
        cfg = md.ModelConfig(name, {**cfg.params, "n_estimators": 15, "min_child_samples": 5})
        _, model = md.fit_predict(cfg, ctx, [], return_model=True)
        out[name] = (ctx, model)
    return out


def test_shap_importance_reproducible_with_fixed_seed(fitted):
    ctx, model = fitted["lightgbm"]
    X, names, _ = ctx.matrix("tree", "val")
    a = ex.shap_importance(model, X, n_sample=300, seed=1)
    b = ex.shap_importance(model, X, n_sample=300, seed=1)
    assert a.equals(b)
    assert set(a.columns) == {"feature", "mean_abs_shap", "rank"}
    assert a["rank"].to_list() == list(range(1, a.height + 1))
    assert a["feature"][0] == "x1"  # la señal real es x1


def test_rank_change_table():
    v = pl.DataFrame({"feature": ["a", "b", "c"], "mean_abs_shap": [3.0, 2.0, 1.0], "rank": [1, 2, 3]})
    o = pl.DataFrame({"feature": ["a", "b", "c"], "mean_abs_shap": [1.0, 2.0, 3.0], "rank": [3, 2, 1]})
    t = ex.rank_change(v, o)
    row = t.filter(pl.col("feature") == "c").row(0, named=True)
    assert row["rank_val"] == 3 and row["rank_oot"] == 1 and row["rank_change"] == -2


def test_logistic_standardized_coefficients(fitted):
    ctx, model = fitted["logistic"]
    X, names, _ = ctx.matrix("linear", "fit")
    t = ex.logistic_std_coefs(model, X)
    assert t.height == len(names)
    coef = model.obj.coef_[0]
    expected = dict(zip(model.names, coef * X.std(axis=0)))
    got = dict(zip(t["feature"].to_list(), t["std_coef"].to_list()))
    assert got == pytest.approx(expected, rel=1e-5)
    assert t["feature"][0] == "x1"


def test_psi_hand_computed_and_identical_is_zero():
    ref = np.array([0] * 50 + [1] * 50, float)
    cur = np.array([0] * 25 + [1] * 75, float)
    assert dr.psi(ref, cur) == pytest.approx(0.2747, abs=1e-3)
    assert dr.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)


def test_psi_continuous_detects_shift_and_handles_nulls():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert dr.psi(ref, rng.normal(size=5000)) < 0.05
    assert dr.psi(ref, rng.normal(loc=1.5, size=5000)) > 0.25
    withnull = np.where(rng.random(5000) < 0.2, np.nan, rng.normal(size=5000))
    assert dr.psi(ref, withnull) > 0.05  # el nulo es un bin propio


def test_weekly_psi_table():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=2000)
    cur = np.concatenate([rng.normal(size=500), rng.normal(loc=2, size=500)])
    weeks = np.repeat([74, 75], 500)
    t = dr.weekly_psi(ref, cur, weeks)
    assert t["WEEK_NUM"].to_list() == [74, 75]
    assert t["psi"][0] < 0.1 < 0.25 < t["psi"][1]


def test_adwin_alerts_on_level_shift_only():
    # ADWIN con delta=0.002 necesita ~100 puntos para detectar un salto grande
    rng = np.random.default_rng(0)
    weeks = np.arange(100)
    shift = np.where(weeks < 50, 0.10, 0.90) + rng.normal(0, 0.02, 100)
    alerts = dr.adwin_alerts(weeks, shift, delta=0.002)
    assert alerts and min(alerts) >= 50
    assert dr.adwin_alerts(weeks, 0.1 + rng.normal(0, 0.02, 100), delta=0.002) == []


def test_psi_with_all_null_reference_is_nan_not_crash():
    assert np.isnan(dr.psi(np.full(10, np.nan), np.arange(10, dtype=float)))
    assert np.isnan(dr.psi(np.arange(10, dtype=float), np.array([])))


def _tab(order):
    return pl.DataFrame({"feature": order, "mean_abs_shap": [4.0, 3.0, 2.0, 1.0], "rank": [1, 2, 3, 4]})


def test_shap_stability_summary_hand_computed():
    same = ex.stability_summary([_tab(list("abcd")), _tab(list("abcd"))], top_k=2)
    assert same["spearman_mean"] == pytest.approx(1.0) and same["jaccard_mean"] == pytest.approx(1.0)
    rev = ex.stability_summary([_tab(list("abcd")), _tab(list("dcba"))], top_k=2)
    assert rev["spearman_mean"] == pytest.approx(-1.0) and rev["spearman_min"] == pytest.approx(-1.0)
    assert rev["jaccard_mean"] == pytest.approx(0.0)
    mixed = ex.stability_summary([_tab(list("abcd")), _tab(list("abdc")), _tab(list("bacd"))], top_k=2)
    # rho: A-B 0.8, A-C 0.8, B-C 0.6; top-2 siempre {a, b} -> Jaccard 1
    assert mixed["spearman_min"] == pytest.approx(0.6)
    assert mixed["spearman_mean"] == pytest.approx((0.8 + 0.8 + 0.6) / 3)
    assert mixed["jaccard_min"] == pytest.approx(1.0)
    per_feature = mixed["per_feature"]
    assert per_feature.filter(pl.col("feature") == "a")["rank_mean"][0] == pytest.approx((1 + 1 + 2) / 3)
    assert per_feature.filter(pl.col("feature") == "a")["rank_std"][0] == pytest.approx(np.std([1, 1, 2], ddof=1))


def test_shap_stability_uses_independent_seeded_samples(fitted):
    ctx, model = fitted["lightgbm"]
    X, _, _ = ctx.matrix("tree", "val")
    r1 = ex.shap_stability(model, X, seeds=(1, 2, 3), n_sample=300, top_k=5)
    r2 = ex.shap_stability(model, X, seeds=(1, 2, 3), n_sample=300, top_k=5)
    scalars = lambda r: {k: v for k, v in r["summary"].items() if k != "per_feature"}
    assert scalars(r1) == scalars(r2)
    assert r1["summary"]["per_feature"].equals(r2["summary"]["per_feature"])
    assert r1["tables"][0].equals(ex.shap_importance(model, X, n_sample=300, seed=1))
    assert -1.0 <= r1["summary"]["spearman_min"] <= r1["summary"]["spearman_mean"] <= 1.0
