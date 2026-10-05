import numpy as np
import polars as pl
import pytest

from dm_credit import baseline


def test_gini_perfect_and_inverse():
    y = np.array([0, 0, 1, 1])
    assert baseline.gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert baseline.gini(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(-1.0)


def test_gini_constant_score_is_zero():
    assert baseline.gini(np.array([0, 1, 0, 1]), np.full(4, 0.5)) == pytest.approx(0.0)


def test_gini_single_class_is_nan():
    assert np.isnan(baseline.gini(np.array([0, 0]), np.array([0.1, 0.2])))


def test_gini_stability_constant_weekly_gini_equals_that_gini():
    # pendiente 0 y sin ruido residual -> estabilidad == Gini medio
    assert baseline.gini_stability([0.5] * 6, [1, 2, 3, 4, 5, 6]) == pytest.approx(0.5)


def test_gini_stability_penalises_degradation():
    flat = baseline.gini_stability([0.5, 0.5, 0.5, 0.5], [1, 2, 3, 4])
    falling = baseline.gini_stability([0.6, 0.5, 0.4, 0.3], [1, 2, 3, 4])
    assert falling < flat
    # una tendencia creciente no se premia: solo se usa min(0, pendiente)
    rising = baseline.gini_stability([0.3, 0.4, 0.5, 0.6], [1, 2, 3, 4])
    assert rising == pytest.approx(np.mean([0.3, 0.4, 0.5, 0.6]))


def test_validate_contract_ok(base_df):
    baseline.validate_contract(base_df)


def test_validate_contract_missing_columns(base_df):
    with pytest.raises(ValueError, match="WEEK_NUM"):
        baseline.validate_contract(base_df.drop("WEEK_NUM"))


def test_validate_contract_duplicate_case_id(base_df):
    with pytest.raises(ValueError, match="case_id"):
        baseline.validate_contract(pl.concat([base_df, base_df.head(1)]))


def test_validate_contract_bad_target(base_df):
    bad = base_df.with_columns(pl.when(pl.col("case_id") == 0).then(2).otherwise(pl.col("target")).alias("target"))
    with pytest.raises(ValueError, match="target"):
        baseline.validate_contract(bad)


def test_validate_contract_target_optional_for_test(base_df):
    baseline.validate_contract(base_df.drop("target"), require_target=False)


def test_oot_split_is_disjoint_and_ordered(base_df):
    train, val = baseline.oot_split(base_df, cutoff_week=15)
    assert train["WEEK_NUM"].max() < 15 <= val["WEEK_NUM"].min()
    assert train.height + val.height == base_df.height


def test_oot_split_rejects_empty_partition(base_df):
    with pytest.raises(ValueError):
        baseline.oot_split(base_df, cutoff_week=0)
    with pytest.raises(ValueError):
        baseline.oot_split(base_df, cutoff_week=1000)


def test_default_feature_columns_excludes_reserved(base_df):
    df = base_df.with_columns(pl.lit("2019-01-01").alias("date_decision"), pl.lit(201901).alias("MONTH"))
    assert baseline.default_feature_columns(df) == ["x_signal", "x_noise", "x_cat"]


def test_prevalence_baseline_scores_constant(base_df):
    res = baseline.run_baseline(base_df, cutoff_week=15, model="prevalence", features=[])
    assert res.scores["score"].n_unique() == 1
    assert res.metrics["gini"] == pytest.approx(0.0)
    assert res.metrics["roc_auc"] == pytest.approx(0.5)


def test_logistic_baseline_learns_signal(base_df):
    res = baseline.run_baseline(base_df, cutoff_week=15, model="logistic", features=["x_signal", "x_noise", "x_cat"])
    assert res.metrics["gini"] > 0.5
    assert res.scores["score"].is_between(0, 1).all()
    assert set(res.weekly["WEEK_NUM"]) == set(range(15, 20))
    assert res.metrics["n_train"] + res.metrics["n_val"] == base_df.height


def test_run_baseline_is_reproducible(base_df):
    a = baseline.run_baseline(base_df, cutoff_week=15, model="logistic", features=["x_signal"], seed=1)
    b = baseline.run_baseline(base_df, cutoff_week=15, model="logistic", features=["x_signal"], seed=1)
    assert a.metrics == b.metrics


def test_run_baseline_does_not_use_target_or_ids(base_df):
    with pytest.raises(ValueError, match="reserved"):
        baseline.run_baseline(base_df, cutoff_week=15, model="logistic", features=["target"])


def test_experiment_record_contains_required_fields(base_df):
    res = baseline.run_baseline(base_df, cutoff_week=15, model="logistic", features=["x_signal"], seed=7)
    rec = res.record
    for key in ("model", "features", "cutoff_week", "seed", "params", "metrics", "date", "data_version"):
        assert key in rec
    assert rec["seed"] == 7
