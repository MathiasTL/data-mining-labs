import polars as pl
import pytest

from dm_credit.preparation import outliers
from dm_credit.preparation.outliers import DomainViolationError, RangeWarning


@pytest.fixture
def train():
    return pl.DataFrame({"WEEK_NUM": [1, 2, 3, 4, 5], "a": [1.0, 2.0, 3.0, 4.0, 100.0], "b": [10.0, 20.0, None, 40.0, 50.0]})


def test_fit_learns_min_max_from_train_only(train):
    pol = outliers.fit_range_policy(train, ["a", "b"])
    assert pol.train_range["a"] == (1.0, 100.0)
    assert pol.train_range["b"] == (10.0, 50.0)  # nulos ignorados


def test_fit_rejects_oot_weeks(train):
    oot = train.with_columns(pl.Series("WEEK_NUM", [70, 71, 72, 73, 74]))  # solo la semana 74 es OOT
    with pytest.raises(ValueError, match="74"):
        outliers.fit_range_policy(oot, ["a"])


def test_fit_accepts_week_73_as_last_train_week(train):
    ok = train.with_columns(pl.Series("WEEK_NUM", [69, 70, 71, 72, 73]))
    outliers.fit_range_policy(ok, ["a"])


def test_fit_rejects_all_null_week_column(train):
    bad = train.with_columns(pl.lit(None, dtype=pl.Int64).alias("WEEK_NUM"))
    with pytest.raises(ValueError, match="WEEK_NUM"):
        outliers.fit_range_policy(bad, ["a"])


def test_fit_rejects_infinite_values(train):
    bad = train.with_columns(pl.Series("a", [1.0, 2.0, 3.0, 4.0, float("inf")]))
    with pytest.raises(ValueError, match="infinite"):
        outliers.fit_range_policy(bad, ["a"])


def test_fit_rejects_train_that_violates_its_own_domain_rule(train):
    with pytest.raises(ValueError, match="domain"):
        outliers.fit_range_policy(train, ["a"], domain={"a": (5.0, None)})


def test_apply_rejects_infinite_values(train):
    pol = outliers.fit_range_policy(train, ["a"], clip_columns=["a"])
    with pytest.raises(DomainViolationError, match="infinite"):
        pol.apply(pl.DataFrame({"a": [1.0, float("inf")]}))


def test_fit_allows_tables_without_week_column(train):
    outliers.fit_range_policy(train.drop("WEEK_NUM"), ["a"])


def test_fit_rejects_non_numeric_column():
    with pytest.raises(ValueError, match="numeric"):
        outliers.fit_range_policy(pl.DataFrame({"s": ["x"]}), ["s"])


def test_out_of_range_warns_and_does_not_modify(train):
    pol = outliers.fit_range_policy(train, ["a"])
    new = pl.DataFrame({"a": [0.0, 50.0, 500.0]})
    with pytest.warns(RangeWarning, match="a"):
        out, report = pol.apply(new)
    assert out["a"].to_list() == [0.0, 50.0, 500.0]  # sin recorte automático
    row = report.row(0, named=True)
    assert row["column"] == "a" and row["n_below"] == 1 and row["n_above"] == 1 and row["n_clipped"] == 0


def test_in_range_emits_no_warning(train, recwarn):
    pol = outliers.fit_range_policy(train, ["a"])
    pol.apply(pl.DataFrame({"a": [1.0, 50.0]}))
    assert not [w for w in recwarn if issubclass(w.category, RangeWarning)]


def test_clip_only_for_approved_columns_with_train_fitted_bounds():
    train = pl.DataFrame({"a": [float(i) for i in range(1, 101)], "b": [float(i) for i in range(1, 101)]})
    pol = outliers.fit_range_policy(train, ["a", "b"], clip_columns=["a"], clip_quantiles=(0.05, 0.95))
    lo, hi = pol.clip_bounds["a"]
    assert (lo, hi) == pytest.approx((5.95, 95.05))
    assert "b" not in pol.clip_bounds
    with pytest.warns(RangeWarning):
        out, report = pol.apply(pl.DataFrame({"a": [-1000.0, 50.0, 1000.0], "b": [-1000.0, 50.0, 1000.0]}))
    assert out["a"].to_list() == pytest.approx([lo, 50.0, hi])
    assert out["b"].to_list() == [-1000.0, 50.0, 1000.0]
    assert report.filter(pl.col("column") == "a")["n_clipped"][0] == 2


def test_domain_rule_violation_raises(train):
    pol = outliers.fit_range_policy(train, ["a"], domain={"a": (0.0, None)})
    with pytest.raises(DomainViolationError, match="a"):
        pol.apply(pl.DataFrame({"a": [-1.0, 2.0]}))


def test_domain_rule_ok_when_respected(train):
    pol = outliers.fit_range_policy(train, ["a"], domain={"a": (0.0, None)})
    with pytest.warns(RangeWarning):
        pol.apply(pl.DataFrame({"a": [0.0, 500.0]}))


def test_apply_requires_fitted_columns(train):
    pol = outliers.fit_range_policy(train, ["a"])
    with pytest.raises(ValueError, match="missing"):
        pol.apply(pl.DataFrame({"zzz": [1.0]}))


def test_apply_preserves_row_count_and_order(train):
    pol = outliers.fit_range_policy(train, ["a"], clip_columns=["a"], clip_quantiles=(0.1, 0.9))
    new = pl.DataFrame({"id": [3, 1, 2], "a": [1e6, -1e6, 3.0]})
    with pytest.warns(RangeWarning):
        out, _ = pol.apply(new)
    assert out["id"].to_list() == [3, 1, 2] and out.height == 3
