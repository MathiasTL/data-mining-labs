"""EDA complementario (rúbrica del examen parcial): pruebas con datos sintéticos calculables a mano."""

import numpy as np
import polars as pl
import pytest

from dm_credit import eda


def test_continuous_summary_reports_nulls_quartiles_and_iqr_outliers():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0, None]})
    row = eda.continuous_summary(df, ["x"]).row(0, named=True)
    assert row["pct_null"] == pytest.approx(100 / 6)
    assert (row["min"], row["median"], row["max"]) == (1.0, 3.0, 100.0)
    # Q1=2, Q3=4, IQR=2: solo 100 supera Q3 + 1,5·IQR = 7
    assert row["pct_iqr_outliers"] == pytest.approx(20.0)


def test_categorical_association_uses_chi_square_and_cramers_v():
    # categoría a: 0 de 50 positivos; b: 50 de 50 -> asociación perfecta
    df = pl.DataFrame({"c": ["a"] * 50 + ["b"] * 50, "target": [0] * 50 + [1] * 50})
    row = eda.categorical_association(df, ["c"]).row(0, named=True)
    assert row["n_categories"] == 2
    assert row["cramers_v"] == pytest.approx(1.0, abs=0.03)
    assert row["p_value"] < 1e-10


def test_categorical_rates_include_missing_as_its_own_level():
    df = pl.DataFrame({"c": ["a", "a", None, None], "target": [1, 0, 1, 1]})
    rates = eda.category_rates(df, "c")
    assert dict(zip(rates["category"], rates["target_rate"])) == {"a": 50.0, "__missing__": 100.0}


def test_ordinal_spearman_follows_the_declared_order():
    order = ["LESS_ONE", "MORE_ONE", "MORE_FIVE"]
    df = pl.DataFrame({"o": ["LESS_ONE", "MORE_ONE", "MORE_FIVE"] * 20, "target": [1, 0, 0] * 20})
    row = eda.ordinal_spearman(df, "o", order)
    assert row["spearman_rho"] < 0 and row["n"] == 60


def test_missing_pattern_separates_structural_absence_from_partial_nulls():
    df = pl.DataFrame({"f1": [1.0, None, None, 2.0], "f2": [1.0, None, 3.0, 2.0]})
    row = eda.missing_pattern(df, {"fam": ["f1", "f2"]}).row(0, named=True)
    assert row["pct_rows_all_null"] == 25.0 and row["pct_rows_any_null"] == 50.0


def test_high_correlation_pairs_and_pca_variance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pl.DataFrame({"a": a, "b": a * 2 + 0.01 * rng.normal(size=500), "c": rng.normal(size=500)})
    pairs = eda.high_correlations(df, ["a", "b", "c"], threshold=0.9)
    assert pairs.select("feature_1", "feature_2").rows() == [("a", "b")]
    pca = eda.pca_variance(df.to_numpy())
    assert pca["components"].to_list()[:2] == [1, 2]
    assert pca["cumulative_variance"][1] == pytest.approx(1.0, abs=0.01)  # dos direcciones bastan


def test_histogram_counts_non_null_values_per_bin():
    h = eda.histogram(pl.DataFrame({"x": [0.0, 1.0, 1.0, 2.0, None]}), "x", bins=2)
    assert h["count"].to_list() == [1, 3] and h["bin_left"].to_list()[0] == 0.0
