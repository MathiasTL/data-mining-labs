from pathlib import Path

import polars as pl

from dm_credit.quality_catalog import build_quality_catalog, summarize_quality


def test_quality_catalog_combines_streaming_profile(tmp_path: Path):
    profile = tmp_path / "profile_train_example"
    profile.mkdir()
    pl.DataFrame(
        {
            "column": ["constant", "sparse", "amount"],
            "dtype": ["Int64", "String", "Float64"],
            "nulls": [0, 96, 0],
            "pct_null": [0.0, 96.0, 0.0],
            "n_unique_approx": [1, 2, 10],
        }
    ).write_csv(profile / "columns.csv")
    pl.DataFrame(
        {
            "column": ["constant", "amount"],
            "min": [1.0, -5.0],
            "max": [1.0, 20.0],
            "mean": [1.0, 3.0],
            "n_zero": [0, 1],
            "n_negative": [0, 2],
        }
    ).write_csv(profile / "numeric.csv")

    catalog = build_quality_catalog([profile])
    by_name = {row["column"]: row for row in catalog.iter_rows(named=True)}

    assert by_name["constant"]["is_constant"]
    assert by_name["sparse"]["missingness_band"] == "critical_95_plus"
    assert by_name["amount"]["has_negative"]
    assert not by_name["sparse"]["has_top_categories"]


def test_quality_summary_counts_flags(tmp_path: Path):
    profile = tmp_path / "profile_train_example"
    profile.mkdir()
    pl.DataFrame(
        {
            "column": ["category", "empty"],
            "dtype": ["String", "String"],
            "nulls": [0, 2],
            "pct_null": [0.0, 100.0],
            "n_unique": [2, 0],
        }
    ).write_csv(profile / "columns.csv")
    pl.DataFrame(
        {
            "column": ["category"],
            "top_value": ["a"],
            "top_count": [2],
            "pct_dominant": [100.0],
            "top_values": ["a=2"],
        }
    ).write_csv(profile / "categorical.csv")

    row = summarize_quality(build_quality_catalog([profile])).row(0, named=True)

    assert row["columns"] == 2
    assert row["columns_95pct_null_or_more"] == 1
    assert row["constant_columns"] == 1
    assert row["categorical_columns_with_top_values"] == 1
