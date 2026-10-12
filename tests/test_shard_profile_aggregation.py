from pathlib import Path

import polars as pl

from dm_credit.shard_profile_aggregation import aggregate_profiles


def write_shard(path: Path, name: str, rows: int, nulls: int, unique: int, mean: float):
    path.mkdir()
    pl.DataFrame(
        {
            "column": ["x"],
            "dtype": ["Float64"],
            "nulls": [nulls],
            "pct_null": [100 * nulls / rows],
            "n_unique_approx": [unique],
        }
    ).write_csv(path / "columns.csv")
    pl.DataFrame(
        {
            "column": ["x"],
            "min": [-1.0],
            "max": [10.0],
            "mean": [mean],
            "n_zero": [1],
            "n_negative": [1],
        }
    ).write_csv(path / "numeric.csv")
    pl.DataFrame(
        {"source_file": [name], "rows_in_source": [rows], "columns": [1]}
    ).write_csv(path / "sources.csv")


def test_aggregate_profiles_combines_exact_stats_and_cardinality_bounds(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_shard(first, "first.parquet", rows=10, nulls=2, unique=5, mean=2.0)
    write_shard(second, "second.parquet", rows=20, nulls=4, unique=7, mean=4.0)

    columns, numeric, sources = aggregate_profiles([first, second])
    column = columns.row(0, named=True)
    number = numeric.row(0, named=True)

    assert sources["rows_in_source"].sum() == 30
    assert column["nulls"] == 6
    assert column["pct_null"] == 20.0
    assert column["n_unique_lower_bound"] == 7
    assert column["n_unique_upper_bound"] == 12
    assert number["mean"] == 10 / 3
    assert number["n_zero"] == 2
    assert number["n_negative"] == 2
