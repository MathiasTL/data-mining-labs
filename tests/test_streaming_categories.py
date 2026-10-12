from pathlib import Path

import polars as pl

from dm_credit.streaming_categories import categorical_columns, count_categories


def test_categories_are_aggregated_across_files(tmp_path: Path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pl.DataFrame(
        {"cat": ["a", "b", None], "flag": [True, True, False], "x": [1, 2, 3]}
    ).write_parquet(first)
    pl.DataFrame(
        {"cat": ["a", "a"], "flag": [False, False], "x": [4, 5]}
    ).write_parquet(second)

    assert categorical_columns(first) == ["cat", "flag"]
    rows = {
        row["column"]: row
        for row in count_categories(
            [first, second], ["cat", "flag"], top_k=2
        ).iter_rows(named=True)
    }

    assert rows["cat"]["top_value"] == "a"
    assert rows["cat"]["top_count"] == 3
    assert rows["cat"]["n_unique"] == 2
    assert rows["flag"]["top_value"] == "False"
    assert rows["flag"]["rows_scanned"] == 5
