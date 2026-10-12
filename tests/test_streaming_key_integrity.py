from pathlib import Path

import polars as pl

from dm_credit.streaming_key_integrity import check_ordered_key


def write(path: Path, rows: list[tuple[int, int, int]]):
    pl.DataFrame(
        rows, schema=["case_id", "num_group1", "num_group2"], orient="row"
    ).write_parquet(path)


def test_ordered_unique_key_across_shards(tmp_path: Path):
    write(tmp_path / "table_0.parquet", [(1, 0, 0), (1, 0, 1)])
    write(tmp_path / "table_1.parquet", [(2, 0, 0), (3, 0, 0)])

    result = check_ordered_key(
        [tmp_path / "table_1.parquet", tmp_path / "table_0.parquet"],
        ["case_id", "num_group1", "num_group2"],
        batch_size=1,
    )

    assert result["candidate_key_unique"]
    assert result["distinct_case_id"] == 3
    assert result["ordering_violations"] == 0


def test_duplicate_and_disorder_are_reported(tmp_path: Path):
    path = tmp_path / "table.parquet"
    write(path, [(2, 0, 0), (2, 0, 0), (1, 0, 0)])

    result = check_ordered_key(
        [path], ["case_id", "num_group1", "num_group2"], batch_size=2
    )

    assert not result["candidate_key_unique"]
    assert result["adjacent_duplicate_rows"] == 1
    assert result["ordering_violations"] == 1
    assert result["distinct_case_id"] is None
