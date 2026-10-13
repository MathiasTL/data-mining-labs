import polars as pl
import pytest

from dm_credit import io


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DM_DATA_DIR", str(tmp_path))
    (tmp_path / "parquet_files" / "train").mkdir(parents=True)
    (tmp_path / "parquet_files" / "test").mkdir(parents=True)
    return tmp_path


def write(data_dir, split, name, rows, **kwargs):
    df = pl.DataFrame(rows, schema={"case_id": pl.Int64, "x": pl.Float64})
    df.write_parquet(data_dir / "parquet_files" / split / name, **kwargs)


def test_shards_are_sorted_numerically_not_alphabetically(data_dir):
    for n in (0, 1, 2, 10):
        write(data_dir, "train", f"train_a_1_{n}.parquet", {"case_id": [n], "x": [1.0]})
    names = [p.name for p in io.shard_paths("train", "a_1")]
    assert names == ["train_a_1_0.parquet", "train_a_1_1.parquet", "train_a_1_2.parquet", "train_a_1_10.parquet"]


def test_single_file_without_index_is_found(data_dir):
    write(data_dir, "train", "train_applprev_2.parquet", {"case_id": [1], "x": [1.0]})
    assert [p.name for p in io.shard_paths("train", "applprev_2")] == ["train_applprev_2.parquet"]


def test_similar_source_names_do_not_mix(data_dir):
    write(data_dir, "train", "train_static_0_0.parquet", {"case_id": [1], "x": [1.0]})
    write(data_dir, "train", "train_static_cb_0.parquet", {"case_id": [2], "x": [1.0]})
    write(data_dir, "train", "train_applprev_1_0.parquet", {"case_id": [3], "x": [1.0]})
    write(data_dir, "train", "train_applprev_1_1.parquet", {"case_id": [4], "x": [1.0]})
    write(data_dir, "train", "train_applprev_2.parquet", {"case_id": [5], "x": [1.0]})
    assert [p.name for p in io.shard_paths("train", "static_0")] == ["train_static_0_0.parquet"]
    assert [p.name for p in io.shard_paths("train", "applprev_1")] == ["train_applprev_1_0.parquet", "train_applprev_1_1.parquet"]


def test_split_selects_the_right_folder_and_prefix(data_dir):
    write(data_dir, "train", "train_p.parquet", {"case_id": [1], "x": [1.0]})
    write(data_dir, "test", "test_p.parquet", {"case_id": [2], "x": [1.0]})
    assert io.shard_paths("test", "p")[0].name == "test_p.parquet"


def test_missing_source_is_an_error_naming_the_data_dir(data_dir):
    with pytest.raises(FileNotFoundError, match="DM_DATA_DIR"):
        io.shard_paths("train", "nope")


def test_invalid_split_is_rejected(data_dir):
    with pytest.raises(ValueError, match="split"):
        io.shard_paths("oot", "p")


def test_scan_concatenates_all_shards_lazily(data_dir):
    write(data_dir, "train", "train_p_0.parquet", {"case_id": [1, 2], "x": [1.0, 2.0]})
    write(data_dir, "train", "train_p_1.parquet", {"case_id": [3], "x": [3.0]})
    lf = io.scan_parquet_shards("train", "p")
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect()["case_id"].to_list() == [1, 2, 3]


def test_scan_projects_columns(data_dir):
    write(data_dir, "train", "train_p.parquet", {"case_id": [1], "x": [1.0]})
    assert io.scan_parquet_shards("train", "p", columns=["case_id"]).collect().columns == ["case_id"]


def test_scan_rejects_unknown_columns_up_front(data_dir):
    write(data_dir, "train", "train_p.parquet", {"case_id": [1], "x": [1.0]})
    with pytest.raises(ValueError, match="nope"):
        io.scan_parquet_shards("train", "p", columns=["case_id", "nope"])


def test_empty_file_with_schema_is_valid(data_dir):
    write(data_dir, "test", "test_p.parquet", {"case_id": [], "x": []})
    out = io.scan_parquet_shards("test", "p").collect()
    assert out.height == 0 and out.columns == ["case_id", "x"]


def test_iter_row_groups_yields_one_group_at_a_time(data_dir):
    write(data_dir, "train", "train_p.parquet", {"case_id": [1, 2, 3, 4, 5], "x": [1.0] * 5}, row_group_size=2)
    groups = list(io.iter_row_groups(data_dir / "parquet_files" / "train" / "train_p.parquet", columns=["case_id"]))
    assert [g.height for g in groups] == [2, 2, 1]
    assert all(g.columns == ["case_id"] for g in groups)
    assert pl.concat(groups)["case_id"].to_list() == [1, 2, 3, 4, 5]


def test_iter_row_groups_missing_file_is_actionable(data_dir):
    with pytest.raises(FileNotFoundError, match="DM_DATA_DIR"):
        list(io.iter_row_groups(data_dir / "nope.parquet"))
