import polars as pl
import pytest

from dm_credit import io


def test_load_parquet_roundtrip(tmp_path):
    path = tmp_path / "t.parquet"
    pl.DataFrame({"case_id": [1, 2], "target": [0, 1]}).write_parquet(path)
    df = io.load_parquet(path)
    assert df.shape == (2, 2)


def test_load_parquet_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="DM_DATA_DIR"):
        io.load_parquet(tmp_path / "nope.parquet")


def test_base_path_uses_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DM_DATA_DIR", str(tmp_path))
    assert io.base_path("train") == tmp_path / "parquet_files" / "train" / "train_base.parquet"
    assert io.base_path("test").name == "test_base.parquet"


def test_base_path_rejects_unknown_split():
    with pytest.raises(ValueError):
        io.base_path("valid")


def test_load_feature_definitions(tmp_path, monkeypatch):
    monkeypatch.setenv("DM_DATA_DIR", str(tmp_path))
    (tmp_path / "feature_definitions.csv").write_text("Variable,Description\nx,desc\n")
    defs = io.load_feature_definitions()
    assert defs.columns == ["Variable", "Description"]
