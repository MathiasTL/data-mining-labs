from pathlib import Path

import polars as pl
import pytest

from dm_credit.preparation.c8_catalog import consolidate_catalogs, write_catalog

COLUMNS = {
    "name": ["alpha__n_records"],
    "role": ["cobertura"],
    "logical_type": ["int"],
    "source_table": ["alpha"],
    "source_columns": [""],
    "aggregation": ["n_records"],
    "window": ["all"],
    "temporal_rule": ["sin fecha de evento"],
    "description": ["conteo de registros"],
}


def _write_catalog(path: Path, name: str) -> None:
    row = {column: values[0] for column, values in COLUMNS.items()}
    row["name"] = name
    pl.DataFrame([row]).write_csv(path)


def test_consolidates_and_sorts_distinct_family_catalogs(tmp_path: Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_catalog(first, "zeta__n_records")
    _write_catalog(second, "alpha__n_records")

    result = consolidate_catalogs([first, second])

    assert result["name"].to_list() == ["alpha__n_records", "zeta__n_records"]
    assert result.columns == list(COLUMNS)


def test_rejects_duplicate_feature_names(tmp_path: Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_catalog(first, "same__n_records")
    _write_catalog(second, "same__n_records")

    with pytest.raises(ValueError, match="duplicate C8 feature names"):
        consolidate_catalogs([first, second])


def test_requires_all_input_catalogs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="C8 input catalog not found"):
        consolidate_catalogs([tmp_path / "missing.csv"])


def test_writes_consolidated_catalog(tmp_path: Path) -> None:
    source, output = tmp_path / "family.csv", tmp_path / "nested" / "C8.csv"
    _write_catalog(source, "alpha__n_records")

    result = write_catalog([source], output)

    assert output.is_file()
    assert pl.read_csv(output).equals(result)
