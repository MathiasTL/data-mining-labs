"""E5: tablas maestras ensambladas desde las piezas de E4, de a un split por vez."""

import json

import polars as pl
import pytest

import test_preparation_validation as tv
from dm_credit.preparation import e5

published = tv.published  # mismas doce familias sintéticas para train, OOT y test


def run(published, **kw):
    return e5.assemble_master_tables(published["bases"], published["artifacts"], output_dir=published["tmp"] / "E5", cutoff_week=tv.g3.CUTOFF, **kw)


def test_one_master_table_per_split_with_every_family(published):
    report = run(published)
    assert report["split"].to_list() == ["train", "oot", "test"]
    assert report["families"].to_list() == [12, 12, 12]
    for row in report.iter_rows(named=True):
        base = published["bases"][row["split"]]
        m = pl.read_parquet(row["data_path"])
        assert m.height == row["rows"] == base.height
        assert m["case_id"].to_list() == base["case_id"].to_list()
        assert row["features"] == row["predictors"] + row["coverage"]


def test_master_tables_share_the_same_features_and_order(published):
    report = run(published)
    cols = [pl.read_parquet(p).columns for p in report["data_path"]]
    train, oot, test = cols
    assert train == oot and test == [c for c in train if c != "target"]


def test_families_are_joined_in_the_fixed_e4_order(published):
    report = run(published)
    manifest = json.loads(open(report.filter(pl.col("split") == "train")["manifest_path"].item()).read())
    families = [json.loads(open(p).read())["family"] for p in manifest["input_manifests"]]
    assert families == [f for f in e5.FAMILY_ORDER if f in families] and len(families) == 12


def test_a_missing_family_stops_the_assembly(published):
    published["artifacts"]["oot"] = published["artifacts"]["oot"][1:]
    with pytest.raises(ValueError, match="oot"):
        run(published)


def test_the_report_records_checksums_and_time(published):
    report = run(published)
    assert report.columns == ["split", "rows", "columns", "features", "predictors", "coverage", "families", "seconds", "data_sha256", "data_path", "manifest_path"]
    assert all(len(h) == 64 for h in report["data_sha256"])


def test_assembly_is_deterministic(published):
    a = run(published)["data_sha256"].to_list()
    b = e5.assemble_master_tables(published["bases"], published["artifacts"], output_dir=published["tmp"] / "E5b", cutoff_week=tv.g3.CUTOFF)["data_sha256"].to_list()
    assert a == b
