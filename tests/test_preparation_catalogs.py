"""Paso de pipeline que regenera los catálogos de familia y el C8 desde las piezas E4 publicadas."""

import polars as pl
import pytest

from dm_credit.preparation import catalogs, contracts as ct
from dm_credit.preparation.validation import load_published


def publish(tmp_path, family, split="train"):
    frame = pl.DataFrame({"case_id": [1, 2], f"{family}__n_records": [1, 0], f"{family}__source_present": [True, False]})
    definitions = ct.describe_features(frame, temporal_rule="regla")
    return ct.write_family_features(frame, definitions, family=family, split=split, output_dir=tmp_path / "E4" / family / split)


def publish_all(tmp_path, families=("static_0", "applprev", "credit_bureau_a", "tax_registry_a_1")):
    for family in families:
        for split in ("train", "oot", "test"):
            publish(tmp_path, family, split)


def test_regenerates_each_family_catalog_and_consolidates_c8(tmp_path):
    publish_all(tmp_path)
    results = tmp_path / "results"

    c8 = catalogs.regenerate_catalogs(tmp_path / "E4", results)

    for name, family in (
        ("depth0", "static_0"),
        ("applications_people", "applprev"),
        ("bureau", "credit_bureau_a"),
        ("products", "tax_registry_a_1"),
    ):
        part = pl.read_csv(results / f"{name}_feature_catalog.csv")
        assert part["source_table"].unique().to_list() == [family]
    assert c8["name"].to_list() == sorted(c8["name"].to_list()) and c8.height == 8
    assert pl.read_csv(results / "C8_feature_catalog.csv").height == 8


def test_a_family_group_without_pieces_is_an_error(tmp_path):
    publish_all(tmp_path, families=("static_0", "applprev"))
    with pytest.raises(ValueError, match="bureau"):
        catalogs.regenerate_catalogs(tmp_path / "E4", tmp_path / "results")
    assert not (tmp_path / "results" / "C8_feature_catalog.csv").exists()


def test_catalogs_equal_what_the_e4_auditor_expects(tmp_path):
    publish_all(tmp_path)
    catalogs.regenerate_catalogs(tmp_path / "E4", tmp_path / "results")
    arts = [a for a in load_published([tmp_path / "E4"])["train"] if a.family == "applprev"]
    expected = ct.feature_catalog(arts)
    got = pl.read_csv(tmp_path / "results" / "applications_people_feature_catalog.csv", schema_overrides=ct.CATALOG_SCHEMA).fill_null("")
    assert got.equals(expected)
