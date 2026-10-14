"""Filtro as-of de productos y registros: la cobertura solo cuenta filas disponibles a ``date_decision``.

Motivo (F7): sin filtro, ``tax_registry_a_1`` tenía el 100 % de sus filas con fecha posterior a la decisión.
"""

from datetime import date

import polars as pl
import pytest

from dm_credit.preparation.families import products_registry as pr

BASE = pl.DataFrame({"case_id": [1, 2, 3], "date_decision": ["2020-01-10", "2020-01-10", "2020-01-10"]})


def rows(source, dates):
    anchor = pr.ANCHORS[source]
    return pl.DataFrame({"case_id": [1, 1, 2, 3], "num_group1": [0, 1, 0, 0], anchor: dates})


@pytest.mark.parametrize("source", [s for s in pr.SOURCES if pr.ANCHORS[s]])
def test_rows_after_decision_or_without_anchor_are_dropped(source):
    # anterior, igual, posterior y nula
    valid, exclusions = pr.filter_as_of(source, BASE, rows(source, ["2020-01-01", "2020-01-10", "2020-01-11", None]))
    assert valid.sort("case_id", "num_group1").select("case_id", "num_group1").rows() == [(1, 0), (1, 1)]
    assert exclusions == {f"{source}_rows_after_decision": 1, f"{source}_rows_null_anchor": 1}


def test_coverage_after_the_filter_counts_only_available_rows():
    valid, _ = pr.filter_as_of("tax_registry_a_1", BASE, rows("tax_registry_a_1", ["2020-01-01", "2020-01-11", "2020-01-11", "2019-12-31"]))
    out = pr.aggregate_coverage("tax_registry_a_1", BASE.select("case_id"), valid).features
    assert out["tax_registry_a_1__n_records"].to_list() == [1, 0, 1]
    assert out["tax_registry_a_1__source_present"].to_list() == [True, False, True]


def test_a_source_without_availability_date_contributes_nothing():
    src = pl.DataFrame({"case_id": [1, 2], "num_group1": [0, 0]})
    valid, exclusions = pr.filter_as_of("other_1", BASE, src)
    assert valid.is_empty()
    assert exclusions == {"other_1_rows_no_anchor": 2}


def test_dates_are_compared_as_dates_not_text():
    base = BASE.with_columns(pl.col("date_decision").str.to_date())
    src = rows("deposit_1", [date(2020, 1, 9), date(2020, 1, 9), date(2020, 1, 9), date(2020, 1, 9)])
    valid, exclusions = pr.filter_as_of("deposit_1", base, src)
    assert valid.height == 4 and sum(exclusions.values()) == 0


def test_missing_anchor_column_is_an_error():
    with pytest.raises(ValueError, match="openingdate_857D"):
        pr.filter_as_of("debitcard_1", BASE, pl.DataFrame({"case_id": [1], "num_group1": [0]}))


def test_every_source_declares_its_rule():
    for source in pr.SOURCES:
        rule = pr.temporal_rule(source)
        assert ("<= date_decision" in rule) == bool(pr.ANCHORS[source])
        if pr.ANCHORS[source]:
            assert pr.ANCHORS[source] in rule


# ---------------------------------------------------- F6: G2 por familia y claves de la fuente completa
import preparation_products_artifacts as ppa  # noqa: E402


def _source_dir(tmp_path, monkeypatch, frame):
    root = tmp_path / "data"
    for split in ("train", "test"):
        (root / "parquet_files" / split).mkdir(parents=True)
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    frame.write_parquet(root / "parquet_files" / "train" / "train_deposit_1.parquet")


def _build(tmp_path, g2_path=None):
    base = pl.DataFrame({"case_id": [1, 2], "WEEK_NUM": [10, 10], "date_decision": ["2020-01-10"] * 2})
    selection = ppa.write_selection(tmp_path / "sel.csv")
    return pr.build_deposit_1_features(
        base, "train", output_dir=tmp_path / "out", selection_path=selection,
        g2_path=g2_path or ppa.write_decisions(tmp_path / "g2.csv"),
    )


def test_a_g2_file_without_rows_for_the_family_is_an_error(tmp_path, monkeypatch):
    _source_dir(tmp_path, monkeypatch, pl.DataFrame({"case_id": [1], "num_group1": [0], "openingdate_313D": ["2020-01-01"]}))
    g2 = tmp_path / "g2_other.csv"
    pl.read_csv(ppa.write_decisions(tmp_path / "g2.csv")).filter(pl.col("family") != "deposit_1").write_csv(g2)
    with pytest.raises(ValueError, match="deposit_1"):
        _build(tmp_path, g2)


def test_source_keys_are_validated_before_keeping_only_base_cases(tmp_path, monkeypatch):
    # la fila con clave nula no pertenece a la base: antes desaparecía sin control
    _source_dir(tmp_path, monkeypatch, pl.DataFrame({"case_id": [1, None], "num_group1": [0, 0], "openingdate_313D": ["2020-01-01"] * 2}))
    with pytest.raises(ValueError, match="null"):
        _build(tmp_path)


def test_rows_of_cases_outside_the_base_are_counted(tmp_path, monkeypatch):
    _source_dir(tmp_path, monkeypatch, pl.DataFrame({"case_id": [1, 99], "num_group1": [0, 0], "openingdate_313D": ["2020-01-01"] * 2}))
    import json
    manifest = json.loads(_build(tmp_path).manifest_path.read_text())
    assert manifest["exclusions"]["source_rows_outside_base"] == 1


def _other_1(tmp_path, monkeypatch, frame):
    root = tmp_path / "data"
    for split in ("train", "test"):
        (root / "parquet_files" / split).mkdir(parents=True)
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    frame.write_parquet(root / "parquet_files" / "train" / "train_other_1.parquet")
    base = pl.DataFrame({"case_id": [1, 2], "WEEK_NUM": [10, 10], "date_decision": ["2020-01-10"] * 2})
    return lambda: pr.build_other_1_features(
        base, "train", output_dir=tmp_path / "out", selection_path=ppa.write_selection(tmp_path / "sel.csv"),
        g2_path=ppa.write_decisions(tmp_path / "g2.csv"),
    )


def test_other_1_is_one_row_per_base_case_but_the_file_key_is_case_and_group(tmp_path, monkeypatch):
    # como en la muestra real de test: un caso ajeno a la base con num_group1 0 y 1 no es error
    build = _other_1(tmp_path, monkeypatch, pl.DataFrame({"case_id": [1, 99, 99], "num_group1": [0, 0, 1]}))
    assert build().case_count == 2
    # un caso de la base con dos filas sí rompe el supuesto 1:0..1
    build = _other_1(tmp_path / "b", monkeypatch, pl.DataFrame({"case_id": [1, 1], "num_group1": [0, 1]}))
    with pytest.raises(ValueError, match="1:0..1"):
        build()
