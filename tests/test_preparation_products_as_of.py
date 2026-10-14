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
