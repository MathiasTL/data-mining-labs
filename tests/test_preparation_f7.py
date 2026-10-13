"""F7: informe de fuga temporal consolidado por familia."""

import polars as pl

import test_preparation_validation as tv
from dm_credit.preparation import f7

published = tv.published

G2 = pl.DataFrame(
    {
        "family": ["applprev_1", "applprev_1", "applprev_2", "person_1", "credit_bureau_a_2", "static_0", "base", "tax_registry_a_1"],
        "column": ["a", "b", "c", "d", "e", "f", "case_id", "g"],
        "g2_decision": ["aprobada", "bloqueada", "excluida", "aplazada", "bloqueada", "bloqueada", "excluida", "excluida"],
    }
)


def summary(published):
    return f7.leakage_by_family(published["artifacts"], G2)


def row(df, family):
    return df.filter(pl.col("family") == family).row(0, named=True)


def test_one_row_per_family_with_the_expected_columns(published):
    s = summary(published)
    assert s.height == 12 and s["family"].is_unique().all()
    assert s.columns == list(f7.SUMMARY_COLUMNS)


def test_row_and_value_exclusions_are_summed_per_split(published):
    r = row(summary(published), "applprev")
    # train (casos 1 y 2): una solicitud posterior y un hijo con padre inválido; una aprobación posterior anulada
    assert (r["rows_excluded_train"], r["values_nulled_train"]) == (2, 1)
    assert r["as_of_row_filter"] is True


def test_coverage_only_families_without_a_row_rule_are_flagged(published):
    r = row(summary(published), "tax_registry_a_1")
    assert r["as_of_row_filter"] is False and r["predictors"] == 0


def test_g2_columns_are_counted_per_family_including_child_tables(published):
    s = summary(published)
    a = row(s, "applprev")
    assert (a["columns_approved"], a["columns_blocked"], a["columns_excluded"], a["columns_deferred"]) == (1, 1, 1, 0)
    assert row(s, "credit_bureau_a")["columns_blocked"] == 1  # a_2 cuenta dentro de bureau A
    assert row(s, "person")["columns_deferred"] == 1


def test_rule_and_evidence_are_filled(published):
    r = row(summary(published), "applprev")
    assert "date_decision" in r["temporal_rule"]
    assert "manifest" in r["evidence"] and "G2" in r["evidence"]
