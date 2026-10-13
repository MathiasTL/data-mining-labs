"""Pruebas B5: selección de columnas de bureau de crédito (Mathias, P4)."""

from pathlib import Path

import polars as pl
import pytest

from dm_credit.preparation import bureau_selection as b5

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "results" / "variable_catalog.csv"
OUTPUT = ROOT / "results" / "preparation" / "B5_bureau_columns.csv"
ACTIONS = {"excluir", "bloquear_hasta_G2", "referencia_as_of", "transformar_as_of", "candidato_G2"}
INCLUDED = {"referencia_as_of", "transformar_as_of", "candidato_G2"}


def row(family: str, column: str, **kw) -> dict:
    base = {
        "family": family,
        "column": column,
        "dtype": "Float64",
        "function": "importe/capacidad financiera",
        "temporal_relevance": "sin referencia temporal explícita",
        "identifier_risk": "no señalado",
        "leakage_review": "sin señal automática; revisar dominio",
        "pct_null": 50.0,
        "n_unique": 100,
    }
    base.update(kw)
    return base


def decide(**kw) -> dict:
    return b5.select_columns(pl.DataFrame([row(**kw)])).row(0, named=True)


@pytest.fixture(scope="module")
def real():
    if not CATALOG.exists():
        pytest.skip("variable_catalog.csv no disponible")
    catalog = pl.read_csv(CATALOG).filter(pl.col("family").is_in(b5.FAMILIES))
    return catalog, b5.select_columns(catalog)


def test_reserved_keys_are_excluded_in_every_family():
    for fam in b5.FAMILIES:
        for key in ("case_id", "num_group1", "num_group2"):
            r = decide(family=fam, column=key, dtype="Int64", function="clave/índice estructural", pct_null=0.0)
            assert r["action"] == "excluir"


def test_institution_and_creditor_names_are_excluded():
    r = decide(family="credit_bureau_a_1", column="financialinstitution_591M", dtype="String",
               identifier_risk="posible cuasi-identificador/alta cardinalidad", pct_null=0.0)
    assert r["action"] == "excluir"
    r = decide(family="credit_bureau_b_1", column="credor_3940957M", dtype="String", pct_null=0.0)
    assert r["action"] == "excluir"


def test_snapshot_dates_are_the_as_of_references():
    for fam, col in b5.AS_OF_REFERENCE:
        r = decide(family=fam, column=col, dtype="String", function="temporal",
                   leakage_review="revisar fecha contra date_decision")
        assert r["action"] == "referencia_as_of"
        assert col in r["temporal_rule"] and "<= date_decision" in r["temporal_rule"]


def test_low_coverage_is_excluded_with_g2_rule_4():
    r = decide(family="credit_bureau_a_1", column="interestrate_508L", pct_null=99.5871)
    assert r["action"] == "excluir" and "regla 4" in r["reason"]


def test_states_payments_and_dpd_are_blocked_with_explicit_as_of_rule():
    cases = [
        ("credit_bureau_a_1", "dpdmax_139P", "mora/incumplimiento histórico", "revisar disponibilidad al decidir"),
        ("credit_bureau_a_1", "contractst_545M", "producto/contrato", "revisar disponibilidad al decidir"),
        ("credit_bureau_b_1", "pmtdaysoverdue_1135P", "mora/incumplimiento histórico", "revisar disponibilidad al decidir"),
        ("credit_bureau_b_2", "pmts_dpdvalue_108P", "mora/incumplimiento histórico", "revisar disponibilidad al decidir"),
    ]
    for fam, col, function, leak in cases:
        r = decide(family=fam, column=col, function=function, leakage_review=leak)
        assert r["action"] == "bloquear_hasta_G2"
        assert "date_decision" in r["temporal_rule"]


def test_balances_without_catalog_flag_are_still_blocked():
    r = decide(family="credit_bureau_a_1", column="outstandingamount_362A")
    assert r["action"] == "bloquear_hasta_G2"


def test_contract_end_dates_are_blocked_as_calendar():
    r = decide(family="credit_bureau_a_1", column="dateofcredend_289D", dtype="String", function="temporal",
               leakage_review="revisar fecha contra date_decision")
    assert r["action"] == "bloquear_hasta_G2" and "regla 3" in r["reason"]


def test_start_dates_become_as_of_transformations():
    r = decide(family="credit_bureau_b_1", column="contractdate_551D", dtype="String", function="temporal",
               leakage_review="revisar fecha contra date_decision")
    assert r["action"] == "transformar_as_of" and "lastupdate_260D" in r["temporal_rule"]


def test_credit_bureau_a_2_is_never_admitted_directly():
    r = decide(family="credit_bureau_a_2", column="pmts_year_1139T", function="producto/contrato",
               leakage_review="revisar disponibilidad al decidir")
    assert r["action"] == "bloquear_hasta_G2" and "D6" in r["note"]
    r = decide(family="credit_bureau_a_2", column="subjectroles_name_838M", dtype="String",
               identifier_risk="posible cuasi-identificador/alta cardinalidad", pct_null=0.0)
    assert r["action"] == "bloquear_hasta_G2"


def test_contract_terms_are_candidates():
    r = decide(family="credit_bureau_b_1", column="credlmt_1052A")
    assert r["action"] == "candidato_G2" and "lastupdate_260D" in r["temporal_rule"]


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unsupported family"):
        b5.select_columns(pl.DataFrame([row("applprev_1", "x")]))


def test_every_real_column_gets_exactly_one_valid_action(real):
    catalog, out = real
    assert out.height == catalog.height == 149
    assert set(out["action"]) <= ACTIONS
    assert out.select(["family", "column"]).is_duplicated().sum() == 0
    counts = {fam: n for fam, n in out.group_by("family").len().rows()}
    assert counts == {"credit_bureau_a_1": 79, "credit_bureau_a_2": 19, "credit_bureau_b_1": 45, "credit_bureau_b_2": 6}


def test_reserved_columns_are_never_included(real):
    _, out = real
    reserved = out.filter(pl.col("column").is_in(list(b5.RESERVED_KEYS)))
    assert reserved.height == 2 + 3 + 2 + 3  # a_1, a_2, b_1, b_2
    assert set(reserved["action"]) == {"excluir"}


def test_every_inclusion_has_as_of_rule_and_every_exclusion_has_evidence(real):
    _, out = real
    included = out.filter(pl.col("action").is_in(list(INCLUDED)))
    assert included.height > 0
    assert included.filter(~pl.col("temporal_rule").str.contains("date_decision")).height == 0
    for action in ("excluir", "bloquear_hasta_G2"):
        part = out.filter(pl.col("action") == action)
        assert part.filter(pl.col("reason").is_null() | (pl.col("reason") == "")).height == 0
        assert part.filter(~pl.col("evidence").str.starts_with("variable_catalog.csv:")).height == 0
    assert out.filter(pl.col("action") == "referencia_as_of").height == 4


def test_published_csv_matches_a_fresh_run(real):
    if not OUTPUT.exists():
        pytest.skip("B5_bureau_columns.csv no generado")
    _, out = real
    published = pl.read_csv(OUTPUT, schema={c: pl.String for c in b5.OUTPUT_COLUMNS})
    assert published.fill_null("").equals(out.fill_null(""))
