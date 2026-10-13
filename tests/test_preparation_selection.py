from pathlib import Path

import polars as pl
import pytest

from dm_credit.preparation import applications_people_selection as sel

ACTIONS = {"excluir", "bloquear_hasta_G2", "referencia_as_of", "transformar_as_of", "candidato_G2"}


def row(family, column, dtype="String", function="persona/demografía", ident="no señalado", leak="sin señal automática; revisar dominio", temporal="histórico implícito"):
    return {
        "family": family,
        "column": column,
        "dtype": dtype,
        "function": function,
        "identifier_risk": ident,
        "leakage_review": leak,
        "temporal_relevance": temporal,
    }


def decide(**kw):
    out = sel.select_columns(pl.DataFrame([row(**kw)]))
    return out.row(0, named=True)


def test_reserved_keys_are_excluded():
    for col in ("case_id", "num_group1", "num_group2"):
        r = decide(family="applprev_2", column=col, function="clave/índice estructural", leak="excluir de predictores")
        assert r["action"] == "excluir"


def test_quasi_identifier_is_excluded():
    r = decide(family="person_1", column="registaddr_zipcode_184M", ident="posible cuasi-identificador/alta cardinalidad")
    assert r["action"] == "excluir"


def test_boolean_flagged_as_quasi_identifier_is_blocked_not_excluded():
    r = decide(family="person_1", column="contaddr_matchlist_1032L", dtype="Boolean", ident="posible cuasi-identificador/alta cardinalidad")
    assert r["action"] == "bloquear_hasta_G2"


def test_creation_date_is_the_as_of_reference():
    r = decide(family="applprev_1", column="creationdate_885D", dtype="String", function="temporal", leak="revisar fecha contra date_decision")
    assert r["action"] == "referencia_as_of"


def test_other_dates_become_as_of_transformations():
    r = decide(family="applprev_1", column="approvaldate_319D", function="temporal", leak="revisar fecha contra date_decision")
    assert r["action"] == "transformar_as_of"


def test_states_and_outcomes_are_blocked_until_g2():
    r = decide(family="applprev_1", column="status_219L", function="producto/contrato", leak="revisar disponibilidad al decidir")
    assert r["action"] == "bloquear_hasta_G2"


def test_applprev_2_has_no_row_date_so_nothing_is_admitted_directly():
    r = decide(family="applprev_2", column="conts_type_509L")
    assert r["action"] == "bloquear_hasta_G2" and "padre" in r["reason"]
    r = decide(family="applprev_2", column="credacc_cards_status_52L", leak="revisar disponibilidad al decidir")
    assert r["action"] == "excluir"


def test_previous_application_state_without_own_date_is_blocked():
    # el estado de un contrato previo sin fecha propia puede reflejar información posterior a su creación
    r = decide(family="applprev_1", column="credacc_actualbalance_314A", dtype="Float64", function="importe/capacidad financiera", temporal="sin referencia temporal explícita")
    assert r["action"] == "bloquear_hasta_G2" and "estado" in r["reason"]


def test_person_attribute_without_row_date_stays_candidate():
    r = decide(family="person_1", column="education_927M", temporal="sin referencia temporal explícita")
    assert r["action"] == "candidato_G2"


def test_plain_attribute_is_candidate_pending_g2():
    r = decide(family="person_1", column="education_927M")
    assert r["action"] == "candidato_G2"


def test_sensitive_attributes_are_flagged_for_team_decision():
    for col in ("gender_992L", "sex_738L"):
        r = decide(family="person_1", column=col)
        assert r["action"] == "candidato_G2" and "sensible" in r["note"]


def test_only_the_four_families_are_accepted():
    with pytest.raises(ValueError, match="family"):
        sel.select_columns(pl.DataFrame([row(family="static_0", column="x")]))


def test_every_real_column_gets_exactly_one_valid_action():
    path = Path(__file__).resolve().parents[1] / "results" / "variable_catalog.csv"
    if not path.exists():
        pytest.skip("variable_catalog.csv no disponible")
    catalog = pl.read_csv(path).filter(pl.col("family").is_in(sel.FAMILIES))
    out = sel.select_columns(catalog)
    assert out.height == catalog.height == 95
    assert set(out["action"]) <= ACTIONS
    assert out.select(["family", "column"]).is_duplicated().sum() == 0
    assert out.filter(pl.col("column") == "creationdate_885D")["action"][0] == "referencia_as_of"
