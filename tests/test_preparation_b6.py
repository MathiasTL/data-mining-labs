"""Pruebas B6: selección de columnas de productos y registros fiscales (Gabriel, P5)."""

import polars as pl
import pytest

from dm_credit.preparation import products_registry_selection as b6


def row(family: str, column: str, **kw) -> dict:
    base = {
        "family": family,
        "column": column,
        "dtype": "Float64",
        "function": "importe/capacidad financiera",
        "temporal_relevance": "sin referencia temporal explícita",
        "identifier_risk": "no señalado",
        "leakage_review": "sin señal automática",
    }
    base.update(kw)
    return base


def test_keys_are_excluded():
    for fam in b6.FAMILIES:
        for key in ("case_id", "num_group1", "num_group2"):
            action, _, _ = b6._decide(row(fam, key))
            assert action == "excluir"


def test_quasi_identifiers_are_excluded():
    action, _, _ = b6._decide(
        row("tax_registry_a_1", "name_4527232M", dtype="String", identifier_risk="posible cuasi-identificador/alta cardinalidad")
    )
    assert action == "excluir"
    action, _, _ = b6._decide(
        row("tax_registry_c_1", "employername_160M", dtype="String", identifier_risk="posible cuasi-identificador/alta cardinalidad")
    )
    assert action == "excluir"


def test_other_1_is_blocked_not_excluded():
    action, reason, _ = b6._decide(row("other_1", "amtdepositbalance_4809441A"))
    assert action == "bloquear_hasta_G2"
    assert "G2" in reason


def test_temporal_columns_transform_as_of():
    action, _, _ = b6._decide(
        row("deposit_1", "openingdate_313D", dtype="String", function="temporal", temporal_relevance="referencia temporal explícita")
    )
    assert action == "transformar_as_of"


def test_windowed_amounts_are_candidates():
    action, _, _ = b6._decide(row("debitcard_1", "last30dayturnover_651A", temporal_relevance="referencia temporal explícita"))
    assert action == "candidato_G2"


def test_amounts_without_reference_are_blocked():
    action, _, _ = b6._decide(row("deposit_1", "amount_416A"))
    assert action == "bloquear_hasta_G2"
    action, _, _ = b6._decide(row("tax_registry_c_1", "pmtamount_36A"))
    assert action == "bloquear_hasta_G2"


def test_null_dtype_does_not_crash_the_selector():
    # tax_registry_c_1 llega vacío en test con columnas físicas Null
    df = pl.DataFrame(
        [
            {"family": "tax_registry_c_1", "column": "x", "dtype": "Null", "function": "temporal",
             "temporal_relevance": "referencia temporal explícita", "identifier_risk": "no señalado", "leakage_review": ""},
            {"family": "tax_registry_c_1", "column": "case_id", "dtype": "Null", "function": "clave/índice estructural",
             "temporal_relevance": "no aplica", "identifier_risk": "identificador técnico", "leakage_review": ""},
        ]
    )
    out = b6.select_columns(df)
    assert out.sort("column")["action"].to_list() == ["excluir", "transformar_as_of"]


def test_unknown_family_is_rejected():
    df = pl.DataFrame([row("applprev_1", "x")])
    with pytest.raises(ValueError, match="unsupported family"):
        b6.select_columns(df)


def test_real_catalog_yields_33_rows_with_no_candidate_lacking_reason(tmp_path=None):
    catalog = pl.read_csv("results/variable_catalog.csv").filter(pl.col("family").is_in(b6.FAMILIES))
    out = b6.select_columns(catalog)
    assert out.height == 33
    assert out["action"].null_count() == 0 and out["reason"].null_count() == 0
    counts = dict(out.group_by("action").len().rows())
    assert counts == {"bloquear_hasta_G2": 9, "candidato_G2": 3, "excluir": 15, "transformar_as_of": 6}
