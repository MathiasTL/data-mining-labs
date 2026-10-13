"""Pruebas A6: faltantes, categorías, coerción, extremos y nombres (Gabriel, P5).

Cada prueba fija un punto de la propuesta
context/03-preparacion-A6-faltantes-categorias.md con datos sintéticos.
"""

import re

import polars as pl
import pytest

import preparation_a6_fixtures as fx
from dm_credit.preparation import coercion, outliers
from dm_credit.preparation.coercion import CoercionError
from dm_credit.preparation.outliers import DomainViolationError, RangeWarning

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(__[a-z0-9_]+){1,4}$")
RESERVED = {"case_id", "num_group1", "num_group2", "date_decision", "MONTH", "WEEK_NUM", "target"}


def valid_name(n: str) -> bool:
    if n in RESERVED:
        return False
    if n != n.lower():
        return False
    if not NAME_RE.match(n):
        return False
    return True


def test_missing_indicator_marks_nulls():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    out = fx.with_missing_indicator(df, "a")
    assert out["a__missing"].to_list() == [0, 1, 0]


def test_nan_is_missing_not_error():
    out = coercion.coerce_to_logical(pl.DataFrame({"a": [1.0, float("nan")]}), {"a": "float"})
    assert out["a"].null_count() == 1


def test_infinite_is_error():
    with pytest.raises(CoercionError, match="inf"):
        coercion.coerce_to_logical(pl.DataFrame({"a": [1.0, float("inf")]}), {"a": "float"})


def test_non_convertible_text_is_error():
    with pytest.raises(CoercionError, match="a"):
        coercion.coerce_to_logical(pl.DataFrame({"a": ["1.5", "abc"]}), {"a": "float"})
    with pytest.raises(CoercionError, match="d"):
        coercion.coerce_to_logical(pl.DataFrame({"d": ["2020-01-31", "31/01/2020"]}), {"d": "date"})
    with pytest.raises(CoercionError, match="b"):
        coercion.coerce_to_logical(pl.DataFrame({"b": ["true", "maybe"]}), {"b": "bool"})


def test_physical_null_adopts_logical_type():
    df = fx.physical_null()
    assert coercion.coerce_to_logical(df, {"x": "float"}).schema["x"] == pl.Float64
    assert coercion.coerce_to_logical(df, {"x": "date"}).schema["x"] == pl.Date


def test_all_null_float_date_becomes_date():
    out = coercion.coerce_to_logical(fx.all_null_float(), {"d": "date"})
    assert out.schema["d"] == pl.Date


def test_unknown_category_maps_to_unknown_token():
    vocab = fx.fit_vocab(fx.categorical_train()["c"].to_list(), min_frequency=0.1)
    assert "rare" not in vocab and fx.UNKNOWN in vocab  # rara colapsa con min_frequency=0.1
    out = fx.apply_vocab(fx.categorical_oot()["c"].to_list(), vocab)
    assert out == ["x", fx.UNKNOWN, fx.MISSING, "y"]


def test_vocab_is_frozen_against_oot():
    vocab = fx.fit_vocab(["a", "b", "a", "b"], min_frequency=0.01)
    before = list(vocab)
    fx.apply_vocab(["a", "zzz"], vocab)
    assert vocab == before  # transform no amplía


def test_fit_rejects_oot_week():
    with pytest.raises(ValueError, match="74"):
        outliers.fit_range_policy(fx.oot_week_table(), ["a"])


def test_out_of_range_warns_without_clipping():
    pol = outliers.fit_range_policy(fx.range_train(), ["a"])
    with pytest.warns(RangeWarning, match="a"):
        out, report = pol.apply(fx.range_oot_values())
    assert out["a"].to_list() == [0.0, 50.0, 500.0]
    row = report.row(0, named=True)
    assert (row["n_below"], row["n_above"], row["n_clipped"]) == (1, 1, 0)


def test_clip_only_when_approved():
    train = pl.DataFrame({"a": [float(i) for i in range(1, 101)]})
    pol = outliers.fit_range_policy(train, ["a"], clip_columns=["a"], clip_quantiles=(0.05, 0.95))
    with pytest.warns(RangeWarning):
        out, report = pol.apply(pl.DataFrame({"a": [-1000.0, 50.0, 1000.0]}))
    lo, hi = pol.clip_bounds["a"]
    assert out["a"].to_list() == pytest.approx([lo, 50.0, hi])
    assert report["n_clipped"][0] == 2


def test_domain_violation_is_error():
    pol = outliers.fit_range_policy(fx.range_train(), ["a"], domain={"a": (0.0, None)})
    with pytest.raises(DomainViolationError, match="a"):
        pol.apply(pl.DataFrame({"a": [-1.0]}))


def test_names_follow_convention():
    for n in fx.VALID_NAMES:
        assert valid_name(n), n
    assert not valid_name("WEEK_NUM")
    assert not valid_name("Deposit__Amt__Mean")
    assert not valid_name("sinseparador")
