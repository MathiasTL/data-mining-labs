import math

import polars as pl
import pytest

from dm_credit.preparation import coercion
from dm_credit.preparation.coercion import CoercionError


def test_string_to_date_strict_format():
    df = pl.DataFrame({"d": ["2020-01-31", None]})
    out = coercion.coerce_to_logical(df, {"d": "date"})
    assert out.schema["d"] == pl.Date
    assert out["d"].to_list()[0].isoformat() == "2020-01-31" and out["d"][1] is None


def test_invalid_date_raises_with_column_and_count():
    df = pl.DataFrame({"d": ["2020-01-31", "31/01/2020", "xx"]})
    with pytest.raises(CoercionError, match=r"d.*2"):
        coercion.coerce_to_logical(df, {"d": "date"})


def test_all_null_float_date_becomes_date():
    # las cuatro fechas de test llegan como Float64 completamente nulo
    df = pl.DataFrame({"d": [None, None]}, schema={"d": pl.Float64})
    assert coercion.coerce_to_logical(df, {"d": "date"}).schema["d"] == pl.Date


def test_float_date_with_values_is_rejected():
    df = pl.DataFrame({"d": [18000.0, None]})
    with pytest.raises(CoercionError, match="d"):
        coercion.coerce_to_logical(df, {"d": "date"})


def test_physical_null_dtype_adopts_logical_type():
    df = pl.DataFrame({"x": [None, None]}, schema={"x": pl.Null})
    for logical, dtype in [("float", pl.Float64), ("int", pl.Int64), ("bool", pl.Boolean), ("string", pl.String), ("date", pl.Date)]:
        assert coercion.coerce_to_logical(df, {"x": logical}).schema["x"] == dtype


def test_float_to_int_only_if_whole_numbers():
    ok = coercion.coerce_to_logical(pl.DataFrame({"n": [1.0, 2.0, None]}), {"n": "int"})
    assert ok.schema["n"] == pl.Int64 and ok["n"].to_list() == [1, 2, None]
    with pytest.raises(CoercionError, match="n"):
        coercion.coerce_to_logical(pl.DataFrame({"n": [1.0, 2.5]}), {"n": "int"})


def test_bool_from_text_accepts_only_known_tokens():
    out = coercion.coerce_to_logical(pl.DataFrame({"b": ["true", "false", "1", "0", None]}), {"b": "bool"})
    assert out["b"].to_list() == [True, False, True, False, None]
    with pytest.raises(CoercionError, match="b"):
        coercion.coerce_to_logical(pl.DataFrame({"b": ["true", "maybe"]}), {"b": "bool"})


def test_non_numeric_text_for_float_raises():
    with pytest.raises(CoercionError, match="a"):
        coercion.coerce_to_logical(pl.DataFrame({"a": ["1.5", "abc"]}), {"a": "float"})


def test_numeric_text_for_float_is_converted():
    out = coercion.coerce_to_logical(pl.DataFrame({"a": ["1.5", None]}), {"a": "float"})
    assert out["a"].to_list() == [1.5, None]


def test_infinite_values_raise():
    for bad in (math.inf, -math.inf):
        with pytest.raises(CoercionError, match="inf"):
            coercion.coerce_to_logical(pl.DataFrame({"a": [1.0, bad]}), {"a": "float"})


def test_nan_is_treated_as_missing():
    out = coercion.coerce_to_logical(pl.DataFrame({"a": [1.0, float("nan")]}), {"a": "float"})
    assert out["a"].null_count() == 1


def test_missing_required_column_raises():
    with pytest.raises(CoercionError, match="missing"):
        coercion.coerce_to_logical(pl.DataFrame({"a": [1.0]}), {"a": "float", "b": "float"})


def test_extra_columns_are_kept_untouched_and_order_follows_input():
    df = pl.DataFrame({"z": [1], "a": ["2020-01-01"]})
    out = coercion.coerce_to_logical(df, {"a": "date"})
    assert out.columns == ["z", "a"] and out["z"].to_list() == [1]


def test_unknown_logical_type_raises():
    with pytest.raises(ValueError, match="logical"):
        coercion.coerce_to_logical(pl.DataFrame({"a": [1]}), {"a": "decimal"})


def test_logical_type_for_uses_suffix_and_train_dtype():
    assert coercion.logical_type_for("creationdate_885D", pl.String) == "date"
    assert coercion.logical_type_for("validfrom_1069D", pl.Float64) == "date"
    assert coercion.logical_type_for("status_219L", pl.String) == "string"
    assert coercion.logical_type_for("profession_152M", pl.String) == "string"
    assert coercion.logical_type_for("credamount_590A", pl.Float64) == "float"
    assert coercion.logical_type_for("isbidproduct_390L", pl.Boolean) == "bool"
    assert coercion.logical_type_for("case_id", pl.Int64) == "int"
    assert coercion.logical_type_for("num_group1", pl.Int64) == "int"
    assert coercion.logical_type_for("WEEK_NUM", pl.Int64) == "int"


def test_lenient_date_formats_are_rejected():
    # '2020-1-5' no cumple %Y-%m-%d: no se acepta en silencio
    with pytest.raises(CoercionError, match="d"):
        coercion.coerce_to_logical(pl.DataFrame({"d": ["2020-1-5"]}), {"d": "date"})


def test_large_integer_text_keeps_exact_value():
    out = coercion.coerce_to_logical(pl.DataFrame({"n": ["9007199254740993"]}), {"n": "int"})
    assert out["n"].to_list() == [9007199254740993]


def test_integer_overflow_raises_coercion_error_not_polars_error():
    with pytest.raises(CoercionError, match="n"):
        coercion.coerce_to_logical(pl.DataFrame({"n": [1e20]}), {"n": "int"})


def test_datetime_with_time_component_is_rejected_for_date():
    from datetime import datetime

    with pytest.raises(CoercionError, match="d"):
        coercion.coerce_to_logical(pl.DataFrame({"d": [datetime(2020, 1, 1, 13, 30)]}), {"d": "date"})


def test_midnight_datetime_becomes_date():
    from datetime import datetime

    out = coercion.coerce_to_logical(pl.DataFrame({"d": [datetime(2020, 1, 1)]}), {"d": "date"})
    assert out.schema["d"] == pl.Date


def test_float_to_string_does_not_write_nan_or_inf_as_text():
    out = coercion.coerce_to_logical(pl.DataFrame({"s": [1.5, float("nan")]}), {"s": "string"})
    assert out["s"].to_list() == ["1.5", None]
    with pytest.raises(CoercionError, match="inf"):
        coercion.coerce_to_logical(pl.DataFrame({"s": [float("inf")]}), {"s": "string"})


def test_all_null_column_of_any_dtype_adopts_logical_type():
    df = pl.DataFrame({"b": [None, None]}, schema={"b": pl.Boolean})
    assert coercion.coerce_to_logical(df, {"b": "int"}).schema["b"] == pl.Int64
    assert coercion.coerce_to_logical(df, {"b": "float"}).schema["b"] == pl.Float64


def test_infinite_text_is_rejected():
    with pytest.raises(CoercionError, match="inf"):
        coercion.coerce_to_logical(pl.DataFrame({"a": ["inf"]}), {"a": "float"})
