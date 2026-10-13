from datetime import date

import polars as pl
import pytest

from dm_credit.schema_compatibility import (
    UNKNOWN_CATEGORY,
    apply_category_vocabulary,
    classify_type_compatibility,
    coerce_to_schema,
    family_coverage,
    fit_category_vocabulary,
    fit_numeric_statistics,
    numeric_range_check,
)


def test_identical_schema_and_different_column_order_are_accepted():
    frame = pl.DataFrame({"category": ["a"], "value": [1]})

    result = coerce_to_schema(frame, {"value": "Int64", "category": "String"})

    assert result.columns == ["value", "category"]
    assert result.schema == {"value": pl.Int64, "category": pl.String}


def test_missing_predictor_and_unexpected_test_column_are_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        coerce_to_schema(pl.DataFrame({"x": [1]}), {"x": "Int64", "y": "Float64"})
    with pytest.raises(ValueError, match="Unexpected columns"):
        coerce_to_schema(pl.DataFrame({"x": [1], "extra": [2]}), {"x": "Int64"})


def test_target_only_in_train_is_an_expected_coverage_difference():
    rows = [
        {"split": "train", "family": "base", "file": "train_base.parquet", "rows": 2},
        {"split": "test", "family": "base", "file": "test_base.parquet", "rows": 1},
    ]
    schemas = {
        ("train", "base"): {"case_id": {"Int64"}, "target": {"Int64"}},
        ("test", "base"): {"case_id": {"Int64"}},
    }
    for family in (
        "static_0",
        "static_cb_0",
        "applprev_1",
        "credit_bureau_a_1",
        "credit_bureau_b_1",
        "debitcard_1",
        "deposit_1",
        "other_1",
        "person_1",
        "tax_registry_a_1",
        "tax_registry_b_1",
        "tax_registry_c_1",
        "applprev_2",
        "credit_bureau_a_2",
        "credit_bureau_b_2",
        "person_2",
    ):
        rows.extend(
            [
                {
                    "split": "train",
                    "family": family,
                    "file": f"train_{family}.parquet",
                    "rows": 0,
                },
                {
                    "split": "test",
                    "family": family,
                    "file": f"test_{family}.parquet",
                    "rows": 0,
                },
            ]
        )
        schemas[("train", family)] = {"case_id": {"Int64"}}
        schemas[("test", family)] = {"case_id": {"Int64"}}

    base = (
        family_coverage(rows, schemas)
        .filter(pl.col("family") == "base")
        .row(0, named=True)
    )

    assert base["train_only_columns"] == "target"
    assert base["logical_schema_match"] is True


def test_integer_float_is_safely_coercible_but_fraction_is_rejected():
    status, _, failures = classify_type_compatibility(
        {"Int64"}, {"Float64"}, "Int64", pl.Series([1.0, None, 2.0])
    )
    assert status == "safe_explicit_coercion"
    assert failures == 0

    with pytest.raises(ValueError, match="cannot be converted"):
        coerce_to_schema(pl.DataFrame({"x": [1.5]}), {"x": "Int64"})


def test_physical_null_column_uses_logical_train_type():
    status, action, failures = classify_type_compatibility(
        {"String"}, {"Null"}, "String", pl.Series("x", [None, None], dtype=pl.Null)
    )
    assert status == "test_physical_null"
    assert "String" in action
    assert failures == 0


def test_valid_date_is_parsed_and_invalid_date_is_rejected():
    result = coerce_to_schema(
        pl.DataFrame({"event_1D": ["2020-01-02"]}), {"event_1D": "Date"}
    )
    assert result["event_1D"].to_list() == [date(2020, 1, 2)]

    with pytest.raises(ValueError, match="cannot be converted to Date"):
        coerce_to_schema(
            pl.DataFrame({"event_1D": ["not-a-date"]}), {"event_1D": "Date"}
        )


def test_unknown_category_does_not_extend_train_vocabulary():
    train = pl.Series("category", ["a", "b", "a"])
    vocabulary = fit_category_vocabulary(train)

    encoded = apply_category_vocabulary(
        pl.Series("category", ["b", "new", None]), vocabulary
    )

    assert encoded.to_list() == ["b", UNKNOWN_CATEGORY, None]
    assert vocabulary == {"a", "b"}


def test_out_of_train_range_is_reported_not_rejected():
    report = numeric_range_check(pl.Series([0.0, 10.0]), pl.Series([-1.0, 5.0, 12.0]))

    assert report["test_outside_train_count"] == 2
    assert report["test_non_finite_count"] == 0


def test_test_values_do_not_change_fitted_train_statistics():
    train = pl.Series([0.0, 2.0])
    fitted = fit_numeric_statistics(train)

    numeric_range_check(train, pl.Series([1_000_000.0]))

    assert fitted == fit_numeric_statistics(train)
    assert fitted["mean"] == 1.0


def test_non_convertible_and_infinite_values_are_rejected():
    with pytest.raises(ValueError, match="cannot be converted"):
        coerce_to_schema(pl.DataFrame({"x": ["bad"]}), {"x": "Float64"})
    with pytest.raises(ValueError, match="infinite"):
        coerce_to_schema(pl.DataFrame({"x": [float("inf")]}), {"x": "Float64"})
