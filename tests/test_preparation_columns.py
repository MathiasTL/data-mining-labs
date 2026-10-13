import pytest

from dm_credit.baseline import RESERVED
from dm_credit.preparation.columns import (
    RESERVED_COLUMNS,
    validate_control_columns,
    validate_predictor_columns,
)


@pytest.mark.parametrize("split", ["train", "oot"])
def test_labeled_partitions_require_target_as_control(split):
    columns = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target", "signal")
    assert validate_control_columns(columns, split) == columns[:-1]
    with pytest.raises(ValueError, match="target"):
        validate_control_columns(columns[:-2] + ("signal",), split)


def test_test_must_not_contain_target_even_if_null():
    columns = ("case_id", "WEEK_NUM", "date_decision", "signal")
    assert validate_control_columns(columns, "test") == columns[:3]
    with pytest.raises(ValueError, match="target"):
        validate_control_columns((*columns, "target"), "test")


@pytest.mark.parametrize("reserved", RESERVED)
def test_every_reserved_column_is_rejected_as_predictor(reserved):
    assert reserved in RESERVED_COLUMNS
    with pytest.raises(ValueError, match="reserved"):
        validate_predictor_columns(
            ("signal", reserved), available_columns=("signal", reserved)
        )


def test_explicit_predictors_preserve_order_and_must_exist():
    assert validate_predictor_columns(
        ("signal_2", "signal_1"), available_columns=("case_id", "signal_1", "signal_2")
    ) == ("signal_2", "signal_1")
    with pytest.raises(ValueError, match="missing predictor"):
        validate_predictor_columns(("not_present",), available_columns=("signal_1",))
    with pytest.raises(ValueError, match="duplicate predictor"):
        validate_predictor_columns(("signal_1", "signal_1"))


def test_controls_must_include_case_week_and_decision_date():
    for missing in ("case_id", "WEEK_NUM", "date_decision"):
        columns = tuple(
            c
            for c in ("case_id", "WEEK_NUM", "date_decision", "target")
            if c != missing
        )
        with pytest.raises(ValueError, match=missing):
            validate_control_columns(columns, "train")
