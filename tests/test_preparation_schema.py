import polars as pl
import pytest

from dm_credit.preparation.schema import validate_matrix_schema, validate_matrix_schemas


def matrices():
    train = pl.DataFrame(
        {
            "case_id": [1, 2],
            "WEEK_NUM": [72, 73],
            "date_decision": ["2020-01-01", "2020-01-02"],
            "MONTH": [202001, 202001],
            "target": [0, 1],
            "source__n_records": [1, 2],
            "source__amount__mean__all": [2.0, 3.0],
        }
    )
    oot = train.with_columns(
        pl.Series("case_id", [3, 4]),
        pl.Series("WEEK_NUM", [74, 75]),
        pl.Series("target", [1, 0]),
    )
    test = oot.drop("target")
    return train, oot, test


def test_matching_matrices_have_ordered_logical_schema_and_separate_controls():
    train, oot, test = matrices()
    oot = oot.with_columns(pl.col("source__n_records").cast(pl.Int32))
    test = test.with_columns(pl.col("source__amount__mean__all").cast(pl.Float32))
    schema = validate_matrix_schemas(train, oot, test)
    assert schema.control_columns == (
        "case_id",
        "WEEK_NUM",
        "date_decision",
        "MONTH",
        "target",
    )
    assert schema.feature_schema == (
        ("source__n_records", "int"),
        ("source__amount__mean__all", "float"),
    )
    assert "target" not in validate_matrix_schema(test, "test").control_columns


def test_rejects_reordered_missing_or_differently_typed_features():
    train, oot, test = matrices()
    names = ["source__n_records", "source__amount__mean__all"]
    controls = [c for c in oot.columns if c not in names]
    for bad in (
        oot.select(*controls, *reversed(names)),
        oot.drop("source__n_records"),
        oot.with_columns(pl.col("source__n_records").cast(pl.String)),
    ):
        with pytest.raises(ValueError, match="feature names, order or logical types"):
            validate_matrix_schemas(train, bad, test)


def test_controls_must_be_first_and_match_excluding_target():
    train, oot, test = matrices()
    with pytest.raises(ValueError, match="precede"):
        validate_matrix_schema(
            train.select(
                "case_id",
                "source__n_records",
                *(
                    c
                    for c in train.columns
                    if c not in ("case_id", "source__n_records")
                ),
            ),
            "train",
        )
    with pytest.raises(ValueError, match="control columns differ"):
        validate_matrix_schemas(train, oot, test.drop("MONTH"))
    with pytest.raises(ValueError, match="num_group1"):
        validate_matrix_schema(
            train.with_columns(pl.lit(0).alias("num_group1")), "train"
        )


def test_target_is_required_binary_for_train_and_oot_absent_in_test():
    train, oot, test = matrices()
    with pytest.raises(ValueError, match="target"):
        validate_matrix_schema(oot.drop("target"), "oot")
    with pytest.raises(ValueError, match="target"):
        validate_matrix_schema(test.with_columns(pl.lit(None).alias("target")), "test")
    for target in ([0, None], [0, 2]):
        with pytest.raises(ValueError, match="target"):
            validate_matrix_schema(
                train.with_columns(pl.Series("target", target)), "train"
            )


def test_invalid_base_and_feature_types_are_rejected_without_coercing():
    train, oot, test = matrices()
    with pytest.raises(ValueError, match="case_id"):
        validate_matrix_schema(
            train.with_columns(pl.Series("case_id", [1, 1])), "train"
        )
    with pytest.raises(ValueError, match="date_decision"):
        validate_matrix_schema(
            test.with_columns(pl.lit("not-a-date").alias("date_decision")), "test"
        )
    with pytest.raises(ValueError, match="WEEK_NUM"):
        validate_matrix_schema(
            test.with_columns(pl.lit(float("inf")).alias("WEEK_NUM")), "test"
        )
    with pytest.raises(ValueError, match="dtype"):
        validate_matrix_schema(
            train.with_columns(pl.lit([1, 2]).alias("source__n_records")), "train"
        )


def test_does_not_modify_input_frames():
    train, oot, test = matrices()
    originals = [df.clone() for df in (train, oot, test)]
    validate_matrix_schemas(train, oot, test)
    assert all(
        df.equals(original) for df, original in zip((train, oot, test), originals)
    )
