import polars as pl
import pytest

from dm_credit import baseline


def test_temporal_partition_at_cutoff_is_disjoint_and_exhaustive(base_df):
    train, oot = baseline.oot_split(base_df, cutoff_week=15)

    assert train["WEEK_NUM"].max() < 15
    assert oot["WEEK_NUM"].min() >= 15
    assert set(train["case_id"].to_list()).isdisjoint(oot["case_id"].to_list())
    assert set(train["case_id"].to_list()) | set(oot["case_id"].to_list()) == set(
        base_df["case_id"].to_list()
    )


@pytest.mark.parametrize(
    "bad_key", [pl.Series("case_id", [1, None, 3]), pl.Series("case_id", [1, 1, 3])]
)
def test_preparation_contract_rejects_null_or_repeated_case_id(base_df, bad_key):
    malformed = base_df.head(3).with_columns(bad_key)

    with pytest.raises(ValueError, match="case_id"):
        baseline.validate_contract(malformed)


def test_target_is_required_for_train_and_oot_but_absent_from_test(base_df):
    baseline.validate_contract(base_df, require_target=True)
    test = base_df.drop("target")
    baseline.validate_contract(test, require_target=False)


def test_reserved_columns_never_become_predictors(base_df):
    with_controls = base_df.with_columns(
        pl.lit("2019-01-01").alias("date_decision"),
        pl.lit(201901).alias("MONTH"),
        pl.lit(0).alias("num_group1"),
        pl.lit(0).alias("num_group2"),
    )

    assert baseline.default_feature_columns(with_controls) == [
        "x_signal",
        "x_noise",
        "x_cat",
    ]
    for reserved in baseline.RESERVED:
        if reserved in with_controls.columns:
            assert reserved not in baseline.default_feature_columns(with_controls)
