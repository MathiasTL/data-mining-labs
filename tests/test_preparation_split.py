from dataclasses import FrozenInstanceError

import polars as pl
import pytest

from dm_credit.preparation.split import partition_base


def bases():
    train = pl.DataFrame(
        {
            "case_id": [10, 11, 12, 13],
            "WEEK_NUM": [74, 73, 75, 72],
            "date_decision": ["2020-01-04", "2020-01-03", "2020-01-05", "2020-01-02"],
            "target": [0, 1, 1, 0],
            "MONTH": [202001] * 4,
        }
    )
    test = pl.DataFrame(
        {
            "case_id": [20, 21],
            "WEEK_NUM": [73, 74],
            "date_decision": ["2020-01-03", "2020-01-04"],
            "MONTH": [202001, 202001],
        }
    )
    return train, test


def test_week_73_74_is_disjoint_exhaustive_and_test_stays_separate():
    train_base, test_base = bases()
    result = partition_base(train_base, test_base)

    assert result.cutoff_week == 74
    assert result.train["case_id"].to_list() == [11, 13]
    assert result.oot["case_id"].to_list() == [10, 12]
    assert result.test.equals(test_base)
    ids = [
        set(part["case_id"].to_list())
        for part in (result.train, result.oot, result.test)
    ]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert ids[0] | ids[1] == set(train_base["case_id"].to_list())
    assert sum(part.height for part in (result.train, result.oot)) == train_base.height
    assert result.train.schema == result.oot.schema == train_base.schema
    assert result.test.schema == test_base.schema
    assert "target" not in result.test.columns


def test_partition_does_not_modify_its_inputs():
    train, test = bases()
    original_train, original_test = train.clone(), test.clone()

    partition_base(train, test)

    assert train.equals(original_train) and test.equals(original_test)


def test_returned_frames_cannot_change_saved_partitions():
    train, test = bases()
    result = partition_base(train, test)
    original = (result.train, result.oot, result.test)
    for part in (result.train, result.oot, result.test):
        part[0, "case_id"] = 999
    assert all(
        current.equals(expected)
        for current, expected in zip((result.train, result.oot, result.test), original)
    )
    with pytest.raises(FrozenInstanceError):
        result.cutoff_week = 75


def test_cutoff_is_explicit_and_empty_partition_is_rejected():
    train, test = bases()
    result = partition_base(train, test, cutoff_week=73)
    assert result.train["WEEK_NUM"].to_list() == [72]
    assert result.oot["WEEK_NUM"].to_list() == [74, 73, 75]
    with pytest.raises(ValueError, match="empty partition"):
        partition_base(train, test, cutoff_week=72)


@pytest.mark.parametrize("week", [None, float("nan"), float("inf"), 73.5])
def test_invalid_week_cannot_disappear_during_filtering(week):
    train, test = bases()
    bad = train.with_columns(pl.Series("WEEK_NUM", [72.0, week, 74.0, 75.0]))
    with pytest.raises(ValueError, match="WEEK_NUM"):
        partition_base(bad, test)


def test_invalid_test_week_is_rejected():
    train, test = bases()
    bad_test = test.with_columns(pl.Series("WEEK_NUM", [73.0, float("inf")]))
    with pytest.raises(ValueError, match="WEEK_NUM"):
        partition_base(train, bad_test)


@pytest.mark.parametrize("bad_date", [None, "not-a-date", "2020-02-30"])
@pytest.mark.parametrize("physical_split", ["train", "test"])
def test_invalid_decision_date_is_rejected_before_split(bad_date, physical_split):
    train, test = bases()
    base = train if physical_split == "train" else test
    dates = base["date_decision"].to_list()
    dates[0] = bad_date
    bad = base.with_columns(pl.Series("date_decision", dates))
    with pytest.raises(ValueError, match="date_decision"):
        partition_base(
            bad if physical_split == "train" else train,
            bad if physical_split == "test" else test,
        )


def test_rejects_repeated_keys_and_test_overlap():
    train, test = bases()
    with pytest.raises(ValueError, match="case_id"):
        partition_base(pl.concat([train, train.head(1)]), test)
    with pytest.raises(ValueError, match="overlapping case_id"):
        partition_base(train, test.with_columns(pl.Series("case_id", [10, 21])))


def test_nan_key_is_rejected_on_physical_train_before_filtering():
    train, test = bases()
    bad = train.with_columns(pl.Series("case_id", [10.0, float("nan"), 12.0, 13.0]))
    with pytest.raises(ValueError, match="case_id"):
        partition_base(bad, test)


def test_test_cannot_have_target_and_train_needs_date():
    train, test = bases()
    with pytest.raises(ValueError, match="target"):
        partition_base(train, test.with_columns(pl.lit(None).alias("target")))
    with pytest.raises(ValueError, match="date_decision"):
        partition_base(train.drop("date_decision"), test)
