import polars as pl
import pytest

from dm_credit.target_analysis import analyze_target_timing, save_report


def test_validation_report_counts_target_and_weeks():
    df = pl.DataFrame(
        {
            "WEEK_NUM": [0, 0, 1, 1, 2, 2],
            "target": [0, 1, 0, 0, 1, 1],
        }
    )

    report = analyze_target_timing(df)

    target = {row["target"]: row for row in report.target_distribution.to_dicts()}
    assert target["0"] == {"target": "0", "cases": 3, "pct": 50.0}
    assert target["1"] == {"target": "1", "cases": 3, "pct": 50.0}
    weekly = report.weekly.to_dicts()
    assert [row["cases"] for row in weekly] == [2, 2, 2]
    assert [row["positives"] for row in weekly] == [1, 0, 2]
    assert [row["default_rate_pct"] for row in weekly] == [50.0, 0.0, 100.0]
    assert report.cutoffs.select("cutoff_week").to_series().to_list() == [1, 2]
    assert report.cutoffs.row(0, named=True)["train_positives"] == 1
    assert report.cutoffs.row(0, named=True)["validation_positives"] == 2


def test_validation_report_surfaces_missing_and_unexpected_values():
    df = pl.DataFrame(
        {
            "WEEK_NUM": [0, 1, None],
            "target": [0, None, 2],
        },
        schema={"WEEK_NUM": pl.Int64, "target": pl.Int64},
    )

    report = analyze_target_timing(df)
    summary = dict(zip(report.summary["metric"], report.summary["value"]))
    target = {row["target"]: row["cases"] for row in report.target_distribution.to_dicts()}

    assert summary["target_missing"] == 1
    assert summary["target_unexpected"] == 1
    assert summary["week_missing"] == 1
    assert target["(null)"] == 1
    assert target["2"] == 1
    assert report.weekly.filter(pl.col("WEEK_NUM").is_null())["cases"][0] == 1


def test_validation_report_requires_columns_and_numeric_values():
    with pytest.raises(ValueError, match="missing required columns"):
        analyze_target_timing(pl.DataFrame({"target": [0]}))
    with pytest.raises(ValueError, match="target must be numeric"):
        analyze_target_timing(pl.DataFrame({"target": ["0"], "WEEK_NUM": [0]}))


def test_save_report_writes_all_csvs(tmp_path):
    report = analyze_target_timing(pl.DataFrame({"WEEK_NUM": [0, 1], "target": [0, 1]}))
    save_report(report, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "summary.csv",
        "target_distribution.csv",
        "weekly.csv",
        "cutoff_candidates.csv",
    }
