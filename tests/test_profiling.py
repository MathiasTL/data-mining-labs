import polars as pl

from dm_credit.profiling import profile_dataframe


def make():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 3],
            "num": [1.0, None, -2.0, 0.0],
            "cat": ["a", "a", "b", None],
            "const": [7, 7, 7, 7],
        }
    )


def test_summary_counts():
    p = profile_dataframe(make())
    assert (p.n_rows, p.n_cols) == (4, 4)
    assert p.n_duplicate_rows == 0
    assert p.memory_mb > 0


def test_duplicates_counted():
    df = pl.concat([make(), make().head(1)])
    assert profile_dataframe(df).n_duplicate_rows == 2  # both copies of the repeated row


def test_columns_table():
    cols = profile_dataframe(make()).columns.to_dicts()
    by_name = {c["column"]: c for c in cols}
    assert by_name["num"]["nulls"] == 1
    assert by_name["num"]["pct_null"] == 25.0
    assert by_name["cat"]["n_unique"] == 2  # nulls excluded
    assert by_name["const"]["n_unique"] == 1


def test_numeric_stats_include_negatives_and_zeros():
    num = profile_dataframe(make()).numeric.filter(pl.col("column") == "num").row(0, named=True)
    assert num["min"] == -2.0 and num["max"] == 1.0
    assert num["n_negative"] == 1 and num["n_zero"] == 1
    assert {"mean", "median", "p01", "p25", "p75", "p99"} <= set(num)


def test_categorical_top_values():
    cat = profile_dataframe(make(), top_k=1).categorical
    row = cat.filter(pl.col("column") == "cat").row(0, named=True)
    assert row["top_value"] == "a" and row["top_count"] == 2
    assert row["pct_dominant"] == 50.0


def test_all_null_column_does_not_crash():
    df = pl.DataFrame({"x": [None, None], "y": [1, 2]}, schema={"x": pl.Float64, "y": pl.Int64})
    p = profile_dataframe(df)
    assert p.columns.filter(pl.col("column") == "x")["pct_null"][0] == 100.0
