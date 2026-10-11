import polars as pl

from dm_credit.integrity import check_integrity


def make(**overrides) -> pl.DataFrame:
    data = {
        "case_id": [1, 2, 3, 4],
        "date_decision": ["2019-01-01", "2019-01-09", "2019-02-01", "2019-02-10"],
        "MONTH": [201901, 201901, 201902, 201902],
        "WEEK_NUM": [0, 1, 4, 5],
        "target": [0, 1, 0, 0],
    }
    data.update(overrides)
    return pl.DataFrame(data)


def finding(report, check: str) -> dict:
    return report.findings.filter(pl.col("check") == check).row(0, named=True)


def metric(report, name: str) -> int:
    return report.key_integrity.filter(pl.col("metric") == name)["value"][0]


def test_clean_table_does_not_block():
    r = check_integrity(make())
    assert (r.n_rows, r.n_cols) == (4, 5)
    assert r.memory_mb > 0
    assert not r.blocks_baseline
    assert metric(r, "null_keys") == 0 and metric(r, "repeated_keys") == 0
    assert finding(r, "month_mismatch")["count"] == 0
    assert finding(r, "week_num_mismatch")["count"] == 0
    assert r.repeated_sample.height == 0


def test_structure_reports_roles_and_kinds():
    s = {row["column"]: row for row in check_integrity(make()).structure.to_dicts()}
    assert s["case_id"]["role"] == "key" and s["target"]["role"] == "target" and s["WEEK_NUM"]["role"] == "week"
    assert s["case_id"]["apparent_kind"] == "numeric"
    assert s["date_decision"]["apparent_kind"] == "temporal"
    assert s["date_decision"]["dtype"] == "String"


def test_repeated_keys_counted_with_frequency_and_sample():
    r = check_integrity(pl.concat([make(), make().head(1), make().head(2)]))
    # claves 1 y 2 repetidas: 1 aparece 3 veces, 2 aparece 2 veces
    assert metric(r, "repeated_keys") == 2
    assert metric(r, "rows_in_repeated_keys") == 5
    assert dict(zip(r.key_frequency["times_repeated"], r.key_frequency["n_keys"])) == {2: 1, 3: 1}
    assert set(r.repeated_sample["case_id"]) == {1, 2}
    assert finding(r, "repeated_keys")["status"] == "blocking"
    assert r.blocks_baseline


def test_null_key_is_blocking():
    r = check_integrity(make(case_id=[1, 2, None, 4]))
    assert metric(r, "null_keys") == 1
    assert finding(r, "null_keys")["status"] == "blocking"
    assert r.blocks_baseline


def test_exact_duplicate_rows_detected():
    df = pl.concat([make(), make().head(1)])
    r = check_integrity(df)
    assert metric(r, "duplicate_rows") == 2  # ambas copias de la fila repetida
    assert finding(r, "duplicate_rows")["status"] == "observation"


def test_date_stored_as_string_is_non_blocking_observation():
    r = check_integrity(make())
    f = finding(r, "dtype_content_mismatch")
    assert f["status"] == "observation" and f["count"] == 1
    assert "date_decision" in f["detail"]
    assert not r.blocks_baseline


def test_number_stored_as_string_flagged():
    r = check_integrity(make(MONTH=["201901", "201901", "201902", "201902"]))
    detail = finding(r, "dtype_content_mismatch")["detail"]
    assert "MONTH: number stored as String" in detail


def test_fully_empty_column_flagged():
    df = make().with_columns(pl.lit(None, dtype=pl.Float64).alias("empty"))
    f = finding(check_integrity(df), "empty_columns")
    assert f["count"] == 1 and f["detail"] == "empty"


def test_column_name_issues_flagged():
    df = make().with_columns(pl.lit(1).alias(" pad"), pl.lit(1).alias("Pad"))
    f = finding(check_integrity(df), "column_name_issues")
    assert f["count"] == 2  # espacio sobrante y colisión por mayúsculas tras normalizar


def test_month_and_week_inconsistency_counted():
    r = check_integrity(make(MONTH=[201901, 201902, 201902, 201902], WEEK_NUM=[0, 1, 4, 9]))
    assert finding(r, "month_mismatch")["count"] == 1
    assert finding(r, "week_num_mismatch")["count"] == 1


def test_target_outside_0_1_is_blocking():
    r = check_integrity(make(target=[0, 1, 2, 0]))
    f = finding(r, "target_outside_0_1")
    assert f["count"] == 1 and f["status"] == "blocking"
    assert r.blocks_baseline


def test_missing_required_column_is_blocking():
    r = check_integrity(make().drop("WEEK_NUM"))
    assert finding(r, "missing_column:WEEK_NUM")["status"] == "blocking"
    assert r.blocks_baseline
    r = check_integrity(make().drop("case_id"))  # sin clave no se evalúa su integridad
    assert finding(r, "missing_column:case_id")["status"] == "blocking"
    assert r.key_integrity.filter(pl.col("metric") == "null_keys").height == 0
