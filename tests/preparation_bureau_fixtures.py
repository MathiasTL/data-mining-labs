"""Synthetic bureau fixtures (C4/C5, E3) with hand-computed expectations. Never read real bureau files.

Five base cases, all decided on 2020-06-01. ``WEEK_NUM`` = 10, 10, 11, 11, 12, so with ``CUTOFF = 11``
train = cases 1 and 2 and OOT = cases 3 to 5. Test files reuse the rows with ``case_id + 100``.

credit_bureau_a_1 (anchors lastupdate_1112D active / lastupdate_388D closed):
    (1,0) active anchor before -> valid      (1,1) closed anchor equal -> valid
    (1,2) active anchor after -> dropped     (1,3) both anchors null -> dropped
    (2,0) valid, start date after the decision -> value nulled and counted
    (4,0) one anchor before, one after -> dropped (conservative row rule)
    (5,0) valid                              (99,0) not in the base -> ignored
credit_bureau_a_2 (monthly anchors; skipped with the default selection):
    (1,0,0) May 2020 -> valid   (1,0,1) June 2020 = decision month -> dropped   (1,1,0) closed Jan 2020 -> valid
    (1,2,0) parent (1,2) invalid -> dropped   (1,0,2) no anchor -> dropped   (2,0,0) Dec 2019 -> valid
credit_bureau_b_1 (anchor lastupdate_260D):
    (1,0) valid  (1,1) after -> dropped  (2,0) null -> dropped  (2,1) valid  (2,2) equal -> valid
credit_bureau_b_2 (anchor pmts_date_1107D, parent in b_1):
    (1,0,0) valid  (1,0,1) after  (1,1,0) invalid parent  (2,0,0) invalid parent  (2,1,0) null date
    (2,1,1) valid  (2,2,0) equal -> valid  (3,0,0) parent missing -> invalid parent
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

DECISION = "2020-06-01"
CASE_IDS = [1, 2, 3, 4, 5]
CUTOFF = 11
TEST_OFFSET = 100

# approved vocabulary for the categorical candidate (G1: shares only for approved categories)
CATEGORY_SHARES = {"credit_bureau_a_1": {"classificationofcontr_13M": ("a", "b")}}

SELECTION_ROWS = [
    # family, column, dtype, action
    ("credit_bureau_a_1", "case_id", "Int64", "excluir"),
    ("credit_bureau_a_1", "num_group1", "Int64", "excluir"),
    ("credit_bureau_a_1", "lastupdate_1112D", "String", "referencia_as_of"),
    ("credit_bureau_a_1", "lastupdate_388D", "String", "referencia_as_of"),
    ("credit_bureau_a_1", "credlmt_935A", "Float64", "candidato_G2"),
    ("credit_bureau_a_1", "dateofcredstart_739D", "String", "transformar_as_of"),
    ("credit_bureau_a_1", "classificationofcontr_13M", "String", "candidato_G2"),
    ("credit_bureau_a_1", "outstandingamount_362A", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_a_1", "financialinstitution_382M", "String", "excluir"),
    ("credit_bureau_a_2", "case_id", "Int64", "excluir"),
    ("credit_bureau_a_2", "num_group1", "Int64", "excluir"),
    ("credit_bureau_a_2", "num_group2", "Int64", "excluir"),
    ("credit_bureau_a_2", "pmts_year_1139T", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_a_2", "pmts_month_158T", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_a_2", "pmts_year_507T", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_a_2", "pmts_month_706T", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_a_2", "pmts_overdue_1140A", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_b_1", "case_id", "Int64", "excluir"),
    ("credit_bureau_b_1", "num_group1", "Int64", "excluir"),
    ("credit_bureau_b_1", "lastupdate_260D", "String", "referencia_as_of"),
    ("credit_bureau_b_1", "amount_1115A", "Float64", "candidato_G2"),
    ("credit_bureau_b_1", "contractdate_551D", "String", "transformar_as_of"),
    ("credit_bureau_b_1", "isbidproduct_1L", "Boolean", "candidato_G2"),
    ("credit_bureau_b_1", "dpdmax_851P", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_b_1", "credor_3940957M", "String", "excluir"),
    ("credit_bureau_b_2", "case_id", "Int64", "excluir"),
    ("credit_bureau_b_2", "num_group1", "Int64", "excluir"),
    ("credit_bureau_b_2", "num_group2", "Int64", "excluir"),
    ("credit_bureau_b_2", "pmts_date_1107D", "String", "referencia_as_of"),
    ("credit_bureau_b_2", "pmts_dpdvalue_108P", "Float64", "bloquear_hasta_G2"),
    ("credit_bureau_b_2", "pmts_pmtsoverdue_635A", "Float64", "bloquear_hasta_G2"),
]
BLOCKED_OR_EXCLUDED = {c for _, c, _, a in SELECTION_ROWS if a in ("bloquear_hasta_G2", "excluir")} - {"case_id", "num_group1", "num_group2"}
A2_ANCHORS = ("pmts_year_1139T", "pmts_month_158T", "pmts_year_507T", "pmts_month_706T")


def selection(*, approve_a2: bool = False) -> pl.DataFrame:
    """Small B5-like selection. ``approve_a2`` simulates a G2 approval of the a_2 anchors and one amount."""
    rows = []
    for family, column, dtype, action in SELECTION_ROWS:
        if approve_a2 and column in A2_ANCHORS:
            action = "referencia_as_of"
        if approve_a2 and column == "pmts_overdue_1140A":
            action = "candidato_G2"
        rows.append((family, column, dtype, action, "", "", "", ""))
    return pl.DataFrame(
        rows, schema=["family", "column", "dtype", "action", "reason", "temporal_rule", "evidence", "note"], orient="row"
    )


def write_selection(path: Path, *, approve_a2: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    selection(approve_a2=approve_a2).write_csv(path)
    return path


def base() -> pl.DataFrame:
    return pl.DataFrame(
        {"case_id": CASE_IDS, "date_decision": [DECISION] * 5, "MONTH": [202006] * 5, "WEEK_NUM": [10, 10, 11, 11, 12], "target": [0, 1, 0, 0, 1]}
    )


def credit_bureau_a_1() -> pl.DataFrame:
    rows = [
        (1, 0, "2020-05-01", None, 100.0, "2019-01-01", 5.0, "X", "a"),
        (1, 1, None, "2020-06-01", 300.0, "2018-01-01", 7.0, "Y", "b"),
        (1, 2, "2020-07-01", None, 1000.0, "2017-01-01", 9.0, "Z", "a"),
        (1, 3, None, None, 5000.0, "2016-01-01", 1.0, "X", "a"),
        (2, 0, "2020-01-01", None, None, "2020-08-01", 2.0, "X", "a"),
        (4, 0, "2020-05-01", "2020-07-01", 70.0, "2019-06-01", 3.0, "Y", "b"),
        (5, 0, "2020-05-01", None, 50.0, "2019-06-01", 4.0, "X", "c"),
        (99, 0, "2020-05-01", None, 1.0, "2019-06-01", 1.0, "X", "a"),
    ]
    schema = {
        "case_id": pl.Int64, "num_group1": pl.Int64, "lastupdate_1112D": pl.String, "lastupdate_388D": pl.String,
        "credlmt_935A": pl.Float64, "dateofcredstart_739D": pl.String, "outstandingamount_362A": pl.Float64,
        "financialinstitution_382M": pl.String, "classificationofcontr_13M": pl.String,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def credit_bureau_a_2() -> pl.DataFrame:
    rows = [
        (1, 0, 0, 2020.0, 5.0, None, None, 10.0),
        (1, 0, 1, 2020.0, 6.0, None, None, 20.0),
        (1, 0, 2, None, None, None, None, 50.0),
        (1, 1, 0, None, None, 2020.0, 1.0, 30.0),
        (1, 2, 0, 2020.0, 1.0, None, None, 40.0),
        (2, 0, 0, 2019.0, 12.0, None, None, 60.0),
    ]
    schema = {
        "case_id": pl.Int64, "num_group1": pl.Int64, "num_group2": pl.Int64, "pmts_year_1139T": pl.Float64,
        "pmts_month_158T": pl.Float64, "pmts_year_507T": pl.Float64, "pmts_month_706T": pl.Float64, "pmts_overdue_1140A": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def credit_bureau_b_1() -> pl.DataFrame:
    rows = [
        (1, 0, "2020-05-01", 10.0, "2020-01-01", True, 3.0, "C"),
        (1, 1, "2020-06-10", 99.0, "2020-02-01", False, 4.0, "C"),
        (2, 0, None, 5.0, "2019-01-01", True, 0.0, "D"),
        (2, 1, "2020-04-01", 20.0, "2019-02-01", False, 1.0, "D"),
        (2, 2, "2020-06-01", 40.0, None, True, 1.0, "D"),
    ]
    schema = {
        "case_id": pl.Int64, "num_group1": pl.Int64, "lastupdate_260D": pl.String, "amount_1115A": pl.Float64,
        "contractdate_551D": pl.String, "isbidproduct_1L": pl.Boolean, "dpdmax_851P": pl.Float64, "credor_3940957M": pl.String,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def credit_bureau_b_2() -> pl.DataFrame:
    rows = [
        (1, 0, 0, "2020-03-01", 0.0, 0.0),
        (1, 0, 1, "2020-06-15", 0.0, 0.0),
        (1, 1, 0, "2020-03-01", 0.0, 0.0),
        (2, 0, 0, "2020-03-01", 0.0, 0.0),
        (2, 1, 0, None, 0.0, 0.0),
        (2, 1, 1, "2020-05-01", 3.0, 1.0),
        (2, 2, 0, "2020-06-01", 0.0, 0.0),
        (3, 0, 0, "2020-01-01", 0.0, 0.0),
    ]
    schema = {
        "case_id": pl.Int64, "num_group1": pl.Int64, "num_group2": pl.Int64, "pmts_date_1107D": pl.String,
        "pmts_dpdvalue_108P": pl.Float64, "pmts_pmtsoverdue_635A": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


TABLES = {
    "credit_bureau_a_1": credit_bureau_a_1,
    "credit_bureau_a_2": credit_bureau_a_2,
    "credit_bureau_b_1": credit_bureau_b_1,
    "credit_bureau_b_2": credit_bureau_b_2,
}

# hand-computed expectations (decision 2020-06-01)
EXPECTED_A1 = {
    "credit_bureau_a_1__n_records": [2, 1, 0, 0, 1],
    "credit_bureau_a_1__source_present": [True, True, False, False, True],
    "credit_bureau_a_1__credlmt_935a__count__all": [2, 0, 0, 0, 1],
    "credit_bureau_a_1__credlmt_935a__min__all": [100.0, None, None, None, 50.0],
    "credit_bureau_a_1__credlmt_935a__max__all": [300.0, None, None, None, 50.0],
    "credit_bureau_a_1__credlmt_935a__mean__all": [200.0, None, None, None, 50.0],
    "credit_bureau_a_1__credlmt_935a__sum__all": [400.0, None, None, None, 50.0],
    "credit_bureau_a_1__credlmt_935a__std__all": [20000.0**0.5, None, None, None, None],
    "credit_bureau_a_1__dateofcredstart_739d__days_since_last__all": [517, None, None, None, 366],
    "credit_bureau_a_1__dateofcredstart_739d__days_since_first__all": [882, None, None, None, 366],
    "credit_bureau_a_1__classificationofcontr_13m__share_a__all": [0.5, 1.0, None, None, 0.0],
    "credit_bureau_a_1__classificationofcontr_13m__share_b__all": [0.5, 0.0, None, None, 0.0],
}
EXPECTED_A1_EXCLUSIONS = {
    "credit_bureau_a_1_rows_null_anchor": 1,
    "credit_bureau_a_1_rows_after_decision": 2,
    "credit_bureau_a_1_dateofcredstart_739D_values_after_decision": 1,
}
EXPECTED_A2_APPROVED = {
    "credit_bureau_a_2__n_records": [2, 1, 0, 0, 0],
    "credit_bureau_a_2__pmts_overdue_1140a__sum__all": [40.0, 60.0, None, None, None],
}
EXPECTED_A2_EXCLUSIONS = {
    "credit_bureau_a_2_rows_null_anchor": 1,
    "credit_bureau_a_2_rows_after_decision": 1,
    "credit_bureau_a_2_rows_invalid_parent": 1,
}
EXPECTED_B = {
    "credit_bureau_b_1__n_records": [1, 2, 0, 0, 0],
    "credit_bureau_b_1__amount_1115a__count__all": [1, 2, 0, 0, 0],
    "credit_bureau_b_1__amount_1115a__mean__all": [10.0, 30.0, None, None, None],
    "credit_bureau_b_1__amount_1115a__std__all": [None, 200.0**0.5, None, None, None],
    "credit_bureau_b_1__contractdate_551d__days_since_last__all": [152, 486, None, None, None],
    "credit_bureau_b_1__isbidproduct_1l__share_true__all": [1.0, 0.5, None, None, None],
    "credit_bureau_b_2__n_records": [1, 2, 0, 0, 0],
    "credit_bureau_b_2__source_present": [True, True, False, False, False],
}
EXPECTED_B_EXCLUSIONS = {
    "credit_bureau_b_1_rows_null_anchor": 1,
    "credit_bureau_b_1_rows_after_decision": 1,
    "credit_bureau_b_1_contractdate_551D_values_after_decision": 0,
    "credit_bureau_b_2_rows_null_anchor": 1,
    "credit_bureau_b_2_rows_after_decision": 1,
    "credit_bureau_b_2_rows_invalid_parent": 3,
}
