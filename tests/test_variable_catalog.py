from pathlib import Path

import polars as pl

from dm_credit.variable_catalog import build_catalog, summarize_catalog


def test_catalog_joins_definitions_and_classifies_columns(tmp_path: Path):
    profile = tmp_path / "profile_train_example"
    profile.mkdir()
    pl.DataFrame(
        {
            "column": ["case_id", "approvaldate_1D", "amount_1A", "unknown_1L"],
            "dtype": ["Int64", "String", "Float64", "Float64"],
            "nulls": [0, 1, 0, 0],
            "pct_null": [0.0, 25.0, 0.0, 0.0],
            "n_unique_approx": [4, 3, 4, 1],
        }
    ).write_csv(profile / "columns.csv")
    definitions = pl.DataFrame(
        {
            "Variable": ["approvaldate_1D", "amount_1A"],
            "Description": ["Approval date.", "Credit amount."],
        }
    )

    catalog = build_catalog(definitions, [profile / "columns.csv"])
    by_name = {row["column"]: row for row in catalog.iter_rows(named=True)}

    assert catalog["family"].unique().to_list() == ["example"]
    assert by_name["case_id"]["identifier_risk"] == "identificador técnico"
    assert by_name["approvaldate_1D"]["function"] == "temporal"
    assert (
        by_name["approvaldate_1D"]["leakage_review"]
        == "revisar fecha contra date_decision"
    )
    assert by_name["amount_1A"]["function"] == "importe/capacidad financiera"
    assert by_name["unknown_1L"]["definition_status"] == "not_in_feature_definitions"


def test_summary_counts_definition_coverage(tmp_path: Path):
    profile = tmp_path / "profile_train_example"
    profile.mkdir()
    pl.DataFrame(
        {
            "column": ["case_id", "status_1M"],
            "dtype": ["Int64", "String"],
            "pct_null": [0.0, 0.0],
            "n_unique": [2, 2],
        }
    ).write_csv(profile / "columns.csv")
    definitions = pl.DataFrame(
        {"Variable": ["status_1M"], "Description": ["Contract status."]}
    )

    summary = summarize_catalog(build_catalog(definitions, [profile / "columns.csv"]))
    row = summary.row(0, named=True)

    assert row["columns"] == 2
    assert row["with_official_definition"] == 1
    assert row["without_official_definition"] == 1
    assert row["availability_review"] == 1


def test_inventory_adds_unprofiled_family_without_misclassifying_t_suffix(
    tmp_path: Path,
):
    definitions = pl.DataFrame(
        {
            "Variable": ["relatedpersons_role_762T"],
            "Description": ["Relationship type of a related person."],
        }
    )
    inventory = pl.DataFrame(
        {
            "split": ["train"],
            "file": ["train_credit_bureau_a_2_0.parquet"],
            "types": ["case_id:int64; relatedpersons_role_762T:string"],
        }
    )

    catalog = build_catalog(definitions, [], inventory)
    by_name = {row["column"]: row for row in catalog.iter_rows(named=True)}

    assert catalog["family"].unique().to_list() == ["credit_bureau_a_2"]
    assert by_name["case_id"]["profile_source"] == "inventory_schema"
    assert by_name["relatedpersons_role_762T"]["function"] == "persona/demografía"
    assert (
        by_name["relatedpersons_role_762T"]["temporal_relevance"]
        == "sin referencia temporal explícita"
    )
