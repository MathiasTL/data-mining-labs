import polars as pl
import pytest

from dm_credit.integration_analysis import (
    aggregate_one_row_per_case,
    analyze_frame,
    cartesian_example,
    safe_left_join,
    summarize_family,
    temporal_review,
)


def frame(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["case_id", "num_group1", "num_group2"], orient="row"
    )


def test_one_to_one_and_optional_one_cardinalities():
    base_ids = {1, 2, 3}
    one = analyze_frame("one", frame([(1, 0, 0), (2, 0, 0), (3, 0, 0)]), 0)
    optional = analyze_frame("optional", frame([(1, 0, 0), (3, 0, 0)]), 0)

    one_summary, _ = summarize_family(one, base_ids)
    optional_summary, _ = summarize_family(optional, base_ids)

    assert one_summary["observed_cardinality"] == "1:1"
    assert one_summary["base_coverage_pct"] == 100.0
    assert optional_summary["observed_cardinality"] == "1:0..1"
    assert optional_summary["base_coverage_pct"] == pytest.approx(66.6667)


def test_one_to_many_expansion_and_orphans():
    result = analyze_frame(
        "many", frame([(1, 0, 0), (1, 1, 0), (2, 0, 0), (99, 0, 0)]), 1
    )

    summary, risk = summarize_family(result, {1, 2, 3})

    assert summary["observed_cardinality"] == "1:N"
    assert summary["orphan_rows"] == 1
    assert summary["candidate_key_unique"] is True
    assert risk["direct_left_join_rows"] == 4
    assert risk["mean_expansion_all_base"] == pytest.approx(4 / 3, abs=1e-4)
    assert risk["max_expansion"] == 2


def test_depth_two_parent_relation_and_final_key():
    result = analyze_frame(
        "depth2", frame([(1, 0, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]), 2
    )

    assert result.parent_groups == 3
    assert result.parent_key_unique is False
    assert result.key_unique is True


def test_repeated_composite_key_is_detected():
    result = analyze_frame("duplicate", frame([(1, 0, 0), (1, 0, 0)]), 2)

    assert result.key_unique is False


def test_cartesian_risk_example_is_deterministic():
    example = cartesian_example()

    assert example["direct_join_rows"].to_list() == [6, 6]
    assert example["direct_join_rows"].sum() == 12


def test_aggregation_returns_one_row_per_case():
    source = pl.DataFrame(
        {"case_id": [1, 1, 2], "num_group1": [0, 1, 0], "amount": [10.0, None, 3.0]}
    )

    aggregate = aggregate_one_row_per_case(source)

    assert aggregate.height == 2
    assert aggregate["case_id"].n_unique() == 2
    assert aggregate.filter(pl.col("case_id") == 1)["record_count"].item() == 2
    assert aggregate.filter(pl.col("case_id") == 1)["amount__null_count"].item() == 1


def test_safe_join_rejects_duplicate_right_key_and_preserves_base():
    base = pl.DataFrame({"case_id": [1, 2], "target": [0, 1]})
    duplicate = pl.DataFrame({"case_id": [1, 1], "count": [1, 2]})

    with pytest.raises(ValueError, match="one non-null row"):
        safe_left_join(base, duplicate)

    joined = safe_left_join(base, pl.DataFrame({"case_id": [1], "count": [2]}))
    assert joined.height == base.height
    assert joined["case_id"].to_list() == base["case_id"].to_list()


def test_temporal_review_requires_as_of_and_excludes_unresolved_states():
    catalog = pl.DataFrame(
        {
            "family": ["history", "history", "history"],
            "column": ["eventdate_1D", "status_1M", "case_id"],
            "temporal_relevance": [
                "referencia temporal explícita",
                "sin referencia temporal explícita",
                "no aplica",
            ],
            "leakage_review": [
                "revisar fecha contra date_decision",
                "revisar disponibilidad al decidir",
                "excluir de predictores",
            ],
            "identifier_risk": ["no señalado", "no señalado", "identificador técnico"],
        }
    )

    review = temporal_review(catalog).row(0, named=True)

    assert review["classification"] == "historical_requires_as_of_cutoff"
    assert "<= date_decision" in review["pre_aggregation_rule"]
    assert "status_1M" in review["state_or_outcome_review_columns"]
    assert "case_id" in review["identifier_or_quasi_identifier_columns"]
