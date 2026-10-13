from pathlib import Path

import polars as pl
import pytest

from dm_credit.preparation import depth0_selection as b3

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalogs():
    catalog = pl.read_csv(ROOT / "results" / "variable_catalog.csv")
    temporal_review = pl.read_csv(
        ROOT / "results" / "integration_analysis" / "temporal_leakage_review.csv"
    )
    return catalog, temporal_review


def test_real_catalog_has_one_decision_for_every_depth0_column(catalogs):
    catalog, temporal_review = catalogs
    result = b3.select_depth0_columns(catalog, temporal_review)

    assert result.height == 226
    assert result.select(["family", "column"]).is_duplicated().sum() == 0
    assert set(result["family"].unique()) == set(b3.FAMILIES)
    assert set(result["proposed_action"].unique()) <= {
        "control_only",
        "exclude_predictor",
        "manual_review",
        "provisional_candidate",
    }


def test_base_fields_are_controls_not_predictors(catalogs):
    catalog, temporal_review = catalogs
    result = b3.select_depth0_columns(catalog, temporal_review)
    base = result.filter(pl.col("family") == "base")

    assert base.height == 5
    assert set(base["proposed_action"]) == {"control_only"}


def test_quasi_identifiers_and_future_default_score_are_excluded(catalogs):
    catalog, temporal_review = catalogs
    result = b3.select_depth0_columns(catalog, temporal_review)

    assert (
        result.filter(
            (pl.col("family") == "static_cb_0")
            & (pl.col("column") == "riskassesment_302T")
        )["proposed_action"][0]
        == "exclude_predictor"
    )
    assert (
        result.filter(
            (pl.col("family") == "static_0")
            & (pl.col("column") == "applicationcnt_361L")
        )["proposed_action"][0]
        == "exclude_predictor"
    )


def test_unverified_temporal_columns_are_not_provisional_candidates(catalogs):
    catalog, temporal_review = catalogs
    result = b3.select_depth0_columns(catalog, temporal_review)
    response_date = result.filter(
        (pl.col("family") == "static_cb_0")
        & (pl.col("column") == "responsedate_1012D")
    )

    assert response_date["proposed_action"][0] == "manual_review"
    assert "date_decision" in response_date["temporal_rule"][0]


def test_as_of_review_of_numeric_feature_does_not_claim_it_is_a_date(catalogs):
    catalog, temporal_review = catalogs
    result = b3.select_depth0_columns(catalog, temporal_review)
    feature = result.filter(
        (pl.col("family") == "static_0")
        & (pl.col("column") == "annuity_780A")
    )

    assert feature["proposed_action"][0] == "manual_review"
    assert "confirmar disponibilidad" in feature["temporal_rule"][0]
    assert "derivar" not in feature["temporal_rule"][0]


def test_unknown_families_in_catalog_are_ignored_and_temporal_review_is_required(catalogs):
    catalog, temporal_review = catalogs
    without_base_review = temporal_review.filter(pl.col("family") != "base")

    with pytest.raises(ValueError, match="missing depth-zero families"):
        b3.select_depth0_columns(catalog, without_base_review)


def test_missing_required_catalog_column_fails_explicitly(catalogs):
    catalog, temporal_review = catalogs

    with pytest.raises(ValueError, match="missing required columns"):
        b3.select_depth0_columns(catalog.drop("description"), temporal_review)


def test_selection_is_deterministic(catalogs):
    catalog, temporal_review = catalogs
    first = b3.select_depth0_columns(catalog, temporal_review)
    second = b3.select_depth0_columns(catalog, temporal_review)

    assert first.equals(second)
