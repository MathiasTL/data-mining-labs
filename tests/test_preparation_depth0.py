import json
from datetime import date

import polars as pl
import pytest

from dm_credit.preparation.families import depth0


@pytest.fixture
def static_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for split in ("train", "test"):
        (root / "parquet_files" / split).mkdir(parents=True)
    monkeypatch.setenv("DM_DATA_DIR", str(root))

    train = root / "parquet_files" / "train"
    pl.DataFrame(
        {"case_id": [1], "static_value": [10.0], "static_flag": [True]}
    ).write_parquet(train / "train_static_0_0.parquet")
    pl.DataFrame({"case_id": [1], "static_cb_value": [5]}).write_parquet(
        train / "train_static_cb_0.parquet"
    )

    test = root / "parquet_files" / "test"
    pl.DataFrame(
        {"case_id": [10], "static_value": [30.0], "static_flag": [None]}
    ).write_parquet(test / "test_static_0.parquet")
    pl.DataFrame({"case_id": [10], "static_cb_value": [7]}).write_parquet(
        test / "test_static_cb_0.parquet"
    )
    return root


@pytest.fixture
def b3_selection(tmp_path):
    selection = pl.DataFrame(
        {
            "family": ["static_0", "static_cb_0", "static_0", "static_0"],
            "column": ["static_value", "static_cb_value", "static_flag", "not_used"],
            "proposed_action": [
                "provisional_candidate",
                "provisional_candidate",
                "provisional_candidate",
                "manual_review",
            ],
            "description": [
                "approved test value",
                "approved test value",
                "approved test value",
                "review",
            ],
            "reason": ["", "", "", ""],
            "temporal_rule": ["all", "all", "all", "review"],
            "evidence": ["fixture", "fixture", "fixture", "fixture"],
        }
    )
    path = tmp_path / "B3_selection.csv"
    selection.write_csv(path)
    return path


@pytest.fixture
def g2_decision(tmp_path):
    path = tmp_path / "G2_depth0_columns.csv"
    pl.DataFrame(
        {
            "family": ["static_0", "static_0", "static_cb_0"],
            "column": ["static_value", "static_flag", "static_cb_value"],
            "g2_decision": ["aprobada", "bloqueada", "aprobada"],
        }
    ).write_csv(path)
    return path


def _base(split="train"):
    ids = [1, 2] if split == "train" else [10]
    return pl.DataFrame(
        {
            "case_id": ids,
            "WEEK_NUM": [20] * len(ids),
            "date_decision": [date(2020, 1, 1)] * len(ids),
        }
    )


@pytest.mark.parametrize(
    ("split", "builder", "family", "source_rows"),
    [
        ("train", depth0.build_static_0_features, "static_0", [1, 0]),
        ("train", depth0.build_static_cb_0_features, "static_cb_0", [1, 0]),
        ("test", depth0.build_static_0_features, "static_0", [1]),
        ("test", depth0.build_static_cb_0_features, "static_cb_0", [1]),
    ],
)
@pytest.mark.usefixtures("static_data")
def test_coverage_artifact_is_one_row_per_base_case(
    b3_selection, g2_decision, tmp_path, split, builder, family, source_rows
):
    artifact = builder(
        _base(split),
        split,
        output_dir=tmp_path / "out",
        selection_path=b3_selection,
        g2_path=g2_decision,
    )
    features = pl.read_parquet(artifact.data_path)

    assert artifact.family == family
    assert features["case_id"].to_list() == _base(split)["case_id"].to_list()
    assert features[f"{family}__n_records"].to_list() == source_rows
    assert features[f"{family}__source_present"].to_list() == [
        count > 0 for count in source_rows
    ]
    manifest = json.loads(artifact.manifest_path.read_text())
    column = "static_cb_value" if family == "static_cb_0" else "static_value"
    expected_values = (
        [5, None]
        if family == "static_cb_0" and split == "train"
        else [10, None]
        if family == "static_0" and split == "train"
        else [7]
        if family == "static_cb_0"
        else [30]
    )
    assert features[f"{family}__{column}__first__all"].to_list() == expected_values
    assert "static_0__static_flag__first__all" not in features.columns
    assert manifest["inputs"]["g2_decision"] == str(g2_decision)
    assert manifest["exclusions"]["manual_review"] == (1 if family == "static_0" else 0)
    assert (
        sum(definition.role == "predictor" for definition in artifact.feature_schema)
        == 1
    )
    assert (
        sum(definition.role == "cobertura" for definition in artifact.feature_schema)
        == 2
    )


@pytest.mark.usefixtures("static_data")
def test_catalog_contains_only_emitted_depth0_features(
    b3_selection, g2_decision, tmp_path
):
    artifacts = [
        depth0.build_static_0_features(
            _base(),
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
            g2_path=g2_decision,
        ),
        depth0.build_static_cb_0_features(
            _base(),
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
            g2_path=g2_decision,
        ),
    ]
    catalog_path = depth0.write_feature_catalog(artifacts, tmp_path / "depth0.csv")
    catalog = pl.read_csv(catalog_path)

    assert set(catalog["name"]) == {
        "static_0__n_records",
        "static_0__source_present",
        "static_0__static_value__first__all",
        "static_cb_0__n_records",
        "static_cb_0__source_present",
        "static_cb_0__static_cb_value__first__all",
    }
    assert set(catalog["role"]) == {"cobertura", "predictor"}
    assert set(catalog["window"]) == {"all"}


@pytest.mark.parametrize(
    ("source", "frame", "error"),
    [
        ("static_0", pl.DataFrame({"case_id": [1, 1]}), "not unique"),
        ("static_cb_0", pl.DataFrame({"case_id": [1, None]}), "contains nulls"),
    ],
)
def test_invalid_source_keys_are_rejected(
    static_data, b3_selection, g2_decision, source, frame, error
):
    folder = static_data / "parquet_files" / "train"
    for existing in folder.glob(f"train_{source}*.parquet"):
        existing.unlink()
    frame.write_parquet(folder / f"train_{source}.parquet")
    builder = (
        depth0.build_static_0_features
        if source == "static_0"
        else depth0.build_static_cb_0_features
    )
    with pytest.raises(ValueError, match=error):
        builder(
            _base(),
            "train",
            output_dir=static_data / "out",
            selection_path=b3_selection,
            g2_path=g2_decision,
        )


def test_missing_source_shards_fail_explicitly(
    static_data, b3_selection, g2_decision, tmp_path
):
    (static_data / "parquet_files" / "train" / "train_static_cb_0.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="static_cb_0"):
        depth0.build_static_cb_0_features(
            _base(),
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
            g2_path=g2_decision,
        )


@pytest.mark.usefixtures("static_data")
def test_base_validation_precedes_source_reads(b3_selection, g2_decision, tmp_path):
    invalid_base = _base().with_columns(pl.lit(75).alias("WEEK_NUM"))

    with pytest.raises(ValueError, match="train split"):
        depth0.build_static_0_features(
            invalid_base,
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
            g2_path=g2_decision,
        )


@pytest.mark.usefixtures("static_data")
def test_missing_g2_decision_fails_before_build(b3_selection, tmp_path):
    with pytest.raises(FileNotFoundError, match="G2 depth-zero decision"):
        depth0.build_static_0_features(
            _base(),
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
            g2_path=tmp_path / "absent.csv",
        )
    assert not (tmp_path / "out").exists()


def test_published_g2_depth0_decision_allows_coverage_only():
    for source in depth0.SOURCES:
        columns, _ = depth0._selected_columns(
            source, depth0.DEFAULT_SELECTION_PATH, depth0.DEFAULT_G2_PATH
        )
        assert columns == []


@pytest.mark.parametrize("decision", ["pendiente_filtro", "bloqueada", "", None])
@pytest.mark.usefixtures("static_data")
def test_no_explicit_approval_builds_coverage_only(b3_selection, tmp_path, decision):
    path = tmp_path / "g2.csv"
    pl.DataFrame(
        {"family": ["static_0"], "column": ["static_value"], "g2_decision": [decision]},
        schema={"family": pl.String, "column": pl.String, "g2_decision": pl.String},
    ).write_csv(path)
    artifact = depth0.build_static_0_features(
        _base(),
        "train",
        output_dir=tmp_path / "out",
        selection_path=b3_selection,
        g2_path=path,
    )
    features = pl.read_parquet(artifact.data_path)
    assert features.columns == [
        "case_id",
        "static_0__n_records",
        "static_0__source_present",
    ]
    assert all(definition.role == "cobertura" for definition in artifact.feature_schema)


@pytest.mark.parametrize(
    "columns, error",
    [
        (["static_value", "static_value"], "duplicate"),
        (["not_used"], "not a B3 candidate"),
        (["case_id"], "reserved"),
    ],
)
@pytest.mark.usefixtures("static_data")
def test_invalid_g2_approvals_fail(b3_selection, tmp_path, columns, error):
    path = tmp_path / "g2.csv"
    pl.DataFrame(
        {
            "family": ["static_0"] * len(columns),
            "column": columns,
            "g2_decision": ["aprobada"] * len(columns),
        }
    ).write_csv(path)
    with pytest.raises(ValueError, match=error):
        depth0.build_static_0_features(
            _base(),
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
            g2_path=path,
        )
