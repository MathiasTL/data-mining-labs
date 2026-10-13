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
    ).write_parquet(
        train / "train_static_0_0.parquet"
    )
    pl.DataFrame({"case_id": [1], "static_cb_value": [5]}).write_parquet(
        train / "train_static_cb_0.parquet"
    )

    test = root / "parquet_files" / "test"
    pl.DataFrame(
        {"case_id": [10], "static_value": [30.0], "static_flag": [None]}
    ).write_parquet(
        test / "test_static_0.parquet"
    )
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
    b3_selection, tmp_path, split, builder, family, source_rows
):
    artifact = builder(
        _base(split),
        split,
        output_dir=tmp_path / "out",
        selection_path=b3_selection,
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
    if family == "static_0":
        assert features["static_0__static_flag__first__all"].dtype == pl.Boolean
        if split == "test":
            assert features["static_0__static_flag__first__all"].to_list() == [None]
    assert manifest["exclusions"]["manual_review"] == (
        1 if family == "static_0" else 0
    )
    assert sum(definition.role == "predictor" for definition in artifact.feature_schema) == (
        2 if family == "static_0" else 1
    )
    assert sum(definition.role == "cobertura" for definition in artifact.feature_schema) == 2


@pytest.mark.usefixtures("static_data")
def test_catalog_contains_only_emitted_depth0_features(b3_selection, tmp_path):
    artifacts = [
        depth0.build_static_0_features(
            _base(), "train", output_dir=tmp_path / "out", selection_path=b3_selection
        ),
        depth0.build_static_cb_0_features(
            _base(), "train", output_dir=tmp_path / "out", selection_path=b3_selection
        ),
    ]
    catalog_path = depth0.write_feature_catalog(artifacts, tmp_path / "depth0.csv")
    catalog = pl.read_csv(catalog_path)

    assert set(catalog["name"]) == {
        "static_0__n_records",
        "static_0__source_present",
        "static_0__static_value__first__all",
        "static_0__static_flag__first__all",
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
    static_data, b3_selection, source, frame, error
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
        )


def test_missing_source_shards_fail_explicitly(static_data, b3_selection, tmp_path):
    (static_data / "parquet_files" / "train" / "train_static_cb_0.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="static_cb_0"):
        depth0.build_static_cb_0_features(
            _base(),
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
        )


@pytest.mark.usefixtures("static_data")
def test_base_validation_precedes_source_reads(b3_selection, tmp_path):
    invalid_base = _base().with_columns(pl.lit(75).alias("WEEK_NUM"))

    with pytest.raises(ValueError, match="train split"):
        depth0.build_static_0_features(
            invalid_base,
            "train",
            output_dir=tmp_path / "out",
            selection_path=b3_selection,
        )
