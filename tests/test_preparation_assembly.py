import json
from pathlib import Path

import polars as pl
import pytest

from dm_credit.preparation import contracts as ct
from dm_credit.preparation.assembly import assemble_feature_matrix


def publish_family(tmp_path, family, ids, values, split="train"):
    frame = pl.DataFrame({"case_id": ids, f"{family}__amount__sum__all": values})
    definitions = ct.describe_features(frame, temporal_rule="available at decision")
    return ct.write_family_features(
        frame, definitions, family=family, split=split, output_dir=tmp_path / family
    )


def base_frame(split="train"):
    data = {
        "case_id": [1, 2, 3],
        "WEEK_NUM": [10, 11, 12],
        "date_decision": ["2020-01-01", "2020-01-02", "2020-01-03"],
    }
    if split != "test":
        data["target"] = [0, 1, 0]
    return pl.DataFrame(data)


def test_assembles_families_and_writes_manifest(tmp_path):
    first = publish_family(tmp_path, "family_a", [1, 2], [1.0, 2.0])
    second = publish_family(tmp_path, "family_b", [1, 2, 3], [3.0, 4.0, 5.0])

    artifact = assemble_feature_matrix(
        base_frame(), [first, second], "train", output_dir=tmp_path / "matrix"
    )

    result = pl.read_parquet(artifact.data_path)
    assert result["case_id"].to_list() == [1, 2, 3]
    assert result["family_a__amount__sum__all"].to_list()[:2] == [1.0, 2.0]
    assert result["family_a__amount__sum__all"][2] is None
    assert result.columns == [
        "case_id",
        "WEEK_NUM",
        "date_decision",
        "target",
        "family_a__amount__sum__all",
        "family_b__amount__sum__all",
    ]
    manifest = json.loads(Path(artifact.manifest_path).read_text())
    assert manifest["row_count"] == 3
    assert manifest["input_manifests"] == [
        str(first.manifest_path),
        str(second.manifest_path),
    ]


def test_rejects_artifact_from_another_split(tmp_path):
    artifact = publish_family(
        tmp_path, "family_a", [1, 2, 3], [1.0, 2.0, 3.0], split="oot"
    )
    with pytest.raises(ValueError, match="split"):
        assemble_feature_matrix(
            base_frame(), [artifact], "train", output_dir=tmp_path / "matrix"
        )


@pytest.mark.parametrize(
    "bad_base",
    [
        pl.DataFrame(
            {
                "case_id": [1, 1],
                "WEEK_NUM": [1, 2],
                "date_decision": ["2020-01-01"] * 2,
                "target": [0, 1],
            }
        ),
        pl.DataFrame(
            {
                "case_id": [1, None],
                "WEEK_NUM": [1, 2],
                "date_decision": ["2020-01-01"] * 2,
                "target": [0, 1],
            }
        ),
    ],
)
def test_rejects_invalid_base_keys(tmp_path, bad_base):
    with pytest.raises(ValueError, match="case_id"):
        assemble_feature_matrix(bad_base, [], "train", output_dir=tmp_path / "matrix")


def test_rejects_feature_collision_and_does_not_publish(tmp_path):
    artifact = publish_family(tmp_path, "family_a", [1, 2, 3], [1.0, 2.0, 3.0])
    conflicting = base_frame().with_columns(
        pl.lit(1.0).alias("family_a__amount__sum__all")
    )
    with pytest.raises(ValueError, match="duplicate|reserved"):
        assemble_feature_matrix(
            conflicting, [artifact], "train", output_dir=tmp_path / "matrix"
        )
    assert not (tmp_path / "matrix" / "feature_matrix__train.parquet").exists()


def test_rejects_physical_type_that_disagrees_with_manifest(tmp_path):
    frame = pl.DataFrame(
        {"case_id": [1, 2, 3], "family_a__amount__sum__all": [1, 2, 3]}
    )
    definitions = ct.describe_features(frame, temporal_rule="available at decision")
    incorrect_definition = (
        ct.FeatureDefinition(
            name=definitions[0].name,
            role=definitions[0].role,
            logical_type="float",
            source_table=definitions[0].source_table,
            source_columns=definitions[0].source_columns,
            aggregation=definitions[0].aggregation,
            window=definitions[0].window,
            temporal_rule=definitions[0].temporal_rule,
            description=definitions[0].description,
        ),
    )
    artifact = ct.write_family_features(
        frame,
        incorrect_definition,
        family="family_a",
        split="train",
        output_dir=tmp_path / "family_a",
    )
    with pytest.raises(ValueError, match="logical type"):
        assemble_feature_matrix(
            base_frame(), [artifact], "train", output_dir=tmp_path / "matrix"
        )


def test_rejects_duplicate_family_artifact(tmp_path):
    artifact = publish_family(tmp_path, "family_a", [1, 2, 3], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="duplicate family"):
        assemble_feature_matrix(
            base_frame(), [artifact, artifact], "train", output_dir=tmp_path / "matrix"
        )
