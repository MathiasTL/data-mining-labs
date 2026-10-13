import json

import polars as pl
import pytest

from dm_credit.preparation import contracts as ct


def frame():
    return pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "applprev_1__n_records": [3, 1, 2],
            "applprev_1__source_present": [True, True, True],
            "applprev_1__credamount_590a__mean__all": [2000.0, 500.0, None],
            "applprev_1__creationdate_885d__days_since_last__all": [1, 121, 0],
            "applprev_1__credtype_587l__share_cal__all": [0.5, 1.0, None],
        }
    )


def defs(df=None):
    return ct.describe_features(
        df if df is not None else frame(),
        temporal_rule="creationdate_885D <= date_decision",
        source_columns={
            "credamount_590a": "credamount_590A",
            "creationdate_885d": "creationdate_885D",
            "credtype_587l": "credtype_587L",
        },
    )


def test_parse_feature_name_splits_the_grammar():
    p = ct.parse_feature_name("applprev_1__credamount_590a__mean__all")
    assert p == {
        "source_table": "applprev_1",
        "variable": "credamount_590a",
        "aggregation": "mean",
        "window": "all",
    }


def test_parse_coverage_names():
    assert ct.parse_feature_name("applprev_2__n_records")["aggregation"] == "n_records"
    assert ct.parse_feature_name("person_1__source_present")["variable"] == ""


@pytest.mark.parametrize(
    "bad",
    [
        "case_id",
        "Applprev__X__mean__all",
        "a__b__c__d__e__f",
        "applprev_1__mean",
        "1abc__x__mean__all",
    ],
)
def test_invalid_names_are_rejected(bad):
    with pytest.raises(ValueError, match="name"):
        ct.parse_feature_name(bad)


def test_describe_features_roles_types_and_sources():
    d = {x.name: x for x in defs()}
    assert (
        d["applprev_1__n_records"].role == "cobertura"
        and d["applprev_1__n_records"].logical_type == "int"
    )
    assert d["applprev_1__source_present"].logical_type == "bool"
    m = d["applprev_1__credamount_590a__mean__all"]
    assert (m.role, m.logical_type, m.aggregation, m.window) == (
        "predictor",
        "float",
        "mean",
        "all",
    )
    assert m.source_table == "applprev_1" and m.source_columns == ("credamount_590A",)
    assert m.temporal_rule == "creationdate_885D <= date_decision"
    assert [x.name for x in defs()] == [
        c for c in frame().columns if c != "case_id"
    ]  # conserva el orden


def test_feature_definition_validates_role_and_type():
    with pytest.raises(ValueError, match="role"):
        ct.FeatureDefinition(
            "a__b__mean__all", "otro", "float", "a", ("b",), "mean", "all", "", ""
        )
    with pytest.raises(ValueError, match="logical_type"):
        ct.FeatureDefinition(
            "a__b__mean__all",
            "predictor",
            "decimal",
            "a",
            ("b",),
            "mean",
            "all",
            "",
            "",
        )


def test_describe_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="dtype"):
        ct.describe_features(
            pl.DataFrame({"case_id": [1], "a__b__mean__all": [[1]]}),
            temporal_rule="x",
            source_columns={},
        )


def test_write_and_load_family_artifact_roundtrip(tmp_path):
    df = frame()
    art = ct.write_family_features(
        df,
        defs(),
        family="applprev",
        split="train",
        output_dir=tmp_path,
        exclusions={"applprev_1_rows_after_decision": 2},
        rules_version="v1",
        inputs={"base": "abc"},
    )
    assert art.family == "applprev" and art.split == "train"
    assert (art.row_count, art.case_count) == (3, 3)
    assert art.data_path.exists() and art.manifest_path.exists()
    assert pl.read_parquet(art.data_path).equals(df)
    manifest = json.loads(art.manifest_path.read_text())
    assert manifest["format_version"] == 1 and manifest["rules_version"] == "v1"
    assert manifest["exclusions"] == {"applprev_1_rows_after_decision": 2}
    assert manifest["inputs"] == {"base": "abc"}
    assert [f["name"] for f in manifest["features"]] == [d.name for d in defs()]
    assert ct.load_family_artifact(art.manifest_path) == art


def test_manifest_has_no_data_samples(tmp_path):
    art = ct.write_family_features(
        frame(), defs(), family="applprev", split="train", output_dir=tmp_path
    )
    text = art.manifest_path.read_text()
    assert "2000.0" not in text and '"case_id": [' not in text


def test_duplicated_case_id_is_rejected_and_nothing_is_published(tmp_path):
    bad = pl.concat([frame(), frame().head(1)])
    with pytest.raises(ValueError, match="case_id"):
        ct.write_family_features(
            bad, defs(), family="applprev", split="train", output_dir=tmp_path
        )
    assert (
        not list(tmp_path.rglob("*.parquet"))
        and not list(tmp_path.rglob("*.json"))
        and not list(tmp_path.rglob("*.tmp"))
    )


def test_null_case_id_is_rejected(tmp_path):
    bad = frame().with_columns(pl.Series("case_id", [1, None, 3]))
    with pytest.raises(ValueError, match="case_id"):
        ct.write_family_features(
            bad, defs(), family="applprev", split="train", output_dir=tmp_path
        )


def test_schema_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        ct.write_family_features(
            frame().drop("applprev_1__n_records"),
            defs(),
            family="applprev",
            split="train",
            output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="schema"):
        ct.write_family_features(
            frame().select(
                ["case_id", *reversed([c for c in frame().columns if c != "case_id"])]
            ),
            defs(),
            family="applprev",
            split="train",
            output_dir=tmp_path,
        )


def test_reserved_column_names_are_rejected(tmp_path):
    bad = (
        frame()
        .with_columns(pl.lit(1).alias("applprev_1__x__sum__all"))
        .rename({"applprev_1__x__sum__all": "WEEK_NUM"})
    )
    with pytest.raises(ValueError, match="reserved|name"):
        ct.write_family_features(
            bad,
            defs(bad.drop("WEEK_NUM")),
            family="applprev",
            split="train",
            output_dir=tmp_path,
        )


def test_invalid_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        ct.write_family_features(
            frame(), defs(), family="applprev", split="valid", output_dir=tmp_path
        )


def test_existing_valid_output_is_not_overwritten_when_replacement_fails(tmp_path):
    first = ct.write_family_features(
        frame(), defs(), family="applprev", split="train", output_dir=tmp_path
    )
    before = first.data_path.read_bytes()
    bad = pl.concat([frame(), frame().head(1)])
    with pytest.raises(ValueError):
        ct.write_family_features(
            bad, defs(), family="applprev", split="train", output_dir=tmp_path
        )
    assert first.data_path.read_bytes() == before


def test_empty_family_with_declared_schema_is_valid(tmp_path):
    empty = frame().clear()
    art = ct.write_family_features(
        empty, defs(empty), family="applprev", split="test", output_dir=tmp_path
    )
    assert (
        art.row_count == 0 and pl.read_parquet(art.data_path).columns == empty.columns
    )


def test_feature_matrix_artifact_fields():
    m = ct.FeatureMatrixArtifact(
        split="train",
        data_path="m.parquet",
        manifest_path="m.json",
        row_count=3,
        control_columns=("case_id",),
        feature_schema=tuple(defs()),
        input_manifests=("a.json",),
    )
    assert m.control_columns == ("case_id",)


def test_unserializable_manifest_does_not_touch_an_existing_valid_pair(tmp_path):
    first = ct.write_family_features(
        frame(), defs(), family="applprev", split="train", output_dir=tmp_path
    )
    data_before, manifest_before = (
        first.data_path.read_bytes(),
        first.manifest_path.read_bytes(),
    )
    with pytest.raises(TypeError):
        ct.write_family_features(
            frame().head(2),
            defs(),
            family="applprev",
            split="train",
            output_dir=tmp_path,
            exclusions={"a": object()},
        )
    assert (
        first.data_path.read_bytes() == data_before
        and first.manifest_path.read_bytes() == manifest_before
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_load_detects_a_parquet_that_does_not_match_its_manifest(tmp_path):
    art = ct.write_family_features(
        frame(), defs(), family="applprev", split="train", output_dir=tmp_path
    )
    frame().head(2).write_parquet(
        art.data_path
    )  # alguien reemplaza el Parquet sin actualizar el manifiesto
    with pytest.raises(ValueError, match="checksum|sha256"):
        ct.load_family_artifact(art.manifest_path)


def test_minimal_family_base_does_not_require_target_but_rejects_invalid_date():
    base = pl.DataFrame(
        {"case_id": [1], "WEEK_NUM": [73], "date_decision": ["2020-01-01"]}
    )
    ct.validate_base(base, "train")  # G1: el adaptador no recibe target
    with pytest.raises(ValueError, match="date_decision"):
        ct.validate_base(
            base.with_columns(pl.lit("2020-02-30").alias("date_decision")), "train"
        )


def test_validate_base_rejects_nan_week_numbers():
    base = pl.DataFrame(
        {
            "case_id": [1, 2],
            "WEEK_NUM": [1.0, float("nan")],
            "date_decision": ["2020-01-01", "2020-01-02"],
        }
    )
    with pytest.raises(ValueError, match="WEEK_NUM"):
        ct.validate_base(base, "train", 74)
