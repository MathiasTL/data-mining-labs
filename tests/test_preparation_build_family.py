import json

import polars as pl
import pytest

import preparation_fixtures as fx
from dm_credit.preparation import build
from dm_credit.preparation import contracts as ct
from dm_credit.preparation.families import applications_people as ap


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Datos sintéticos con la estructura de archivos real, repartidos en shards."""
    root = tmp_path / "data"
    for split in ("train", "test"):
        (root / "parquet_files" / split).mkdir(parents=True)
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    tr = root / "parquet_files" / "train"
    a1 = fx.applprev_1()
    a1.head(4).write_parquet(tr / "train_applprev_1_0.parquet")
    a1.tail(4).write_parquet(tr / "train_applprev_1_1.parquet")
    fx.applprev_2().write_parquet(tr / "train_applprev_2.parquet")
    fx.person_1().write_parquet(tr / "train_person_1.parquet")
    fx.person_2().write_parquet(tr / "train_person_2.parquet")
    te = root / "parquet_files" / "test"
    fx.applprev_1().write_parquet(te / "test_applprev_1_0.parquet")
    fx.applprev_2().write_parquet(te / "test_applprev_2.parquet")
    fx.person_1().write_parquet(te / "test_person_1.parquet")
    fx.person_2().write_parquet(te / "test_person_2.parquet")
    return root


def base_for(split, cutoff=11):
    b = fx.base().select("case_id", "WEEK_NUM", "date_decision")
    if split == "oot":
        return b.filter(pl.col("WEEK_NUM") >= cutoff)
    if split == "train":
        return b.filter(pl.col("WEEK_NUM") < cutoff)
    return b


SMALL_APP = ap.ApplicationsConfig(
    numeric={"credamount_590A": ("count", "sum")}, categorical_distinct=("credtype_587L",), category_shares={},
    bool_cols=("isbidproduct_390L",), date_cols=("approvaldate_319D",), children_category_shares={},
)
SMALL_PEOPLE = ap.PeopleConfig(
    main_numeric=("mainoccupationinc_384A",), main_categorical=(), birth_col="birth_259D", employment_col="empl_employedfrom_271D",
    contact_category_shares={}, child_category_shares={},
)


def test_train_artifact_matches_the_pure_core(data, tmp_path):
    base = base_for("train", cutoff=74)
    art = ap.build_applprev_features(base, "train", output_dir=tmp_path / "out", config=SMALL_APP)
    expected = ap.aggregate_applications(fx.base(), fx.applprev_1(), fx.applprev_2(), SMALL_APP)
    out = pl.read_parquet(art.data_path)
    # los casos con cobertura coinciden exactamente con el núcleo
    assert out.filter(pl.col("applprev_1__source_present")).equals(expected)
    # y la salida trae una fila por cada caso de la base (como bureau y productos)
    assert out["case_id"].to_list() == base_for("train", 74)["case_id"].sort().to_list()
    assert art.family == "applprev" and art.split == "train" and art.case_count == 6


def test_manifest_records_exclusions_inputs_and_rules(data, tmp_path):
    art = ap.build_applprev_features(base_for("train", 74), "train", output_dir=tmp_path / "out", config=SMALL_APP)
    m = json.loads(art.manifest_path.read_text())
    assert m["exclusions"]["applprev_1_rows_after_decision"] == 2
    assert m["exclusions"]["applprev_2_rows_invalid_parent"] == 3
    assert m["rules_version"] == ap.RULES_VERSION and m["inputs"]["cutoff_week"] == "74"
    assert len(m["inputs"]["base_sha256"]) == 64 and "train_applprev_1_0.parquet" in m["inputs"]["shards"]
    assert any(f["name"] == "applprev_1__n_records" and f["role"] == "cobertura" for f in m["features"])


def test_oot_only_aggregates_the_cases_of_its_base(data, tmp_path):
    art = ap.build_applprev_features(base_for("oot", 11), "oot", cutoff_week=11, output_dir=tmp_path / "out", config=SMALL_APP)
    out = pl.read_parquet(art.data_path)
    assert out["case_id"].to_list() == [3, 4, 5, 6]  # todos los casos de la base OOT
    assert out.filter(pl.col("applprev_1__source_present"))["case_id"].to_list() == [4]  # el caso 5 solo tiene solicitudes posteriores


def test_train_and_oot_do_not_leak_cases_into_each_other(data, tmp_path):
    tr = ap.build_applprev_features(base_for("train", 11), "train", cutoff_week=11, output_dir=tmp_path / "t", config=SMALL_APP)
    oot = ap.build_applprev_features(base_for("oot", 11), "oot", cutoff_week=11, output_dir=tmp_path / "o", config=SMALL_APP)
    assert set(pl.read_parquet(tr.data_path)["case_id"]).isdisjoint(pl.read_parquet(oot.data_path)["case_id"])


def test_test_split_reads_test_files(data, tmp_path):
    art = ap.build_person_features(base_for("test"), "test", output_dir=tmp_path / "out", config=SMALL_PEOPLE)
    assert art.split == "test" and art.case_count == 6


def test_person_artifact_matches_the_pure_core(data, tmp_path):
    art = ap.build_person_features(base_for("train", 74), "train", output_dir=tmp_path / "out", config=SMALL_PEOPLE)
    expected = ap.aggregate_people(fx.base(), fx.person_1(), fx.person_2(), SMALL_PEOPLE)
    out = pl.read_parquet(art.data_path)
    assert out.filter(pl.col("person_1__source_present")).equals(expected)
    assert out.height == 6
    assert art.family == "person"


@pytest.mark.parametrize(
    "mutate,split,msg",
    [
        (lambda b: b.with_columns(WEEK_NUM=pl.lit(80)), "train", "74"),
        (lambda b: b.with_columns(WEEK_NUM=pl.lit(10)), "oot", "74"),
        (lambda b: b.with_columns(target=pl.lit(0)), "test", "target"),
        (lambda b: b.with_columns(date_decision=pl.when(pl.col("case_id") == 1).then(None).otherwise(pl.col("date_decision"))), "train", "date_decision"),
        (lambda b: b.with_columns(WEEK_NUM=pl.when(pl.col("case_id") == 1).then(None).otherwise(pl.col("WEEK_NUM"))), "train", "WEEK_NUM"),
        (lambda b: pl.concat([b, b.head(1)]), "train", "case_id"),
        (lambda b: b.drop("date_decision"), "train", "date_decision"),
    ],
)
def test_invalid_base_is_rejected(data, tmp_path, mutate, split, msg):
    with pytest.raises(ValueError, match=msg):
        ap.build_applprev_features(mutate(base_for(split, 74 if split != "oot" else 11)), split, cutoff_week=74 if split != "oot" else 74, output_dir=tmp_path / "out", config=SMALL_APP)


def test_invalid_split_is_rejected(data, tmp_path):
    with pytest.raises(ValueError, match="split"):
        ap.build_applprev_features(base_for("train", 74), "valid", output_dir=tmp_path / "out", config=SMALL_APP)


def test_missing_source_files_are_an_error_not_an_empty_aggregate(data, tmp_path):
    (data / "parquet_files" / "train" / "train_applprev_2.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="applprev_2"):
        ap.build_applprev_features(base_for("train", 74), "train", output_dir=tmp_path / "out", config=SMALL_APP)


def test_nothing_is_published_when_validation_fails(data, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        ap.build_applprev_features(base_for("train", 74).with_columns(WEEK_NUM=pl.lit(99)), "train", output_dir=out, config=SMALL_APP)
    assert not out.exists() or not list(out.rglob("*"))


def test_dispatch_by_family_name(data, tmp_path):
    art = build.build_family_features("applprev", base_for("train", 74), "train", output_dir=tmp_path / "out", config=SMALL_APP)
    assert isinstance(art, ct.FamilyFeatureArtifact) and art.family == "applprev"
    art2 = build.build_family_features("person", base_for("train", 74), "train", output_dir=tmp_path / "out2", config=SMALL_PEOPLE)
    assert art2.family == "person"


def test_unknown_family_is_rejected(data, tmp_path):
    with pytest.raises(ValueError, match="family"):
        build.build_family_features("nope", base_for("train", 74), "train", output_dir=tmp_path / "out")


def test_rebuild_is_deterministic(data, tmp_path):
    a = ap.build_applprev_features(base_for("train", 74), "train", output_dir=tmp_path / "a", config=SMALL_APP)
    b = ap.build_applprev_features(base_for("train", 74), "train", output_dir=tmp_path / "b", config=SMALL_APP)
    assert pl.read_parquet(a.data_path).equals(pl.read_parquet(b.data_path))
    assert json.loads(a.manifest_path.read_text()) == json.loads(b.manifest_path.read_text())


def test_shard_fingerprint_changes_when_a_shard_is_modified_in_place(data, tmp_path):
    import os

    a = ap.build_applprev_features(base_for("train", 74), "train", output_dir=tmp_path / "a", config=SMALL_APP)
    shard = data / "parquet_files" / "train" / "train_applprev_2.parquet"
    stat = shard.stat()
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))  # mismo tamaño, otra fecha
    b = ap.build_applprev_features(base_for("train", 74), "train", output_dir=tmp_path / "b", config=SMALL_APP)
    assert json.loads(a.manifest_path.read_text())["inputs"]["shards"] != json.loads(b.manifest_path.read_text())["inputs"]["shards"]



@pytest.mark.parametrize("family", ["applprev", "person"])
def test_uncovered_cases_get_zero_records_and_false_presence(data, tmp_path, family):
    build = ap.build_applprev_features if family == "applprev" else ap.build_person_features
    cfg = SMALL_APP if family == "applprev" else SMALL_PEOPLE
    out = pl.read_parquet(build(base_for("train", 74), "train", output_dir=tmp_path / "o", config=cfg).data_path)
    parent, child = ap.TABLES[family]
    uncovered = out.filter(~pl.col(f"{parent}__source_present"))
    assert uncovered.height > 0
    for table in (parent, child):
        assert uncovered[f"{table}__n_records"].to_list() == [0] * uncovered.height
        assert uncovered[f"{table}__source_present"].to_list() == [False] * uncovered.height
    # la cobertura nunca queda nula; el resto de características sí, para que D2 las trate
    for table in (parent, child):
        assert out[f"{table}__n_records"].null_count() == 0 and out[f"{table}__source_present"].null_count() == 0
    predictors = [c for c in out.columns if c != "case_id" and not c.endswith(("__n_records", "__source_present", "__n_contacts__all"))]
    assert all(uncovered[c].null_count() == uncovered.height for c in predictors)


def test_assembled_matrix_has_no_null_coverage_for_applprev_and_person(data, tmp_path):
    from dm_credit.preparation import assembly

    oot = fx.base().filter(pl.col("WEEK_NUM") >= 11)  # casos 3 a 6, con target
    arts = [
        ap.build_applprev_features(oot, "oot", cutoff_week=11, output_dir=tmp_path / "a", config=SMALL_APP),
        ap.build_person_features(oot, "oot", cutoff_week=11, output_dir=tmp_path / "p", config=SMALL_PEOPLE),
    ]
    m = pl.read_parquet(assembly.assemble_feature_matrix(oot, arts, "oot", output_dir=tmp_path / "m").data_path)
    coverage = [c for c in m.columns if c.endswith(("__n_records", "__source_present"))]
    assert len(coverage) == 8
    assert all(m[c].null_count() == 0 for c in coverage)
