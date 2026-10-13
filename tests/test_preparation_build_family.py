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
    assert pl.read_parquet(art.data_path).equals(expected)
    assert art.family == "applprev" and art.split == "train" and art.case_count == 3


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
    assert pl.read_parquet(art.data_path)["case_id"].to_list() == [4]  # el caso 5 solo tiene solicitudes posteriores


def test_train_and_oot_do_not_leak_cases_into_each_other(data, tmp_path):
    tr = ap.build_applprev_features(base_for("train", 11), "train", cutoff_week=11, output_dir=tmp_path / "t", config=SMALL_APP)
    oot = ap.build_applprev_features(base_for("oot", 11), "oot", cutoff_week=11, output_dir=tmp_path / "o", config=SMALL_APP)
    assert set(pl.read_parquet(tr.data_path)["case_id"]).isdisjoint(pl.read_parquet(oot.data_path)["case_id"])


def test_test_split_reads_test_files(data, tmp_path):
    art = ap.build_person_features(base_for("test"), "test", output_dir=tmp_path / "out", config=SMALL_PEOPLE)
    assert art.split == "test" and art.case_count == 5


def test_person_artifact_matches_the_pure_core(data, tmp_path):
    art = ap.build_person_features(base_for("train", 74), "train", output_dir=tmp_path / "out", config=SMALL_PEOPLE)
    expected = ap.aggregate_people(fx.base(), fx.person_1(), fx.person_2(), SMALL_PEOPLE)
    assert pl.read_parquet(art.data_path).equals(expected)
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
