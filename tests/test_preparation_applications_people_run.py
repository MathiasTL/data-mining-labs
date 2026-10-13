import json

import polars as pl
import pytest

import preparation_artifacts as pa
from dm_credit.preparation import contracts
from dm_credit.preparation.families import applications_people_run as run


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    pa.write_sample_data(root)
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    return root


@pytest.fixture
def artifacts(data, tmp_path):
    return pa.build_small_artifacts(tmp_path / "artifacts")


def test_small_artifacts_cover_every_family_and_split(artifacts):
    assert {(a.family, a.split) for a in artifacts} == {(f, s) for f in ("applprev", "person") for s in ("train", "oot", "test")}


def test_small_artifacts_load_back_with_valid_checksums(artifacts):
    for a in artifacts:
        assert contracts.load_family_artifact(a.manifest_path) == a


def test_feature_schema_is_identical_across_splits(artifacts):
    run.assert_same_feature_schema(artifacts)  # no lanza


def test_schema_drift_between_splits_is_detected(artifacts):
    train = next(a for a in artifacts if (a.family, a.split) == ("applprev", "train"))
    oot = next(a for a in artifacts if (a.family, a.split) == ("applprev", "oot"))
    drifted = type(oot)(**{**oot.__dict__, "feature_schema": oot.feature_schema[:-1]})
    with pytest.raises(ValueError, match="applprev"):
        run.assert_same_feature_schema([train, drifted])


def test_catalog_has_one_row_per_feature_and_the_shared_columns(artifacts):
    cat = contracts.feature_catalog(artifacts)
    assert cat.columns == ["name", "role", "logical_type", "source_table", "source_columns", "aggregation", "window", "temporal_rule", "description"]
    assert cat["name"].is_unique().all()
    assert cat["name"].to_list() == sorted(cat["name"].to_list())
    assert {"applprev_1__n_records", "applprev_2__source_present", "person_1__source_present", "person_2__n_records"} <= set(cat["name"])
    assert set(cat["role"]) == {"cobertura", "predictor"}


def test_empty_catalog_is_not_written(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        contracts.write_feature_catalog([], tmp_path / "c.csv")


def test_run_writes_catalog_report_and_exclusions(data, tmp_path):
    out = tmp_path / "out"
    arts = run.run_real_data(
        ("train", "oot", "test"),
        output_dir=out / "artifacts",
        catalog_path=out / "catalog.csv",
        run_report_path=out / "run.csv",
        exclusions_path=out / "exclusions.csv",
        cutoff_week=pa.CUTOFF,
        applications_config=pa.SMALL_APP,
        people_config=pa.SMALL_PEOPLE,
    )
    assert len(arts) == 6
    report = pl.read_csv(out / "run.csv")
    assert report.columns == ["split", "family", "base_cases", "cases_with_source", "cases_without_source", "valid_rows_table_1", "valid_rows_table_2", "predictor_features", "coverage_features"]
    row = report.filter((pl.col("split") == "train") & (pl.col("family") == "applprev")).row(0, named=True)
    assert (row["base_cases"], row["cases_with_source"], row["cases_without_source"]) == (2, 2, 0)  # casos 1 y 2
    oot = report.filter((pl.col("split") == "oot") & (pl.col("family") == "applprev")).row(0, named=True)
    assert (oot["base_cases"], oot["cases_with_source"], oot["cases_without_source"]) == (4, 1, 3)  # solo el caso 4 tiene cobertura válida
    exc = pl.read_csv(out / "exclusions.csv")
    assert exc.columns == ["split", "family", "metric", "value"]
    v = exc.filter((pl.col("split") == "train") & (pl.col("metric") == "applprev_1_rows_after_decision"))["value"].item()
    assert v == 1  # la solicitud posterior del caso 2
    assert pl.read_csv(out / "catalog.csv")["name"].is_unique().all()


def test_run_is_deterministic(data, tmp_path):
    kw = dict(cutoff_week=pa.CUTOFF, applications_config=pa.SMALL_APP, people_config=pa.SMALL_PEOPLE)
    for name in ("a", "b"):
        o = tmp_path / name
        run.run_real_data(("train", "oot", "test"), output_dir=o / "art", catalog_path=o / "c.csv", run_report_path=o / "r.csv", exclusions_path=o / "e.csv", **kw)
    for f in ("c.csv", "r.csv", "e.csv"):
        assert (tmp_path / "a" / f).read_text() == (tmp_path / "b" / f).read_text()


def test_unknown_split_is_rejected(data, tmp_path):
    with pytest.raises(ValueError, match="split"):
        run.run_real_data(("valid",), output_dir=tmp_path / "a", catalog_path=tmp_path / "c.csv", run_report_path=tmp_path / "r.csv", exclusions_path=tmp_path / "e.csv", cutoff_week=pa.CUTOFF)


@pytest.mark.parametrize("field,value", [("aggregation", "XXX"), ("source_table", "other"), ("role", "cobertura"), ("temporal_rule", "otra regla"), ("window", "3m")])
def test_any_definition_drift_between_splits_is_detected(artifacts, field, value):
    from dataclasses import replace

    train = next(a for a in artifacts if (a.family, a.split) == ("applprev", "train"))
    oot = next(a for a in artifacts if (a.family, a.split) == ("applprev", "oot"))
    idx = next(i for i, f in enumerate(oot.feature_schema) if f.role == "predictor")  # un predictor, para que el cambio de rol sea real
    changed = replace(oot.feature_schema[idx], **{field: value})
    drifted = replace(oot, feature_schema=(*oot.feature_schema[:idx], changed, *oot.feature_schema[idx + 1 :]))
    with pytest.raises(ValueError, match="applprev"):
        run.assert_same_feature_schema([train, drifted])


def test_repeated_splits_are_rejected(data, tmp_path):
    with pytest.raises(ValueError, match="repeated"):
        run.run_real_data(("train", "train"), output_dir=tmp_path / "a", catalog_path=tmp_path / "c.csv", run_report_path=tmp_path / "r.csv", exclusions_path=tmp_path / "e.csv", cutoff_week=pa.CUTOFF)


def test_default_report_paths_do_not_depend_on_the_working_directory():
    for p in (run.DEFAULT_CATALOG, run.DEFAULT_RUN_REPORT, run.DEFAULT_EXCLUSIONS):
        assert p.is_absolute() and p.parent.name == "preparation" and p.parent.parent.name == "results"
