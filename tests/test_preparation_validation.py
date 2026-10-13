"""F3/F4/F7: validaciones sobre artefactos publicados, resultados por familia e informe de fuga."""

import json

import polars as pl
import pytest

import test_preparation_e3_integration as e3
import test_preparation_g3_integration as g3
from dm_credit.preparation import validation as val

SPLITS = ("train", "oot", "test")


@pytest.fixture
def published(tmp_path, monkeypatch):
    """Todas las familias publicadas para train, OOT y test con las fuentes sintéticas de G3."""
    root = tmp_path / "data"
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    g3._write_sources(root)
    base = e3._base([1, 2, 3, 4, 5, 6], [10, 10, 11, 11, 12, 12])
    bases = {
        "train": base.filter(pl.col("WEEK_NUM") < g3.CUTOFF),
        "oot": base.filter(pl.col("WEEK_NUM") >= g3.CUTOFF),
        "test": e3._base([101, 102], [20, 20], labeled=False),
    }
    app_cfg, people_cfg = g3._configs()
    artifacts = {s: g3._artifacts(s, bases[s], tmp_path / "art" / s, app_cfg, people_cfg) for s in SPLITS}
    return {"bases": bases, "artifacts": artifacts, "tmp": tmp_path}


def run(published, **kw):
    return val.validate_pipeline(published["bases"], published["artifacts"], output_dir=published["tmp"] / "val", cutoff_week=g3.CUTOFF, **kw)


def status(checks, name, split="all"):
    rows = checks.filter((pl.col("check") == name) & (pl.col("split") == split))
    assert rows.height == 1, f"{name}/{split} no está en el informe"
    return rows.row(0, named=True)


def test_all_checks_pass_on_a_consistent_pipeline(published):
    checks = run(published)
    assert checks.columns == ["check", "split", "status", "detail"]
    failed = checks.filter(pl.col("status") != "ok")
    assert failed.is_empty(), failed.to_dicts()


def test_expected_counts_are_checked_when_given(published):
    ok = run(published, expected={"train": (2, 1), "oot": (4, 2)})
    assert status(ok, "base_cases", "train")["status"] == "ok"
    bad = run(published, expected={"train": (99, 1), "oot": (4, 2)})
    assert status(bad, "base_cases", "train")["status"] == "fail"


def test_every_split_reports_its_core_checks(published):
    checks = run(published)
    for split in SPLITS:
        for name in ("base_cases", "same_base_fingerprint", "artifacts_cover_base", "assembled_matrix", "finite_values", "features_in_catalog", "no_reserved_predictors"):
            status(checks, name, split)
    for name in ("train_oot_partition", "schema_parity", "baseline_contract", "views_fit_on_train_only"):
        status(checks, name)


def test_an_artifact_built_from_another_base_is_detected(published):
    arts = published["artifacts"]["oot"]
    manifest = json.loads(arts[0].manifest_path.read_text())
    manifest["inputs"]["base_sha256"] = "otra-base"
    arts[0].manifest_path.write_text(json.dumps(manifest))
    row = status(run(published), "same_base_fingerprint", "oot")
    assert row["status"] == "fail" and arts[0].family in row["detail"]


def test_a_missing_family_artifact_is_reported_not_silently_skipped(published):
    published["artifacts"]["test"] = published["artifacts"]["test"][1:]
    row = status(run(published), "family_set")
    assert row["status"] == "fail"


def test_leakage_report_collects_exclusions_per_family_and_split(published):
    report = val.leakage_report([a for s in SPLITS for a in published["artifacts"][s]])
    assert report.columns == ["split", "family", "rules_version", "metric", "value"]
    v = report.filter((pl.col("split") == "train") & (pl.col("metric") == "applprev_1_rows_after_decision"))["value"].item()
    assert v == 1  # la solicitud del caso 2 creada después de la decisión
    assert report.select(["split", "family", "metric"]).is_duplicated().sum() == 0


def test_family_results_report_cases_coverage_and_roles(published):
    res = val.family_results(published["artifacts"]["oot"], published["bases"]["oot"])
    assert res.columns == ["split", "family", "rules_version", "cases", "cases_with_source", "coverage_pct", "predictor_features", "coverage_features"]
    row = res.filter(pl.col("family") == "applprev").row(0, named=True)
    assert (row["cases"], row["cases_with_source"]) == (4, 1)  # solo el caso 4 tiene historial válido en OOT
    assert row["predictor_features"] > 0
    assert res.filter(pl.col("family") == "static_0")["predictor_features"].item() == 0  # G2: solo cobertura


def test_run_real_writes_the_three_reports(published, monkeypatch):
    out = published["tmp"] / "reports"
    val.write_reports(published["bases"], published["artifacts"], out_dir=out, work_dir=published["tmp"] / "work", cutoff_week=g3.CUTOFF)
    for name in ("integration_checks.csv", "temporal_leakage_report.csv", "family_results.csv"):
        assert (out / name).exists()
    assert pl.read_csv(out / "integration_checks.csv").filter(pl.col("status") != "ok").is_empty()
