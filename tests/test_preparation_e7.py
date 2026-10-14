"""E7: configuración versionada, comando único y manifiesto final."""

import json

import polars as pl
import pytest

import test_preparation_validation as tv
from dm_credit.preparation import build, contracts, e5, e6

CONFIG = build.ROOT / "configs" / "preparation.json"


def test_versioned_config_loads_and_is_complete():
    cfg = build.load_config(CONFIG)
    assert cfg["cutoff_week"] == 74 and cfg["polars_max_threads"] == 1
    assert set(cfg["versions"]) == {"e4", "e5", "e6"}


@pytest.mark.parametrize("drop", ["cutoff_week", "versions", "e4", "catalog", "decisions"])
def test_incomplete_config_is_rejected(tmp_path, drop):
    cfg = json.loads(CONFIG.read_text())
    cfg.pop(drop)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=drop):
        build.load_config(path)


def test_steps_reuse_e4_by_default_and_rebuild_from_scratch():
    cfg = build.load_config(CONFIG)
    reuse = [s.name for s in build.plan_steps(cfg, from_scratch=False)]
    scratch = [s.name for s in build.plan_steps(cfg, from_scratch=True)]
    assert reuse == ["e4_audit", "e4_checkpoint_audit", "e5", "e6", "validation"]
    assert scratch == ["e4_rebuild", "e4_checkpoints", "e4_audit", "e4_checkpoint_audit", "e5", "e6", "validation"]
    validation = build.plan_steps(cfg, from_scratch=False)[-1]
    assert validation.args[:4] == ["--artifacts", str(build._dirs(cfg)["e4"]), "--matrix-dir", str(build._dirs(cfg)["e5"])]


def test_configured_cutoff_results_dir_and_catalog_reach_every_step():
    # F6: antes E5, E6 y validation usaban sus valores por defecto aunque la configuración dijera otra cosa
    cfg = build.load_config(CONFIG)
    steps = {s.name: s.args for s in build.plan_steps(cfg, from_scratch=False)}
    results = str(build.ROOT / cfg["results_dir"])
    for name in ("e5", "e6", "validation"):
        args = steps[name]
        assert args[args.index("--cutoff-week") + 1] == str(cfg["cutoff_week"])
    assert steps["e5"][steps["e5"].index("--report") + 1] == f"{results}/E5_master_tables.csv"
    assert steps["e6"][steps["e6"].index("--report") + 1] == f"{results}/E6_views.csv"
    assert steps["e6"][steps["e6"].index("--catalog") + 1] == str(build.ROOT / cfg["catalog"])
    assert steps["validation"][steps["validation"].index("--out") + 1] == results
    assert steps["e4_audit"][steps["e4_audit"].index("--results-dir") + 1] == results


@pytest.mark.parametrize(("key", "value"), [("cutoff_week", 70), ("results_dir", "otros/resultados")])
def test_values_that_e4_cannot_honour_are_rejected(tmp_path, key, value):
    # E4 reconstruye y audita con el corte 74 y escribe en results/preparation: otro valor sería ignorado en silencio
    cfg = json.loads(CONFIG.read_text())
    cfg[key] = value
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=key):
        build.load_config(path)


def test_module_steps_run_serially_with_one_thread_and_stop_on_failure(tmp_path):
    cfg = build.load_config(CONFIG)
    calls = []

    def runner(step, log):
        calls.append((step.name, step.env["POLARS_MAX_THREADS"]))
        return 1 if step.name == "e5" else 0

    with pytest.raises(RuntimeError, match="e5"):
        build.run_pipeline(cfg, from_scratch=False, runner=runner, log_dir=tmp_path)
    assert calls == [("e4_audit", "1"), ("e4_checkpoint_audit", "1"), ("e5", "1")]  # nada corre después del fallo


def test_final_schema_lists_controls_then_features_with_roles(published):
    report = e5.assemble_master_tables(published["bases"], published["artifacts"], output_dir=published["tmp"] / "E5", cutoff_week=tv.g3.CUTOFF)
    train = pl.read_parquet(report.filter(pl.col("split") == "train")["data_path"].item())
    catalog = contracts.feature_catalog(published["artifacts"]["train"])
    schema = build.final_schema(train, catalog)
    assert schema.columns == ["position", "column", "dtype", "role", "logical_type"]
    assert schema["column"].to_list() == train.columns
    assert schema.filter(pl.col("column") == "case_id")["role"].item() == "control"
    assert set(schema["role"]) == {"control", "predictor", "cobertura"}


published = tv.published


def test_manifest_records_config_partitions_and_artifact_checksums(published, tmp_path):
    report5 = e5.assemble_master_tables(published["bases"], published["artifacts"], output_dir=tmp_path / "E5", cutoff_week=tv.g3.CUTOFF)
    tables = {r["split"]: pl.read_parquet(r["data_path"]) for r in report5.iter_rows(named=True)}
    catalog = contracts.feature_catalog(published["artifacts"]["train"])
    report6 = e6.fit_and_transform(tables, catalog, output_dir=tmp_path / "E6", cutoff_week=tv.g3.CUTOFF)
    manifest = build.final_manifest(
        config_path=CONFIG, bases=published["bases"], e4_artifacts=published["artifacts"],
        e5_report=report5, e6_report=report6, e6_dir=tmp_path / "E6", steps=[{"name": "e5", "seconds": 1.0, "returncode": 0}],
    )
    assert len(manifest["config_sha256"]) == 64
    assert manifest["partitions"]["train"]["cases"] == published["bases"]["train"].height
    assert manifest["partitions"]["test"]["positives"] is None
    assert len(manifest["artifacts"]["e4"]) == 36 and all(len(a["data_sha256"]) == 64 for a in manifest["artifacts"]["e4"])
    assert {r["split"] for r in manifest["artifacts"]["e5"]} == {"train", "oot", "test"}
    assert len(manifest["artifacts"]["e6"]["outputs"]) == 6 and len(manifest["artifacts"]["e6"]["state_sha256"]) == 64
    assert manifest["steps"][0]["name"] == "e5"
    json.dumps(manifest)  # serializable
