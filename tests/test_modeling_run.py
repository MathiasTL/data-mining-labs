"""Pruebas de extremo a extremo y de las guardas del protocolo (matriz sintética en Parquet temporal)."""

import json
import shutil

import numpy as np
import polars as pl
import pytest

from dm_credit.modeling import models as md
from dm_credit.modeling import pipeline as pp
from dm_credit.modeling import run as rn

from modeling_fixtures import synthetic_matrix, synthetic_spec

pytestmark = pytest.mark.filterwarnings("ignore")

TINY = {
    "logistic": [md.ModelConfig("logistic", {"C": 1.0}, 0)],
    "lightgbm": [
        md.ModelConfig("lightgbm", {**md.LGBM_FIXED, "num_leaves": 7, "min_child_samples": 5, "n_estimators": 10}, 0),
        md.ModelConfig("lightgbm", {**md.LGBM_FIXED, "num_leaves": 7, "min_child_samples": 5, "n_estimators": 20}, 1),
    ],
}


@pytest.fixture(scope="module")
def matrix_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("E5")
    df = synthetic_matrix(per_week=250)
    spec = synthetic_spec()
    schema = [{"name": c, "logical_type": "float" if c in spec.numeric else "string"} for c in spec.columns]
    df.filter(pl.col("WEEK_NUM") <= 73).write_parquet(d / "feature_matrix__train.parquet")
    df.filter(pl.col("WEEK_NUM") >= 74).write_parquet(d / "feature_matrix__oot.parquet")
    (d / "feature_matrix__train.manifest.json").write_text(json.dumps({"feature_schema": schema}))
    (d / "feature_matrix__oot.manifest.json").write_text(json.dumps({"data_sha256": "abc"}))
    return d


@pytest.fixture(scope="module")
def frozen(matrix_dir, tmp_path_factory):
    """select + ablate hechos con el OOT inexistente (prueba de que no se lee)."""
    d = tmp_path_factory.mktemp("dev_only")
    for name in ("feature_matrix__train.parquet", "feature_matrix__train.manifest.json"):
        shutil.copy(matrix_dir / name, d / name)
    out = tmp_path_factory.mktemp("frozen_out")
    rn.run(d, out, "select", grids=TINY)
    rn.run(d, out, "ablate", grids=TINY)
    return out


def _with_decisions(frozen, tmp_path, retire: dict):
    """Copia el directorio congelado y fija las decisiones de ablación por modelo."""
    out = tmp_path / "out"
    shutil.copytree(frozen, out)
    sel = json.loads((out / "selection.json").read_text())["selected"]
    groups = list(rn.ABLATION_GROUPS)
    rows = [
        {"group": g, "model": m, "config_id": sel[m], "retire": bool(retire.get(m, ()) and g in retire[m]),
         "rule_hash": rn.sel.RULE_HASH}
        for g in groups for m in sel
    ]
    pl.DataFrame(rows).write_csv(out / "ablation_decisions.csv")
    return out


def test_select_and_ablate_run_with_oot_files_absent(frozen):
    assert (frozen / "fold_metrics.csv").exists() and (frozen / "ablation.csv").exists()
    sel = json.loads((frozen / "selection.json").read_text())
    assert set(sel["selected"]) == {"logistic", "lightgbm"} and sel["grid_hash"]
    dec = pl.read_csv(frozen / "ablation_decisions.csv")
    assert set(dec["group"]) == set(rn.ABLATION_GROUPS)  # incluido no_contacts con entrada real
    assert dec.filter(pl.col("group") == "no_contacts")["retire"].is_not_null().all()


def test_select_records_constant_floor_cost_and_predictions(frozen):
    fm = pl.read_csv(frozen / "fold_metrics.csv").filter(pl.col("arm") == "grid")
    const = fm.filter(pl.col("model") == "constant")
    assert const.height == 3 and const["gini_stability"].abs().max() < 1e-9
    assert (const["pr_auc"] > 0).all()
    for col in ("pr_auc", "fit_seconds", "predict_seconds", "peak_rss_mib"):
        assert col in fm.columns and fm[col].null_count() == 0, col
    assert set(json.loads((frozen / "selection.json").read_text())["selected"]) == {"logistic", "lightgbm"}
    fp = pl.read_parquet(frozen / "fold_predictions.parquet")
    assert fp.schema["score"] == pl.Float32 and set(fp["fold"]) == {"P1", "P2", "P3"}
    assert {"constant", "logistic", "lightgbm"} == set(fp["model"].cast(pl.String))
    n_val = sum(fm.filter((pl.col("model") == "logistic") & (pl.col("fold") == f))["n"][0] for f in ("P1", "P2", "P3"))
    assert fp.filter(pl.col("model") == "logistic").height == n_val
    res = json.loads((frozen / "config.json").read_text())["resources"]
    assert res["select"]["peak_rss_mib"] > 0 and res["select"]["wall_seconds"] > 0 and "ablate" in res


def test_oof_metrics_and_selection_inputs_are_recorded(frozen):
    oof = pl.read_csv(frozen / "oof_metrics.csv")
    for col in ("oof_gini_stability", "mean_weekly_gini", "slope", "residual_std", "std_weekly_gini", "auc", "pr_auc"):
        assert col in oof.columns and oof[col].null_count() == 0, col
    grid = oof.filter(pl.col("stage") == "select")
    assert {"constant", "logistic", "lightgbm"} == set(grid["model"])
    assert grid.filter(pl.col("model") == "constant")["oof_gini_stability"].abs().max() < 1e-9
    assert (grid["n_weeks"] == 34).all()  # semanas 40-73 contiguas
    abl = oof.filter(pl.col("stage") == "ablate")
    assert set(abl["group"]) == {"none", *rn.ABLATION_GROUPS}
    sel = json.loads((frozen / "selection.json").read_text())
    assert sel["rule_hash"] == rn.sel.RULE_HASH and "trace" in sel and "oof" in sel
    dec = pl.read_csv(frozen / "ablation_decisions.csv")
    for col in ("oof_gs_base", "oof_gs_without", "folds_mwg_not_worse", "oof_std_base", "oof_std_without", "rule_hash"):
        assert col in dec.columns, col


def test_group_constant_in_every_fit_block_is_reported_as_no_effect(frozen):
    abl = pl.read_csv(frozen / "ablation.csv")
    tax = abl.filter(pl.col("group") == "no_tax_registry_b_1")
    assert (tax["n_dropped"] == 1).all() and (tax["n_columns_removed"] == 0).all()
    assert abl.filter(pl.col("group") == "no_person_2")["n_columns_removed"].min() == 2
    dec = pl.read_csv(frozen / "ablation_decisions.csv").filter(pl.col("group") == "no_tax_registry_b_1")
    assert dec["note"].str.starts_with("group has no effect").all() and dec["note"].str.contains("constant").all()
    assert (dec["retire"] == False).all()  # noqa: E712 - no hay evidencia para retirar


def test_selection_made_under_another_rule_is_rejected(matrix_dir, frozen, tmp_path):
    out = _with_decisions(frozen, tmp_path, {})
    sel = json.loads((out / "selection.json").read_text())
    sel["rule_hash"] = "0" * 64
    (out / "selection.json").write_text(json.dumps(sel))
    with pytest.raises(rn.ProtocolError, match="rule"):
        rn.run(matrix_dir, out, "final", grids=TINY, shap_sample=300)


def test_absent_group_is_reported_not_fatal(matrix_dir, tmp_path):
    d = tmp_path / "dev"
    d.mkdir()
    for name in ("feature_matrix__train.parquet", "feature_matrix__train.manifest.json"):
        shutil.copy(matrix_dir / name, d / name)
    out = tmp_path / "o"
    rn.run(d, out, "select", grids=TINY)
    rn.run(d, out, "ablate", grids=TINY, groups={"ghost": ["nope__*"]})
    dec = pl.read_csv(out / "ablation_decisions.csv")
    assert dec["note"][0] == "group absent in matrix"


def test_contacts_group_uses_exact_column_name():
    assert rn.ABLATION_GROUPS["no_contacts"] == ["person_1__n_contact_persons__all"]


def test_final_requires_ablation_unless_explicitly_waived(matrix_dir, frozen, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(frozen, out)
    (out / "ablation_decisions.csv").unlink()
    with pytest.raises(rn.ProtocolError, match="ablat"):
        rn.run(matrix_dir, out, "final", grids=TINY, shap_sample=300, refit_variant=False)
    assert not (out / "final.done").exists()  # falla antes de leer el OOT
    rn.run(matrix_dir, out, "final", grids=TINY, shap_sample=300, refit_variant=False, no_ablation=True)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["final"]["no_ablation"] is True and cfg["final"]["excluded_columns"] == {"logistic": [], "lightgbm": []}


def test_final_applies_frozen_decisions_per_model(matrix_dir, frozen, tmp_path):
    out = _with_decisions(frozen, tmp_path, {"lightgbm": ["no_person_2"], "logistic": ["no_tax_registry_b_1", "no_contacts"]})
    rn.run(matrix_dir, out, "final", grids=TINY, shap_sample=300, refit_variant=False)
    cfg = json.loads((out / "config.json").read_text())["final"]
    assert set(cfg["excluded_columns"]["lightgbm"]) == {"person_2__n_records", "person_2__source_present"}
    assert set(cfg["excluded_columns"]["logistic"]) == {"tax_registry_b_1__n_records", "person_1__n_contact_persons__all"}
    shap = pl.read_csv(out / "shap_rank.csv")
    lgb_feats = set(shap.filter(pl.col("model") == "lightgbm")["feature"])
    lr_feats = set(shap.filter(pl.col("model") == "logistic")["feature"])
    assert not any(f.startswith("person_2__") for f in lgb_feats)
    assert any(f.startswith("person_2__") for f in lr_feats)
    assert not any(f.startswith("tax_registry_b_1__") for f in lr_feats)
    marker = json.loads((out / "final.done").read_text())
    assert marker["excluded_groups"] == {"lightgbm": ["no_person_2"], "logistic": ["no_contacts", "no_tax_registry_b_1"]}


def test_marker_blocks_second_final_and_force_logs(matrix_dir, frozen, tmp_path):
    out = _with_decisions(frozen, tmp_path, {})
    rn.run(matrix_dir, out, "final", grids=TINY, shap_sample=300, refit_variant=False)
    marker = json.loads((out / "final.done").read_text())
    assert marker["matrix_sha256"]["oot"] and marker["selection_sha256"] and marker["status"] == "completed"
    with pytest.raises(rn.ProtocolError, match="OOT"):
        rn.run(matrix_dir, out, "final", grids=TINY, shap_sample=300, refit_variant=False)
    assert not (out / "oot_reconsultations.log").exists()
    rn.run(matrix_dir, out, "final", grids=TINY, shap_sample=300, refit_variant=False, force_reconsult=True)
    log = (out / "oot_reconsultations.log").read_text().strip().splitlines()
    assert len(log) == 1 and "force-reconsult" in log[0]


def test_selection_missing_or_grid_mismatch_is_fatal(matrix_dir, frozen, tmp_path):
    out = _with_decisions(frozen, tmp_path, {})
    other = {**TINY, "logistic": [md.ModelConfig("logistic", {"C": 0.5}, 0)]}
    for stage in ("ablate", "final"):
        with pytest.raises(rn.ProtocolError, match="grid"):
            rn.run(matrix_dir, out, stage, grids=other, shap_sample=300)
    (out / "selection.json").unlink()
    for stage in ("ablate", "final"):
        with pytest.raises(rn.ProtocolError, match="selection"):
            rn.run(matrix_dir, out, stage, grids=TINY, shap_sample=300)
    assert not (out / "final.done").exists()


def test_full_pipeline_writes_all_outputs_with_in_sample_flags(matrix_dir, tmp_path):
    rn.run(matrix_dir, tmp_path, "all", grids=TINY, shap_sample=500)
    for name in ["config.json", "fold_metrics.csv", "weekly_gini.csv", "calibration.csv", "thresholds.csv",
                 "oot_metrics.csv", "segment_metrics.csv", "shap_rank.csv", "drift_psi.csv", "drift_alerts.csv",
                 "ablation.csv", "final.done"]:
        assert (tmp_path / name).exists(), name
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["seeds"]["model"] == 42 and cfg["matrix_sha256"]["oot"] and cfg["n_jobs"] == 8
    assert cfg["folds"]["oot"] == [74, 91]
    oot = pl.read_csv(tmp_path / "oot_metrics.csv")
    assert set(oot["variant"]) == {"p3_model", "refit_0_73"}
    for name in ("oot_metrics", "calibration", "thresholds", "segment_metrics"):
        t = pl.read_csv(tmp_path / f"{name}.csv")
        assert (t["in_sample"] == (t["period"] == "P3_val")).all(), name
    assert set(pl.read_csv(tmp_path / "thresholds.csv")["recall_target"]) == {0.7, 0.8, 0.9}
    assert {"limited_history", "rest", "all"} == set(pl.read_csv(tmp_path / "segment_metrics.csv")["segment"])
    assert {"P3_val", "OOT", "P3_fit"} == set(pl.read_csv(tmp_path / "shap_rank.csv")["period"])
    fm = pl.read_csv(tmp_path / "fold_metrics.csv")
    assert "lr_converged" in fm.columns
    # referencia constante en OOT: piso con Gini 0 y FPR a recall indefinido
    c = oot.filter((pl.col("model") == "constant") & (pl.col("variant") == "p3_model"))
    assert set(c["period"]) == {"P3_val", "OOT"} and (c["auc"] - 0.5).abs().max() < 1e-9
    assert (c["pr_auc"] > 0).all() and "pr_auc" in oot.columns
    th = pl.read_csv(tmp_path / "thresholds.csv").filter(pl.col("model") == "constant")
    assert th["fpr_fixed"].is_nan().all() and th["note"].str.contains("constant").all()
    assert "constant" not in set(pl.read_csv(tmp_path / "segment_metrics.csv")["model"])
    assert "pr_auc" in pl.read_csv(tmp_path / "weekly_gini.csv").columns
    # predicciones guardadas
    pr = pl.read_parquet(tmp_path / "predictions.parquet")
    assert pr.columns[:7] == ["case_id", "WEEK_NUM", "period", "model", "raw_score", "calibrated_score", "target"]
    assert pr.schema["raw_score"] == pl.Float32 and pr.schema["calibrated_score"] == pl.Float32
    assert {"constant", "logistic", "lightgbm", "logistic_refit_0_73", "lightgbm_refit_0_73"} == set(pr["model"].cast(pl.String))
    assert set(pr.filter(pl.col("model").cast(pl.String).str.contains("refit"))["period"].cast(pl.String)) == {"OOT"}
    n_oot = oot.filter((pl.col("model") == "logistic") & (pl.col("period") == "OOT") & (pl.col("variant") == "p3_model"))["n"][0]
    assert pr.filter((pl.col("model") == "lightgbm") & (pl.col("period") == "OOT")).height == n_oot
    assert pr["raw_score"].is_between(0, 1).all() and pr["calibrated_score"].is_between(0, 1).all()
    # estabilidad SHAP (K=5 semillas 42..46) por periodo
    st = pl.read_csv(tmp_path / "shap_stability.csv")
    per = st.filter(pl.col("level") == "period")
    assert set(per["period"]) == {"P3_val", "OOT"} and (per["n_samples"] == 5).all()
    for col in ("spearman_mean", "spearman_min", "jaccard_top20_mean", "jaccard_top20_min"):
        assert per[col].is_between(-1, 1).all(), col
    feat = st.filter((pl.col("level") == "feature") & (pl.col("period") == "OOT"))
    assert feat["rank_mean"].null_count() == 0 and feat["rank_std"].null_count() == 0
    cfg_final = json.loads((tmp_path / "config.json").read_text())["final"]
    assert cfg_final["shap_seeds"] == [42, 43, 44, 45, 46]
    assert cfg_final["cost"]["lightgbm"]["fit_seconds"] > 0 and "predict_seconds" in cfg_final["cost"]["logistic"]
    assert json.loads((tmp_path / "config.json").read_text())["resources"]["final"]["peak_rss_mib"] > 0


def test_final_outputs_do_not_depend_on_oot_labels_or_features(matrix_dir, frozen, tmp_path):
    """Calibrador, umbrales, SHAP de P3 val, deriva (top-20), selección y exclusiones: idénticos con OOT mutado."""
    mutated = tmp_path / "mut"
    shutil.copytree(matrix_dir, mutated)
    oot = pl.read_parquet(mutated / "feature_matrix__oot.parquet")
    rng = np.random.default_rng(7)
    oot = oot.with_columns(
        pl.Series("target", rng.integers(0, 2, oot.height)),
        (-pl.col("x1") * 3 + 1).alias("x1"),
    )
    oot.write_parquet(mutated / "feature_matrix__oot.parquet")
    outs = []
    for name, src in (("a", matrix_dir), ("b", mutated)):
        sub = tmp_path / f"run_{name}"
        sub.mkdir()
        out = _with_decisions(frozen, sub, {"lightgbm": ["no_person_2"]})
        rn.run(src, out, "final", grids=TINY, shap_sample=300, refit_variant=False)
        outs.append(out)
    a, b = outs

    def part(out, file, **flt):
        t = pl.read_csv(out / file)
        for k, v in flt.items():
            t = t.filter(pl.col(k) == v)
        return t

    for file in ("thresholds.csv", "calibration.csv", "shap_rank.csv", "shap_stability.csv"):
        x, y = part(a, file, period="P3_val"), part(b, file, period="P3_val")
        assert x.fill_nan(None).equals(y.fill_nan(None)), file  # NaN del piso constante cuenta como igual
    timing = ("fit_seconds", "predict_seconds")  # los tiempos no son deterministas
    xo = part(a, "oot_metrics.csv", period="P3_val").drop(*timing)
    assert xo.equals(part(b, "oot_metrics.csv", period="P3_val").drop(*timing))
    pa, pb = (pl.read_parquet(o / "predictions.parquet").filter(pl.col("period").cast(pl.String) == "P3_val") for o in (a, b))
    assert pa.equals(pb)
    # umbrales de P3 val que se aplican al OOT: mismo umbral fijo en ambas corridas
    ta, tb = part(a, "thresholds.csv", period="OOT"), part(b, "thresholds.csv", period="OOT")
    assert ta["threshold"].fill_nan(None).to_list() == tb["threshold"].fill_nan(None).to_list()
    assert ta["fpr_curve"].fill_nan(None).to_list() != tb["fpr_curve"].fill_nan(None).to_list()  # la mutación sí llegó al OOT
    feats = lambda out: sorted(pl.read_csv(out / "drift_psi.csv").filter(pl.col("series").str.starts_with("feature:"))["series"].unique())
    assert feats(a) == feats(b) and len(feats(a)) <= 20
    assert (a / "selection.json").read_bytes() == (b / "selection.json").read_bytes()
    ma, mb = (json.loads((o / "final.done").read_text()) for o in (a, b))
    assert ma["excluded_groups"] == mb["excluded_groups"] and ma["selection_sha256"] == mb["selection_sha256"]
    assert ma["matrix_sha256"]["oot"] != mb["matrix_sha256"]["oot"]


def test_columns_not_in_manifest_are_dropped_and_recorded(matrix_dir, tmp_path):
    d = tmp_path / "E5old"
    shutil.copytree(matrix_dir, d)
    t = pl.read_parquet(d / "feature_matrix__train.parquet").with_columns(old__n_records=pl.lit(1))
    t.write_parquet(d / "feature_matrix__train.parquet")
    m = pp.load_matrix(d)
    assert m.undeclared == ["old__n_records"] and "old__n_records" not in m.train.columns
    oot = pl.read_parquet(d / "feature_matrix__oot.parquet").with_columns(old__n_records=pl.lit(1))
    assert "old__n_records" not in pp.align_oot(oot, m.train).columns
