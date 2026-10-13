import numpy as np
import polars as pl
import pytest

from dm_credit.preparation import leakage_screen as ls


def labeled(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    week = np.sort(rng.integers(0, 60, n))
    target = (rng.random(n) < 0.1).astype(int)
    return pl.DataFrame({"case_id": np.arange(n), "WEEK_NUM": week, "target": target})


def feats(lab, **cols):
    return pl.DataFrame({"case_id": lab["case_id"], **cols})


def row(out, name):
    return out.filter(pl.col("feature") == name).row(0, named=True)


def test_perfect_leak_is_flagged():
    lab = labeled()
    out = ls.screen_features(feats(lab, leak=lab["target"].cast(pl.Float64)), lab)
    r = row(out, "leak")
    assert r["gini_all"] == pytest.approx(1.0) and r["flag"] == "leak_suspect"


def test_noise_is_ok_and_reports_coverage():
    lab = labeled()
    rng = np.random.default_rng(1)
    out = ls.screen_features(feats(lab, noise=pl.Series(rng.normal(size=lab.height))), lab)
    r = row(out, "noise")
    assert abs(r["gini_all"]) < 0.1 and r["flag"] == "ok" and r["n_nonnull"] == lab.height


def test_legit_moderate_signal_is_not_flagged():
    lab = labeled()
    rng = np.random.default_rng(2)
    signal = lab["target"].to_numpy() * 0.8 + rng.normal(size=lab.height)  # AUC ~0.71, Gini ~0.4
    r = row(ls.screen_features(feats(lab, mod=pl.Series(signal)), lab), "mod")
    assert 0.2 < r["gini_all"] < 0.5 and r["flag"] == "ok"


def test_missingness_that_encodes_the_target_is_flagged():
    lab = labeled()
    v = pl.Series([None if t == 1 else 1.0 for t in lab["target"]], dtype=pl.Float64)
    r = row(ls.screen_features(feats(lab, absent=v), lab), "absent")
    assert abs(r["missing_gini"]) > 0.5 and r["flag"] == "leak_suspect"


def test_cases_without_source_count_as_missing():
    lab = labeled()
    partial = pl.DataFrame({"case_id": lab["case_id"].head(1000), "x": pl.Series(np.random.default_rng(3).normal(size=1000))})
    r = row(ls.screen_features(partial, lab), "x")
    assert r["n_nonnull"] == 1000 and r["pct_null"] == pytest.approx(75.0)


def test_sign_flip_across_time_blocks_is_unstable():
    lab = labeled(6000)
    rng = np.random.default_rng(4)
    t = lab["target"].to_numpy()
    early = lab["WEEK_NUM"].to_numpy() < 30
    x = np.where(early, t * 1.0, -t * 1.0) + rng.normal(scale=1.0, size=lab.height)
    r = row(ls.screen_features(feats(lab, flip=pl.Series(x)), lab), "flip")
    assert r["flag"] == "unstable"


def test_string_features_are_not_screened():
    lab = labeled()
    r = row(ls.screen_features(feats(lab, s=pl.Series(["a"] * lab.height)), lab), "s")
    assert r["flag"] == "not_screened" and r["gini_all"] is None


def test_too_few_rows_or_one_class_is_insufficient():
    lab = labeled(500)
    r = row(ls.screen_features(feats(lab, x=pl.Series(np.arange(lab.height, dtype=float))), lab, min_rows=1000), "x")
    assert r["flag"] == "insufficient"
    one = lab.with_columns(target=pl.lit(0))
    assert row(ls.screen_features(feats(one, x=pl.Series(np.arange(one.height, dtype=float))), one, min_rows=10), "x")["flag"] == "insufficient"


def test_constant_feature_has_zero_gini():
    lab = labeled()
    r = row(ls.screen_features(feats(lab, c=pl.Series([1.0] * lab.height)), lab), "c")
    assert r["gini_all"] == pytest.approx(0.0)


def test_bool_features_are_screened():
    lab = labeled()
    out = ls.screen_features(feats(lab, b=(lab["target"] == 1)), lab)
    assert row(out, "b")["flag"] == "leak_suspect"


def test_screen_never_uses_weeks_beyond_the_cutoff():
    lab = labeled().with_columns(WEEK_NUM=pl.lit(80))
    with pytest.raises(ValueError, match="74"):
        ls.screen_features(feats(lab, x=pl.Series(np.zeros(lab.height))), lab)


def test_output_is_deterministic_and_sorted_by_feature():
    lab = labeled()
    f = feats(lab, b=pl.Series(np.arange(lab.height, dtype=float)), a=pl.Series(np.ones(lab.height)))
    one, two = ls.screen_features(f, lab), ls.screen_features(f, lab)
    assert one.equals(two) and one["feature"].to_list() == ["a", "b"]


# ----------------------------------------------------------------- filtro sobre artefactos publicados
def test_screen_artifacts_reads_published_train_artifacts(tmp_path, monkeypatch):
    import preparation_artifacts as pa
    from dm_credit import io

    root = tmp_path / "data"
    pa.write_sample_data(root)
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    arts = pa.build_small_artifacts(tmp_path / "art")
    train = [a for a in arts if a.split == "train"]
    out = ls.screen_artifacts([a.manifest_path for a in train], io.load_base("train"), cutoff_week=pa.CUTOFF, min_rows=2)
    assert out["feature"].is_unique().all()
    assert any(f.startswith("applprev_1__") for f in out["feature"]) and any(f.startswith("person_1__") for f in out["feature"])


def test_screen_artifacts_rejects_non_train_artifacts(tmp_path, monkeypatch):
    import preparation_artifacts as pa
    from dm_credit import io

    root = tmp_path / "data"
    pa.write_sample_data(root)
    monkeypatch.setenv("DM_DATA_DIR", str(root))
    oot = next(a for a in pa.build_small_artifacts(tmp_path / "art") if a.split == "oot")
    with pytest.raises(ValueError, match="train"):
        ls.screen_artifacts([oot.manifest_path], io.load_base("train"), cutoff_week=pa.CUTOFF, min_rows=2)


def test_null_week_or_target_in_labeled_is_rejected_with_a_clear_error():
    lab = labeled()
    f = feats(lab, x=pl.Series(np.zeros(lab.height)))
    with pytest.raises(ValueError, match="WEEK_NUM"):
        ls.screen_features(f, lab.with_columns(pl.when(pl.col("case_id") == 0).then(None).otherwise(pl.col("WEEK_NUM")).alias("WEEK_NUM")))
    with pytest.raises(ValueError, match="target"):
        ls.screen_features(f, lab.with_columns(pl.when(pl.col("case_id") == 0).then(None).otherwise(pl.col("target")).alias("target")))
