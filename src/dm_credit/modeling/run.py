"""CLI del modelado: select, ablate, final.

    python -m dm_credit.modeling.run --matrix-dir data/artifacts/preparation/E5-v2 \
        --out results/modeling/<run> [--stage select|ablate|final|all] [--sample-frac F]

El OOT solo se lee en la etapa ``final`` (una sola vez, con hiperparámetros congelados en select).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm
import numpy as np
import polars as pl
import sklearn

from dm_credit.modeling import calibration as cb
from dm_credit.modeling import drift as dr
from dm_credit.modeling import evaluation as ev
from dm_credit.modeling import explain as ex
from dm_credit.modeling import folds as fd
from dm_credit.modeling import metrics as mt
from dm_credit.modeling import models as md
from dm_credit.modeling import pipeline as pp
from dm_credit.modeling import selection as sel

MODELS = ("logistic", "lightgbm")
SEED = md.SEED
SHAP_SAMPLE = 20000
TOP_FEATURES = 20
ADWIN_DELTA = 0.002

# Pruebas de quitar variables (protocolo). Un grupo ausente en la matriz se omite sin fallar.
ABLATION_GROUPS = {
    "no_coverage": ["*__n_records", "*__source_present"],
    "no_person_2": ["person_2__*"],
    "no_contacts": ["person_1__n_contact_persons__all"],
    "no_tax_registry_b_1": ["tax_registry_b_1__*"],
}


class ProtocolError(RuntimeError):
    """Se violaría una regla del protocolo (OOT una sola vez, selección congelada, ablación aplicada)."""


def grid_hash(grids) -> str:
    payload = json.dumps({m: [asdict(c) for c in grids[m]] for m in sorted(grids)}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _peak_rss_mib() -> float:
    """Pico de memoria residente del proceso hasta ahora (ru_maxrss: bytes en macOS, KiB en Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1 << 20) if sys.platform == "darwin" else peak / 1024, 1)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _upsert(path: Path, df: pl.DataFrame, stage: str) -> None:
    """Escribe ``df`` con columna ``stage``, reemplazando solo las filas previas de esa etapa."""
    df = df.with_columns(pl.lit(stage).alias("stage"))
    if path.exists():
        old = pl.read_csv(path, infer_schema_length=0)
        old = old.filter(pl.col("stage") != stage)
        if old.height:
            df = pl.concat([old.cast({c: df.schema[c] for c in old.columns if c in df.schema}, strict=False), df], how="diagonal_relaxed")
    df.write_csv(path)


def _grids(grids) -> dict:
    return grids or {m: md.grid(m) for m in MODELS}


def _metric_row(model, cfg, fold, y, p, weeks, extra=None) -> dict:
    row = {"model": model, "config_id": cfg.config_id, "complexity": cfg.complexity, "fold": fold}
    row.update(mt.summarize(y, p, weeks))
    row.update(extra or {})
    return row


class _OOF:
    """Reúne las predicciones de validación de P1-P3 por (grupo, modelo, config) para la serie fuera de partición."""

    def __init__(self):
        self.parts, self.meta = {}, {}

    def add(self, group, model, cfg, y, p, weeks):
        key = (group, model, cfg.config_id)
        self.meta[key] = cfg.complexity
        self.parts.setdefault(key, []).append((np.asarray(y), np.asarray(p, float), np.asarray(weeks)))

    def summary(self, group, model, config_id) -> dict:
        y, p, w = (np.concatenate(x) for x in zip(*self.parts[(group, model, config_id)]))
        return mt.oof_summary(y, p, w)

    def frame(self) -> pl.DataFrame:
        rows = [{"group": g, "arm": "grid" if g == "grid" else ("with" if g == "none" else "without"), "model": m,
                 "config_id": c, "complexity": self.meta[(g, m, c)], **self.summary(g, m, c)} for (g, m, c) in self.parts]
        return pl.DataFrame(rows)


def _eval_models(ctx, grids, models=MODELS, tag=None, collect=None, oof=None, oof_group="grid"):
    """Entrena cada config en el bloque de ajuste y mide en validación. Un entrenamiento LGBM cubre los prefijos."""
    rows, weekly = [], []
    for model in models:
        cfgs = grids[model]
        groups: dict[str, list] = {}
        for c in cfgs:
            key = json.dumps({k: v for k, v in c.params.items() if k != "n_estimators"}, sort_keys=True)
            groups.setdefault(key if model == "lightgbm" else c.config_id, []).append(c)
        for members in groups.values():
            t0 = time.perf_counter()
            if model == "lightgbm":
                trees = tuple(sorted(c.params["n_estimators"] for c in members))
                preds, fitted = md.fit_predict(members[0], ctx, ["val"], n_estimators_eval=trees, return_model=True)
                get = lambda c: preds[("val", c.params["n_estimators"])]
            else:
                preds, fitted = md.fit_predict(members[0], ctx, ["val"], return_model=True)
                get = lambda c: preds["val"]
            for c in members:
                p = get(c)
                info = fitted.info or {}
                # fit_seconds: entrenamiento; en LGBM es el de la corrida compartida (máx. de árboles)
                extra = {"n_fit": len(ctx.y_fit), "n_features": len(fitted.names),
                         "fit_seconds": round(info["fit_seconds"], 2), "predict_seconds": round(info["predict_seconds"], 2),
                         "peak_rss_mib": _peak_rss_mib(),
                         "lr_n_iter": info.get("n_iter"), "lr_converged": info.get("converged")}
                if tag:
                    extra.update(tag)
                rows.append(_metric_row(model, c, ctx.fold.name, ctx.y_val, p, ctx.val_weeks, extra))
                wt = mt.weekly_table(ctx.y_val, p, ctx.val_weeks).with_columns(
                    pl.lit(model).alias("model"), pl.lit(c.config_id).alias("config_id"), pl.lit(ctx.fold.name).alias("fold")
                )
                weekly.append(wt)
                if collect is not None:
                    collect.append(_pred_frame(ctx, model, c.config_id, p))
                if oof is not None:
                    oof.add(oof_group, model, c, ctx.y_val, p, ctx.val_weeks)
            del fitted
    return rows, weekly


def _pred_frame(ctx, model, config_id, p) -> pl.DataFrame:
    v = ctx.raw["val"]
    return pl.DataFrame({
        "case_id": v["case_id"], "WEEK_NUM": v["WEEK_NUM"].cast(pl.Int16),
        "fold": pl.Series([ctx.fold.name] * v.height), "model": pl.Series([model] * v.height),
        "config_id": pl.Series([config_id] * v.height),
        "score": pl.Series(np.asarray(p, np.float32)), "target": pl.Series(ctx.y_val.astype(np.int8)),
    }).with_columns(pl.col("fold", "model", "config_id").cast(pl.Categorical))


def _eval_constant(ctx, collect=None, oof=None):
    """Piso del protocolo: tasa base del bloque de ajuste. No participa en la selección."""
    cfg = md.constant_config()
    preds, fitted = md.fit_predict(cfg, ctx, ["val"], return_model=True)
    extra = {"n_fit": len(ctx.y_fit), "n_features": 0, "fit_seconds": 0.0, "predict_seconds": 0.0,
             "peak_rss_mib": _peak_rss_mib(), "lr_n_iter": None, "lr_converged": None}
    row = _metric_row(md.CONSTANT, cfg, ctx.fold.name, ctx.y_val, preds["val"], ctx.val_weeks, extra)
    wt = mt.weekly_table(ctx.y_val, preds["val"], ctx.val_weeks).with_columns(
        pl.lit(md.CONSTANT).alias("model"), pl.lit(cfg.config_id).alias("config_id"), pl.lit(ctx.fold.name).alias("fold"))
    if collect is not None:
        collect.append(_pred_frame(ctx, md.CONSTANT, cfg.config_id, preds["val"]))
    if oof is not None:
        oof.add("grid", md.CONSTANT, cfg, ctx.y_val, preds["val"], ctx.val_weeks)
    return row, wt


def stage_select(matrix, out: Path, grids) -> dict:
    rows, weekly, preds, oof = [], [], [], _OOF()
    for fold in fd.FOLDS:
        _log(f"select {fold.name}: ajuste de vistas con semanas {fold.fit}")
        ctx = pp.prepare_fold(matrix.train, matrix.spec, fold)
        r, w = _eval_models(ctx, grids, collect=preds, oof=oof)
        rc, wc = _eval_constant(ctx, preds, oof)
        rows += r + [rc]
        weekly += w + [wc]
        del ctx
    table = pl.DataFrame(rows)
    _upsert(out / "fold_metrics.csv", table.with_columns(pl.lit("grid").alias("arm")), "select")
    _upsert(out / "weekly_gini.csv", pl.concat(weekly).with_columns(pl.lit("P_val").alias("period")), "select")
    pl.concat(preds).write_parquet(out / "fold_predictions.parquet", compression="zstd")
    oof_df = oof.frame()
    _upsert(out / "oof_metrics.csv", oof_df, "select")
    chosen, trace = {}, []
    for m in MODELS:  # constant: piso, nunca candidato
        if m in grids:
            chosen[m], tr = sel.select_config(oof_df, table, m, return_trace=True)
            trace += [{"model": m, **t} for t in tr]
    keep = ["model", "config_id", "complexity", "oof_gini_stability", "mean_weekly_gini", "slope", "residual_std", "std_weekly_gini", "auc", "pr_auc"]
    (out / "selection.json").write_text(json.dumps({
        "selected": chosen, "grid_hash": grid_hash(grids), "rule_hash": sel.RULE_HASH, "rule": sel.RULE_TEXT,
        "trace": trace, "oof": oof_df.select(keep).to_dicts(),
        "fold_mean_weekly_gini": table.select("model", "config_id", "fold", "mean_weekly_gini", "gini_stability").to_dicts(),
    }, indent=2))
    _log(f"selección: {chosen}")
    return chosen


def _selected(out: Path, grids) -> dict:
    """Configuración congelada por select. Falla si falta o si la rejilla actual no es la de select."""
    path = out / "selection.json"
    if not path.exists():
        raise ProtocolError(f"selection.json not found in {out}: run the select stage first")
    data = json.loads(path.read_text())
    if data.get("grid_hash") != grid_hash(grids):
        raise ProtocolError("grid hash differs from the one stored in selection.json: the grids changed after select")
    if data.get("rule_hash") != sel.RULE_HASH:
        raise ProtocolError("selection rule hash differs from the one stored in selection.json: it was made under another rule")
    by_id = {c.config_id: c for m in grids for c in grids[m]}
    missing = {m: i for m, i in data["selected"].items() if i not in by_id}
    if missing or set(data["selected"]) != set(grids):
        raise ProtocolError(f"selected configs are not in the current grids: {missing or sorted(set(grids) ^ set(data['selected']))}")
    return {m: by_id[i] for m, i in data["selected"].items()}


def stage_ablate(matrix, out: Path, grids, groups=None) -> None:
    groups = groups or ABLATION_GROUPS
    cfg = _selected(out, grids)
    one = {m: [cfg[m]] for m in cfg}
    rows, skipped, oof, matched = [], [], _OOF(), {}
    for fold in fd.FOLDS:
        _log(f"ablate {fold.name}: arm con todas las variables")
        ctx = pp.prepare_fold(matrix.train, matrix.spec, fold)
        r, _ = _eval_models(ctx, one, tag={"group": "none", "arm": "with", "n_dropped": 0, "n_columns_removed": 0},
                            oof=oof, oof_group="none")
        rows += r
        constant_in_fit = set(ctx.views.dropped_features)  # las vistas descartan las constantes del bloque de ajuste
        del ctx
        for name, patterns in groups.items():
            df, spec, dropped = pp.exclude_columns(matrix.train, matrix.spec, patterns)
            if not dropped:
                skipped.append(name)
                continue
            effective = sorted(set(dropped) - constant_in_fit)  # columnas que el modelo "con" sí usaba
            matched.setdefault(name, {"dropped": sorted(dropped), "removed": []})["removed"].append(effective)
            _log(f"ablate {fold.name}: sin {name} ({len(dropped)} columnas, {len(effective)} efectivas)")
            ctx = pp.prepare_fold(df, spec, fold)
            r, _ = _eval_models(ctx, one, tag={"group": name, "arm": "without", "n_dropped": len(dropped),
                                               "n_columns_removed": len(effective)}, oof=oof, oof_group=name)
            rows += r
            del ctx, df
    table = pl.DataFrame(rows)
    decisions = []
    for name in groups:
        if name not in matched:
            continue
        info = matched[name]
        no_effect = all(len(r) == 0 for r in info["removed"])
        for model in cfg:
            cid = cfg[model].config_id
            base_f = table.filter((pl.col("group") == "none") & (pl.col("model") == model)).rename({"config_id": "config_id"})
            w_f = table.filter((pl.col("group") == name) & (pl.col("model") == model))
            d = sel.ablation_decision(oof.summary("none", model, cid), base_f, oof.summary(name, model, cid), w_f)
            row = {"group": name, "model": model, "config_id": cid, "n_columns_matched": len(info["dropped"]),
                   "n_columns_removed_max": max(len(r) for r in info["removed"]), **d, "note": None, "rule_hash": sel.RULE_HASH}
            if no_effect:
                row["retire"] = False
                row["note"] = (f"group has no effect: all {len(info['dropped'])} matched columns are constant in every fit block "
                               f"and the views already drop them: {', '.join(info['dropped'])}")
            decisions.append(row)
    for name in sorted(set(skipped) - set(matched)):
        decisions.append({"group": name, "model": "all", "config_id": "", "retire": None, "note": "group absent in matrix",
                          "rule_hash": sel.RULE_HASH})
    _upsert(out / "ablation.csv", table, "ablate")
    _upsert(out / "oof_metrics.csv", oof.frame(), "ablate")
    pl.from_dicts(decisions, infer_schema_length=None).write_csv(out / "ablation_decisions.csv")


def _calibrated_extra(y, p, cal, base_rate) -> dict:
    q = cal.predict(p)
    return {
        "cal_brier": mt.brier(y, q),
        "cal_logloss": mt.logloss(y, q),
        "cal_ece": mt.ece(y, q),
        "cal_ratio": mt.calibration_ratio(y, q),
        "brier_base_rate": mt.brier(y, np.full(len(y), base_rate)),
        "ratio_rel_dev": abs(mt.calibration_ratio(y, q) - 1.0),
    }


def _excluded_groups(out: Path, cfgs, groups, no_ablation: bool) -> dict:
    """Grupos con retire=True por modelo, leídos de ablation_decisions.csv (congelado por ablate)."""
    path = out / "ablation_decisions.csv"
    excluded = {m: [] for m in cfgs}
    if no_ablation:
        _log("final: --no-ablation, no se excluye ningún grupo")
        return excluded
    if not path.exists():
        raise ProtocolError("ablation_decisions.csv not found: run the ablate stage first or pass --no-ablation explicitly")
    dec = pl.read_csv(path)
    if "rule_hash" not in dec.columns or set(dec["rule_hash"].drop_nulls()) != {sel.RULE_HASH}:
        raise ProtocolError("ablation decisions were made under another rule (rule_hash differs): re-run the ablate stage")
    for r in dec.iter_rows(named=True):
        if r["model"] not in cfgs:
            continue
        if r["config_id"] != cfgs[r["model"]].config_id:
            raise ProtocolError("ablation decisions were made with a different config than the selected one")
        if r["retire"]:
            if r["group"] not in groups:
                raise ProtocolError(f"retired group {r['group']!r} is unknown")
            excluded[r["model"]].append(r["group"])
    return {m: sorted(g) for m, g in excluded.items()}


def _check_marker(out: Path, force: bool) -> None:
    marker = out / "final.done"
    if marker.exists():
        if not force:
            raise ProtocolError(f"{marker} exists: the OOT was already consulted. Use --force-reconsult to re-run (it is logged)")
        with (out / "oot_reconsultations.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} force-reconsult previous_marker={marker.read_text().replace(chr(10), ' ')}\n")


def _write_marker(out: Path, matrix, selection_sha, excluded, status) -> dict:
    marker = {
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "matrix_sha256": matrix.hashes,
        "selection_sha256": selection_sha,
        "excluded_groups": excluded,
    }
    (out / "final.done").write_text(json.dumps(marker, indent=2))
    return marker


def _psi_series(ctx, model, fitted, scores, top_features):
    """PSI semanal del score (y de las variables top-20 en LightGBM). Referencias: fit 50-59 y val 60-73."""
    ref_mask = (ctx.fit_weeks >= 50) & (ctx.fit_weeks <= 59)
    weeks_all = np.concatenate([ctx.val_weeks, ctx.weeks("oot")])
    rows = []

    def add(series, ref_name, ref, cur):
        rows.append(dr.weekly_psi(ref, cur, weeks_all).with_columns(pl.lit(series).alias("series"), pl.lit(ref_name).alias("reference")))

    Xf, names, _ = ctx.matrix(fitted.view, "fit")
    cur = np.concatenate([scores["val"], scores["oot"]])
    add(f"score:{model}", "fit_50_59", fitted.predict(Xf[ref_mask]), cur)
    add(f"score:{model}", "val_60_73", scores["val"], cur)
    if top_features:
        Xv, _, _ = ctx.matrix(fitted.view, "val")
        Xo, _, _ = ctx.matrix(fitted.view, "oot")
        for f in top_features:
            j = names.index(f)
            cur = np.concatenate([Xv[:, j], Xo[:, j]])
            add(f"feature:{f}", "fit_50_59", Xf[ref_mask, j], cur)
            add(f"feature:{f}", "val_60_73", Xv[:, j], cur)
        del Xv, Xo
    del Xf
    return rows


def _alerts(out: Path, models, psi_df: pl.DataFrame) -> None:
    """ADWIN sobre log loss semanal (val + OOT) y PSI de score >= 0.25, con Gini y calibración de esa semana."""
    weekly = pl.read_csv(out / "weekly_gini.csv").filter(pl.col("stage") == "final")
    alerts = []

    def row_for(w, wk, detector, model, value):
        row = w.filter(pl.col("WEEK_NUM") == wk).row(0, named=True)
        return {"model": model, "detector": detector, "WEEK_NUM": wk, "value": value, "gini": row["gini"],
                "mean_pred": row["mean_pred"], "obs_rate": row["obs_rate"],
                "calib_ratio": row["mean_pred"] / row["obs_rate"] if row["obs_rate"] else None}

    for model in models:
        w = weekly.filter((pl.col("model") == model) & (pl.col("variant") == "p3_model")).sort("WEEK_NUM")
        for wk in dr.adwin_alerts(w["WEEK_NUM"].to_list(), w["logloss"].to_list(), ADWIN_DELTA):
            alerts.append(row_for(w, wk, f"adwin_logloss_delta{ADWIN_DELTA}", model, w.filter(pl.col("WEEK_NUM") == wk)["logloss"][0]))
        s = psi_df.filter((pl.col("series") == f"score:{model}") & (pl.col("reference") == "val_60_73") & (pl.col("psi") >= dr.PSI_ALERT))
        for r in s.iter_rows(named=True):
            alerts.append(row_for(w, r["WEEK_NUM"], "psi_score_ref_val_60_73>=0.25", model, r["psi"]))
    schema = {"model": pl.String, "detector": pl.String, "WEEK_NUM": pl.Int64, "value": pl.Float64, "gini": pl.Float64,
              "mean_pred": pl.Float64, "obs_rate": pl.Float64, "calib_ratio": pl.Float64}
    pl.DataFrame(alerts, schema=schema).write_csv(out / "drift_alerts.csv")


class _Acc:
    """Acumuladores de salidas de la etapa final."""

    def __init__(self):
        self.oot_rows, self.weekly, self.calib, self.thr_rows = [], [], [], []
        self.seg_rows, self.shap_rows, self.stab_rows, self.psi_rows, self.preds = [], [], [], [], []
        self.cost = {}


def _record(acc, label, variant, cfg_id, preds, cal, thresholds, periods, base_rate, seg_masks, *, constant=False, full=True, timing=None):
    """Registra métricas, calibración, umbrales, segmento y predicciones de un modelo (o variante)."""
    timing = timing or {"fit_seconds": 0.0, "predict_seconds": 0.0}
    for period, (block, y, wk, ids) in periods.items():
        p = preds[block]
        q = cal.predict(p)
        acc.oot_rows.append({"model": label if variant == "p3_model" else label.removesuffix("_refit_0_73"), "variant": variant,
                             "period": period, "config_id": cfg_id, **mt.summarize(y, p, wk),
                             **_calibrated_extra(y, p, cal, base_rate),
                             "fit_seconds": round(timing["fit_seconds"], 2), "predict_seconds": round(timing["predict_seconds"], 2)})
        acc.preds.append(pl.DataFrame({
            "case_id": ids, "WEEK_NUM": pl.Series(wk).cast(pl.Int16),
            "period": pl.Series([period] * len(p)), "model": pl.Series([label] * len(p)),
            "raw_score": pl.Series(np.asarray(p, np.float32)), "calibrated_score": pl.Series(np.asarray(q, np.float32)),
            "target": pl.Series(np.asarray(y, np.int8)), "variant": pl.Series([variant] * len(p)),
        }).with_columns(pl.col("period", "model", "variant").cast(pl.Categorical)))
        model_name = label.removesuffix("_refit_0_73")
        if full:
            wt = mt.weekly_table(y, p, wk).with_columns(
                pl.lit(model_name).alias("model"), pl.lit(period).alias("period"), pl.lit(variant).alias("variant"))
            wq = mt.weekly_table(y, q, wk).select("WEEK_NUM", pl.col("mean_pred").alias("mean_pred_cal"))
            acc.weekly.append(wt.join(wq, on="WEEK_NUM"))
            for kind, sc in (("raw", p), ("isotonic", q)):
                acc.calib.append(mt.calibration_table(y, sc).with_columns(
                    pl.lit(model_name).alias("model"), pl.lit(period).alias("period"), pl.lit(kind).alias("calibration"),
                    pl.lit(variant).alias("variant")))
        rows = ev.undefined_threshold_rows(tuple(thresholds)) if constant else ev.threshold_metrics(y, p, thresholds)
        for r in rows:
            acc.thr_rows.append({"model": model_name, "variant": variant, "period": period,
                                 "threshold_source": "P3_val" if variant == "p3_model" else "P3_val_of_p3_model", **r})
        if full and not constant and seg_masks[period] is not None:
            for r in ev.segment_metrics(y, p, seg_masks[period], thresholds):
                acc.seg_rows.append({"model": model_name, "period": period, **r})


def _shap_outputs(acc, fitted, ctx, periods, shap_sample):
    """SHAP global (semilla 42, shap_rank) y estabilidad en K muestras independientes (shap_stability)."""
    stab = {}
    for period, (block, _, _, _) in periods.items():
        X, _, _ = ctx.matrix("tree", block)
        stab[period] = ex.shap_stability(fitted, X, seeds=ex.SHAP_SEEDS, n_sample=shap_sample, top_k=TOP_FEATURES)
        del X
    sv = {period: r["tables"][0] for period, r in stab.items()}  # semilla 42
    change = ex.rank_change(sv["P3_val"], sv["OOT"])
    for period, t in sv.items():
        for r in t.iter_rows(named=True):
            rc = change.filter(pl.col("feature") == r["feature"])["rank_change"][0] if period == "OOT" else None
            acc.shap_rows.append({"model": "lightgbm", "period": period, **r, "coef": None, "std_coef": None, "rank_change": rc})
    for period, r in stab.items():
        sm = r["summary"]
        base = {"model": "lightgbm", "period": period, "n_samples": sm["n_samples"], "n_rows": sm["n_rows"],
                "sample_size": min(shap_sample, sm["n_rows"]), "exhaustive": sm["exhaustive"],
                "seeds": ",".join(map(str, ex.SHAP_SEEDS))}
        acc.stab_rows.append({**base, "level": "period", "feature": None, "spearman_mean": sm["spearman_mean"],
                              "spearman_min": sm["spearman_min"], "jaccard_top20_mean": sm["jaccard_mean"],
                              "jaccard_top20_min": sm["jaccard_min"], "rank_mean": None, "rank_std": None})
        for f in sm["per_feature"].iter_rows(named=True):
            acc.stab_rows.append({**base, "level": "feature", "feature": f["feature"], "spearman_mean": None,
                                  "spearman_min": None, "jaccard_top20_mean": None, "jaccard_top20_min": None,
                                  "rank_mean": f["rank_mean"], "rank_std": f["rank_std"]})
    return sv["P3_val"].head(TOP_FEATURES)["feature"].to_list()  # deriva: top-20 de P3 val, sin OOT


def stage_final(matrix, matrix_dir, out: Path, grids, *, sample_frac=None, groups=None, shap_sample=SHAP_SAMPLE,
                refit_variant=True, no_ablation=False, force_reconsult=False) -> dict:
    """Única consulta del OOT: modelo P3, calibrador y umbrales de 60-73, exclusiones de ablación congeladas."""
    groups = groups or ABLATION_GROUPS
    _check_marker(out, force_reconsult)  # antes de leer el OOT
    cfgs = _selected(out, grids)
    excluded = _excluded_groups(out, cfgs, groups, no_ablation)
    selection_sha = hashlib.sha256((out / "selection.json").read_bytes()).hexdigest()
    oot_raw, matrix.hashes["oot"] = pp.load_oot(matrix_dir, sample_frac=sample_frac, seed=SEED)
    oot_full = pp.align_oot(oot_raw, matrix.train)
    _write_marker(out, matrix, selection_sha, excluded, "started")  # si algo falla, el OOT ya fue consultado

    fold = fd.get_fold("P3")
    # el segmento usa las banderas de la matriz completa, aunque un grupo de cobertura se excluya del modelo
    val_full = matrix.train.filter(pl.col("WEEK_NUM").is_between(*fold.val))
    seg_masks = {"P3_val": ev.limited_history_mask(val_full), "OOT": ev.limited_history_mask(oot_full)}
    del val_full

    acc = _Acc()
    excluded_columns, lr_info = {}, {}
    constant_done = False
    for model, cfg in cfgs.items():
        patterns = [p for g in excluded[model] for p in groups[g]]
        df, spec, dropped = pp.exclude_columns(matrix.train, matrix.spec, patterns)
        excluded_columns[model] = sorted(dropped)
        _log(f"final: {model} {cfg.config_id}; excluye {len(dropped)} columnas {excluded_groups_str(excluded[model])}")
        oot = oot_full.select([c for c in oot_full.columns if c in df.columns])
        ctx = pp.prepare_fold(df, spec, fold, oot=oot)
        periods = {"P3_val": ("val", ctx.y_val, ctx.val_weeks, ctx.raw["val"]["case_id"]),
                   "OOT": ("oot", ctx.y("oot"), ctx.weeks("oot"), ctx.raw["oot"]["case_id"])}
        base_rate = float(ctx.y_fit.mean())
        if not constant_done:  # y y semanas son las mismas para todos los modelos: el piso usa el primer contexto
            cc = md.constant_config()
            cp, cm = md.fit_predict(cc, ctx, ["val", "oot"], return_model=True)
            _record(acc, md.CONSTANT, "p3_model", cc.config_id, cp, cb.fit_calibrator(ctx.y_val, cp["val"]),
                    ev.fit_thresholds_constant(), periods, base_rate, seg_masks, constant=True, timing=cm.info)
            acc.cost[md.CONSTANT] = cm.info
            constant_done = True
        preds, fitted = md.fit_predict(cfg, ctx, ["val", "oot"], return_model=True)
        acc.cost[model] = dict(fitted.info)
        if "n_iter" in fitted.info:
            lr_info[model] = {"n_iter": fitted.info["n_iter"], "converged": fitted.info["converged"]}
        cal = cb.fit_calibrator(ctx.y_val, preds["val"])
        thresholds = ev.fit_thresholds(ctx.y_val, preds["val"])
        _record(acc, model, "p3_model", cfg.config_id, preds, cal, thresholds, periods, base_rate, seg_masks, timing=fitted.info)
        top = None
        if model == "lightgbm":
            top = _shap_outputs(acc, fitted, ctx, periods, shap_sample)
        else:
            Xf, _, _ = ctx.matrix("linear", "fit")
            t = ex.logistic_std_coefs(fitted, Xf)
            del Xf
            for i, r in enumerate(t.iter_rows(named=True), 1):
                acc.shap_rows.append({"model": model, "period": "P3_fit", "feature": r["feature"], "mean_abs_shap": None,
                                      "rank": i, "coef": r["coef"], "std_coef": r["std_coef"], "rank_change": None})
        acc.psi_rows += _psi_series(ctx, model, fitted, {"val": preds["val"], "oot": preds["oot"]}, top)
        oot_period = {"OOT": periods["OOT"]}
        del ctx, fitted

        if refit_variant:
            _log(f"final: {model} variante reajuste 0-73 sin recalibrar (contraste, no es el modelo final)")
            refit_fold = fd.Fold("REFIT", fd.DEV_WEEKS, (fd.OOT_WEEKS[0], fd.OOT_WEEKS[0] - 1))
            rctx = pp.prepare_fold(df, spec, refit_fold, oot=oot)
            rp, rm = md.fit_predict(cfg, rctx, ["oot"], return_model=True)
            _record(acc, f"{model}_refit_0_73", "refit_0_73", cfg.config_id, rp, cal, thresholds, oot_period,
                    float(rctx.y_fit.mean()), seg_masks, full=False, timing=rm.info)
            acc.cost[f"{model}_refit_0_73"] = dict(rm.info)
            del rctx
        del df, oot

    _write_final(out, acc)
    psi_df = pl.concat(acc.psi_rows)
    psi_df.write_csv(out / "drift_psi.csv")
    _alerts(out, list(cfgs), psi_df)
    marker = _write_marker(out, matrix, selection_sha, excluded, "completed")
    cost = {m: {k: round(float(v), 2) for k, v in info.items() if k.endswith("_seconds")} for m, info in acc.cost.items()}
    return {"no_ablation": no_ablation, "excluded_groups": excluded, "excluded_columns": excluded_columns,
            "logistic_fit": lr_info, "cost": cost, "shap_seeds": list(ex.SHAP_SEEDS), "marker": marker}


def excluded_groups_str(groups) -> str:
    return f"({', '.join(groups)})" if groups else "(ninguno)"


def _flag(rows: list[dict]) -> list[dict]:
    """``in_sample``: en P3_val el calibrador y los umbrales se ajustaron con esas mismas filas."""
    return [{**r, "in_sample": r["period"] == "P3_val"} for r in rows]


def _write_final(out, acc) -> None:
    pl.DataFrame(_flag(acc.oot_rows)).write_csv(out / "oot_metrics.csv")
    _upsert(out / "weekly_gini.csv", pl.concat(acc.weekly, how="diagonal_relaxed"), "final")
    pl.concat(acc.calib).with_columns((pl.col("period") == "P3_val").alias("in_sample")).write_csv(out / "calibration.csv")
    pl.DataFrame(_flag(acc.thr_rows)).write_csv(out / "thresholds.csv")
    if acc.seg_rows:
        pl.DataFrame(_flag(acc.seg_rows)).write_csv(out / "segment_metrics.csv")
    else:
        pl.DataFrame({"note": ["limited-history flags absent in matrix"]}).write_csv(out / "segment_metrics.csv")
    pl.DataFrame(acc.shap_rows, infer_schema_length=None).write_csv(out / "shap_rank.csv")
    pl.DataFrame(acc.stab_rows, infer_schema_length=None).write_csv(out / "shap_stability.csv")
    pl.concat(acc.preds).write_parquet(out / "predictions.parquet", compression="zstd")


def write_config(out: Path, matrix, grids, stage, sample_frac, final_info=None, resources=None) -> None:
    path = out / "config.json"
    cfg = json.loads(path.read_text()) if path.exists() else {}
    cfg.update(
        seeds={"model": SEED, "shap_sample": SEED, "row_sample": SEED},
        grids={m: [asdict(c) for c in grids[m]] for m in grids},
        grid_hash=grid_hash(grids),
        matrix_sha256=matrix.hashes,
        folds=fd.fold_bounds(),
        sample_frac=sample_frac,
        shap_sample=SHAP_SAMPLE,
        n_jobs=md.N_JOBS,
        adwin_delta=ADWIN_DELTA,
        psi_references=["fit_50_59 (P3 fit block, in-sample for the score)", "val_60_73"],
        ablation_groups=ABLATION_GROUPS,
        versions={"python": platform.python_version(), "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__, "polars": pl.__version__},
        predictors=len(matrix.spec.columns),
        undeclared_columns_dropped=matrix.undeclared,
    )
    if final_info is not None:
        cfg["final"] = final_info
    if resources is not None:
        cfg.setdefault("resources", {})[stage] = resources
    cfg.setdefault("stages_run", [])
    if stage not in cfg["stages_run"]:
        cfg["stages_run"].append(stage)
    path.write_text(json.dumps(cfg, indent=2, default=str))


def run(matrix_dir, out, stage="all", sample_frac=None, grids=None, groups=None, shap_sample=SHAP_SAMPLE,
        refit_variant=True, no_ablation=False, force_reconsult=False, n_jobs=None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grids = _grids(grids)
    if n_jobs is not None:
        md.set_n_jobs(n_jobs)
    stages = ["select", "ablate", "final"] if stage == "all" else [stage]
    if stages[0] != "select":  # validar lo congelado antes de cargar nada pesado
        _selected(out, grids)
    if "final" in stages:
        _check_marker_dry(out, force_reconsult)
    matrix = pp.load_matrix(matrix_dir, include_oot=False, sample_frac=sample_frac, seed=SEED)
    for st in stages:
        t0 = time.perf_counter()
        info = None
        if st == "select":
            stage_select(matrix, out, grids)
        elif st == "ablate":
            _selected(out, grids)
            stage_ablate(matrix, out, grids, groups)
        else:
            info = stage_final(matrix, matrix_dir, out, grids, sample_frac=sample_frac, groups=groups,
                               shap_sample=shap_sample, refit_variant=refit_variant,
                               no_ablation=no_ablation, force_reconsult=force_reconsult)
        wall = time.perf_counter() - t0
        write_config(out, matrix, grids, st, sample_frac, info,
                     {"wall_seconds": round(wall, 1), "peak_rss_mib": _peak_rss_mib()})
        _log(f"etapa {st}: {wall:.0f}s, pico RSS {_peak_rss_mib():.0f} MiB")


def _check_marker_dry(out: Path, force: bool) -> None:
    if (out / "final.done").exists() and not force:
        raise ProtocolError(f"{out / 'final.done'} exists: the OOT was already consulted. Use --force-reconsult to re-run (it is logged)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=("select", "ablate", "final", "all"), default="all")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--no-refit-variant", action="store_true")
    ap.add_argument("--no-ablation", action="store_true", help="final sin aplicar exclusiones (queda registrado)")
    ap.add_argument("--force-reconsult", action="store_true", help="repite final aunque exista final.done (se registra en oot_reconsultations.log)")
    a = ap.parse_args()
    run(a.matrix_dir, a.out, a.stage, a.sample_frac, refit_variant=not a.no_refit_variant, no_ablation=a.no_ablation,
        force_reconsult=a.force_reconsult, n_jobs=a.n_jobs)


if __name__ == "__main__":
    main()
