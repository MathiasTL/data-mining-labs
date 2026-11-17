"""Phase 4 modeling run: logistic regression, random forest and LightGBM.

Design: fit on WEEK_NUM < 64, select/early-stop on 64..73, refit on < 74,
evaluate out of time (OOT) on WEEK_NUM >= 74. Writes CSV results and figures.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(sys.argv[1])  # directory holding E6/
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
SEED = 42
import os
SMOKE = os.environ.get("SMOKE") == "1"
CUT, VAL_START = 74, 64
CONTROL = ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"]
COLORS = {"Logística L2": "#2a78d6", "Random Forest": "#eb6834", "LightGBM": "#1baf7a"}
LOG: list[dict] = []


def log(**row) -> None:
    LOG.append(row)
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


# ---------------------------------------------------------------- data
def load(view: str):
    tr = pl.read_parquet(ROOT / f"E6/{view}__train.parquet")
    oot = pl.read_parquet(ROOT / f"E6/{view}__oot.parquet")
    feats = [c for c in tr.columns if c not in CONTROL]
    if SMOKE:
        tr = tr.filter(pl.col("case_id") % 20 == 0)
        oot = oot.filter(pl.col("case_id") % 4 == 0)
    return tr, oot, feats


def encode_tree(tr: pl.DataFrame, oot: pl.DataFrame, feats: list[str]):
    """Booleans -> 0/1, strings -> integer codes from the train vocabulary."""
    cats = []
    exprs_tr, exprs_oot = [], []
    for c in feats:
        dt = tr.schema[c]
        if dt == pl.Boolean:
            exprs_tr.append(pl.col(c).cast(pl.Float32))
            exprs_oot.append(pl.col(c).cast(pl.Float32))
        elif isinstance(dt, (pl.Enum, pl.Categorical)):
            e = pl.col(c).to_physical().cast(pl.Float32)
            exprs_tr.append(e)
            exprs_oot.append(e)
            cats.append(c)
        elif dt == pl.String:
            vocab = sorted(v for v in tr[c].unique().to_list() if v is not None)
            mapping = {v: i for i, v in enumerate(vocab)}
            e = pl.col(c).replace_strict(mapping, default=None, return_dtype=pl.Float32)
            exprs_tr.append(e)
            exprs_oot.append(e)
            cats.append(c)
        else:
            exprs_tr.append(pl.col(c).cast(pl.Float32))
            exprs_oot.append(pl.col(c).cast(pl.Float32))
    xtr = tr.select(exprs_tr).to_numpy()
    xoot = oot.select(exprs_oot).to_numpy()
    return xtr, xoot, cats


# ---------------------------------------------------------------- metrics
def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def weekly_gini(weeks, y, p) -> pl.DataFrame:
    rows = []
    for w in np.unique(weeks):
        m = weeks == w
        rows.append({"WEEK_NUM": int(w), "cases": int(m.sum()), "positives": int(y[m].sum()),
                     "gini": gini(y[m], p[m])})
    return pl.DataFrame(rows)


def gini_stability(wg: pl.DataFrame) -> tuple[float, float, float, float]:
    x = wg["WEEK_NUM"].to_numpy().astype(float)
    g = wg["gini"].to_numpy()
    a, b = np.polyfit(x, g, 1)
    res = g - (a * x + b)
    return float(g.mean() + 88 * min(0.0, a) - 0.5 * res.std()), float(g.mean()), float(a), float(res.std())


def at_recall(y, p, target=0.80):
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr >= target))
    t = thr[i]
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {"threshold": float(t), "recall": float(tpr[i]), "fpr": float(fpr[i]),
            "precision": tp / max(tp + fp, 1), "flagged_pct": float(pred.mean()),
            "false_positives": fp, "true_positives": tp}


def apply_threshold(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return {"recall": tp / max(int(y.sum()), 1), "fpr": fp / max(int((y == 0).sum()), 1),
            "precision": tp / max(tp + fp, 1), "flagged_pct": float(pred.mean()),
            "false_positives": fp, "true_positives": tp}


def calibration_table(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return pl.DataFrame({"bin": idx, "y": y, "p": p}).group_by("bin").agg(
        pl.len().alias("cases"), pl.col("p").mean().alias("mean_predicted"),
        pl.col("y").mean().alias("observed_rate")).sort("bin")


def psi(ref, cur, edges) -> float:
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, 1e-6, None); c = np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def adwin_detections(stream: np.ndarray, delta=0.002, check_every=500, min_side=2000):
    """ADWIN (Bifet & Gavalda, 2007) with prefix sums; returns indices of detected changes."""
    det, start = [], 0
    cs = np.concatenate([[0.0], np.cumsum(stream)])
    for end in range(min_side * 2, len(stream) + 1, check_every):
        n = end - start
        if n < 2 * min_side:
            continue
        cuts = np.arange(start + min_side, end - min_side + 1, check_every)
        n0 = cuts - start; n1 = end - cuts
        mu0 = (cs[cuts] - cs[start]) / n0; mu1 = (cs[end] - cs[cuts]) / n1
        var = stream[start:end].var()
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        dprime = delta / n
        eps = np.sqrt(2.0 / m * var * np.log(2.0 / dprime)) + 2.0 / (3.0 * m) * np.log(2.0 / dprime)
        hit = np.abs(mu0 - mu1) > eps
        if hit.any():
            det.append(int(end))
            start = int(cuts[np.argmax(hit)])
    return det


# ---------------------------------------------------------------- run
def main():
    t0 = time.time()
    results, weekly_all, curves = [], [], {}

    # ---------- logistic regression on the linear view
    tr, oot, feats_lin = load("linear")
    w_tr = tr["WEEK_NUM"].to_numpy(); y_tr = tr["target"].to_numpy()
    w_oot = oot["WEEK_NUM"].to_numpy(); y_oot = oot["target"].to_numpy()
    xtr = tr.select(feats_lin).to_numpy().astype(np.float32)
    xoot = oot.select(feats_lin).to_numpy().astype(np.float32)
    fit_m, val_m = w_tr < VAL_START, w_tr >= VAL_START
    best = None
    for C in [0.01, 0.1, 1.0]:
        s = time.time()
        m = LogisticRegression(C=C, max_iter=300, solver="lbfgs")
        m.fit(xtr[fit_m], y_tr[fit_m])
        auc = roc_auc_score(y_tr[val_m], m.predict_proba(xtr[val_m])[:, 1])
        log(stage="select", model="Logística L2", config=f"C={C}", val_auc=auc,
            iters=int(m.n_iter_[0]), seconds=round(time.time() - s, 1))
        if best is None or auc > best[1]:
            best = (C, auc)
    s = time.time()
    lr = LogisticRegression(C=best[0], max_iter=300, solver="lbfgs").fit(xtr, y_tr)
    p_lr_tr = lr.predict_proba(xtr)[:, 1]; p_lr = lr.predict_proba(xoot)[:, 1]
    p_lr_val = LogisticRegression(C=best[0], max_iter=300, solver="lbfgs").fit(
        xtr[fit_m], y_tr[fit_m]).predict_proba(xtr[val_m])[:, 1]
    log(stage="refit", model="Logística L2", config=f"C={best[0]}", seconds=round(time.time() - s, 1))
    scores = {"Logística L2": (p_lr_val, p_lr, p_lr_tr, f"C={best[0]}")}
    del xtr, xoot

    # ---------- tree view for RF and LightGBM
    tr, oot, feats_tree = load("tree")
    assert (tr["case_id"].to_numpy() == tr["case_id"].to_numpy()).all()
    xtr, xoot, cats = encode_tree(tr, oot, feats_tree)
    thin_oot = (oot["applprev_1__source_present"].cast(pl.String) == "false").to_numpy()
    w_tr = tr["WEEK_NUM"].to_numpy(); y_tr = tr["target"].to_numpy()
    fit_m, val_m = w_tr < VAL_START, w_tr >= VAL_START
    cat_idx = [feats_tree.index(c) for c in cats]
    base = dict(objective="binary", metric="auc", seed=SEED, verbose=-1, num_threads=8,
                min_child_samples=200, lambda_l2=1.0)

    def ds(mask):
        return lgb.Dataset(xtr[mask], y_tr[mask], feature_name=feats_tree,
                           categorical_feature=cat_idx, free_raw_data=False)

    dfit, dval = ds(fit_m), None
    dval = lgb.Dataset(xtr[val_m], y_tr[val_m], reference=dfit)
    dall = ds(np.ones_like(fit_m))

    # LightGBM
    best = None
    for leaves in [31, 63]:
        s = time.time()
        prm = {**base, "learning_rate": 0.05, "num_leaves": leaves, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 1}
        b = lgb.train(prm, dfit, 60 if SMOKE else 3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        auc = b.best_score["valid_0"]["auc"]
        log(stage="select", model="LightGBM", config=f"leaves={leaves}", val_auc=auc,
            trees=b.best_iteration, seconds=round(time.time() - s, 1))
        if best is None or auc > best[1]:
            best = (prm, auc, b.best_iteration, b)
    prm, _, n_trees, b_fit = best
    p_gb_val = b_fit.predict(xtr[val_m], num_iteration=n_trees)
    s = time.time()
    gb = lgb.train(prm, dall, n_trees)
    log(stage="refit", model="LightGBM", config=f"leaves={prm['num_leaves']}, trees={n_trees}",
        seconds=round(time.time() - s, 1))
    p_gb = gb.predict(xoot); p_gb_tr = gb.predict(xtr)
    scores["LightGBM"] = (p_gb_val, p_gb, p_gb_tr, f"{prm['num_leaves']} hojas, {n_trees} árboles")

    # Random forest (LightGBM rf mode: bagging without boosting)
    s = time.time()
    rf_prm = {**base, "boosting": "rf", "num_leaves": 255, "bagging_fraction": 0.632,
              "bagging_freq": 1, "feature_fraction": 0.5}
    rf_fit = lgb.train(rf_prm, dfit, 300)
    p_rf_val = rf_fit.predict(xtr[val_m])
    log(stage="select", model="Random Forest", config="255 hojas, 300 árboles",
        val_auc=roc_auc_score(y_tr[val_m], p_rf_val), seconds=round(time.time() - s, 1))
    s = time.time()
    rf = lgb.train(rf_prm, dall, 300)
    log(stage="refit", model="Random Forest", config="255 hojas, 300 árboles",
        seconds=round(time.time() - s, 1))
    scores["Random Forest"] = (p_rf_val, rf.predict(xoot), rf.predict(xtr), "255 hojas, 300 árboles")

    # ---------- evaluation
    y_val = y_tr[val_m]
    prev = float(y_oot.mean())
    for name in ["Logística L2", "Random Forest", "LightGBM"]:
        pv, po, ptr, cfg = scores[name]
        wg = weekly_gini(w_oot, y_oot, po).with_columns(pl.lit(name).alias("model"))
        weekly_all.append(wg)
        gs, gmean, slope, rstd = gini_stability(wg)
        thr_val = at_recall(y_val, pv)["threshold"]
        op_val = apply_threshold(y_oot, po, thr_val)
        op_eq = at_recall(y_oot, po)
        thin = at_recall(y_oot[thin_oot], po[thin_oot])
        results.append({
            "model": name, "config": cfg,
            "val_auc": roc_auc_score(y_val, pv), "train_auc": roc_auc_score(y_tr, ptr),
            "oot_auc": roc_auc_score(y_oot, po), "oot_gini": gini(y_oot, po),
            "gini_stability": gs, "weekly_gini_mean": gmean, "weekly_gini_slope": slope,
            "weekly_gini_resid_std": rstd,
            "weekly_gini_min": float(wg["gini"].min()), "weekly_gini_max": float(wg["gini"].max()),
            "oot_pr_auc": average_precision_score(y_oot, po), "oot_prevalence": prev,
            "oot_brier": brier_score_loss(y_oot, po), "oot_logloss": log_loss(y_oot, po),
            "oot_mean_score": float(po.mean()),
            "fpr_at_recall80": op_eq["fpr"], "precision_at_recall80": op_eq["precision"],
            "flagged_pct_at_recall80": op_eq["flagged_pct"], "fp_at_recall80": op_eq["false_positives"],
            "valthr_recall": op_val["recall"], "valthr_fpr": op_val["fpr"],
            "valthr_precision": op_val["precision"],
            "thin_cases": int(thin_oot.sum()), "thin_positives": int(y_oot[thin_oot].sum()),
            "thin_auc": roc_auc_score(y_oot[thin_oot], po[thin_oot]),
            "thin_fpr_at_recall80": thin["fpr"],
        })
        curves[name] = roc_curve(y_oot, po)
        calibration_table(y_oot, po).with_columns(pl.lit(name).alias("model")).write_csv(
            OUT / f"calibration_{name.split()[0].lower()}.csv")

    res = pl.DataFrame(results)
    res.write_csv(OUT / "model_comparison.csv")
    weekly = pl.concat(weekly_all)
    weekly.write_csv(OUT / "weekly_gini.csv")

    # ---------- threshold sweep (OM3)
    sweep = []
    for name in scores:
        fpr, tpr, _ = curves[name]
        for r in [0.5, 0.6, 0.7, 0.8, 0.9]:
            i = int(np.argmax(tpr >= r))
            sweep.append({"model": name, "recall": r, "fpr": float(fpr[i])})
    pl.DataFrame(sweep).write_csv(OUT / "threshold_sweep.csv")

    # ---------- SHAP-type contributions (OM2) on LightGBM
    rng = np.random.default_rng(SEED)
    i_tr = rng.choice(len(xtr), min(40000, len(xtr)), replace=False)
    i_oot = rng.choice(len(xoot), min(40000, len(xoot)), replace=False)
    c_tr = gb.predict(xtr[i_tr], pred_contrib=True)[:, :-1]
    c_oot = gb.predict(xoot[i_oot], pred_contrib=True)[:, :-1]
    imp = pl.DataFrame({"feature": feats_tree,
                        "mean_abs_train": np.abs(c_tr).mean(0),
                        "mean_abs_oot": np.abs(c_oot).mean(0),
                        "gain": gb.feature_importance("gain")})
    imp = imp.with_columns(
        pl.col("mean_abs_train").rank(descending=True).alias("rank_train"),
        pl.col("mean_abs_oot").rank(descending=True).alias("rank_oot"),
        (pl.col("mean_abs_oot") / pl.col("mean_abs_train") - 1).alias("rel_change"),
    ).sort("mean_abs_oot", descending=True)
    imp.write_csv(OUT / "shap_importance.csv")
    top_tr = set(imp.sort("rank_train").head(10)["feature"]); top_oot = set(imp.head(10)["feature"])
    rho = float(np.corrcoef(imp["rank_train"].to_numpy(), imp["rank_oot"].to_numpy())[0, 1])
    # example case: highest-scored OOT defaulter in the sample
    po_s = p_gb[i_oot]; ys = y_oot[i_oot]
    k = int(np.argmax(np.where(ys == 1, po_s, -1)))
    order = np.argsort(-np.abs(c_oot[k]))[:6]
    example = pl.DataFrame({"feature": [feats_tree[j] for j in order],
                            "value": [None if np.isnan(xoot[i_oot][k, j]) else float(xoot[i_oot][k, j]) for j in order],
                            "contribution_logodds": [float(c_oot[k, j]) for j in order]})
    example.write_csv(OUT / "shap_example_case.csv")
    log(stage="shap", top10_overlap=len(top_tr & top_oot), spearman_rank=rho,
        example_case_id=int(oot["case_id"].to_numpy()[i_oot][k]), example_score=float(po_s[k]))

    # ---------- drift and retraining windows (OM4)
    edges = np.unique(np.quantile(p_gb_tr, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    top_feats = imp.head(5)["feature"].to_list()
    drift = []
    for w in np.unique(w_oot):
        m = w_oot == w
        row = {"WEEK_NUM": int(w), "score_psi": psi(p_gb_tr, p_gb[m], edges)}
        for f in top_feats:
            j = feats_tree.index(f)
            ref = xtr[:, j]; cur = xoot[m, j]
            ref_nan, cur_nan = np.isnan(ref).mean(), np.isnan(cur).mean()
            ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
            fe = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
            fe[0], fe[-1] = -np.inf, np.inf
            row[f"psi__{f}"] = psi(ref, cur, fe) if len(cur) else float("nan")
            row[f"null_delta__{f}"] = float(cur_nan - ref_nan)
        drift.append(row)
    drift = pl.DataFrame(drift).join(weekly.filter(pl.col("model") == "LightGBM").select("WEEK_NUM", "gini"),
                                     on="WEEK_NUM")
    # ADWIN streams ordered by decision date: validation weeks then OOT
    dates = np.concatenate([tr["date_decision"].to_numpy()[val_m], oot["date_decision"].to_numpy()])
    weeks_s = np.concatenate([w_tr[val_m], w_oot])
    p_s = np.concatenate([gb.predict(xtr[val_m]), p_gb])
    y_s = np.concatenate([y_val, y_oot])
    o = np.argsort(dates, kind="stable")
    p_s, y_s, weeks_s = p_s[o], y_s[o], weeks_s[o]
    eps = 1e-6
    loss = -(y_s * np.log(np.clip(p_s, eps, 1)) + (1 - y_s) * np.log(np.clip(1 - p_s, eps, 1)))
    adw = []
    for stream_name, stream in [("score", p_s), ("logloss", loss)]:
        for d in adwin_detections(stream):
            adw.append({"stream": stream_name, "index": d, "WEEK_NUM": int(weeks_s[d - 1])})
    adw_df = pl.DataFrame(adw, schema={"stream": pl.String, "index": pl.Int64, "WEEK_NUM": pl.Int64})
    adw_df.write_csv(OUT / "adwin_detections.csv")
    drift.write_csv(OUT / "drift_weekly.csv")

    # ---------- figures
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name in ["Logística L2", "Random Forest", "LightGBM"]:
        d = weekly.filter(pl.col("model") == name)
        ax.plot(d["WEEK_NUM"], d["gini"], color=COLORS[name], lw=2, marker="o", ms=5, label=name)
    ax.set_xlabel("Semana (WEEK_NUM)"); ax.set_ylabel("Gini semanal")
    ax.set_title("Gini semanal fuera de tiempo (semanas 74–91)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "04_weekly_gini.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for name in scores:
        fpr, tpr, _ = curves[name]
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=name)
    ax.axhline(0.8, color="#8a8984", lw=1, ls="--"); ax.text(0.62, 0.765, "recall = 0,80", color="#52514e")
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1)
    ax.set_xlabel("Tasa de falsos positivos (FPR)"); ax.set_ylabel("Detección de incumplimientos (recall)")
    ax.set_title("Curva ROC en OOT", loc="left"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "04_roc_oot.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for name in scores:
        c = pl.read_csv(OUT / f"calibration_{name.split()[0].lower()}.csv")
        ax.plot(c["mean_predicted"], c["observed_rate"], color=COLORS[name], lw=2, marker="o", ms=5, label=name)
    lim = 0.12
    ax.plot([0, lim], [0, lim], color="#c3c2b7", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Probabilidad media predicha (decil)"); ax.set_ylabel("Tasa observada de incumplimiento")
    ax.set_title("Curva de confiabilidad en OOT", loc="left"); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "04_calibration_oot.png", dpi=200); plt.close(fig)

    top = imp.head(12).reverse()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    yy = np.arange(len(top))
    ax.barh(yy + 0.2, top["mean_abs_train"], height=0.38, color="#86b6ef", label="Entrenamiento")
    ax.barh(yy - 0.2, top["mean_abs_oot"], height=0.38, color="#2a78d6", label="OOT")
    ax.set_yticks(yy, [f.replace("__", " · ")[:48] for f in top["feature"]], fontsize=8.5)
    ax.set_xlabel("Contribución media absoluta (log-odds)")
    ax.set_title("Factores con mayor contribución (LightGBM)", loc="left")
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "04_shap_top.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(drift["WEEK_NUM"], drift["score_psi"], color="#2a78d6", width=0.7, label="PSI del score")
    ax.axhline(0.10, color="#eda100", lw=1.2, ls="--"); ax.text(73.6, 0.104, "0,10 vigilancia", color="#52514e", fontsize=9)
    ax.axhline(0.25, color="#e34948", lw=1.2, ls="--"); ax.text(73.6, 0.254, "0,25 alerta", color="#52514e", fontsize=9)
    for _, r in enumerate(adw_df.filter(pl.col("WEEK_NUM") >= CUT).iter_rows(named=True)):
        ax.axvline(r["WEEK_NUM"], color="#6250d6" if r["stream"] == "score" else "#008300", lw=1, alpha=0.7)
    ax.set_xlabel("Semana (WEEK_NUM)"); ax.set_ylabel("PSI frente a entrenamiento")
    ax.set_title("Cambio en la población: PSI semanal y detecciones ADWIN", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "04_drift_psi_adwin.png", dpi=200); plt.close(fig)

    pl.DataFrame(LOG, infer_schema_length=None).write_csv(OUT / "experiment_log.csv") if False else \
        (OUT / "experiment_log.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in LOG))
    log(stage="done", seconds=round(time.time() - t0, 1))
    print(res.select("model", "oot_auc", "oot_gini", "gini_stability", "fpr_at_recall80", "oot_brier"))


if __name__ == "__main__":
    main()
