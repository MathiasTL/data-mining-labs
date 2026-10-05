"""Baseline skeleton: contract validation, OOT split, metrics, experiment record.

Contract (stage A):
    case_id      unique key, one row per case
    WEEK_NUM     time axis, used for the out-of-time (OOT) split, never as predictor
    target       0/1 label, training data only
    predictors   everything else except the reserved technical/temporal columns

Usage:
    python -m dm_credit.baseline --cutoff-week 80 --model prevalence
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from dm_credit import io

KEY, WEEK, TARGET = "case_id", "WEEK_NUM", "target"
# Technical / temporal columns that are never predictors in the baseline.
RESERVED = (KEY, WEEK, TARGET, "date_decision", "MONTH")
MISSING = "__missing__"
STABILITY_FALLING_RATE_PENALTY = 88.0  # Kaggle metric constant (to be confirmed, see doc)


# ---------------------------------------------------------------- metrics
def gini(y_true, y_score) -> float:
    """Gini = 2*AUC - 1. NaN if only one class is present."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def gini_stability(weekly_gini, weeks) -> float:
    """Competition-style stability: mean + 88*min(0, slope) - 0.5*std(residuals)."""
    g = np.asarray(weekly_gini, dtype=float)
    w = np.asarray(weeks, dtype=float)
    mask = np.isfinite(g)
    g, w = g[mask], w[mask]
    if len(g) == 0:
        return float("nan")
    if len(g) < 2:
        return float(g.mean())
    slope, intercept = np.polyfit(w, g, 1)
    residuals = g - (slope * w + intercept)
    return float(g.mean() + STABILITY_FALLING_RATE_PENALTY * min(0.0, slope) - 0.5 * residuals.std())


# --------------------------------------------------------------- contract
def validate_contract(df: pl.DataFrame, require_target: bool = True) -> None:
    required = [KEY, WEEK] + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df[KEY].null_count():
        raise ValueError(f"{KEY} contains nulls")
    if df[KEY].n_unique() != df.height:
        raise ValueError(f"{KEY} is not unique: expected one row per case")
    if df[WEEK].null_count():
        raise ValueError(f"{WEEK} contains nulls")
    if require_target and not set(df[TARGET].unique().to_list()) <= {0, 1}:
        raise ValueError(f"{TARGET} must contain only 0/1 (no nulls)")


def default_feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in RESERVED]


def oot_split(df: pl.DataFrame, cutoff_week: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train: WEEK_NUM < cutoff. Validation (OOT): WEEK_NUM >= cutoff."""
    train = df.filter(pl.col(WEEK) < cutoff_week)
    val = df.filter(pl.col(WEEK) >= cutoff_week)
    if train.is_empty() or val.is_empty():
        raise ValueError(
            f"cutoff_week={cutoff_week} leaves an empty partition "
            f"(weeks available: {df[WEEK].min()}..{df[WEEK].max()})"
        )
    return train, val


# ------------------------------------------------------------------ model
def _to_float(a):
    return a.astype(float)


def _split_features(df: pl.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in features:
        dt = df.schema[c]
        if dt.is_numeric():
            num.append(c)
        elif dt in (pl.String, pl.Categorical, pl.Boolean):
            cat.append(c)
        else:
            raise ValueError(f"unsupported dtype {dt} for feature {c!r}; transform it first")
    return num, cat


def _matrix(df: pl.DataFrame, num: list[str], cat: list[str]) -> np.ndarray:
    cols = [df[c].cast(pl.Float64).to_numpy() for c in num]
    cols += [df[c].cast(pl.String).fill_null(MISSING).to_numpy() for c in cat]
    return np.column_stack(cols).astype(object)


def _build_logistic(num: list[str], cat: list[str], seed: int, params: dict) -> Pipeline:
    n_num = len(num)
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("cast", FunctionTransformer(_to_float)),
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(range(n_num)),
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False),
                list(range(n_num, n_num + len(cat))),
            )
        )
    return Pipeline(
        [
            ("prep", ColumnTransformer(transformers)),
            ("clf", LogisticRegression(random_state=seed, **params)),
        ]
    )


@dataclass
class BaselineResult:
    scores: pl.DataFrame  # case_id, WEEK_NUM, target, score  (validation partition)
    weekly: pl.DataFrame  # WEEK_NUM, n, n_pos, gini
    metrics: dict
    record: dict


def run_baseline(
    df: pl.DataFrame,
    cutoff_week: int,
    model: str = "logistic",
    features: list[str] | None = None,
    seed: int = 42,
    data_version: str = "unspecified",
) -> BaselineResult:
    validate_contract(df)
    features = default_feature_columns(df) if features is None else list(features)
    reserved = [c for c in features if c in RESERVED]
    if reserved:
        raise ValueError(f"reserved columns cannot be predictors: {reserved}")
    if model not in ("prevalence", "logistic"):
        raise ValueError("model must be 'prevalence' or 'logistic'")

    train, val = oot_split(df, cutoff_week)
    y_train, y_val = train[TARGET].to_numpy(), val[TARGET].to_numpy()

    params: dict = {}
    if model == "prevalence" or not features:
        score = np.full(val.height, y_train.mean())
        used_features: list[str] = []
        model_name = "prevalence"
    else:
        params = {"max_iter": 1000, "C": 1.0}
        num, cat = _split_features(train, features)
        pipe = _build_logistic(num, cat, seed, params)
        pipe.fit(_matrix(train, num, cat), y_train)  # fitted on train only: no OOT leakage
        score = pipe.predict_proba(_matrix(val, num, cat))[:, 1]
        used_features, model_name = features, "logistic"

    scores = val.select(KEY, WEEK, TARGET).with_columns(pl.Series("score", score))
    weekly = (
        scores.group_by(WEEK)
        .agg(
            pl.len().alias("n"),
            pl.col(TARGET).sum().alias("n_pos"),
            pl.struct(TARGET, "score").map_batches(
                lambda s: pl.Series([gini(s.struct.field(TARGET).to_numpy(), s.struct.field("score").to_numpy())]),
                returns_scalar=True,
            ).alias("gini"),
        )
        .sort(WEEK)
    )
    metrics = {
        "gini": gini(y_val, score),
        "roc_auc": float(roc_auc_score(y_val, score)) if len(np.unique(y_val)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_val, score)) if y_val.sum() else float("nan"),
        "gini_stability": gini_stability(weekly["gini"].to_numpy(), weekly[WEEK].to_numpy()),
        "mean_weekly_gini": float(np.nanmean(weekly["gini"].to_numpy())) if weekly.height else float("nan"),
        "min_weekly_gini": float(np.nanmin(weekly["gini"].to_numpy())) if weekly.height else float("nan"),
        "prevalence_train": float(y_train.mean()),
        "prevalence_val": float(y_val.mean()),
        "n_train": train.height,
        "n_val": val.height,
    }
    record = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": data_version,
        "model": model_name,
        "features": used_features,
        "cutoff_week": cutoff_week,
        "seed": seed,
        "params": params,
        "metrics": metrics,
    }
    return BaselineResult(scores=scores, weekly=weekly, metrics=metrics, record=record)


def append_record(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff-week", type=int, required=True, help="OOT cut: train < cutoff <= validation")
    ap.add_argument("--model", choices=["prevalence", "logistic"], default="prevalence")
    ap.add_argument("--features", nargs="*", help="Predictor columns (default: all non-reserved)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=Path, default=Path("results/experiments.jsonl"))
    args = ap.parse_args()

    path = io.base_path("train")
    df = io.load_parquet(path)
    version = f"{path.name}|{path.stat().st_size}B|{df.height}rows"
    res = run_baseline(df, args.cutoff_week, args.model, args.features, args.seed, version)
    print(json.dumps(res.record, indent=2, default=str))
    append_record(res.record, args.log)
    print(f"appended to {args.log}")


if __name__ == "__main__":
    main()
