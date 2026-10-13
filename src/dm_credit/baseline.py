"""Esqueleto del baseline: validación del contrato, corte OOT, métricas y registro del experimento.

Contrato (etapa A):
    case_id      clave única, una fila por caso
    WEEK_NUM     eje temporal para el corte fuera de tiempo (OOT); nunca es predictor
    target       etiqueta 0/1, solo en entrenamiento
    predictors   el resto, salvo las columnas técnicas/temporales reservadas

Uso:
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from dm_credit import io

KEY, WEEK, TARGET = "case_id", "WEEK_NUM", "target"
# Columnas técnicas/temporales que nunca son predictores en el baseline.
RESERVED = (KEY, WEEK, TARGET, "date_decision", "MONTH")
MISSING = "__missing__"
STABILITY_FALLING_RATE_PENALTY = 88.0
STABILITY_RESIDUAL_PENALTY = 0.5


# ---------------------------------------------------------------- métricas
@dataclass(frozen=True)
class StabilityComponents:
    mean_weekly_gini: float
    slope: float
    intercept: float
    residual_std: float
    slope_penalty: float
    variability_penalty: float
    score: float


def _validated_metric_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    score = np.asarray(y_score, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or len(y) == 0:
        raise ValueError(
            "y_true and y_score must be non-empty one-dimensional arrays of equal length"
        )
    if not np.isfinite(score).all():
        raise ValueError("y_score must contain only finite values")
    try:
        y_float = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must contain only 0/1") from exc
    if not np.isfinite(y_float).all() or not set(np.unique(y_float)) <= {0.0, 1.0}:
        raise ValueError("y_true must contain only 0/1")
    if len(np.unique(y_float)) < 2:
        raise ValueError("Gini is undefined when y_true contains a single class")
    return y_float, score


def gini(y_true, y_score) -> float:
    """Return 2 * ROC AUC - 1 after validating the binary inputs."""
    y, score = _validated_metric_arrays(y_true, y_score)
    return float(2.0 * roc_auc_score(y, score) - 1.0)


def weekly_gini(y_true, y_score, weeks) -> pl.DataFrame:
    """Calculate auditable weekly Gini values; every observed week must have both classes."""
    y = np.asarray(y_true)
    score = np.asarray(y_score)
    week = np.asarray(weeks)
    if week.ndim != 1 or len(week) != len(y):
        raise ValueError(
            "weeks must be one-dimensional and have the same length as y_true"
        )
    if (
        len(week) == 0
        or pl.Series(week).null_count()
        or not np.isfinite(week.astype(float)).all()
    ):
        raise ValueError(
            "weeks must be non-empty and contain only finite, non-null values"
        )
    rows = []
    for value in np.unique(week):
        mask = week == value
        rows.append(
            {
                WEEK: value,
                "n": int(mask.sum()),
                "n_pos": int(np.asarray(y[mask], dtype=float).sum()),
                "gini": gini(y[mask], score[mask]),
            }
        )
    return pl.DataFrame(rows).sort(WEEK)


def gini_stability_components(weekly_gini_values, weeks) -> StabilityComponents:
    """Return the official score and each component used to calculate it."""
    values = np.asarray(weekly_gini_values, dtype=float)
    week = np.asarray(weeks, dtype=float)
    if (
        values.ndim != 1
        or week.ndim != 1
        or len(values) != len(week)
        or len(values) < 2
    ):
        raise ValueError(
            "weekly Gini and weeks must have equal length and contain at least two weeks"
        )
    if not np.isfinite(values).all() or not np.isfinite(week).all():
        raise ValueError("weekly Gini and weeks must contain only finite values")
    if len(np.unique(week)) != len(week):
        raise ValueError("weeks must be unique")
    slope, intercept = np.polyfit(week, values, 1)
    residuals = values - (slope * week + intercept)
    mean = float(values.mean())
    residual_std = float(residuals.std(ddof=0))
    slope_penalty = float(STABILITY_FALLING_RATE_PENALTY * min(0.0, slope))
    variability_penalty = float(STABILITY_RESIDUAL_PENALTY * residual_std)
    score = mean + slope_penalty - variability_penalty
    return StabilityComponents(
        mean,
        float(slope),
        float(intercept),
        residual_std,
        slope_penalty,
        variability_penalty,
        score,
    )


def gini_stability(weekly_gini_values, weeks) -> float:
    """Return the official Gini Stability score."""
    return gini_stability_components(weekly_gini_values, weeks).score


# --------------------------------------------------------------- contrato
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
    """Entrenamiento: WEEK_NUM < corte. Validación (OOT): WEEK_NUM >= corte."""
    train = df.filter(pl.col(WEEK) < cutoff_week)
    val = df.filter(pl.col(WEEK) >= cutoff_week)
    if train.is_empty() or val.is_empty():
        raise ValueError(
            f"cutoff_week={cutoff_week} leaves an empty partition "
            f"(weeks available: {df[WEEK].min()}..{df[WEEK].max()})"
        )
    return train, val


# ------------------------------------------------------------------ modelo
def _to_float(a):
    return a.astype(float)


def _split_features(
    df: pl.DataFrame, features: list[str]
) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in features:
        dt = df.schema[c]
        if dt.is_numeric():
            num.append(c)
        elif dt in (pl.String, pl.Categorical, pl.Boolean):
            cat.append(c)
        else:
            raise ValueError(
                f"unsupported dtype {dt} for feature {c!r}; transform it first"
            )
    return num, cat


def _matrix(df: pl.DataFrame, num: list[str], cat: list[str]) -> np.ndarray:
    cols = [df[c].cast(pl.Float64).to_numpy() for c in num]
    cols += [df[c].cast(pl.String).fill_null(MISSING).to_numpy() for c in cat]
    return np.column_stack(cols).astype(object)


def _build_logistic(
    num: list[str], cat: list[str], seed: int, params: dict
) -> Pipeline:
    n_num = len(num)
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("cast", FunctionTransformer(_to_float)),
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
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
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
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
    scores: pl.DataFrame  # case_id, WEEK_NUM, target, score  (partición de validación)
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
        train_score = np.full(train.height, y_train.mean())
        score = np.full(val.height, y_train.mean())
        used_features: list[str] = []
        model_name = "prevalence"
    else:
        params = {"max_iter": 1000, "C": 1.0}
        num, cat = _split_features(train, features)
        pipe = _build_logistic(num, cat, seed, params)
        pipe.fit(
            _matrix(train, num, cat), y_train
        )  # se ajusta solo con entrenamiento: sin fuga desde OOT
        train_score = pipe.predict_proba(_matrix(train, num, cat))[:, 1]
        score = pipe.predict_proba(_matrix(val, num, cat))[:, 1]
        used_features, model_name = features, "logistic"

    scores = val.select(KEY, WEEK, TARGET).with_columns(pl.Series("score", score))
    weekly = weekly_gini(y_val, score, val[WEEK].to_numpy())
    stability = gini_stability_components(
        weekly["gini"].to_numpy(), weekly[WEEK].to_numpy()
    )
    weekly_values = weekly["gini"].to_numpy()
    metrics = {
        "train_roc_auc": float(roc_auc_score(y_train, train_score)),
        "train_gini": gini(y_train, train_score),
        "roc_auc": float(roc_auc_score(y_val, score)),
        "gini": gini(y_val, score),
        "gini_stability": stability.score,
        "mean_weekly_gini": stability.mean_weekly_gini,
        "weekly_gini_std": float(weekly_values.std(ddof=0)),
        "weekly_gini_slope": stability.slope,
        "residual_std": stability.residual_std,
        "log_loss": float(log_loss(y_val, score)),
        "brier_score": float(brier_score_loss(y_val, score)),
        "pr_auc": float(average_precision_score(y_val, score)),
        "min_weekly_gini": float(weekly_values.min()),
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cutoff-week",
        type=int,
        required=True,
        help="OOT cut: train < cutoff <= validation",
    )
    ap.add_argument("--model", choices=["prevalence", "logistic"], default="prevalence")
    ap.add_argument(
        "--features", nargs="*", help="Predictor columns (default: all non-reserved)"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=Path, default=Path("results/experiments.jsonl"))
    args = ap.parse_args()

    path = io.base_path("train")
    df = io.load_parquet(path)
    version = f"{path.name}|{path.stat().st_size}B|{df.height}rows"
    res = run_baseline(
        df, args.cutoff_week, args.model, args.features, args.seed, version
    )
    print(json.dumps(res.record, indent=2, default=str))
    append_record(res.record, args.log)
    print(f"appended to {args.log}")


if __name__ == "__main__":
    main()
