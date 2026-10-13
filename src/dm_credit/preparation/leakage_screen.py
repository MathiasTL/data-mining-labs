"""Filtro empírico de fuga para G2, calculado solo con la partición de entrenamiento.

Un nombre limpio no prueba que una variable estuviera disponible al decidir. Este filtro mira
los datos: una variable que por sí sola separa el objetivo casi perfectamente casi siempre lo
conoce de antemano. Para cada característica se calcula:

    gini_all      Gini univariado (2*AUC - 1) sobre los casos con valor
    gini_block_i  el mismo Gini en bloques de semanas (estabilidad temporal)
    missing_gini  Gini del indicador de ausencia (la falta de dato también puede delatar el objetivo)

Banderas: ``leak_suspect`` (|Gini| mayor que el umbral), ``unstable`` (el signo cambia entre
bloques), ``insufficient`` (pocas filas o una sola clase), ``not_screened`` (texto) y ``ok``.
El umbral por defecto, 0,5, es holgado: en riesgo de crédito una variable sola rara vez supera
~0,4; un valor mayor exige explicación antes de aprobarla. OOT y test **no** se usan jamás.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from dm_credit import io
from dm_credit.preparation import contracts

KEY, WEEK, TARGET = "case_id", "WEEK_NUM", "target"
CUTOFF_WEEK = 74
SIGN_FLIP_MIN = 0.05  # un Gini menor que esto en un bloque se considera ruido, no cambio de signo
MIN_BLOCK_ROWS = 200


def _gini(values: np.ndarray, target: np.ndarray, min_rows: int) -> float | None:
    if len(values) < min_rows or len(np.unique(target)) < 2:
        return None
    return float(2.0 * roc_auc_score(target, values) - 1.0)


def screen_features(
    features: pl.DataFrame,
    labeled: pl.DataFrame,
    *,
    leak_threshold: float = 0.5,
    min_rows: int = 1000,
    blocks: int = 3,
    cutoff_week: int = CUTOFF_WEEK,
) -> pl.DataFrame:
    """Evalúa cada característica contra el objetivo. ``labeled`` trae ``case_id``, ``WEEK_NUM`` y ``target`` de train."""
    missing = [c for c in (KEY, WEEK, TARGET) if c not in labeled.columns]
    if missing:
        raise ValueError(f"labeled: missing required columns {missing}")
    for col in (WEEK, TARGET):
        if labeled[col].null_count():
            raise ValueError(f"labeled: {col} contains nulls")
    if labeled.height and labeled[WEEK].max() >= cutoff_week:
        raise ValueError(f"the screen only uses the train partition (WEEK_NUM < {cutoff_week})")

    df = labeled.select(KEY, WEEK, TARGET).join(features, on=KEY, how="left")
    cuts = [df[WEEK].quantile(i / blocks) for i in range(1, blocks)]
    block = np.zeros(df.height, dtype=int)
    weeks = df[WEEK].to_numpy()
    for cut in cuts:
        block += (weeks >= cut).astype(int)
    target = df[TARGET].to_numpy()

    rows = []
    for col in sorted(c for c in features.columns if c != KEY):
        s = df[col]
        row = {
            "feature": col,
            "n_rows": df.height,
            "n_nonnull": df.height - s.null_count(),
            "pct_null": round(100.0 * s.null_count() / df.height, 4) if df.height else 0.0,
        }
        block_names = [f"gini_block_{i + 1}" for i in range(blocks)]
        if not (s.dtype.is_numeric() or s.dtype == pl.Boolean):
            rows.append({**row, "gini_all": None, **dict.fromkeys(block_names), "missing_gini": None, "flag": "not_screened"})
            continue
        v = s.cast(pl.Float64).fill_nan(None).replace([float("inf"), float("-inf")], None).to_numpy()
        present = ~np.isnan(v)
        gini_all = _gini(v[present], target[present], min_rows)
        missing_gini = _gini((~present).astype(float), target, min_rows)
        block_ginis = [_gini(v[present & (block == i)], target[present & (block == i)], MIN_BLOCK_ROWS) for i in range(blocks)]

        strong = [g for g in (gini_all, missing_gini) if g is not None and abs(g) > leak_threshold]
        informative = [g for g in block_ginis if g is not None and abs(g) > SIGN_FLIP_MIN]
        flipped = any(a > 0 > b or a < 0 < b for a in informative for b in informative)
        if strong:
            flag = "leak_suspect"
        elif gini_all is None:
            flag = "insufficient"
        elif flipped:
            flag = "unstable"
        else:
            flag = "ok"
        rows.append({**row, "gini_all": gini_all, **dict(zip(block_names, block_ginis)), "missing_gini": missing_gini, "flag": flag})

    schema = {"feature": pl.String, "n_rows": pl.Int64, "n_nonnull": pl.Int64, "pct_null": pl.Float64, "gini_all": pl.Float64,
              **{f"gini_block_{i + 1}": pl.Float64 for i in range(blocks)}, "missing_gini": pl.Float64, "flag": pl.String}
    return pl.DataFrame(rows, schema=schema).sort("feature")


def screen_artifacts(
    manifests: Sequence[Path],
    train_base: pl.DataFrame,
    *,
    cutoff_week: int = CUTOFF_WEEK,
    **options,
) -> pl.DataFrame:
    """Filtro sobre artefactos de familia ya publicados. Solo acepta artefactos de ``train``."""
    frames = []
    labeled = train_base.select(KEY, WEEK, TARGET).filter(pl.col(WEEK) < cutoff_week)
    for manifest in manifests:
        art = contracts.load_family_artifact(Path(manifest))
        if art.split != "train":
            raise ValueError(f"{manifest}: the leakage screen only accepts train artifacts, got {art.split!r}")
        frames.append(screen_features(pl.read_parquet(art.data_path), labeled, cutoff_week=cutoff_week, **options))
    return pl.concat(frames).sort("feature") if frames else pl.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description="Filtro de fuga G2 sobre artefactos de familia de train ya publicados.")
    ap.add_argument("manifests", nargs="+", type=Path, help="manifiestos *.manifest.json de artefactos de train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff-week", type=int, default=CUTOFF_WEEK)
    ap.add_argument("--leak-threshold", type=float, default=0.5)
    args = ap.parse_args()
    result = screen_artifacts(args.manifests, io.load_base("train"), cutoff_week=args.cutoff_week, leak_threshold=args.leak_threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.out)
    print(result.group_by("flag").len().sort("flag"))
    print(f"escrito en {args.out}")


if __name__ == "__main__":
    main()
