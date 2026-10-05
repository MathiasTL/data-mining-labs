"""Generic DataFrame profiler used by every team member.

Usage:
    python -m dm_credit.profiling                       # train_base
    python -m dm_credit.profiling path/to/file.parquet --out results/profile_x
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dm_credit import io


@dataclass
class Profile:
    n_rows: int
    n_cols: int
    memory_mb: float
    n_duplicate_rows: int
    columns: pl.DataFrame  # column, dtype, nulls, pct_null, n_unique
    numeric: pl.DataFrame  # column, min, max, mean, median, std, percentiles, n_zero, n_negative
    categorical: pl.DataFrame  # column, n_unique, top_value, top_count, pct_dominant, top_values


def profile_dataframe(df: pl.DataFrame, top_k: int = 5) -> Profile:
    n = df.height
    pct = lambda x: round(100.0 * x / n, 4) if n else 0.0  # noqa: E731

    columns = pl.DataFrame(
        [
            {
                "column": c,
                "dtype": str(df.schema[c]),
                "nulls": (nulls := df[c].null_count()),
                "pct_null": pct(nulls),
                "n_unique": df[c].drop_nulls().n_unique(),
            }
            for c in df.columns
        ],
        schema={"column": pl.String, "dtype": pl.String, "nulls": pl.Int64, "pct_null": pl.Float64, "n_unique": pl.Int64},
    )

    num_rows = []
    for c in df.columns:
        if not df.schema[c].is_numeric():
            continue
        s = df[c].drop_nulls().cast(pl.Float64)
        q = lambda p: s.quantile(p, "linear") if s.len() else None  # noqa: E731
        num_rows.append(
            {
                "column": c,
                "min": s.min(),
                "max": s.max(),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "p01": q(0.01),
                "p25": q(0.25),
                "p75": q(0.75),
                "p99": q(0.99),
                "n_zero": int((s == 0).sum()),
                "n_negative": int((s < 0).sum()),
            }
        )
    numeric = pl.DataFrame(num_rows) if num_rows else pl.DataFrame(schema={"column": pl.String})

    cat_rows = []
    for c in df.columns:
        if df.schema[c] not in (pl.String, pl.Categorical, pl.Boolean):
            continue
        vc = df[c].drop_nulls().value_counts(sort=True)
        count_col = vc.columns[-1]
        total = int(vc[count_col].sum()) if vc.height else 0
        top = vc.head(top_k)
        cat_rows.append(
            {
                "column": c,
                "n_unique": vc.height,
                "top_value": str(top[c][0]) if vc.height else None,
                "top_count": int(top[count_col][0]) if vc.height else 0,
                "pct_dominant": pct(int(top[count_col][0])) if vc.height else 0.0,
                "top_values": "; ".join(f"{v}={k}" for v, k in zip(top[c].cast(pl.String), top[count_col])),
            }
        )
    categorical = pl.DataFrame(cat_rows) if cat_rows else pl.DataFrame(schema={"column": pl.String})

    return Profile(
        n_rows=n,
        n_cols=df.width,
        memory_mb=df.estimated_size("mb"),
        n_duplicate_rows=int(df.is_duplicated().sum()),
        columns=columns,
        numeric=numeric,
        categorical=categorical,
    )


def save_profile(profile: Profile, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.columns.write_csv(out_dir / "columns.csv")
    profile.numeric.write_csv(out_dir / "numeric.csv")
    profile.categorical.write_csv(out_dir / "categorical.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="Parquet file (default: train_base)")
    ap.add_argument("--out", type=Path, help="Directory where CSV reports are written")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    df = io.load_parquet(args.path) if args.path else io.load_base("train")
    p = profile_dataframe(df, top_k=args.top_k)
    print(f"rows={p.n_rows:,} cols={p.n_cols} memory={p.memory_mb:.1f}MB duplicate_rows={p.n_duplicate_rows:,}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60):
        print(p.columns)
        print(p.numeric)
        print(p.categorical)
    if args.out:
        save_profile(p, args.out)
        print(f"reports written to {args.out}")


if __name__ == "__main__":
    main()
