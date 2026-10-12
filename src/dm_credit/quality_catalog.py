"""Consolida perfiles existentes en un catálogo de calidad sin releer Parquet.

Uso:
    uv run -m dm_credit.quality_catalog
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _read_optional(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.read_csv(path) if path.is_file() else pl.DataFrame(schema=schema)


def build_quality_catalog(profile_dirs: list[Path]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for directory in sorted(profile_dirs):
        columns_path = directory / "columns.csv"
        if not columns_path.is_file():
            continue
        family = directory.name.removeprefix("profile_train_")
        columns = pl.read_csv(columns_path)
        unique_column = next(
            name
            for name in ("n_unique", "n_unique_approx", "n_unique_lower_bound")
            if name in columns.columns
        )
        columns = columns.with_columns(
            pl.lit(family).alias("family"),
            pl.col(unique_column).alias("n_unique_profiled"),
        )
        numeric = _read_optional(
            directory / "numeric.csv",
            {
                "column": pl.String,
                "min": pl.Float64,
                "max": pl.Float64,
                "mean": pl.Float64,
                "n_zero": pl.Int64,
                "n_negative": pl.Int64,
            },
        )
        numeric = numeric.select(
            [
                name
                for name in ("column", "min", "max", "mean", "n_zero", "n_negative")
                if name in numeric.columns
            ]
        )
        categorical = _read_optional(
            directory / "categorical.csv",
            {
                "column": pl.String,
                "top_value": pl.String,
                "top_count": pl.Int64,
                "pct_dominant": pl.Float64,
                "top_values": pl.String,
            },
        )
        if (
            "pct_dominant_non_null" in categorical.columns
            and "pct_dominant" not in categorical.columns
        ):
            categorical = categorical.rename({"pct_dominant_non_null": "pct_dominant"})
        categorical = categorical.select(
            [
                name
                for name in (
                    "column",
                    "top_value",
                    "top_count",
                    "pct_dominant",
                    "top_values",
                )
                if name in categorical.columns
            ]
        )
        frame = columns.join(numeric, on="column", how="left").join(
            categorical, on="column", how="left"
        )
        optional_columns = {
            "min": pl.Float64,
            "max": pl.Float64,
            "mean": pl.Float64,
            "n_zero": pl.Int64,
            "n_negative": pl.Int64,
            "top_value": pl.String,
            "top_count": pl.Int64,
            "pct_dominant": pl.Float64,
            "top_values": pl.String,
        }
        frame = frame.with_columns(
            [
                pl.lit(None, dtype=dtype).alias(name)
                for name, dtype in optional_columns.items()
                if name not in frame.columns
            ]
        )
        frame = frame.with_columns(
            pl.when(pl.col("pct_null") >= 95)
            .then(pl.lit("critical_95_plus"))
            .when(pl.col("pct_null") >= 80)
            .then(pl.lit("very_high_80_95"))
            .when(pl.col("pct_null") >= 50)
            .then(pl.lit("high_50_80"))
            .when(pl.col("pct_null") > 0)
            .then(pl.lit("partial_below_50"))
            .otherwise(pl.lit("complete"))
            .alias("missingness_band"),
            (pl.col("n_unique_profiled") <= 1).alias("is_constant"),
            (pl.col("n_negative").fill_null(0) > 0).alias("has_negative"),
            pl.col("top_value").is_not_null().alias("has_top_categories"),
        )
        frames.append(frame)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("family", "column")


def summarize_quality(catalog: pl.DataFrame) -> pl.DataFrame:
    return (
        catalog.group_by("family")
        .agg(
            pl.len().alias("columns"),
            (pl.col("pct_null") > 0).sum().alias("columns_with_nulls"),
            (pl.col("pct_null") >= 50).sum().alias("columns_50pct_null_or_more"),
            (pl.col("pct_null") >= 95).sum().alias("columns_95pct_null_or_more"),
            pl.col("is_constant").sum().alias("constant_columns"),
            pl.col("has_negative").sum().alias("numeric_columns_with_negatives"),
            pl.col("has_top_categories")
            .sum()
            .alias("categorical_columns_with_top_values"),
            pl.col("pct_null").max().alias("max_pct_null"),
        )
        .sort("family")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/quality_catalog.csv"))
    args = parser.parse_args()

    catalog = build_quality_catalog(list(args.profiles.glob("profile_train_*")))
    if catalog.is_empty():
        raise FileNotFoundError(f"No profiles found below {args.profiles}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_csv(args.out)
    summary_path = args.out.with_name(f"{args.out.stem}_summary.csv")
    summarize_quality(catalog).write_csv(summary_path)
    print(
        f"rows={catalog.height} families={catalog['family'].n_unique()} catalog={args.out} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
