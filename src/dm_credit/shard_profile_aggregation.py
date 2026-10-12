"""Agrega checkpoints de perfiles por fragmento sin releer los Parquet.

Las estadísticas aditivas y numéricas se combinan exactamente. La cardinalidad
se expresa como límites porque las estimaciones por fragmento no permiten saber
cuántos valores aparecen en más de un fragmento.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def aggregate_profiles(
    shard_dirs: list[Path],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not shard_dirs:
        raise ValueError("Provide at least one shard profile directory")

    column_frames: list[pl.DataFrame] = []
    numeric_frames: list[pl.DataFrame] = []
    source_frames: list[pl.DataFrame] = []
    for shard in sorted(shard_dirs):
        sources = pl.read_csv(shard / "sources.csv")
        rows = int(sources["rows_in_source"].sum())
        columns = pl.read_csv(shard / "columns.csv").with_columns(
            pl.lit(rows).alias("rows_in_shard"),
            (pl.lit(rows) - pl.col("nulls")).alias("non_null_in_shard"),
        )
        numeric = pl.read_csv(shard / "numeric.csv").join(
            columns.select("column", "non_null_in_shard"), on="column", how="left"
        )
        column_frames.append(columns)
        numeric_frames.append(numeric)
        source_frames.append(sources)

    all_columns = pl.concat(column_frames)
    total_rows = int(pl.concat(source_frames)["rows_in_source"].sum())
    unique_name = "n_unique_approx"
    columns = (
        all_columns.group_by("column", "dtype")
        .agg(
            pl.col("nulls").sum().alias("nulls"),
            pl.col(unique_name).max().alias("n_unique_lower_bound"),
            pl.col(unique_name).sum().alias("unique_sum"),
            pl.col("non_null_in_shard").sum().alias("non_nulls"),
        )
        .with_columns(
            (100 * pl.col("nulls") / total_rows).round(4).alias("pct_null"),
            pl.min_horizontal("unique_sum", "non_nulls").alias("n_unique_upper_bound"),
        )
        .drop("unique_sum", "non_nulls")
        .select(
            "column",
            "dtype",
            "nulls",
            "pct_null",
            "n_unique_lower_bound",
            "n_unique_upper_bound",
        )
        .sort("column")
    )

    all_numeric = pl.concat(numeric_frames)
    numeric = (
        all_numeric.group_by("column")
        .agg(
            pl.col("min").min().alias("min"),
            pl.col("max").max().alias("max"),
            (
                (pl.col("mean") * pl.col("non_null_in_shard")).sum()
                / pl.col("non_null_in_shard").sum()
            ).alias("mean"),
            pl.col("n_zero").sum().alias("n_zero"),
            pl.col("n_negative").sum().alias("n_negative"),
        )
        .sort("column")
    )
    sources = pl.concat(source_frames).sort("source_file")
    return columns, numeric, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    columns, numeric, sources = aggregate_profiles(args.shard_dirs)
    args.out.mkdir(parents=True, exist_ok=True)
    columns.write_csv(args.out / "columns.csv")
    numeric.write_csv(args.out / "numeric.csv")
    sources.write_csv(args.out / "sources.csv")
    print(
        f"rows={int(sources['rows_in_source'].sum()):,} cols={columns.height} shards={sources.height} reports written to {args.out}"
    )


if __name__ == "__main__":
    main()
