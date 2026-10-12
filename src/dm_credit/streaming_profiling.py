"""Perfil descriptivo por escaneo, adecuado para Parquet grandes.

Calcula cobertura, cardinalidad aproximada y estadísticas numéricas básicas sin
materializar el DataFrame completo. No calcula duplicados ni categorías top-k.

Uso:
    uv run -m dm_credit.streaming_profiling file1.parquet file2.parquet --out results/profile_family
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def profile_paths(paths: list[Path], out_dir: Path) -> tuple[int, int]:
    if not paths:
        raise ValueError("Provide at least one Parquet path")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Parquet file(s) not found: {', '.join(map(str, missing))}"
        )

    frame = pl.scan_parquet([str(path) for path in paths])
    schema = frame.collect_schema()
    numeric_columns = [name for name, dtype in schema.items() if dtype.is_numeric()]

    expressions: list[pl.Expr] = [pl.len().alias("rows")]
    expressions.extend(
        pl.col(name).null_count().alias(f"nulls__{name}") for name in schema
    )
    expressions.extend(
        pl.col(name).approx_n_unique().alias(f"unique__{name}") for name in schema
    )
    expressions.extend(
        pl.col(name).min().alias(f"min__{name}") for name in numeric_columns
    )
    expressions.extend(
        pl.col(name).max().alias(f"max__{name}") for name in numeric_columns
    )
    expressions.extend(
        pl.col(name).mean().alias(f"mean__{name}") for name in numeric_columns
    )
    expressions.extend(
        (pl.col(name) == 0).sum().alias(f"zero__{name}") for name in numeric_columns
    )
    expressions.extend(
        (pl.col(name) < 0).sum().alias(f"negative__{name}") for name in numeric_columns
    )
    summary = frame.select(expressions).collect(engine="streaming")
    n_rows = int(summary["rows"][0])

    columns = pl.DataFrame(
        [
            {
                "column": name,
                "dtype": str(dtype),
                "nulls": (nulls := int(summary[f"nulls__{name}"][0])),
                "pct_null": round(100 * nulls / n_rows, 4) if n_rows else 0.0,
                "n_unique_approx": int(summary[f"unique__{name}"][0]),
            }
            for name, dtype in schema.items()
        ]
    )
    numeric = pl.DataFrame(
        [
            {
                "column": name,
                "min": summary[f"min__{name}"][0],
                "max": summary[f"max__{name}"][0],
                "mean": summary[f"mean__{name}"][0],
                "n_zero": int(summary[f"zero__{name}"][0]),
                "n_negative": int(summary[f"negative__{name}"][0]),
            }
            for name in numeric_columns
        ],
        schema_overrides={"column": pl.String},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    columns.write_csv(out_dir / "columns.csv")
    numeric.write_csv(out_dir / "numeric.csv")
    pl.DataFrame(
        {
            "source_file": [path.name for path in paths],
            "rows_in_source": [
                int(
                    pl.scan_parquet(str(path))
                    .select(pl.len())
                    .collect(engine="streaming")
                    .item()
                )
                for path in paths
            ],
            "columns": [len(schema)] * len(paths),
        }
    ).write_csv(out_dir / "sources.csv")
    return n_rows, len(schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Parquet files with compatible schemas"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output directory for CSV reports"
    )
    args = parser.parse_args()
    rows, columns = profile_paths(args.paths, args.out)
    print(f"rows={rows:,} cols={columns} reports written to {args.out}")


if __name__ == "__main__":
    main()
