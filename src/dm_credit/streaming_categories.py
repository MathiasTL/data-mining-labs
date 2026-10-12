"""Calcula categorías frecuentes exactas por lotes con PyArrow."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def categorical_columns(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    return [
        field.name
        for field in schema
        if pa.types.is_string(field.type)
        or pa.types.is_large_string(field.type)
        or pa.types.is_boolean(field.type)
    ]


def count_categories(
    paths: list[Path], columns: list[str], top_k: int = 5, batch_size: int = 262_144
) -> pl.DataFrame:
    counters = {column: Counter() for column in columns}
    non_null = Counter()
    total_rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=columns, use_threads=False
        ):
            total_rows += batch.num_rows
            for name, array in zip(columns, batch.columns):
                counts = pc.value_counts(array.drop_null())
                for item in counts.to_pylist():
                    counters[name][item["values"]] += item["counts"]
                    non_null[name] += item["counts"]
    rows = []
    for column in columns:
        top = counters[column].most_common(top_k)
        rows.append(
            {
                "column": column,
                "n_unique": len(counters[column]),
                "top_value": str(top[0][0]) if top else None,
                "top_count": top[0][1] if top else 0,
                "pct_dominant_non_null": round(100 * top[0][1] / non_null[column], 4)
                if top
                else 0.0,
                "top_values": "; ".join(f"{value}={count}" for value, count in top),
                "rows_scanned": total_rows,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    columns = categorical_columns(args.paths[0])
    result = count_categories(args.paths, columns, args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.out)
    print(
        f"rows={result['rows_scanned'][0]:,} categorical_columns={result.height} report={args.out}"
    )


if __name__ == "__main__":
    main()
