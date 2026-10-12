"""Verifica claves ordenadas en Parquet por lotes y con memoria constante.

Si no hay violaciones de orden, toda clave duplicada debe aparecer de forma
adyacente; en ese caso ``candidate_key_unique`` es una conclusión exacta.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def natural_shard_key(path: Path) -> tuple[str, int]:
    stem, _, suffix = path.stem.rpartition("_")
    return (stem, int(suffix)) if suffix.isdigit() else (path.stem, -1)


def check_ordered_key(
    paths: list[Path], key_columns: list[str], batch_size: int = 262_144
) -> dict[str, Any]:
    if not paths or not key_columns:
        raise ValueError("Provide paths and at least one key column")

    rows = 0
    null_rows = 0
    adjacent_duplicate_rows = 0
    ordering_violations = 0
    distinct_case_ids = 0
    previous_key: tuple[Any, ...] | None = None
    previous_case: Any = None

    for path in sorted(paths, key=natural_shard_key):
        parquet = pq.ParquetFile(path)
        missing = set(key_columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing key columns in {path}: {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=key_columns, use_threads=False
        ):
            rows += batch.num_rows
            valid = np.ones(batch.num_rows, dtype=bool)
            for column in batch.columns:
                if column.null_count:
                    valid &= ~column.is_null().to_numpy(zero_copy_only=False)
            null_rows += int(batch.num_rows - valid.sum())
            if not valid.any():
                continue

            values = [
                column.to_numpy(zero_copy_only=False)[valid] for column in batch.columns
            ]
            if previous_key is not None:
                values = [
                    np.concatenate(([previous_key[index]], value))
                    for index, value in enumerate(values)
                ]

            equal = np.ones(len(values[0]) - 1, dtype=bool)
            less = np.zeros(len(values[0]) - 1, dtype=bool)
            equal_prefix = np.ones(len(values[0]) - 1, dtype=bool)
            for value in values:
                current_equal = value[1:] == value[:-1]
                less |= equal_prefix & (value[1:] < value[:-1])
                equal &= current_equal
                equal_prefix &= current_equal
            adjacent_duplicate_rows += int(equal.sum())
            ordering_violations += int(less.sum())

            cases = values[0]
            if previous_key is None:
                distinct_case_ids += 1
            distinct_case_ids += int((cases[1:] != cases[:-1]).sum())
            previous_case = cases[-1]
            previous_key = tuple(value[-1].item() for value in values)

    ordered = ordering_violations == 0
    return {
        "rows": rows,
        "key_columns": "+".join(key_columns),
        "key_null_rows": null_rows,
        "ordering_violations": ordering_violations,
        "adjacent_duplicate_rows": adjacent_duplicate_rows,
        "candidate_key_unique": ordered
        and null_rows == 0
        and adjacent_duplicate_rows == 0,
        "distinct_case_id": distinct_case_ids if ordered else None,
        "method": "ordered_batch_scan",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=262_144)
    args = parser.parse_args()

    result = check_ordered_key(args.paths, args.keys, args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(result)


if __name__ == "__main__":
    main()
