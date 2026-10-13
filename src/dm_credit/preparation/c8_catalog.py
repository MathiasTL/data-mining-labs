"""Consolida los catálogos C8 generados por los adaptadores C1–C7."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from dm_credit.preparation.contracts import CATALOG_SCHEMA

DEFAULT_INPUTS = (
    Path("results/preparation/depth0_feature_catalog.csv"),
    Path("results/preparation/applications_people_feature_catalog.csv"),
    Path("results/preparation/bureau_feature_catalog.csv"),
    Path("results/preparation/products_feature_catalog.csv"),
)
DEFAULT_OUTPUT = Path("results/preparation/C8_feature_catalog.csv")


def consolidate_catalogs(paths: Sequence[Path]) -> pl.DataFrame:
    """Lee catálogos de familia, comprueba el contrato C8 y devuelve filas únicas."""
    if not paths:
        raise ValueError("at least one C8 input catalog is required")

    frames: list[pl.DataFrame] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"C8 input catalog not found: {path}")
        try:
            frame = pl.read_csv(path, schema_overrides=CATALOG_SCHEMA)
        except Exception as exc:
            raise ValueError(f"could not read C8 catalog {path}: {exc}") from exc
        missing = sorted(set(CATALOG_SCHEMA) - set(frame.columns))
        if missing:
            raise ValueError(f"C8 catalog {path} is missing columns: {missing}")
        if frame.is_empty():
            raise ValueError(f"C8 input catalog is empty: {path}")
        frames.append(frame.select(list(CATALOG_SCHEMA)))

    catalog = pl.concat(frames, how="vertical")
    duplicates = catalog.group_by("name").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        names = duplicates["name"].sort().to_list()
        raise ValueError(f"duplicate C8 feature names: {names}")
    if catalog.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("C8 catalog contains null values in required fields")
    return catalog.sort("name")


def write_catalog(paths: Sequence[Path], output: Path) -> pl.DataFrame:
    catalog = consolidate_catalogs(paths)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_csv(output)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = write_catalog(args.inputs, args.out)
    print(f"C8: {catalog.height} características consolidadas en {args.out}")


if __name__ == "__main__":
    main()
