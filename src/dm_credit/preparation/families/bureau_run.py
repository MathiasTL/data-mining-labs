"""Ejecución real reproducible de los adaptadores de bureau C4/C5.

Uso: ``POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.bureau_run``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.families import bureau
from dm_credit.preparation.split import partition_base

SPLITS = ("train", "oot", "test")
DEFAULT_OUTPUT_DIR = io.data_dir() / "artifacts" / "preparation" / "C4-C5-v1"
DEFAULT_CATALOG = Path("results/preparation/bureau_feature_catalog.csv")


def run_real_data(
    splits: Sequence[str] = SPLITS,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    catalog_path: Path = DEFAULT_CATALOG,
    cutoff_week: int = 74,
) -> list[contracts.FamilyFeatureArtifact]:
    """Construye los artefactos C4/C5 y publica el catálogo de sus esquemas."""
    invalid_splits = sorted(set(splits) - set(SPLITS))
    if invalid_splits:
        raise ValueError(f"invalid splits {invalid_splits}; expected {list(SPLITS)}")
    if not splits:
        raise ValueError("at least one split is required")

    partitions = partition_base(
        io.load_base("train"), io.load_base("test"), cutoff_week=cutoff_week
    )
    bases = {"train": partitions.train, "oot": partitions.oot, "test": partitions.test}
    output_dir = Path(output_dir)
    cache_dir = output_dir / "checkpoints"
    builders = (
        ("credit_bureau_a", bureau.build_credit_bureau_a_features),
        ("credit_bureau_b", bureau.build_credit_bureau_b_features),
    )
    artifacts: list[contracts.FamilyFeatureArtifact] = []
    for split in splits:
        for family, builder in builders:
            artifact = builder(
                bases[split],
                split,
                cutoff_week=cutoff_week,
                output_dir=output_dir / family / split,
                cache_dir=cache_dir,
            )
            artifacts.append(artifact)
            print(
                f"{family}/{split}: {artifact.case_count} casos, "
                f"{len(artifact.feature_schema)} características",
                flush=True,
            )

    contracts.write_feature_catalog(artifacts, Path(catalog_path))
    print(f"catálogo C4/C5 escrito en {catalog_path}")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--cutoff-week", type=int, default=74)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    run_real_data(
        args.splits,
        output_dir=args.output_dir,
        catalog_path=args.catalog,
        cutoff_week=args.cutoff_week,
    )


if __name__ == "__main__":
    main()
