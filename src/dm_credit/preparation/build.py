"""Despacho por familia (decisión G1, punto 1).

``build_family_features(family, base, split, *, cutoff_week, output_dir)`` delega en el constructor
registrado de cada familia. Las familias se registran en ``BUILDERS``; P1 amplía este registro
con depth 0, productos y bureau.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dm_credit.preparation.contracts import FamilyFeatureArtifact
from dm_credit.preparation.families import applications_people

BUILDERS = {**applications_people.BUILDERS}


def build_family_features(
    family: str,
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    **options,
) -> FamilyFeatureArtifact:
    if family not in BUILDERS:
        raise ValueError(f"unknown family {family!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[family](base, split, cutoff_week=cutoff_week, output_dir=output_dir, **options)
