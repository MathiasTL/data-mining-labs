"""Despacho por familia (decisión G1, punto 1).

``build_family_features(family, base, split, *, cutoff_week, output_dir)`` delega en el constructor
registrado de cada familia. Las familias se registran en ``BUILDERS``; P1 amplía este registro
con depth 0, productos y bureau.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dm_credit.preparation.contracts import FamilyFeatureArtifact
from dm_credit.preparation.families import applications_people, applications_people_g2, bureau

BUILDERS = {**applications_people.BUILDERS, **bureau.BUILDERS}
G2_CONFIG_INDEX = {"applprev": 0, "person": 1}  # posición de la familia en load_g2_configs()


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
    if options.get("config") is None and family in G2_CONFIG_INDEX:
        # sin configuración explícita solo se usan las columnas aprobadas en G2; si falta la decisión, error
        options["config"] = applications_people_g2.load_g2_configs()[G2_CONFIG_INDEX[family]]
    return BUILDERS[family](base, split, cutoff_week=cutoff_week, output_dir=output_dir, **options)
