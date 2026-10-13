"""E1/E2 — ensamblado validado de artefactos de familias."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from dm_credit.preparation.columns import (
    validate_control_columns,
    validate_predictor_columns,
)
from dm_credit.preparation.contracts import (
    FORMAT_VERSION,
    KEY,
    FamilyFeatureArtifact,
    FeatureDefinition,
    FeatureMatrixArtifact,
    _sha256,
    load_family_artifact,
    logical_type,
    validate_base_keys,
)


def _read_verified_artifact(
    artifact: FamilyFeatureArtifact, split: str
) -> pl.DataFrame:
    if artifact.split != split:
        raise ValueError(
            f"family {artifact.family!r} has split {artifact.split!r}, expected {split!r}"
        )
    loaded = load_family_artifact(Path(artifact.manifest_path))
    if loaded != artifact:
        raise ValueError(
            f"family {artifact.family!r}: supplied artifact differs from its manifest"
        )
    try:
        frame = pl.read_parquet(artifact.data_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(
            f"cannot read family {artifact.family!r} data at {artifact.data_path}: {exc}"
        ) from exc
    if (
        frame.height != artifact.row_count
        or frame[KEY].n_unique() != artifact.case_count
    ):
        raise ValueError(
            f"family {artifact.family!r}: row/case counts do not match its manifest"
        )
    if frame.columns != [
        KEY,
        *(definition.name for definition in artifact.feature_schema),
    ]:
        raise ValueError(
            f"family {artifact.family!r}: schema does not match its manifest"
        )
    for definition in artifact.feature_schema:
        actual_type = logical_type(frame.schema[definition.name], definition.name)
        if actual_type != definition.logical_type:
            raise ValueError(
                f"family {artifact.family!r}: {definition.name!r} declares "
                f"{definition.logical_type!r}, but data has logical type {actual_type!r}"
            )
    if frame[KEY].null_count() or frame[KEY].n_unique() != frame.height:
        raise ValueError(
            f"family {artifact.family!r}: {KEY} must be non-null and unique"
        )
    return frame


def assemble_feature_matrix(
    base: pl.DataFrame,
    family_artifacts: Sequence[FamilyFeatureArtifact],
    split: str,
    *,
    output_dir: Path,
) -> FeatureMatrixArtifact:
    """Une familias 1:1 con la base, preservando orden y población caso a caso."""
    controls = validate_control_columns(base.columns, split)
    validate_base_keys(base)
    if split not in ("train", "oot", "test"):
        raise ValueError(f"invalid split {split!r}")
    base_keys = base[KEY].to_list()
    result = base.clone()
    definitions: list[FeatureDefinition] = []
    seen_names = set(base.columns)
    seen_families: set[str] = set()

    for artifact in family_artifacts:
        if artifact.family in seen_families:
            raise ValueError(f"duplicate family artifact {artifact.family!r}")
        seen_families.add(artifact.family)
        family_frame = _read_verified_artifact(artifact, split)
        names = family_frame.columns[1:]
        validate_predictor_columns(names)
        collisions = seen_names.intersection(names)
        if collisions:
            raise ValueError(
                f"duplicate or reserved feature names: {sorted(collisions)}"
            )
        if len({definition.name for definition in artifact.feature_schema}) != len(
            artifact.feature_schema
        ):
            raise ValueError(
                f"family {artifact.family!r}: duplicate feature definitions"
            )
        joined = result.join(
            family_frame, on=KEY, how="left", validate="1:1", maintain_order="left"
        )
        if joined.height != result.height:
            raise ValueError(f"join with {artifact.family!r} changed row count")
        if joined[KEY].to_list() != base_keys:
            raise ValueError(
                f"join with {artifact.family!r} changed the base case set/order"
            )
        if joined[KEY].n_unique() != len(base_keys) or joined[KEY].null_count():
            raise ValueError(
                f"join with {artifact.family!r} violated case_id uniqueness"
            )
        result = joined
        seen_names.update(names)
        definitions.extend(artifact.feature_schema)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"feature_matrix__{split}.parquet"
    manifest_path = output_dir / f"feature_matrix__{split}.manifest.json"
    tmp_data, tmp_manifest = (
        data_path.with_name(data_path.name + ".tmp"),
        manifest_path.with_name(manifest_path.name + ".tmp"),
    )
    input_manifests = tuple(str(Path(a.manifest_path)) for a in family_artifacts)
    try:
        result.write_parquet(tmp_data)
        manifest = {
            "format_version": FORMAT_VERSION,
            "split": split,
            "data_path": data_path.name,
            "data_sha256": _sha256(tmp_data),
            "row_count": result.height,
            "control_columns": list(controls),
            "feature_schema": [
                {
                    **definition.__dict__,
                    "source_columns": list(definition.source_columns),
                }
                for definition in definitions
            ],
            "input_manifests": list(input_manifests),
            "validations": [
                "base case_id no nulo/único",
                "joins 1:1",
                "filas y orden de casos preservados",
            ],
        }
        tmp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(tmp_data, data_path)
        os.replace(tmp_manifest, manifest_path)
    finally:
        tmp_data.unlink(missing_ok=True)
        tmp_manifest.unlink(missing_ok=True)
    return FeatureMatrixArtifact(
        split=split,
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=result.height,
        control_columns=controls,
        feature_schema=tuple(definitions),
        input_manifests=input_manifests,
    )
