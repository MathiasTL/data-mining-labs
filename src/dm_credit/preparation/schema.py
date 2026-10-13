"""D6: contrato de esquema para matrices finales train, OOT y test.

No ensambla ni transforma datos: valida tablas ya construidas antes de publicarlas.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dm_credit import baseline
from dm_credit.preparation.columns import (
    validate_control_columns,
    validate_predictor_columns,
)
from dm_credit.preparation.contracts import logical_type, validate_base
from dm_credit.preparation.split import validate_weeks


@dataclass(frozen=True)
class MatrixSchema:
    control_columns: tuple[str, ...]
    feature_schema: tuple[tuple[str, str], ...]  # (nombre, tipo lógico), en orden


def validate_matrix_schema(
    matrix: pl.DataFrame, split: str, *, cutoff_week: int = 74
) -> MatrixSchema:
    """Valida una matriz etiquetada; a diferencia de la base mínima, exige target en train/OOT."""
    controls = validate_control_columns(matrix.columns, split)
    if any(name in controls for name in ("num_group1", "num_group2")):
        raise ValueError(
            f"{split}: num_group1 and num_group2 must not reach the matrix"
        )
    if matrix.columns[0] != "case_id":
        raise ValueError(f"{split}: case_id must be the first column")
    if tuple(matrix.columns[: len(controls)]) != controls:
        raise ValueError(f"{split}: control columns must precede all features")
    features = tuple(matrix.columns[len(controls) :])
    validate_predictor_columns(features)

    validate_base(matrix, split, cutoff_week)
    validate_weeks(matrix, f"{split} matrix")
    if split != "test":
        baseline.validate_contract(matrix)
    return MatrixSchema(
        control_columns=controls,
        feature_schema=tuple(
            (name, logical_type(matrix.schema[name], name)) for name in features
        ),
    )


def validate_matrix_schemas(
    train: pl.DataFrame,
    oot: pl.DataFrame,
    test: pl.DataFrame,
    *,
    cutoff_week: int = 74,
) -> MatrixSchema:
    """Exige los mismos controles (salvo target) y características lógicas en los tres splits."""
    schemas = {
        "train": validate_matrix_schema(train, "train", cutoff_week=cutoff_week),
        "oot": validate_matrix_schema(oot, "oot", cutoff_week=cutoff_week),
        "test": validate_matrix_schema(test, "test", cutoff_week=cutoff_week),
    }
    reference = schemas["train"]
    expected_controls = tuple(c for c in reference.control_columns if c != "target")
    for split in ("oot", "test"):
        current = schemas[split]
        controls = tuple(c for c in current.control_columns if c != "target")
        if controls != expected_controls:
            raise ValueError(
                f"{split}: control columns differ from train: {controls} != {expected_controls}"
            )
        if current.feature_schema != reference.feature_schema:
            raise ValueError(
                f"{split}: feature names, order or logical types differ from train"
            )
    return reference
