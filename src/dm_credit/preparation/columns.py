"""Validación central de columnas de control y predictores de preparación."""

from __future__ import annotations

from collections.abc import Iterable

from dm_credit.baseline import RESERVED

SPLITS = ("train", "oot", "test")
REQUIRED_CONTROLS = ("case_id", "WEEK_NUM", "date_decision")
RESERVED_COLUMNS = frozenset(RESERVED)


def validate_control_columns(columns: Iterable[str], split: str) -> tuple[str, ...]:
    """Comprueba controles obligatorios y presencia de target según la partición.

    El nombre ``train`` también identifica las filas físicas de train antes del corte OOT.
    Devuelve los controles presentes en el mismo orden que las columnas recibidas.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    names = tuple(columns)
    if len(set(names)) != len(names):
        raise ValueError("duplicate column names")
    required = REQUIRED_CONTROLS + (("target",) if split != "test" else ())
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"missing required control columns: {missing}")
    if split == "test" and "target" in names:
        raise ValueError("test must not contain target")
    return tuple(name for name in names if name in RESERVED_COLUMNS)


def validate_predictor_columns(
    predictors: Iterable[str], *, available_columns: Iterable[str] | None = None
) -> tuple[str, ...]:
    """Rechaza reservadas o nombres desconocidos en una selección explícita."""
    names = tuple(predictors)
    if len(set(names)) != len(names):
        raise ValueError("duplicate predictor columns")
    reserved = [name for name in names if name in RESERVED_COLUMNS]
    if reserved:
        raise ValueError(f"reserved columns cannot be predictors: {reserved}")
    if available_columns is not None:
        available = set(available_columns)
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"missing predictor columns: {missing}")
    return names
