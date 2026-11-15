"""Particiones temporales del protocolo: ventana creciente sobre WEEK_NUM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEEK = "WEEK_NUM"
OOT_WEEKS = (74, 91)
DEV_WEEKS = (0, 73)


@dataclass(frozen=True)
class Fold:
    name: str
    fit: tuple[int, int]  # semanas inclusivas
    val: tuple[int, int]

    @property
    def cutoff_week(self) -> int:
        """Corte exclusivo que exige ``fit_matrix_views`` (WEEK_NUM < corte)."""
        return self.fit[1] + 1


FOLDS = (
    Fold("P1", (0, 39), (40, 49)),
    Fold("P2", (0, 49), (50, 59)),
    Fold("P3", (0, 59), (60, 73)),
)


def get_fold(name: str) -> Fold:
    for f in FOLDS:
        if f.name == name:
            return f
    raise KeyError(name)


def _between(weeks, bounds) -> np.ndarray:
    w = np.asarray(weeks)
    return (w >= bounds[0]) & (w <= bounds[1])


def split_masks(weeks, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras booleanas (ajuste, validación) sobre WEEK_NUM."""
    return _between(weeks, fold.fit), _between(weeks, fold.val)


def oot_mask(weeks) -> np.ndarray:
    return _between(weeks, OOT_WEEKS)


def fold_bounds() -> dict:
    return {
        "folds": {f.name: {"fit": list(f.fit), "val": list(f.val)} for f in FOLDS},
        "oot": list(OOT_WEEKS),
    }
