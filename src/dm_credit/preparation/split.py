"""Partición reproducible de la base física de train y separación de test."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dm_credit import baseline
from dm_credit.preparation.columns import validate_control_columns
from dm_credit.preparation.contracts import (
    validate_base,
    validate_base_keys,
    validate_decision_date,
)


def validate_weeks(base: pl.DataFrame, label: str) -> None:
    weeks = base["WEEK_NUM"]
    if not (weeks.dtype.is_integer() or weeks.dtype.is_float()):
        raise ValueError(f"{label}: WEEK_NUM must be numeric")
    if weeks.dtype.is_float() and (
        weeks.is_nan().any() or weeks.is_infinite().any() or ((weeks % 1) != 0).any()
    ):
        raise ValueError(f"{label}: WEEK_NUM must contain finite whole weeks")


@dataclass(frozen=True)
class BasePartitions:
    """Instantánea de las particiones; cada acceso devuelve un DataFrame independiente."""

    _train: pl.DataFrame
    _oot: pl.DataFrame
    _test: pl.DataFrame
    cutoff_week: int

    @property
    def train(self) -> pl.DataFrame:
        return self._train.clone()

    @property
    def oot(self) -> pl.DataFrame:
        return self._oot.clone()

    @property
    def test(self) -> pl.DataFrame:
        return self._test.clone()


def partition_base(
    train_base: pl.DataFrame,
    test_base: pl.DataFrame,
    *,
    cutoff_week: int = 74,
) -> BasePartitions:
    """Separa la base de entrenamiento por semana sin mezclar los archivos de test.

    Conserva filas, orden, columnas y tipos. Las bases de entrada no se modifican;
    los tres resultados se validan antes de entregarse a los adaptadores.
    """
    if isinstance(cutoff_week, bool) or not isinstance(cutoff_week, int):
        raise ValueError("cutoff_week must be an integer")
    validate_control_columns(train_base.columns, "train")
    validate_control_columns(test_base.columns, "test")
    validate_base_keys(train_base)
    baseline.validate_contract(train_base)
    validate_decision_date(train_base)
    validate_weeks(train_base, "train base")
    validate_base(test_base, "test", cutoff_week)
    validate_weeks(test_base, "test base")

    train, oot = baseline.oot_split(train_base, cutoff_week)
    validate_base(train, "train", cutoff_week)
    validate_base(oot, "oot", cutoff_week)

    if (
        test_base.select("case_id")
        .join(train_base.select("case_id"), on="case_id", how="semi")
        .height
    ):
        raise ValueError("train and test bases contain overlapping case_id values")

    return BasePartitions(train.clone(), oot.clone(), test_base.clone(), cutoff_week)
