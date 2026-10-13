"""Fixtures sintéticos A6: faltantes, categorías, coerción y rangos (Gabriel, P5).

Tablas pequeñas con resultados calculados a mano. No leen Parquet reales.
"""

from __future__ import annotations

import polars as pl

UNKNOWN = "__UNKNOWN__"
MISSING = "__missing__"

TRAIN_WEEKS = [70, 71, 72, 73]
OOT_WEEK = 74


def numeric_mixed() -> pl.DataFrame:
    """Numérica con nulo, NaN, infinito, texto convertible y texto inválido."""
    return pl.DataFrame(
        {
            "a": [1.5, None, float("nan"), float("inf"), "2.5", "abc"],
        }
    )


def categorical_train() -> pl.DataFrame:
    """Train categórico: 'x' frecuente, 'y' frecuente, 'rare' rara (1/12)."""
    return pl.DataFrame(
        {"c": ["x", "x", "x", "x", "x", "y", "y", "y", "y", "y", "y", "rare"]}
    )


def categorical_oot() -> pl.DataFrame:
    """OOT con conocida, desconocida ('new') y nula."""
    return pl.DataFrame({"c": ["x", "new", None, "y"]})


def bool_text() -> pl.DataFrame:
    return pl.DataFrame({"b": ["true", "false", "1", "0", None, "maybe"]})


def dates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ok": ["2020-01-31", None],
            "bad": ["2020-01-31", "31/01/2020"],
        }
    )


def all_null_float() -> pl.DataFrame:
    """Las cuatro fechas de test llegan como Float64 totalmente nulo."""
    return pl.DataFrame({"d": [None, None]}, schema={"d": pl.Float64})


def physical_null() -> pl.DataFrame:
    """Columnas de test_tax_registry_c_1: tipo físico Null (parquet vacío)."""
    return pl.DataFrame({"x": [None, None]}, schema={"x": pl.Null})


def range_train() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "WEEK_NUM": TRAIN_WEEKS + [73],
            "a": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )


def range_oot_values() -> pl.DataFrame:
    return pl.DataFrame({"a": [0.0, 50.0, 500.0]})


def oot_week_table() -> pl.DataFrame:
    return pl.DataFrame({"WEEK_NUM": [70, 71, 72, 73, OOT_WEEK], "a": [1.0] * 5})


def fit_vocab(cats: list[str], min_frequency: float = 0.01) -> list[str]:
    """Vocabulario de referencia: frecuentes ordenadas + UNKNOWN al final."""
    n = len(cats)
    freq = {c: cats.count(c) / n for c in set(cats)}
    kept = sorted(c for c, f in freq.items() if f >= min_frequency)
    return kept + [UNKNOWN]


def apply_vocab(values: list, vocab: list[str]) -> list[str]:
    """Mapeo de referencia: nulo -> MISSING, fuera de vocab -> UNKNOWN."""
    known = set(vocab) - {UNKNOWN}
    out = []
    for v in values:
        if v is None:
            out.append(MISSING)
        elif v in known:
            out.append(v)
        else:
            out.append(UNKNOWN)
    return out


def with_missing_indicator(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Indicador de referencia <col>__missing (1 si nulo)."""
    return df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(f"{col}__missing"))


VALID_NAMES = [
    "deposit_1__amt__mean__all",
    "applprev_1__n_records",
    "tax_registry_a_1__amt__sum__all",
    "applprev_1__credamount_590a__mean__all__missing",  # sufijo __missing de D2: 4 segmentos
]

INVALID_NAMES = [
    "WEEK_NUM",  # reservada
    "Deposit__Amt__Mean",  # mayúsculas
    "sinseparador",  # sin separador __
    "familia__col__agreg__UNKNoWN__extra",  # mayúsculas en segmento
]
