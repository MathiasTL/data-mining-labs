"""D4 — Coerción estricta de tipos hacia el esquema lógico.

Reglas (contrato de compatibilidad train/test del capítulo 02):
    - Una columna de tipo físico ``Null`` o totalmente nula adopta el tipo lógico de train.
    - Fechas: texto ``%Y-%m-%d``; cualquier otro formato es un error.
    - Entero desde flotante solo si todo valor finito tiene parte fraccionaria cero.
    - Booleano desde texto solo admite ``true``, ``false``, ``1`` o ``0``.
    - Nunca se convierte un valor inválido en nulo en silencio: se lanza ``CoercionError``.
    - Infinitos son un error; ``NaN`` se trata como ausente.
"""

from __future__ import annotations

import re
from datetime import time

import polars as pl

LOGICAL_TYPES = {
    "date": pl.Date,
    "int": pl.Int64,
    "float": pl.Float64,
    "bool": pl.Boolean,
    "string": pl.String,
}
DATE_FORMAT = "%Y-%m-%d"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"  # estricto: sin ceros omitidos ni otros separadores
MAX_EXACT_INT = 2**53  # un flotante representa enteros exactos solo hasta aquí
BOOL_TOKENS = {"true": True, "false": False, "1": True, "0": False}
INTEGER_COLUMNS = {"case_id", "num_group1", "num_group2", "WEEK_NUM", "MONTH", "target"}


class CoercionError(ValueError):
    """Un valor no se puede convertir al tipo lógico sin perder información."""


def logical_type_for(column: str, train_dtype: pl.DataType) -> str:
    """Tipo lógico de una columna según claves conocidas, sufijo Home Credit y tipo de train.

    ``D`` es fecha y ``M`` es texto categórico; el resto conserva la representación de train.
    """
    if column in INTEGER_COLUMNS:
        return "int"
    suffix = column.rsplit("_", 1)[-1][-1:]
    if suffix == "D":
        return "date"
    if suffix == "M":
        return "string"
    if train_dtype == pl.Boolean:
        return "bool"
    if train_dtype in (pl.String, pl.Categorical):
        return "string"
    if train_dtype.is_integer():
        return "int"
    if train_dtype.is_float():
        return "float"
    raise ValueError(f"cannot infer the logical type of {column!r} from dtype {train_dtype}")


def _fail(column: str, detail: str) -> CoercionError:
    return CoercionError(f"column {column!r}: {detail}")


def _numeric(s: pl.Series, column: str) -> pl.Series:
    """Serie Float64 sin NaN (ausente) y sin infinitos."""
    if s.dtype == pl.String:
        parsed = s.cast(pl.Float64, strict=False)
        bad = int((s.is_not_null() & parsed.is_null()).sum())
        if bad:
            raise _fail(column, f"{bad} text values are not numeric")
        s = parsed
    elif s.dtype == pl.Boolean or not (s.dtype.is_numeric()):
        raise _fail(column, f"dtype {s.dtype} cannot be converted to a number")
    s = s.cast(pl.Float64).fill_nan(None)
    n_inf = int(s.is_infinite().sum())
    if n_inf:
        raise _fail(column, f"{n_inf} infinite values")
    return s


def _to_date(s: pl.Series, column: str) -> pl.Series:
    if s.dtype == pl.Null or s.null_count() == s.len() and s.dtype != pl.String:
        return pl.Series(column, [None] * s.len(), dtype=pl.Date)
    if s.dtype == pl.Date:
        return s
    if s.dtype == pl.Datetime:
        with_time = int((s.dt.time() != time(0)).sum() or 0)
        if with_time:
            raise _fail(column, f"{with_time} datetime values have a time component that would be truncated")
        return s.cast(pl.Date)
    if s.dtype == pl.String:
        strict = s.str.contains(DATE_PATTERN)
        parsed = s.str.to_date(DATE_FORMAT, strict=False)
        bad = int((s.is_not_null() & (strict.not_() | parsed.is_null())).sum())
        if bad:
            raise _fail(column, f"{bad} values do not match {DATE_FORMAT}")
        return parsed
    raise _fail(column, f"dtype {s.dtype} with values cannot be converted to date")


def _to_int(s: pl.Series, column: str) -> pl.Series:
    if s.dtype.is_integer():
        try:
            return s.cast(pl.Int64)
        except pl.exceptions.InvalidOperationError as exc:
            raise _fail(column, "integer values do not fit in Int64") from exc
    if s.dtype == pl.String:
        exact = s.cast(pl.Int64, strict=False)  # evita pasar por flotante y perder precisión
        if not int((s.is_not_null() & exact.is_null()).sum()):
            return exact
    f = _numeric(s, column)
    fractional = int(((f % 1) != 0).sum())
    if fractional:
        raise _fail(column, f"{fractional} values have a fractional part and cannot become integers")
    too_big = int((f.abs() > MAX_EXACT_INT).sum() or 0)
    if too_big:
        raise _fail(column, f"{too_big} values exceed the exact integer range")
    return f.cast(pl.Int64)


def _to_bool(s: pl.Series, column: str) -> pl.Series:
    if s.dtype == pl.Boolean:
        return s
    if s.dtype == pl.String:
        mapped = s.str.to_lowercase().replace_strict(BOOL_TOKENS, default=None, return_dtype=pl.Boolean)
        bad = int((s.is_not_null() & mapped.is_null()).sum())
        if bad:
            raise _fail(column, f"{bad} text values are not in {sorted(BOOL_TOKENS)}")
        return mapped
    f = _numeric(s, column)
    bad = int((f.is_not_null() & ~f.is_in([0.0, 1.0])).sum())
    if bad:
        raise _fail(column, f"{bad} numeric values are not 0 or 1")
    return f.cast(pl.Boolean)


def _coerce_series(s: pl.Series, column: str, logical: str) -> pl.Series:
    target = LOGICAL_TYPES[logical]
    if s.dtype == pl.Null or s.null_count() == s.len():
        # sin observaciones: adopta el tipo lógico de train, sea cual sea el tipo físico
        if logical == "date":
            return pl.Series(column, [None] * s.len(), dtype=pl.Date)
        return s.cast(target)
    if logical == "date":
        return _to_date(s, column)
    if logical == "int":
        return _to_int(s, column)
    if logical == "float":
        return _numeric(s, column)
    if logical == "bool":
        return _to_bool(s, column)
    if s.dtype.is_float():
        s = _numeric(s, column)  # NaN pasa a nulo e infinito es error; nunca se escribe como texto
    return s.cast(pl.String)


def coerce_to_logical(df: pl.DataFrame, schema: dict[str, str]) -> pl.DataFrame:
    """Convierte las columnas de ``schema`` a su tipo lógico; las demás pasan sin cambios."""
    unknown = {c: t for c, t in schema.items() if t not in LOGICAL_TYPES}
    if unknown:
        raise ValueError(f"unknown logical type(s): {unknown}; valid: {sorted(LOGICAL_TYPES)}")
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise CoercionError(f"missing required columns: {missing}")
    converted = [_coerce_series(df[c], c, t).alias(c) for c, t in schema.items()]
    return df.with_columns(converted)
