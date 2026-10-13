"""D5 — Política de extremos ajustada solo con entrenamiento temporal.

    - Fuera del rango de train: advertencia (``RangeWarning``), sin recorte automático.
    - Recorte (winsorización): únicamente para columnas aprobadas, con límites aprendidos en train.
    - Regla de dominio explícita violada (por ejemplo, importe negativo): error.
    - Infinitos: error, tanto al ajustar como al aplicar.
Los límites se aprenden una sola vez en ``fit_range_policy``; ``apply`` nunca los reajusta.

Notas: las columnas con recorte se devuelven como ``Float64`` y su ``NaN`` pasa a nulo; las columnas
sin recorte no se modifican. La guarda contra semanas OOT solo actúa si la tabla trae ``WEEK_NUM``:
quien llama debe pasar la partición de entrenamiento ya validada (tarea D1).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import polars as pl

CUTOFF_WEEK = 74
WEEK = "WEEK_NUM"


class RangeWarning(UserWarning):
    """Hay valores fuera del rango observado en entrenamiento."""


class DomainViolationError(ValueError):
    """Un valor viola una regla de dominio explícita."""


def _clean(s: pl.Series) -> pl.Series:
    """Serie Float64 sin NaN: la ausencia se representa con nulo."""
    return s.cast(pl.Float64).fill_nan(None)


@dataclass(frozen=True)
class RangePolicy:
    train_range: dict[str, tuple[float, float]]
    clip_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    domain: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Devuelve ``(df_tratado, reporte)``. Conserva filas y orden."""
        missing = [c for c in self.train_range if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

        rows, out_cols, flagged = [], [], []
        for col, (tmin, tmax) in self.train_range.items():
            s = _clean(df[col])
            n_inf = int(s.is_infinite().sum())
            if n_inf:
                raise DomainViolationError(f"column {col!r}: {n_inf} infinite values")
            lo_d, hi_d = self.domain.get(col, (None, None))
            n_dom = int((s < lo_d).sum() or 0) if lo_d is not None else 0
            n_dom += int((s > hi_d).sum() or 0) if hi_d is not None else 0
            if n_dom:
                raise DomainViolationError(f"column {col!r}: {n_dom} values violate the domain rule {(lo_d, hi_d)}")

            n_below, n_above = int((s < tmin).sum() or 0), int((s > tmax).sum() or 0)
            n_clipped = 0
            if col in self.clip_bounds:
                lo, hi = self.clip_bounds[col]
                n_clipped = int(((s < lo) | (s > hi)).sum() or 0)
                out_cols.append(s.clip(lo, hi).alias(col))
            if n_below or n_above:
                flagged.append(f"{col}: {n_below} below, {n_above} above train range [{tmin}, {tmax}]")
            rows.append({"column": col, "n_below": n_below, "n_above": n_above, "n_clipped": n_clipped})

        if flagged:
            warnings.warn("values outside the train range -> " + "; ".join(flagged), RangeWarning, stacklevel=2)
        report = pl.DataFrame(
            rows,
            schema={"column": pl.String, "n_below": pl.Int64, "n_above": pl.Int64, "n_clipped": pl.Int64},
        )
        return (df.with_columns(out_cols) if out_cols else df), report


def fit_range_policy(
    train: pl.DataFrame,
    columns: list[str],
    *,
    clip_columns: list[str] | tuple[str, ...] = (),
    clip_quantiles: tuple[float, float] = (0.001, 0.999),
    domain: dict[str, tuple[float | None, float | None]] | None = None,
    cutoff_week: int | None = CUTOFF_WEEK,
) -> RangePolicy:
    """Aprende rangos (y límites de recorte aprobados) solo con semanas de entrenamiento."""
    if cutoff_week is not None and WEEK in train.columns and train.height:
        if train[WEEK].null_count() == train.height:
            raise ValueError(f"{WEEK} has no observed values: cannot verify the train partition")
        if train[WEEK].max() >= cutoff_week:
            raise ValueError(f"fit received {WEEK} >= {cutoff_week}: only the temporal train partition may be used")
    domain = domain or {}
    for name, group in (("clip_columns", clip_columns), ("domain", domain)):
        extra = [c for c in group if c not in columns]
        if extra:
            raise ValueError(f"{name} contains columns not in `columns`: {extra}")
    lo_q, hi_q = clip_quantiles
    if not 0.0 <= lo_q <= hi_q <= 1.0:
        raise ValueError("clip_quantiles must satisfy 0 <= low <= high <= 1")

    train_range, clip_bounds = {}, {}
    for col in columns:
        if col not in train.columns:
            raise ValueError(f"missing required columns: [{col!r}]")
        if not train.schema[col].is_numeric():
            raise ValueError(f"column {col!r} is not numeric ({train.schema[col]})")
        s = _clean(train[col]).drop_nulls()
        if s.is_empty():
            raise ValueError(f"column {col!r} has no observed values to learn a range from")
        if int(s.is_infinite().sum()):
            raise ValueError(f"column {col!r} contains infinite values in train")
        if col in domain:
            lo_d, hi_d = domain[col]
            n_bad = (int((s < lo_d).sum()) if lo_d is not None else 0) + (int((s > hi_d).sum()) if hi_d is not None else 0)
            if n_bad:
                raise ValueError(f"column {col!r}: train has {n_bad} values violating its domain rule {(lo_d, hi_d)}")
        train_range[col] = (float(s.min()), float(s.max()))
        if col in clip_columns:
            clip_bounds[col] = (float(s.quantile(lo_q, "linear")), float(s.quantile(hi_q, "linear")))
    return RangePolicy(train_range=train_range, clip_bounds=clip_bounds, domain=dict(domain))
