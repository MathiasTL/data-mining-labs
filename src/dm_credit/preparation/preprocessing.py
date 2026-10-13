"""D1-D3 — Preprocesador ajustado solo con train temporal (Gabriel, P5).

Propuesta A6 (``context/03-preparacion-A6-faltantes-categorias.md``) hecha código:

- D1: ``fit_preprocessor`` aprende **solo** de ``WEEK_NUM < corte`` (exige la
  columna y falla ante OOT); ``transform`` no recalcula nada y sirve
  train/OOT/test con idéntico esquema y orden. Estado serializable a JSON.
- D2: cada columna del spec genera ``<feat>__missing``; la imputación numérica
  usa la mediana de train. ``NaN`` es ausencia; infinito o no convertible es
  error (vía ``coercion.py`` de D4, sin duplicarlo).
- D3: vocabulario categórico congelado y ordenado en ``fit``; rara (bajo
  ``min_frequency``) o desconocida en OOT/test → ``__UNKNOWN__``; nulo →
  ``__missing__``. Las categóricas salen como texto (condición para alojar
  ``__UNKNOWN__``); booleanos como ``"true"``/``"false"``.
- Extremos vía ``outliers.RangePolicy`` de D5: fuera de rango avisa
  (``RangeWarning``) sin recorte, salvo columnas aprobadas.

Límite con D7: aquí sale la tabla limpia con tokens; el one-hot/escalado de la
vista lineal y los nativos de la vista de árboles pertenecen a las vistas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from dm_credit.preparation.coercion import LOGICAL_TYPES, coerce_to_logical
from dm_credit.preparation.outliers import RangePolicy, fit_range_policy

WEEK = "WEEK_NUM"
UNKNOWN = "__UNKNOWN__"
MISSING = "__missing__"  # coincide con dm_credit.baseline.MISSING


@dataclass(frozen=True)
class PreprocessingSpec:
    """Qué columnas entran y con qué parámetros (todo aprendido en train)."""

    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    min_frequency: float = 0.01
    clip_columns: tuple[str, ...] = ()
    clip_quantiles: tuple[float, float] = (0.001, 0.999)
    domain: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        both = sorted(set(self.numeric) & set(self.categorical))
        if both:
            raise ValueError(f"columns with two roles (numeric and categorical): {both}")
        if not 0.0 <= self.min_frequency <= 1.0:
            raise ValueError(f"min_frequency must be in [0, 1], got {self.min_frequency}")
        extra = [c for c in self.clip_columns if c not in self.numeric]
        if extra:
            raise ValueError(f"clip_columns not in numeric: {extra}")
        extra_domain = [c for c in self.domain if c not in self.numeric]
        if extra_domain:
            raise ValueError(f"domain keys not in numeric: {extra_domain}")

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.numeric, *self.categorical)


def build_spec_from_catalog(
    catalog: pl.DataFrame, *, min_frequency: float = 0.01
) -> PreprocessingSpec:
    """Spec desde un catálogo C8 (columnas ``name`` y ``logical_type``).

    Numéricas: ``int``/``float``. Categóricas: ``string``/``bool``. Las
    ``date`` no entran como predictores directos (en el capítulo 03 solo viven
    como antigüedad/recencia ya agregada).
    """
    required = {"name", "logical_type"}
    if required - set(catalog.columns):
        raise ValueError(
            f"catalog is missing columns {sorted(required - set(catalog.columns))}"
        )
    numeric = tuple(catalog.filter(pl.col("logical_type").is_in(["int", "float"]))["name"].to_list())
    categorical = tuple(catalog.filter(pl.col("logical_type").is_in(["string", "bool"]))["name"].to_list())
    if catalog.filter(pl.col("logical_type") == "date").height:
        raise ValueError("catalog contains date features: they are not direct predictors")
    return PreprocessingSpec(numeric=numeric, categorical=categorical, min_frequency=min_frequency)


def _logical_schema(spec: PreprocessingSpec) -> dict[str, str]:
    schema = {}
    for col in spec.numeric:
        schema[col] = "float"
    for col in spec.categorical:
        schema[col] = "string"
    unknown = {c: t for c, t in schema.items() if t not in LOGICAL_TYPES}
    if unknown:  # inalcanzable con el mapa fijo, pero explícito
        raise ValueError(f"unknown logical type(s): {unknown}")
    return schema


def _check_spec_columns(df: pl.DataFrame, spec: PreprocessingSpec, label: str) -> None:
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def _require_train_weeks(df: pl.DataFrame, cutoff_week: int) -> None:
    if WEEK not in df.columns:
        raise ValueError(f"fit requires {WEEK!r} to prove the train partition")
    weeks = df[WEEK]
    if weeks.null_count() or (weeks.dtype.is_float() and weeks.is_nan().any()):
        raise ValueError(f"fit: {WEEK} contains nulls (explicit failure, no implicit partition)")
    if df.height and weeks.max() >= cutoff_week:
        raise ValueError(
            f"fit received {WEEK} >= {cutoff_week}: only the temporal train partition may be used"
        )


@dataclass(frozen=True)
class FittedPreprocessor:
    """Estado ajustado solo con train: medianas, vocabularios y rangos."""

    spec: PreprocessingSpec
    medians: dict[str, float]
    vocabs: dict[str, tuple[str, ...]]  # frecuentes ordenadas (sin tokens)
    ranges: RangePolicy
    cutoff_week: int = 74

    def _transform_frame(self, df: pl.DataFrame, label: str) -> pl.DataFrame:
        _check_spec_columns(df, self.spec, label)
        frame = coerce_to_logical(df, _logical_schema(self.spec))
        out = frame.select([*self.spec.numeric, *self.spec.categorical])
        indicators = [
            pl.col(c).is_null().cast(pl.Int8).alias(f"{c}__missing")
            for c in self.spec.columns
        ]
        out = out.with_columns(
            *[pl.col(c).fill_null(self.medians[c]) for c in self.spec.numeric],
            *[self._map_categorical(c) for c in self.spec.categorical],
            *indicators,
        )
        out, _ = self.ranges.apply(out)
        return out.select([*self.spec.columns, *(f"{c}__missing" for c in self.spec.columns)])

    def _map_categorical(self, column: str) -> pl.Expr:
        kept = set(self.vocabs[column])
        return (
            pl.when(pl.col(column).is_null())
            .then(pl.lit(MISSING))
            .when(pl.col(column).is_in(sorted(kept)))
            .then(pl.col(column))
            .otherwise(pl.lit(UNKNOWN))
            .alias(column)
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplica el estado ajustado sin aprender nada (train, OOT o test)."""
        return self._transform_frame(df, "transform")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "cutoff_week": self.cutoff_week,
            "spec": {
                "numeric": list(self.spec.numeric),
                "categorical": list(self.spec.categorical),
                "min_frequency": self.spec.min_frequency,
                "clip_columns": list(self.spec.clip_columns),
                "clip_quantiles": list(self.spec.clip_quantiles),
                "domain": {c: list(v) for c, v in self.spec.domain.items()},
            },
            "medians": dict(self.medians),
            "vocabs": {c: list(v) for c, v in self.vocabs.items()},
            "train_range": {c: list(v) for c, v in self.ranges.train_range.items()},
            "clip_bounds": {c: list(v) for c, v in self.ranges.clip_bounds.items()},
            "domain": {c: list(v) for c, v in self.ranges.domain.items()},
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FittedPreprocessor":
        if data.get("version") != 1:
            raise ValueError(f"unsupported preprocessor state version: {data.get('version')}")
        spec_data = data["spec"]
        spec = PreprocessingSpec(
            numeric=tuple(spec_data["numeric"]),
            categorical=tuple(spec_data["categorical"]),
            min_frequency=spec_data["min_frequency"],
            clip_columns=tuple(spec_data["clip_columns"]),
            clip_quantiles=tuple(spec_data["clip_quantiles"]),
            domain={c: tuple(v) for c, v in spec_data["domain"].items()},
        )
        return FittedPreprocessor(
            spec=spec,
            medians=dict(data["medians"]),
            vocabs={c: tuple(v) for c, v in data["vocabs"].items()},
            ranges=RangePolicy(
                train_range={c: tuple(v) for c, v in data["train_range"].items()},
                clip_bounds={c: tuple(v) for c, v in data["clip_bounds"].items()},
                domain={c: tuple(v) for c, v in data["domain"].items()},
            ),
            cutoff_week=int(data["cutoff_week"]),
        )


def fit_preprocessor(
    train_features: pl.DataFrame,
    spec: PreprocessingSpec,
    *,
    cutoff_week: int = 74,
) -> FittedPreprocessor:
    """Ajusta solo con la partición temporal de entrenamiento."""
    _require_train_weeks(train_features, cutoff_week)
    _check_spec_columns(train_features, spec, "fit")
    frame = coerce_to_logical(train_features, _logical_schema(spec))

    medians = {}
    for col in spec.numeric:
        values = frame[col].drop_nulls()
        if values.is_empty():
            raise ValueError(f"fit: column {col!r} has no observed values to learn a median from")
        medians[col] = float(values.median())

    vocabs = {}
    for col in spec.categorical:
        observed = frame[col].drop_nulls()
        total = observed.len()
        if total == 0:
            vocabs[col] = ()
            continue
        freq = observed.value_counts().with_columns(
            (pl.col("count") / total).alias("freq")
        )
        kept = sorted(freq.filter(pl.col("freq") >= spec.min_frequency)[col].to_list())
        vocabs[col] = tuple(kept)

    numeric_frame = frame.select(list(spec.numeric)) if spec.numeric else frame.head(0)
    ranges = fit_range_policy(
        numeric_frame,
        list(spec.numeric),
        clip_columns=list(spec.clip_columns),
        clip_quantiles=spec.clip_quantiles,
        domain=dict(spec.domain),
        cutoff_week=None,  # la guarda OOT ya la hizo _require_train_weeks con WEEK presente
    )
    return FittedPreprocessor(spec=spec, medians=medians, vocabs=vocabs, ranges=ranges, cutoff_week=cutoff_week)


def save_preprocessor(fitted: FittedPreprocessor, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fitted.to_dict(), ensure_ascii=False, indent=2))
    return path


def load_preprocessor(path: Path) -> FittedPreprocessor:
    return FittedPreprocessor.from_dict(json.loads(Path(path).read_text()))
