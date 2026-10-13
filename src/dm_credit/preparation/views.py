"""D7 — Vistas de la matriz final ajustadas solo con train temporal (Giancarlo, P2).

Dos vistas de la misma matriz ensamblada, con las mismas filas, el mismo orden
de casos y las mismas características de origen (decisión 12 de G1):

- **Lineal:** parte de la salida limpia de D1-D3 (``preprocessing.py``):
  numéricas imputadas con la mediana de train y escaladas con media y
  desviación de train; indicador ``<feat>__missing`` de cada numérica;
  categóricas en one-hot sobre el vocabulario congelado más ``__missing__`` y
  ``__UNKNOWN__``.
- **Árboles:** parte de la matriz **antes** de imputar: numéricas ``Float64``
  con sus nulos; categóricas ``pl.Enum`` con el vocabulario congelado más
  ``__UNKNOWN__`` (rara o desconocida), y el nulo se conserva nulo.

Ninguna vista recorta extremos. El ajuste reutiliza ``fit_preprocessor``, que
exige ``WEEK_NUM < corte``; ``linear`` y ``tree`` no aprenden nada. Las
columnas de control pasan sin cambios y nunca son predictores. Una columna de
la matriz que no está en el spec ni es de control es un error: ninguna
característica se descarta en silencio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from dm_credit.preparation.coercion import INTEGER_COLUMNS, coerce_to_logical
from dm_credit.preparation.columns import (
    validate_control_columns,
    validate_predictor_columns,
)
from dm_credit.preparation.preprocessing import (
    MISSING,
    UNKNOWN,
    FittedPreprocessor,
    PreprocessingSpec,
    fit_preprocessor,
)

VIEWS = ("linear", "tree")
TARGET = "target"
TECHNICAL_KEYS = ("num_group1", "num_group2")
ONEHOT_SEPARATOR = "__is__"
SPEC_SCHEMA = {
    "view": pl.String,
    "role": pl.String,
    "output_column": pl.String,
    "source_feature": pl.String,
    "logical_type": pl.String,
    "transformation": pl.String,
    "parameters": pl.String,
    "fitted_on": pl.String,
}


def _controls(columns: list[str], split: str) -> tuple[str, ...]:
    controls = validate_control_columns(columns, split)
    technical = [c for c in controls if c in TECHNICAL_KEYS]
    if technical:
        raise ValueError(f"{split}: {technical} must not reach the matrix views")
    return controls


def _reject_undeclared(
    columns: list[str], controls: tuple[str, ...], spec: PreprocessingSpec, split: str
) -> None:
    declared = {*controls, *spec.columns}
    undeclared = [c for c in columns if c not in declared]
    if undeclared:
        raise ValueError(
            f"{split}: columns are neither controls nor in the spec: {undeclared}"
        )


@dataclass(frozen=True)
class FittedMatrixViews:
    """Estado ajustado solo con train: preprocesador D1-D3, medias y escalas."""

    preprocessor: FittedPreprocessor
    means: dict[str, float]
    scales: dict[str, float]  # 1.0 en columnas de varianza cero
    control_columns: tuple[str, ...]  # controles de train, en su orden

    @property
    def spec(self) -> PreprocessingSpec:
        return self.preprocessor.spec

    def _levels(self, column: str) -> tuple[str, ...]:
        return (*self.preprocessor.vocabs[column], MISSING, UNKNOWN)

    def _tree_dtype(self, column: str) -> pl.Enum:
        return pl.Enum([*self.preprocessor.vocabs[column], UNKNOWN])

    def predictor_columns(self, view: str) -> tuple[str, ...]:
        """Predictores de la vista, en el orden congelado tras el ajuste."""
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        if view == "tree":
            return self.spec.columns
        return (
            *self.spec.numeric,
            *(f"{c}__missing" for c in self.spec.numeric),
            *(
                f"{c}{ONEHOT_SEPARATOR}{level}"
                for c in self.spec.categorical
                for level in self._levels(c)
            ),
        )

    def _control_frame(self, matrix: pl.DataFrame, split: str) -> pl.DataFrame:
        present = _controls(matrix.columns, split)
        _reject_undeclared(matrix.columns, present, self.spec, split)
        expected = tuple(
            c for c in self.control_columns if not (split == "test" and c == TARGET)
        )
        if set(present) != set(expected):
            raise ValueError(
                f"{split}: control columns differ from train: {present} != {expected}"
            )
        return matrix.select(expected)

    def linear(self, matrix: pl.DataFrame, split: str) -> pl.DataFrame:
        """Vista lineal: sin nulos, numéricas escaladas y categóricas en one-hot."""
        controls = self._control_frame(matrix, split)
        clean = self.preprocessor.transform(matrix)
        predictors = clean.select(
            *[
                ((pl.col(c) - self.means[c]) / self.scales[c]).alias(c)
                for c in self.spec.numeric
            ],
            *[pl.col(f"{c}__missing") for c in self.spec.numeric],
            *[
                (pl.col(c) == level).cast(pl.Int8).alias(f"{c}{ONEHOT_SEPARATOR}{level}")
                for c in self.spec.categorical
                for level in self._levels(c)
            ],
        )
        not_finite = [
            c for c in self.spec.numeric if not predictors[c].is_finite().all()
        ]
        if not_finite:
            raise ValueError(f"{split}: scaled values are not finite in {not_finite}")
        return pl.concat([controls, predictors], how="horizontal")

    def tree(self, matrix: pl.DataFrame, split: str) -> pl.DataFrame:
        """Vista de árboles: nulos nativos y categóricas ``pl.Enum`` congeladas."""
        controls = self._control_frame(matrix, split)
        validate_predictor_columns(self.spec.columns, available_columns=matrix.columns)
        schema = {
            **{c: "float" for c in self.spec.numeric},
            **{c: "string" for c in self.spec.categorical},
        }
        frame = coerce_to_logical(matrix.select(self.spec.columns), schema)
        predictors = frame.with_columns(
            [
                pl.when(pl.col(c).is_null())
                .then(None)
                .when(pl.col(c).is_in(list(self.preprocessor.vocabs[c])))
                .then(pl.col(c))
                .otherwise(pl.lit(UNKNOWN))
                .cast(self._tree_dtype(c))
                .alias(c)
                for c in self.spec.categorical
            ]
        )
        return pl.concat([controls, predictors], how="horizontal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "preprocessor": self.preprocessor.to_dict(),
            "means": dict(self.means),
            "scales": dict(self.scales),
            "control_columns": list(self.control_columns),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FittedMatrixViews":
        if data.get("version") != 1:
            raise ValueError(f"unsupported matrix views state version: {data.get('version')}")
        preprocessor = FittedPreprocessor.from_dict(data["preprocessor"])
        means = {c: float(v) for c, v in data["means"].items()}
        scales = {c: float(v) for c, v in data["scales"].items()}
        numeric = set(preprocessor.spec.numeric)
        if set(means) != numeric or set(scales) != numeric:
            raise ValueError("matrix views state: means and scales must cover the numeric columns")
        return FittedMatrixViews(
            preprocessor=preprocessor,
            means=means,
            scales=scales,
            control_columns=tuple(data["control_columns"]),
        )


def fit_matrix_views(
    train_matrix: pl.DataFrame,
    spec: PreprocessingSpec,
    *,
    cutoff_week: int = 74,
) -> FittedMatrixViews:
    """Ajusta ambas vistas solo con la partición temporal de entrenamiento.

    La media y la desviación (poblacional) se calculan sobre las numéricas ya
    imputadas de train. Una columna constante usa escala 1.0 y su valor como
    media: sale exactamente en cero, sin dividir entre cero.
    """
    if spec.clip_columns:
        raise ValueError(
            f"matrix views do not clip outliers; clip_columns must be empty, got {list(spec.clip_columns)}"
        )
    controls = _controls(train_matrix.columns, "train")
    validate_predictor_columns(spec.columns, available_columns=train_matrix.columns)
    _reject_undeclared(train_matrix.columns, controls, spec, "train")
    preprocessor = fit_preprocessor(train_matrix, spec, cutoff_week=cutoff_week)

    clean = preprocessor.transform(train_matrix)
    means, scales = {}, {}
    for col in spec.numeric:
        values = clean[col]
        if values.min() == values.max():
            means[col], scales[col] = float(values.min()), 1.0
            continue
        means[col], scales[col] = float(values.mean()), float(values.std(ddof=0))

    fitted = FittedMatrixViews(
        preprocessor=preprocessor, means=means, scales=scales, control_columns=controls
    )
    for col in spec.categorical:
        levels = fitted._levels(col)
        if len(set(levels)) != len(levels):
            raise ValueError(f"fit: column {col!r} has a category equal to a reserved token")
    validate_predictor_columns(fitted.predictor_columns("linear"))
    return fitted


def matrix_views_spec(fitted: FittedMatrixViews) -> pl.DataFrame:
    """Especificación ``matrix_views.csv``: una fila por columna de salida y vista.

    ``target`` aparece como control aunque la partición de test no lo tenga.
    """
    spec = fitted.spec
    fitted_on = f"train WEEK_NUM < {fitted.preprocessor.cutoff_week}"
    rows: list[tuple[str, str, str, str, str, str, str, str]] = []

    def add(view, role, output, source, logical, transformation, parameters, learned):
        rows.append(
            (
                view,
                role,
                output,
                source,
                logical,
                transformation,
                json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                learned,
            )
        )

    for view in VIEWS:
        for col in fitted.control_columns:
            logical = "int" if col in INTEGER_COLUMNS else "date"
            add(view, "control", col, col, logical, "passthrough", {}, "not fitted")

    for col in spec.numeric:
        low, high = fitted.preprocessor.ranges.train_range[col]
        constant = low == high
        add(
            "linear",
            "predictor",
            col,
            col,
            "float",
            "median_impute+center_only_zero_variance" if constant else "median_impute+standard_scale",
            {
                "median": fitted.preprocessor.medians[col],
                "mean": fitted.means[col],
                "scale": fitted.scales[col],
            },
            fitted_on,
        )
    for col in spec.numeric:
        add("linear", "predictor", f"{col}__missing", col, "int", "missing_indicator", {}, "not fitted")
    for col in spec.categorical:
        for level in fitted._levels(col):
            add(
                "linear",
                "predictor",
                f"{col}{ONEHOT_SEPARATOR}{level}",
                col,
                "int",
                "one_hot",
                {"level": level, "min_frequency": spec.min_frequency},
                fitted_on,
            )

    for col in spec.numeric:
        add("tree", "predictor", col, col, "float", "native_nulls", {}, "not fitted")
    for col in spec.categorical:
        add(
            "tree",
            "predictor",
            col,
            col,
            "string",
            "native_enum_nulls",
            {
                "categories": [*fitted.preprocessor.vocabs[col], UNKNOWN],
                "min_frequency": spec.min_frequency,
            },
            fitted_on,
        )
    return pl.DataFrame(rows, schema=SPEC_SCHEMA, orient="row")


def write_matrix_views_spec(fitted: FittedMatrixViews, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_views_spec(fitted).write_csv(path)
    return path


def save_matrix_views(fitted: FittedMatrixViews, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fitted.to_dict(), ensure_ascii=False, indent=2))
    return path


def load_matrix_views(path: Path) -> FittedMatrixViews:
    return FittedMatrixViews.from_dict(json.loads(Path(path).read_text()))
