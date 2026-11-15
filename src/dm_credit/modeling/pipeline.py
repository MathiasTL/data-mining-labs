"""Pipeline por partición: las vistas se ajustan SOLO con el bloque de ajuste.

Reutiliza ``preparation.views.fit_matrix_views`` sin editarla: su ``cutoff_week`` exige
``WEEK_NUM < corte``, así que pasar ``fold.cutoff_week`` basta para ajustar por partición.
Los controles (``case_id``, ``WEEK_NUM``, ``date_decision``, ``MONTH``, ``target``) viajan
aparte de X y nunca entran como predictores.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

from dm_credit.modeling import folds as fd
from dm_credit.preparation.preprocessing import PreprocessingSpec, build_spec_from_catalog
from dm_credit.preparation.views import FittedMatrixViews, fit_matrix_views

CONTROLS = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target")
TRAIN_FILE = "feature_matrix__train.parquet"
OOT_FILE = "feature_matrix__oot.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_from_schema(rows, available) -> PreprocessingSpec:
    """Spec de preprocesamiento desde ``feature_schema`` del manifiesto, solo columnas presentes."""
    keep = set(available)
    catalog = pl.DataFrame(
        [{"name": r["name"], "logical_type": r["logical_type"]} for r in rows if r["name"] in keep],
        schema={"name": pl.String, "logical_type": pl.String},
    )
    return build_spec_from_catalog(catalog)


@dataclass
class Matrix:
    train: pl.DataFrame
    oot: pl.DataFrame | None
    spec: PreprocessingSpec
    hashes: dict = field(default_factory=dict)
    undeclared: list = field(default_factory=list)


def _sample(df: pl.DataFrame, sample_frac, seed) -> pl.DataFrame:
    if sample_frac is not None and sample_frac < 1.0:
        return df.sample(fraction=sample_frac, seed=seed).sort("case_id")
    return df


def load_matrix(matrix_dir: Path, *, include_oot: bool = False, sample_frac: float | None = None, seed: int = 42) -> Matrix:
    """Carga la matriz E5 de desarrollo. El OOT solo se lee con ``include_oot`` (usar ``load_oot``)."""
    matrix_dir = Path(matrix_dir)
    manifest = json.loads((matrix_dir / "feature_matrix__train.manifest.json").read_text())
    train = pl.read_parquet(matrix_dir / TRAIN_FILE)
    hashes = {"train": sha256_file(matrix_dir / TRAIN_FILE)}
    # el train de E5 son las semanas de desarrollo; el OOT (74+) nunca debe colarse en él
    train = _sample(train.filter(pl.col("WEEK_NUM") <= fd.DEV_WEEKS[1]), sample_frac, seed)
    spec = spec_from_schema(manifest["feature_schema"], available=train.columns)
    # columnas del Parquet que el manifiesto no declara (p. ej. matriz vieja): fuera, y se registran
    undeclared = [c for c in train.columns if c not in CONTROLS and c not in spec.columns]
    train = train.drop(undeclared)
    matrix = Matrix(train=train, oot=None, spec=spec, hashes=hashes, undeclared=undeclared)
    oot_manifest = matrix_dir / "feature_matrix__oot.manifest.json"
    if oot_manifest.exists():
        hashes["oot"] = json.loads(oot_manifest.read_text()).get("data_sha256")
    if include_oot:
        matrix.oot, hashes["oot"] = load_oot(matrix_dir, sample_frac=sample_frac, seed=seed)
    return matrix


def load_oot(matrix_dir: Path, *, sample_frac: float | None = None, seed: int = 42) -> tuple[pl.DataFrame, str]:
    """Lee el OOT (semanas 74-91) y su SHA-256. Solo la etapa final lo llama."""
    path = Path(matrix_dir) / OOT_FILE
    oot = pl.read_parquet(path)
    return _sample(oot, sample_frac, seed), sha256_file(path)


def align_oot(oot: pl.DataFrame, train: pl.DataFrame, extra_keep=()) -> pl.DataFrame:
    """Deja en el OOT las mismas columnas que el desarrollo (más las banderas del segmento)."""
    keep = [c for c in oot.columns if c in train.columns or c in extra_keep]
    return oot.select(keep)


def exclude_columns(df: pl.DataFrame, spec: PreprocessingSpec, patterns) -> tuple[pl.DataFrame, PreprocessingSpec, tuple[str, ...]]:
    """Quita del frame y del spec las columnas que casan con los patrones (grupo ausente: no falla)."""
    dropped = tuple(
        c for c in spec.columns if any(fnmatch.fnmatch(c, p) for p in patterns)
    )
    gone = set(dropped)
    new = replace(
        spec,
        numeric=tuple(c for c in spec.numeric if c not in gone),
        categorical=tuple(c for c in spec.categorical if c not in gone),
        clip_columns=tuple(c for c in spec.clip_columns if c not in gone),
        domain={c: v for c, v in spec.domain.items() if c not in gone},
    )
    return df.drop(list(dropped)), new, dropped


def _to_float32(frame: pl.DataFrame, columns, view: str) -> tuple[np.ndarray, list[str], list[int]]:
    names = list(columns)
    exprs, cat_idx = [], []
    for i, c in enumerate(names):
        dtype = frame.schema[c]
        if isinstance(dtype, pl.Enum):
            exprs.append(pl.col(c).to_physical().cast(pl.Float32))
            cat_idx.append(i)
        else:
            exprs.append(pl.col(c).cast(pl.Float32))
    X = frame.select(exprs).to_numpy(order="c")
    return X.astype(np.float32, copy=False), names, cat_idx


@dataclass
class FoldContext:
    fold: fd.Fold
    views: FittedMatrixViews
    raw: dict  # bloque -> frame crudo
    y_fit: np.ndarray
    y_val: np.ndarray
    fit_weeks: np.ndarray
    val_weeks: np.ndarray

    def matrix(self, view: str, block: str):
        """(X float32, nombres, índices categóricos) de la vista para un bloque."""
        if block not in self.raw:
            raise ValueError(f"block {block!r} was not loaded for this fold")
        split = "oot" if block == "oot" else "train"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = getattr(self.views, view)(self.raw[block], split)
        names = [c for c in self.views.predictor_columns(view)]
        assert not set(names) & set(CONTROLS)
        return _to_float32(out, names, view)

    def y(self, block: str) -> np.ndarray:
        return {"fit": self.y_fit, "val": self.y_val}.get(block) if block != "oot" else self.raw["oot"]["target"].to_numpy()

    def weeks(self, block: str) -> np.ndarray:
        if block == "oot":
            return self.raw["oot"]["WEEK_NUM"].to_numpy()
        return {"fit": self.fit_weeks, "val": self.val_weeks}[block]


def prepare_fold(dev: pl.DataFrame, spec: PreprocessingSpec, fold: fd.Fold, *, oot: pl.DataFrame | None = None) -> FoldContext:
    """Ajusta las vistas solo con el bloque de ajuste y deja listos ajuste/validación (y OOT si se pide)."""
    weeks = dev["WEEK_NUM"].to_numpy()
    fit_m, val_m = fd.split_masks(weeks, fold)
    raw_fit = dev.filter(pl.Series(fit_m))
    raw_val = dev.filter(pl.Series(val_m))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        views = fit_matrix_views(raw_fit, spec, cutoff_week=fold.cutoff_week)
    raw = {"fit": raw_fit, "val": raw_val}
    if oot is not None:
        if oot["WEEK_NUM"].min() < fd.OOT_WEEKS[0]:
            raise ValueError("oot frame contains development weeks")
        raw["oot"] = oot
    return FoldContext(
        fold=fold,
        views=views,
        raw=raw,
        y_fit=raw_fit["target"].to_numpy(),
        y_val=raw_val["target"].to_numpy(),
        fit_weeks=raw_fit["WEEK_NUM"].to_numpy(),
        val_weeks=raw_val["WEEK_NUM"].to_numpy(),
    )
