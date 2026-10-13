"""E6 — Preprocesador y vistas ajustados solo con train, aplicados a las tablas maestras de E5.

Ajusta una sola vez (``views.fit_matrix_views``, que exige ``WEEK_NUM < corte`` y reutiliza el
preprocesador D1-D3 con D4 y D5), produce la vista lineal y la de árboles de train, OOT y test, y
comprueba que transformar OOT y test no cambia el estado y que el esquema es idéntico en los tres splits.

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e6
"""

from __future__ import annotations

import argparse
import hashlib
import time
import warnings
from pathlib import Path
from typing import Mapping

import polars as pl

from dm_credit import io
from dm_credit.preparation.preprocessing import build_spec_from_catalog
from dm_credit.preparation.views import fit_matrix_views, save_matrix_views, write_matrix_views_spec

SPLITS = ("train", "oot", "test")
VIEWS = ("linear", "tree")
CONTROLS = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target")
RESULTS = Path(__file__).resolve().parents[3] / "results" / "preparation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_and_transform(
    tables: Mapping[str, pl.DataFrame],
    catalog: pl.DataFrame,
    *,
    output_dir: Path,
    cutoff_week: int = 74,
) -> pl.DataFrame:
    """Ajusta con train, transforma los tres splits en las dos vistas y devuelve el informe de E6."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = build_spec_from_catalog(catalog)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        views = fit_matrix_views(tables["train"], spec, cutoff_week=cutoff_week)
    state = views.to_dict()

    rows, schemas = [], {}
    for split in SPLITS:
        for view in VIEWS:
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = getattr(views, view)(tables[split], split)
            predictors = out.drop([c for c in out.columns if c in CONTROLS])
            schemas.setdefault(view, {})[split] = out.drop([c for c in ("target",) if c in out.columns]).schema
            path = output_dir / f"{view}__{split}.parquet"
            out.write_parquet(path)
            rows.append(
                {
                    "view": view,
                    "split": split,
                    "rows": out.height,
                    "columns": out.width,
                    "predictors": predictors.width,
                    "predictor_nulls": int(sum(predictors[c].null_count() for c in predictors.columns)),
                    "range_warnings": sum(1 for w in caught if w.category.__name__ == "RangeWarning"),
                    "seconds": round(time.perf_counter() - started, 2),
                    "data_sha256": _sha256(path),
                    "data_path": str(path),
                }
            )

    if views.to_dict() != state:
        raise ValueError("transforming OOT or test changed the fitted state")
    for view, by_split in schemas.items():
        if not (by_split["train"] == by_split["oot"] == by_split["test"]):
            raise ValueError(f"{view}: schema differs between splits")
    save_matrix_views(views, output_dir / "matrix_views_state.json")
    write_matrix_views_spec(views, output_dir / "matrix_views_spec.csv")
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masters", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "E5-v1")
    ap.add_argument("--catalog", type=Path, default=RESULTS / "C8_feature_catalog.csv")
    ap.add_argument("--output-dir", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "E6-v1")
    ap.add_argument("--report", type=Path, default=RESULTS / "E6_views.csv")
    ap.add_argument("--cutoff-week", type=int, default=74)
    args = ap.parse_args()
    tables = {}
    for split in SPLITS:
        path = args.masters / f"feature_matrix__{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `uv run -m dm_credit.preparation.e5` first")
        tables[split] = pl.read_parquet(path)
    report = fit_and_transform(tables, pl.read_csv(args.catalog), output_dir=args.output_dir, cutoff_week=args.cutoff_week)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(args.report)
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(report.drop("data_path", "data_sha256"))
    print(f"informe: {args.report}")


if __name__ == "__main__":
    main()
