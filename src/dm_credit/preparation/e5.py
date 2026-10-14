"""E5 — Tablas maestras de train, OOT y test ensambladas desde las piezas de E4.

Un split por vez (ejecución restringida [R]) y las familias en el orden fijo de E4. El ensamblador
(``assembly.assemble_feature_matrix``) verifica cada pieza por su suma de verificación y, después de
cada join, que no cambien el número, el orden ni la unicidad de los casos. Al final se exige el mismo
esquema en los tres splits (``schema.validate_matrix_schemas``).

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.e5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.assembly import assemble_feature_matrix
from dm_credit.preparation.e4_rebuild import FAMILY_ORDER
from dm_credit.preparation.schema import validate_matrix_schemas
from dm_credit.preparation.split import partition_base
from dm_credit.preparation.validation import load_published

SPLITS = ("train", "oot", "test")
CONTROLS = ("case_id", "WEEK_NUM", "date_decision", "MONTH", "target")
RESULTS = Path(__file__).resolve().parents[3] / "results" / "preparation"


def _ordered(artifacts: Sequence[contracts.FamilyFeatureArtifact], split: str) -> list[contracts.FamilyFeatureArtifact]:
    by_family = {a.family: a for a in artifacts}
    missing = [f for f in FAMILY_ORDER if f not in by_family]
    extra = sorted(set(by_family) - set(FAMILY_ORDER))
    if missing or extra or len(by_family) != len(artifacts):
        raise ValueError(f"{split}: E4 pieces do not match the 12 families (missing {missing}, extra {extra})")
    return [by_family[f] for f in FAMILY_ORDER]


def assemble_master_tables(
    bases: Mapping[str, pl.DataFrame],
    artifacts: Mapping[str, Sequence[contracts.FamilyFeatureArtifact]],
    *,
    output_dir: Path,
    cutoff_week: int = 74,
) -> pl.DataFrame:
    """Ensambla las tres tablas maestras y devuelve el informe de E5 (una fila por split)."""
    ordered = {s: _ordered(artifacts[s], s) for s in SPLITS}  # todo se comprueba antes de escribir
    rows, matrices, written = [], {}, []
    try:
        _assemble(bases, ordered, Path(output_dir), rows, matrices, written)
        validate_matrix_schemas(matrices["train"], matrices["oot"], matrices["test"], cutoff_week=cutoff_week)
    except Exception:
        # los tres splits se publican como unidad: un fallo tardío no deja tablas que parezcan válidas
        for path in written:
            Path(path).unlink(missing_ok=True)
        raise
    return pl.DataFrame(rows)


def _assemble(bases, ordered, output_dir: Path, rows: list, matrices: dict, written: list) -> None:
    for split in SPLITS:
        started = time.perf_counter()
        art = assemble_feature_matrix(bases[split], ordered[split], split, output_dir=output_dir)
        written += [art.data_path, art.manifest_path]
        matrix = pl.read_parquet(art.data_path)
        matrices[split] = matrix
        definitions = [d for a in ordered[split] for d in a.feature_schema]
        predictors = sum(d.role == "predictor" for d in definitions)
        features = matrix.width - sum(c in CONTROLS for c in matrix.columns)
        rows.append(
            {
                "split": split,
                "rows": matrix.height,
                "columns": matrix.width,
                "features": features,
                "predictors": predictors,
                "coverage": len(definitions) - predictors,
                "families": len(ordered[split]),
                "seconds": round(time.perf_counter() - started, 2),
                "data_sha256": json.loads(Path(art.manifest_path).read_text())["data_sha256"],
                "data_path": str(art.data_path),
                "manifest_path": str(art.manifest_path),
            }
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "E4-v1")
    ap.add_argument("--output-dir", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "E5-v1")
    ap.add_argument("--report", type=Path, default=RESULTS / "E5_master_tables.csv")
    ap.add_argument("--cutoff-week", type=int, default=74)
    args = ap.parse_args()
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=args.cutoff_week)
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}
    report = assemble_master_tables(bases, load_published([args.artifacts]), output_dir=args.output_dir, cutoff_week=args.cutoff_week)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(args.report)
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=30):
        print(report.drop("data_path", "manifest_path", "data_sha256"))
    print(f"informe: {args.report}")


if __name__ == "__main__":
    main()
