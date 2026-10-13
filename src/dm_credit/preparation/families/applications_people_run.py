"""Ejecución real de ``applprev`` y ``person`` (C2, C3, C8 y aporte a F7).

Construye los artefactos de train, OOT y test con la partición compartida, comprueba que las
características sean idénticas en los tres splits y escribe:
    - el catálogo C8 de las dos familias;
    - un informe de ejecución con cobertura por split y familia;
    - las exclusiones por regla temporal (totales para el informe de fuga F7).

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.preparation.families.applications_people_run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import polars as pl

from dm_credit import io
from dm_credit.preparation import contracts
from dm_credit.preparation.families import applications_people as ap
from dm_credit.preparation.split import partition_base

SPLITS = ("train", "oot", "test")
RESULTS = Path(__file__).resolve().parents[4] / "results" / "preparation"  # raíz del repo, no el directorio de trabajo
DEFAULT_CATALOG = RESULTS / "applications_people_feature_catalog.csv"
DEFAULT_RUN_REPORT = RESULTS / "C2_C3_applications_people_run.csv"
DEFAULT_EXCLUSIONS = RESULTS / "F7_applications_people_exclusions.csv"
REPORT_SCHEMA = {
    "split": pl.String, "family": pl.String, "base_cases": pl.Int64, "cases_with_source": pl.Int64, "cases_without_source": pl.Int64,
    "valid_rows_table_1": pl.Int64, "valid_rows_table_2": pl.Int64, "predictor_features": pl.Int64, "coverage_features": pl.Int64,
}
EXCLUSIONS_SCHEMA = {"split": pl.String, "family": pl.String, "metric": pl.String, "value": pl.Int64}
# tablas físicas de cada familia: (tabla padre, tabla hija)
TABLES = {"applprev": ("applprev_1", "applprev_2"), "person": ("person_1", "person_2")}


def assert_same_feature_schema(artifacts: Sequence[contracts.FamilyFeatureArtifact]) -> None:
    """Cada familia debe tener las mismas características, con la misma definición completa y orden, en todos sus splits."""
    reference: dict[str, tuple] = {}
    for a in artifacts:
        schema = tuple(tuple(sorted(asdict(f).items())) for f in a.feature_schema)
        if reference.setdefault(a.family, schema) != schema:
            raise ValueError(f"{a.family}: feature definitions or order in split {a.split!r} differ from the first split")


def _sum(path: Path, column: str) -> int:
    return int(pl.scan_parquet(path).select(pl.col(column).sum()).collect().item() or 0)


def run_real_data(
    splits: Sequence[str] = SPLITS,
    *,
    output_dir: Path,
    catalog_path: Path,
    run_report_path: Path,
    exclusions_path: Path,
    cutoff_week: int = 74,
    applications_config: ap.ApplicationsConfig | None = None,
    people_config: ap.PeopleConfig | None = None,
) -> list[contracts.FamilyFeatureArtifact]:
    bad = [s for s in splits if s not in SPLITS]
    if bad:
        raise ValueError(f"split must be one of {SPLITS}, got {bad}")
    if len(set(splits)) != len(splits):
        raise ValueError(f"splits has repeated entries: {list(splits)}")
    parts = partition_base(io.load_base("train"), io.load_base("test"), cutoff_week=cutoff_week)
    bases = {"train": parts.train, "oot": parts.oot, "test": parts.test}

    artifacts, report, exclusions = [], [], []
    for family, build in (("applprev", ap.build_applprev_features), ("person", ap.build_person_features)):
        config = applications_config if family == "applprev" else people_config
        for split in splits:
            art = build(bases[split], split, cutoff_week=cutoff_week, output_dir=Path(output_dir) / family / split, config=config)
            artifacts.append(art)
            t1, t2 = TABLES[family]
            predictors = sum(f.role == "predictor" for f in art.feature_schema)
            report.append(
                {
                    "split": split,
                    "family": family,
                    "base_cases": bases[split].height,
                    "cases_with_source": art.case_count,
                    "cases_without_source": bases[split].height - art.case_count,
                    "valid_rows_table_1": _sum(art.data_path, f"{t1}__n_records"),
                    "valid_rows_table_2": _sum(art.data_path, f"{t2}__n_records"),
                    "predictor_features": predictors,
                    "coverage_features": len(art.feature_schema) - predictors,
                }
            )
            for metric, value in json.loads(art.manifest_path.read_text())["exclusions"].items():
                exclusions.append({"split": split, "family": family, "metric": metric, "value": value})

    assert_same_feature_schema(artifacts)
    # se construye todo antes de escribir, para no dejar salidas parciales si algo falla
    catalog = contracts.feature_catalog(artifacts)
    if catalog.is_empty():
        raise ValueError("cannot write an empty feature catalog")
    outputs = (
        (catalog_path, catalog),
        (run_report_path, pl.DataFrame(report, schema=REPORT_SCHEMA).sort(["split", "family"])),
        (exclusions_path, pl.DataFrame(exclusions, schema=EXCLUSIONS_SCHEMA).sort(["split", "family", "metric"])),
    )
    for path, frame in outputs:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(path)
    return artifacts


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap_.add_argument("--cutoff-week", type=int, default=74)
    ap_.add_argument("--output-dir", type=Path, default=io.data_dir() / "artifacts" / "preparation" / "C2-C3-v1")
    ap_.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap_.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    ap_.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    args = ap_.parse_args()
    arts = run_real_data(
        args.splits, output_dir=args.output_dir, catalog_path=args.catalog, run_report_path=args.run_report,
        exclusions_path=args.exclusions, cutoff_week=args.cutoff_week,
    )
    for a in arts:
        print(f"{a.family}/{a.split}: {a.case_count} cases with source, {len(a.feature_schema)} features")
    print(f"catalog: {args.catalog}\nrun report: {args.run_report}\nexclusions: {args.exclusions}")


if __name__ == "__main__":
    main()
