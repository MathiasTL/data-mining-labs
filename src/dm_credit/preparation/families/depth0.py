"""Adaptadores de profundidad cero (static_0 y static_cb_0)."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from dm_credit import io
from dm_credit.preparation import coercion, contracts
from dm_credit.preparation.split import partition_base

KEY = contracts.KEY
RULES_VERSION = "depth0-v3"
SOURCES = ("static_0", "static_cb_0")
TEMPORAL_RULE = (
    "Ventana all de G1. Valor identidad de la única fila estática por case_id; "
    "solo columnas aprobadas explícitamente en G2, cuya política exige revisión "
    "temporal y filtro de fuga en train; sin fecha individual por columna."
)
DEFAULT_SELECTION_PATH = (
    Path(__file__).resolve().parents[4]
    / "results"
    / "preparation"
    / "B3_depth0_columns.csv"
)
DEFAULT_G2_PATH = DEFAULT_SELECTION_PATH.with_name("G2_depth0_columns.csv")
OUTPUT_COLUMNS = {
    "family",
    "column",
    "proposed_action",
    "description",
    "reason",
    "temporal_rule",
    "evidence",
}


def _base_fingerprint(base: pl.DataFrame) -> str:
    ordered = base.select(KEY, "WEEK_NUM", "date_decision").sort(KEY)
    return hashlib.sha256(ordered.write_csv().encode()).hexdigest()


def _shard_fingerprint(file_split: str, source: str) -> str:
    return ";".join(
        sorted(
            f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            for path in io.shard_paths(file_split, source)
        )
    )


def _read_and_validate_source(
    file_split: str,
    source: str,
    cases: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    keys = io.scan_parquet_shards(file_split, source, columns=[KEY]).collect()[KEY]
    if keys.null_count() or (keys.dtype.is_float() and keys.is_nan().any()):
        raise ValueError(f"source {source!r}: {KEY} contains nulls")
    if keys.n_unique() != len(keys):
        raise ValueError(f"source {source!r}: {KEY} is not unique")

    return (
        io.scan_parquet_shards(file_split, source, columns=[KEY, *columns])
        .join(cases.lazy(), on=KEY, how="semi")
        .collect()
    )


def _selected_columns(
    source: str, selection_path: Path, g2_path: Path
) -> tuple[list[str], pl.DataFrame]:
    selection_path = Path(selection_path)
    if not selection_path.is_file():
        raise FileNotFoundError(f"B3 selection manifest not found: {selection_path}")
    selection = pl.read_csv(selection_path)
    missing = sorted(OUTPUT_COLUMNS - set(selection.columns))
    if missing:
        raise ValueError(f"B3 selection manifest is missing columns {missing}")
    family_rows = selection.filter(pl.col("family") == source)
    if family_rows.is_empty():
        raise ValueError(f"B3 selection manifest contains no rows for {source!r}")

    candidates = family_rows.filter(
        pl.col("proposed_action") == "provisional_candidate"
    )["column"].to_list()
    if len(candidates) != len(set(candidates)) or KEY in candidates:
        raise ValueError(
            f"B3 selection manifest has duplicate or reserved columns for {source!r}"
        )

    g2_path = Path(g2_path)
    if not g2_path.is_file():
        raise FileNotFoundError(f"G2 depth-zero decision not found: {g2_path}")
    decisions = pl.read_csv(g2_path)
    required = {"family", "column", "g2_decision"}
    if required - set(decisions.columns):
        raise ValueError(
            f"G2 depth-zero decision is missing columns {sorted(required - set(decisions.columns))}"
        )
    family_decisions = decisions.filter(pl.col("family") == source)
    names = family_decisions["column"].to_list()
    if len(names) != len(set(names)) or any(name is None for name in names):
        raise ValueError(
            f"G2 depth-zero decision has duplicate or invalid columns for {source!r}"
        )
    approved = family_decisions.filter(pl.col("g2_decision") == "aprobada")[
        "column"
    ].to_list()
    if KEY in approved:
        raise ValueError(f"G2 depth-zero decision approves reserved column {KEY!r}")

    if set(approved) - set(candidates):
        raise ValueError(
            f"G2 depth-zero approval is not a B3 candidate for {source!r}: {sorted(set(approved) - set(candidates))}"
        )
    return sorted(approved), family_rows


def _build_static_features(
    base: pl.DataFrame,
    split: str,
    source: str,
    *,
    cutoff_week: int,
    output_dir: Path,
    selection_path: Path,
    g2_path: Path,
) -> contracts.FamilyFeatureArtifact:
    contracts.validate_base(base, split, cutoff_week)
    file_split = "test" if split == "test" else "train"
    cases = base.select(KEY)
    columns, selection = _selected_columns(source, selection_path, g2_path)
    source_rows = _read_and_validate_source(file_split, source, cases, columns)
    available_columns = set(
        io.scan_parquet_shards(file_split, source).collect_schema().names()
    )
    training_schema = io.scan_parquet_shards(
        "train", source, columns=[KEY, *columns]
    ).collect_schema()
    logical_schema = {
        column: coercion.logical_type_for(column, training_schema[column])
        for column in columns
    }
    source_rows = coercion.coerce_to_logical(source_rows, logical_schema)

    counts = source_rows.group_by(KEY).agg(pl.len().alias(f"{source}__n_records"))
    features = (
        cases.join(source_rows, on=KEY, how="left")
        .join(counts, on=KEY, how="left")
        .with_columns(
            pl.col(f"{source}__n_records").fill_null(0),
            (pl.col(f"{source}__n_records").fill_null(0) > 0).alias(
                f"{source}__source_present"
            ),
        )
        .rename(
            {column: f"{source}__{column.lower()}__first__all" for column in columns}
        )
        .sort(KEY)
    )
    source_columns = available_columns
    definitions = contracts.describe_features(
        features,
        temporal_rule=TEMPORAL_RULE,
        source_columns={column.lower(): column for column in columns},
    )
    exclusions = {
        "manual_review": int((selection["proposed_action"] == "manual_review").sum()),
        "exclude_predictor": int(
            (selection["proposed_action"] == "exclude_predictor").sum()
        ),
        "not_selected_from_source": max(0, len(source_columns) - 1 - len(columns)),
    }

    return contracts.write_family_features(
        features,
        definitions,
        family=source,
        split=split,
        output_dir=output_dir,
        exclusions=exclusions,
        rules_version=RULES_VERSION,
        inputs={
            "cutoff_week": str(cutoff_week),
            "base_sha256": _base_fingerprint(base),
            "shards": _shard_fingerprint(file_split, source),
            "selection_manifest": str(Path(selection_path)),
            "g2_decision": str(Path(g2_path)),
        },
    )


def build_static_0_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Build G2-approved identity features and coverage from ``static_0``."""
    return _build_static_features(
        base,
        split,
        "static_0",
        cutoff_week=cutoff_week,
        output_dir=output_dir,
        selection_path=selection_path,
        g2_path=g2_path,
    )


def build_static_cb_0_features(
    base: pl.DataFrame,
    split: str,
    *,
    cutoff_week: int = 74,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
) -> contracts.FamilyFeatureArtifact:
    """Build G2-approved identity features and coverage from ``static_cb_0``."""
    return _build_static_features(
        base,
        split,
        "static_cb_0",
        cutoff_week=cutoff_week,
        output_dir=output_dir,
        selection_path=selection_path,
        g2_path=g2_path,
    )


BUILDERS = {
    "static_0": build_static_0_features,
    "static_cb_0": build_static_cb_0_features,
}


def feature_catalog(
    artifacts: Sequence[contracts.FamilyFeatureArtifact],
) -> pl.DataFrame:
    """Return C8 catalog rows for depth-zero artifacts without touching the shared catalog."""
    rows = [
        {
            "name": feature.name,
            "role": feature.role,
            "logical_type": feature.logical_type,
            "source_table": feature.source_table,
            "source_columns": ";".join(feature.source_columns),
            "aggregation": feature.aggregation,
            "window": feature.window,
            "temporal_rule": feature.temporal_rule,
            "description": feature.description,
        }
        for artifact in artifacts
        for feature in artifact.feature_schema
    ]
    rows = list({row["name"]: row for row in rows}.values())
    rows.sort(key=lambda row: row["name"])
    return pl.DataFrame(
        rows,
        schema={
            "name": pl.String,
            "role": pl.String,
            "logical_type": pl.String,
            "source_table": pl.String,
            "source_columns": pl.String,
            "aggregation": pl.String,
            "window": pl.String,
            "temporal_rule": pl.String,
            "description": pl.String,
        },
    )


def write_feature_catalog(
    artifacts: Sequence[contracts.FamilyFeatureArtifact], path: Path
) -> Path:
    """Write a family-owned C8 catalog, keeping the shared catalog under its owner's control."""
    catalog = feature_catalog(artifacts)
    if catalog.is_empty():
        raise ValueError("cannot write an empty depth-zero feature catalog")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_csv(path)
    return path


def run_real_data(
    splits: Sequence[str],
    *,
    output_dir: Path,
    catalog_path: Path,
    run_report_path: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    g2_path: Path = DEFAULT_G2_PATH,
    cutoff_week: int = 74,
) -> list[contracts.FamilyFeatureArtifact]:
    """Build requested depth-zero artifacts from the project's actual Parquet inputs."""
    train_base = io.load_base("train")
    test_base = io.load_base("test")
    partitions = partition_base(train_base, test_base, cutoff_week=cutoff_week)
    bases = {
        "train": partitions.train,
        "oot": partitions.oot,
        "test": partitions.test,
    }
    artifacts = []
    for split in splits:
        if split not in bases:
            raise ValueError(f"split must be one of {tuple(bases)}, got {split!r}")
        for source, builder in (
            ("static_0", build_static_0_features),
            ("static_cb_0", build_static_cb_0_features),
        ):
            artifacts.append(
                builder(
                    bases[split],
                    split,
                    cutoff_week=cutoff_week,
                    output_dir=Path(output_dir) / source / split,
                    selection_path=selection_path,
                    g2_path=g2_path,
                )
            )
    write_feature_catalog(artifacts, catalog_path)
    summary_rows = []
    for artifact in artifacts:
        source = artifact.family
        coverage = (
            pl.scan_parquet(artifact.data_path)
            .select(
                pl.col(f"{source}__n_records").sum().alias("source_rows_for_base"),
                pl.col(f"{source}__source_present").sum().alias("cases_with_source"),
            )
            .collect()
        )
        predictor_count = sum(
            feature.role == "predictor" for feature in artifact.feature_schema
        )
        summary_rows.append(
            {
                "split": artifact.split,
                "family": source,
                "base_cases": artifact.case_count,
                "source_rows_for_base": int(coverage["source_rows_for_base"][0] or 0),
                "cases_with_source": int(coverage["cases_with_source"][0] or 0),
                "cases_without_source": artifact.case_count
                - int(coverage["cases_with_source"][0] or 0),
                "predictor_features": predictor_count,
                "coverage_features": len(artifact.feature_schema) - predictor_count,
            }
        )
    report_path = Path(run_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).sort(["split", "family"]).write_csv(report_path)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "oot", "test"),
        default=("train", "oot"),
    )
    parser.add_argument("--cutoff-week", type=int, default=74)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=io.data_dir() / "artifacts" / "preparation" / "C1-v1",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("results/preparation/depth0_feature_catalog.csv"),
    )
    parser.add_argument(
        "--run-report",
        type=Path,
        default=Path("results/preparation/C1_depth0_run.csv"),
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--g2-decision", type=Path, default=DEFAULT_G2_PATH)
    args = parser.parse_args()
    artifacts = run_real_data(
        args.splits,
        output_dir=args.output_dir,
        catalog_path=args.catalog,
        run_report_path=args.run_report,
        selection_path=args.selection,
        g2_path=args.g2_decision,
        cutoff_week=args.cutoff_week,
    )
    for artifact in artifacts:
        print(
            f"{artifact.family}/{artifact.split}: "
            f"{artifact.case_count} cases, {len(artifact.feature_schema)} features"
        )
    print(f"wrote C8 depth-zero catalog to {args.catalog}")
    print(f"wrote C1 execution summary to {args.run_report}")


if __name__ == "__main__":
    main()
