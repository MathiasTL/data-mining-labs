"""Audita cardinalidad, riesgo de join y disponibilidad temporal por familia.

Lee únicamente claves de los Parquet, una familia y un lote a la vez. No materializa
joins entre fuentes multirregistro. Las conclusiones temporales se derivan del
catálogo y sus definiciones oficiales; no sustituyen una validación de dominio.

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.integration_analysis
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from dm_credit import io
from dm_credit.streaming_key_integrity import natural_shard_key

FAMILY_PATTERN = re.compile(r"^train_(.+?)(?:_\d+)?\.parquet$")
DEPTHS = {
    "base": 0,
    "static_0": 0,
    "static_cb_0": 0,
    "applprev_1": 1,
    "credit_bureau_a_1": 1,
    "credit_bureau_b_1": 1,
    "debitcard_1": 1,
    "deposit_1": 1,
    "other_1": 1,
    "person_1": 1,
    "tax_registry_a_1": 1,
    "tax_registry_b_1": 1,
    "tax_registry_c_1": 1,
    "applprev_2": 2,
    "credit_bureau_a_2": 2,
    "credit_bureau_b_2": 2,
    "person_2": 2,
}
RESERVED = {
    "case_id",
    "num_group1",
    "num_group2",
    "date_decision",
    "MONTH",
    "WEEK_NUM",
    "target",
}


@dataclass(frozen=True)
class FamilyCounts:
    family: str
    depth: int
    rows: int
    counts: dict[int, int]
    key_unique: bool
    parent_groups: int | None = None
    parent_key_unique: bool | None = None


def candidate_key(depth: int) -> list[str]:
    return (
        ["case_id"]
        + (["num_group1"] if depth >= 1 else [])
        + (["num_group2"] if depth >= 2 else [])
    )


def group_paths(paths: Iterable[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {family: [] for family in DEPTHS}
    for path in paths:
        for family in sorted(DEPTHS, key=len, reverse=True):
            if path.name == f"train_{family}.parquet" or path.name.startswith(
                f"train_{family}_"
            ):
                grouped[family].append(path)
                break
    missing = [family for family, family_paths in grouped.items() if not family_paths]
    if missing:
        raise FileNotFoundError(f"Missing train families: {missing}")
    return {
        family: sorted(family_paths, key=natural_shard_key)
        for family, family_paths in grouped.items()
    }


def analyze_frame(family: str, frame: pl.DataFrame, depth: int) -> FamilyCounts:
    keys = candidate_key(depth)
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"{family}: missing key columns {sorted(missing)}")
    if any(frame[name].null_count() for name in keys):
        raise ValueError(f"{family}: candidate key contains nulls")
    counts = dict(frame.group_by("case_id").len().iter_rows())
    key_unique = frame.select(keys).n_unique() == frame.height
    parent_groups = (
        frame.select("case_id", "num_group1").n_unique() if depth == 2 else None
    )
    parent_unique = parent_groups == frame.height if depth == 2 else None
    return FamilyCounts(
        family, depth, frame.height, counts, key_unique, parent_groups, parent_unique
    )


def scan_ordered_family(
    family: str, paths: list[Path], depth: int, batch_size: int = 1_048_576
) -> FamilyCounts:
    """Cuenta filas por caso con memoria O(n_casos), sin asumir orden físico."""
    keys = candidate_key(depth)
    counts: dict[int, int] = {}
    rows = 0
    key_unique = True
    parent_groups = None
    parent_unique = True if depth == 2 else None
    previous_key: tuple[Any, ...] | None = None

    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = set(keys) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{family}: {path.name} lacks {sorted(missing)}")
        for batch in parquet.iter_batches(
            columns=keys, batch_size=batch_size, use_threads=False
        ):
            if any(column.null_count for column in batch.columns):
                raise ValueError(f"{family}: candidate key contains nulls")
            arrays = [column.to_numpy(zero_copy_only=False) for column in batch.columns]
            original_rows = len(arrays[0])
            if previous_key is not None:
                arrays = [
                    np.concatenate(([previous_key[index]], values))
                    for index, values in enumerate(arrays)
                ]

            equal_key = np.ones(len(arrays[0]) - 1, dtype=bool)
            for values in arrays:
                equal_key &= values[1:] == values[:-1]
            if equal_key.any():
                key_unique = False

            batch_cases, batch_counts = np.unique(
                arrays[0][-original_rows:], return_counts=True
            )
            for case, count in zip(batch_cases, batch_counts, strict=True):
                case_int = int(case)
                counts[case_int] = counts.get(case_int, 0) + int(count)

            if depth == 2:
                equal_parent = (arrays[0][1:] == arrays[0][:-1]) & (
                    arrays[1][1:] == arrays[1][:-1]
                )
                if equal_parent.any():
                    parent_unique = False
            previous_key = tuple(values[-1].item() for values in arrays)
            rows += original_rows
    return FamilyCounts(
        family, depth, rows, counts, key_unique, parent_groups, parent_unique
    )


def classify_cardinality(depth: int, counts: dict[int, int], base_ids: set[int]) -> str:
    covered = len(base_ids.intersection(counts))
    maximum = max((counts[case] for case in base_ids if case in counts), default=0)
    if depth == 0 and maximum <= 1:
        return "1:1" if covered == len(base_ids) else "1:0..1"
    return "1:N" if maximum > 1 else ("1:1" if covered == len(base_ids) else "1:0..1")


def summarize_family(
    result: FamilyCounts, base_ids: set[int], high_threshold: int = 10
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = np.asarray(list(result.counts.values()), dtype=np.int64)
    in_base = np.asarray(
        [count for case, count in result.counts.items() if case in base_ids],
        dtype=np.int64,
    )
    covered = len(in_base)
    orphan_rows = sum(
        count for case, count in result.counts.items() if case not in base_ids
    )
    joined_rows = len(base_ids) - covered + int(in_base.sum())
    quantiles = (
        np.quantile(observed, [0.5, 0.9, 0.95, 0.99]) if len(observed) else np.zeros(4)
    )
    cardinality = {
        "family": result.family,
        "physical_files": None,
        "depth": result.depth,
        "candidate_key": "+".join(candidate_key(result.depth)),
        "rows": result.rows,
        "distinct_case_id": len(result.counts),
        "covered_base_cases": covered,
        "base_coverage_pct": round(100 * covered / len(base_ids), 4),
        "orphan_rows": orphan_rows,
        "rows_per_case_min": int(observed.min()) if len(observed) else 0,
        "rows_per_case_max": int(observed.max()) if len(observed) else 0,
        "rows_per_case_mean": round(float(observed.mean()), 4)
        if len(observed)
        else 0.0,
        "rows_per_case_median": float(quantiles[0]),
        "rows_per_case_p90": float(quantiles[1]),
        "rows_per_case_p95": float(quantiles[2]),
        "rows_per_case_p99": float(quantiles[3]),
        "case_id_unique": result.rows == len(result.counts),
        "candidate_key_unique": result.key_unique,
        "parent_case_num_group1_groups": result.parent_groups,
        "parent_case_num_group1_unique": result.parent_key_unique,
        "observed_cardinality": classify_cardinality(
            result.depth, result.counts, base_ids
        ),
    }
    risk = {
        "family": result.family,
        "direct_left_join_rows": joined_rows,
        "mean_expansion_all_base": round(joined_rows / len(base_ids), 4),
        "mean_rows_per_covered_case": round(float(in_base.mean()), 4)
        if covered
        else 0.0,
        "max_expansion": int(in_base.max()) if covered else 0,
        "cases_multiplicity_ge_10": int((in_base >= high_threshold).sum()),
        "cases_multiplicity_ge_100": int((in_base >= 100).sum()),
        "row_increase_vs_base": joined_rows - len(base_ids),
        "cartesian_warning": "aggregate separately before join"
        if (in_base > 1).any()
        else "none",
    }
    return cardinality, risk


def aggregate_one_row_per_case(frame: pl.DataFrame) -> pl.DataFrame:
    if "case_id" not in frame.columns:
        raise ValueError("missing case_id")
    values = [
        name
        for name in frame.columns
        if name not in {"case_id", "num_group1", "num_group2"}
    ]
    expressions: list[pl.Expr] = [pl.len().alias("record_count")]
    expressions.extend(
        pl.col(name).null_count().alias(f"{name}__null_count") for name in values
    )
    result = frame.group_by("case_id").agg(expressions).sort("case_id")
    assert_unique_case_id(result, "aggregation")
    return result


def assert_unique_case_id(frame: pl.DataFrame, stage: str) -> None:
    if "case_id" not in frame.columns:
        raise ValueError(f"{stage}: missing case_id")
    if frame["case_id"].null_count() or frame["case_id"].n_unique() != frame.height:
        raise ValueError(f"{stage}: expected exactly one non-null row per case_id")


def safe_left_join(
    base: pl.DataFrame, aggregate: pl.DataFrame, *, suffix: str = "_right"
) -> pl.DataFrame:
    assert_unique_case_id(base, "base before join")
    assert_unique_case_id(aggregate, "aggregate before join")
    joined = base.join(aggregate, on="case_id", how="left", suffix=suffix)
    if joined.height != base.height:
        raise ValueError("join changed row count")
    assert_unique_case_id(joined, "after join")
    return joined


def temporal_review(catalog: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, str]] = []
    for family in sorted(catalog["family"].unique().to_list()):
        current = catalog.filter(pl.col("family") == family)
        temporal = current.filter(
            pl.col("temporal_relevance").is_in(
                ["referencia temporal explícita", "tiempo de decisión/partición"]
            )
        )["column"].to_list()
        states = current.filter(
            pl.col("leakage_review").str.starts_with("revisar disponibilidad")
        )["column"].to_list()
        identifiers = current.filter(pl.col("identifier_risk") != "no señalado")[
            "column"
        ].to_list()
        unknown = current.filter(
            (~pl.col("column").is_in(list(RESERVED)))
            & (pl.col("leakage_review") == "sin señal automática; revisar dominio")
        )["column"].to_list()
        if family == "base":
            category = "available_at_decision_or_reserved"
            rule = "Use date_decision only as the as-of reference; split on WEEK_NUM before fitting; exclude all reserved fields from predictors."
            risk = "target is an outcome; temporal and identifier fields can create shortcuts"
        elif temporal:
            category = "historical_requires_as_of_cutoff"
            rule = "Before aggregation, retain only records whose applicable event/snapshot date is <= date_decision; define a retrospective window. If no reliable row timestamp maps to the measure, exclude the measure pending domain review."
            risk = "dates, statuses and current/outcome measures may describe information updated after the decision"
        else:
            category = "insufficient_temporal_evidence"
            rule = "No verified row-level as-of reference: exclude state/outcome fields and do not admit remaining measures until domain lineage confirms availability at date_decision."
            risk = "official definitions describe meaning, not extraction time"
        rows.append(
            {
                "family": family,
                "classification": category,
                "temporal_columns": "; ".join(temporal),
                "state_or_outcome_review_columns": "; ".join(states),
                "identifier_or_quasi_identifier_columns": "; ".join(identifiers),
                "other_domain_review_count": str(len(unknown)),
                "temporal_reference": "date_decision in base; source dates listed above"
                if temporal
                else "date_decision in base only",
                "pre_aggregation_rule": rule,
                "leakage_risk": risk,
                "recommended_action": "exclude reserved identifiers; enforce as-of rule; exclude unresolved state/outcome and quasi-identifiers pending explicit approval",
                "evidence": "official feature definition plus column name; data availability time not independently verified",
            }
        )
    return pl.DataFrame(rows)


def integration_contract() -> pl.DataFrame:
    rules = [
        (
            1,
            "base_grain",
            "Start from train_base with exactly one non-null row per case_id.",
            "fail on null/duplicate case_id",
        ),
        (
            2,
            "split_first",
            "Create train WEEK_NUM < 74 and OOT WEEK_NUM >= 74 before fitting transformations.",
            "fail on overlap, omission, or fitted OOT statistics",
        ),
        (
            3,
            "as_of_filter",
            "Filter events by availability at date_decision and the approved retrospective window before aggregation.",
            "fail on event dates after date_decision or missing approved time rule",
        ),
        (
            4,
            "independent_aggregation",
            "Aggregate each logical family independently to one row per case_id.",
            "fail on duplicate/null case_id after aggregation",
        ),
        (
            5,
            "coverage_features",
            "Preserve source-present indicators, record counts and meaningful missingness counts.",
            "fail when required audit columns are absent",
        ),
        (
            6,
            "aggregate_key",
            "Validate case_id uniqueness after every family aggregation.",
            "fail on any duplicate/null case_id",
        ),
        (
            7,
            "one_to_one_join",
            "Join only validated 1:1 aggregates to base.",
            "fail when right side is not unique on case_id",
        ),
        (
            8,
            "join_invariants",
            "After each left join, preserve base row count and unique case_id set.",
            "fail on row-count, uniqueness, or key-set change",
        ),
        (
            9,
            "fit_train_only",
            "Fit imputation, encoding, selection and all learned statistics only on WEEK_NUM < 74.",
            "fail when fitted rows include WEEK_NUM >= 74",
        ),
        (
            10,
            "reserved_predictors",
            "Exclude case_id, num_group1, num_group2, date_decision, MONTH and WEEK_NUM from predictors unless explicitly revised.",
            "fail when any reserved field enters the feature matrix",
        ),
    ]
    return pl.DataFrame(
        rules, schema=["order", "rule", "requirement", "explicit_failure"], orient="row"
    )


def cartesian_example() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": [101, 102],
            "source_a_rows": [2, 3],
            "source_b_rows": [3, 2],
        }
    ).with_columns(
        (pl.col("source_a_rows") * pl.col("source_b_rows")).alias("direct_join_rows")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=io.data_dir())
    parser.add_argument(
        "--catalog", type=Path, default=Path("results/variable_catalog.csv")
    )
    parser.add_argument(
        "--key-integrity",
        type=Path,
        default=Path("results/key_integrity_all_families.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/integration_analysis")
    )
    parser.add_argument("--batch-size", type=int, default=1_048_576)
    args = parser.parse_args()

    train_dir = args.data_dir / "parquet_files" / "train"
    grouped = group_paths(train_dir.glob("train_*.parquet"))
    base = pl.read_parquet(grouped["base"][0], columns=["case_id"])
    assert_unique_case_id(base, "train_base")
    base_ids = set(base["case_id"].to_list())
    integrity = {
        row["family"]: row
        for row in pl.read_csv(args.key_integrity).iter_rows(named=True)
    }

    cardinality_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for family, depth in DEPTHS.items():
        print(
            f"scanning {family} ({len(grouped[family])} physical file(s))", flush=True
        )
        result = scan_ordered_family(family, grouped[family], depth, args.batch_size)
        expected = integrity[family]
        if (
            result.rows != expected["rows"]
            or result.key_unique != expected["candidate_key_unique"]
        ):
            raise ValueError(f"{family}: scan disagrees with key-integrity report")
        cardinality, risk = summarize_family(result, base_ids)
        cardinality["physical_files"] = len(grouped[family])
        cardinality_rows.append(cardinality)
        risk_rows.append(risk)

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cardinality_rows).write_csv(args.out / "family_cardinality.csv")
    pl.DataFrame(risk_rows).write_csv(args.out / "join_risk.csv")
    temporal_review(pl.read_csv(args.catalog)).write_csv(
        args.out / "temporal_leakage_review.csv"
    )
    integration_contract().write_csv(args.out / "integration_contract.csv")
    cartesian_example().write_csv(args.out / "cartesian_example.csv")
    with (args.out / "run_metadata.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [
                ["setting", "value"],
                ["batch_size", args.batch_size],
                ["threads", 1],
                ["unit", "logical train family"],
                ["modeling_grain", "one row per case_id"],
            ]
        )
    print(f"reports written to {args.out}")


if __name__ == "__main__":
    main()
