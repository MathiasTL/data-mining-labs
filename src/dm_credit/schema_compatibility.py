"""Compara esquemas lógicos train/test sin aprender transformaciones desde test.

Reutiliza inventario y perfiles de train; únicamente lee los pequeños Parquet de test
para comprobar convertibilidad, categorías y rangos observados.

Uso:
    POLARS_MAX_THREADS=1 uv run -m dm_credit.schema_compatibility
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from dm_credit import io
from dm_credit.integration_analysis import DEPTHS
from dm_credit.streaming_key_integrity import natural_shard_key

DATE_FORMAT = "%Y-%m-%d"
UNKNOWN_CATEGORY = "__UNKNOWN__"
STRUCTURAL_TYPES = {
    "case_id": "Int64",
    "num_group1": "Int64",
    "num_group2": "Int64",
    "MONTH": "Int64",
    "WEEK_NUM": "Int64",
    "target": "Int64",
    "date_decision": "Date",
}
TYPE_ALIASES = {
    "bool": "Boolean",
    "boolean": "Boolean",
    "double": "Float64",
    "float": "Float64",
    "float64": "Float64",
    "int64": "Int64",
    "null": "Null",
    "string": "String",
}


def family_from_filename(filename: str, split: str) -> str:
    """Resuelve una familia lógica sin confundir profundidad con número de shard."""
    for family in sorted(DEPTHS, key=len, reverse=True):
        prefix = f"{split}_{family}"
        if filename == f"{prefix}.parquet" or re.fullmatch(
            re.escape(prefix) + r"_\d+\.parquet", filename
        ):
            return family
    raise ValueError(f"Unrecognized {split} parquet family: {filename}")


def parse_types(serialized: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in serialized.split("; "):
        name, dtype = item.split(":", 1)
        fields[name] = TYPE_ALIASES.get(dtype.lower(), dtype)
    return fields


def logical_type(column: str, train_types: Iterable[str]) -> str:
    """Deriva el tipo lógico de train y la convención oficial de sufijos."""
    if column in STRUCTURAL_TYPES:
        return STRUCTURAL_TYPES[column]
    if re.search(r"_\d+D$", column):
        return "Date"
    if re.search(r"_\d+M$", column):
        return "String"
    types = {TYPE_ALIASES.get(str(t).lower(), str(t)) for t in train_types} - {"Null"}
    if types <= {"Int64"}:
        return "Int64"
    if types <= {"Int64", "Float64"}:
        return "Float64" if "Float64" in types else "Int64"
    if types <= {"Boolean"}:
        return "Boolean"
    if types <= {"String"}:
        return "String"
    return "inconsistent_train_schema"


def _non_null_count(series: pl.Series) -> int:
    return series.len() - series.null_count()


def _conversion_failures(series: pl.Series, expected: str) -> int:
    source = series.drop_nulls()
    if not source.len():
        return 0
    if expected == "Date":
        if source.dtype == pl.Date:
            return 0
        converted = source.cast(pl.String).str.strptime(
            pl.Date, DATE_FORMAT, strict=False
        )
    elif expected == "Int64":
        converted = source.cast(pl.Int64, strict=False)
        if source.dtype.is_float():
            return converted.null_count() + int((source != source.floor()).sum())
    elif expected == "Float64":
        converted = source.cast(pl.Float64, strict=False)
    elif expected == "Boolean":
        if source.dtype == pl.Boolean:
            return 0
        normalized = source.cast(pl.String).str.to_lowercase()
        return int((~normalized.is_in(["true", "1", "false", "0"])).sum())
    elif expected == "String":
        converted = source.cast(pl.String, strict=False)
    else:
        return source.len()
    return converted.null_count()


def classify_type_compatibility(
    train_types: Iterable[str],
    test_types: Iterable[str],
    expected: str,
    observed: pl.Series | None = None,
) -> tuple[str, str, int]:
    train = set(train_types)
    test = set(test_types)
    if expected == "inconsistent_train_schema":
        return "confirmed_incompatibility", "resolve inconsistent train shard schema", 0
    non_null = _non_null_count(observed) if observed is not None else None
    if test == {"Null"} or (non_null == 0 and "Null" in test):
        return (
            "test_physical_null",
            f"cast explicitly to {expected}; no test values observed",
            0,
        )
    failures = _conversion_failures(observed, expected) if observed is not None else 0
    if failures:
        return (
            "confirmed_incompatibility",
            f"fail: {failures} non-convertible value(s)",
            failures,
        )
    normalized_expected = "String" if expected == "Date" else expected
    if test == {normalized_expected} and train == test and expected != "Date":
        return "exact", "preserve type", 0
    if non_null == 0:
        return (
            "potential_requires_real_test",
            f"cast explicitly to {expected}; sample has no values",
            0,
        )
    return (
        "safe_explicit_coercion",
        f"cast explicitly to {expected} with strict validation",
        0,
    )


def coerce_to_schema(
    frame: pl.DataFrame,
    schema: Mapping[str, str],
    *,
    allow_extra: bool = False,
) -> pl.DataFrame:
    """Reordena y coerciona estrictamente; nunca convierte fallos en nulos silenciosos."""
    missing = [column for column in schema if column not in frame.columns]
    extra = [column for column in frame.columns if column not in schema]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra and not allow_extra:
        raise ValueError(f"Unexpected columns: {extra}")
    expressions: list[pl.Expr] = []
    for column, expected in schema.items():
        failures = _conversion_failures(frame[column], expected)
        if failures:
            raise ValueError(
                f"{column}: {failures} value(s) cannot be converted to {expected}"
            )
        source = pl.col(column)
        if expected == "Date":
            expr = source.cast(pl.String).str.strptime(
                pl.Date, DATE_FORMAT, strict=True
            )
        elif expected == "Boolean" and frame.schema[column] != pl.Boolean:
            normalized = source.cast(pl.String).str.to_lowercase()
            expr = pl.when(normalized.is_in(["true", "1"])).then(True).otherwise(False)
        else:
            expr = source.cast(getattr(pl, expected), strict=True)
        expressions.append(expr.alias(column))
    result = frame.select(expressions)
    numeric = [name for name, dtype in result.schema.items() if dtype.is_float()]
    if numeric:
        non_finite = result.select(
            pl.sum_horizontal([pl.col(c).is_infinite().sum() for c in numeric]).alias(
                "n"
            )
        ).item()
        if non_finite:
            raise ValueError(f"Feature matrix contains {non_finite} infinite value(s)")
    return result


def fit_numeric_statistics(train: pl.Series) -> dict[str, float | None]:
    """Ajusta estadísticas únicamente sobre la serie de entrenamiento recibida."""
    values = train.drop_nulls().cast(pl.Float64)
    return {"mean": values.mean(), "std": values.std()}


def fit_category_vocabulary(train: pl.Series) -> set[Any]:
    return set(train.drop_nulls().unique().to_list())


def apply_category_vocabulary(
    values: pl.Series, vocabulary: set[Any], unknown: str = UNKNOWN_CATEGORY
) -> pl.Series:
    return values.map_elements(
        lambda value: value if value is None or value in vocabulary else unknown,
        return_dtype=pl.String,
    )


def numeric_range_check(train: pl.Series, test: pl.Series) -> dict[str, Any]:
    """Compara rangos; salir de train es advertencia y los infinitos son error."""
    train_clean = train.drop_nulls().cast(pl.Float64)
    test_clean = test.drop_nulls().cast(pl.Float64, strict=False)
    train_min = train_clean.min()
    train_max = train_clean.max()
    finite = test_clean.filter(test_clean.is_finite())
    outside = 0
    if train_min is not None and train_max is not None:
        outside = int(((finite < train_min) | (finite > train_max)).sum())
    return {
        "train_min": train_min,
        "train_max": train_max,
        "test_min": finite.min(),
        "test_max": finite.max(),
        "test_outside_train_count": outside,
        "test_non_finite_count": int(test_clean.is_infinite().sum()),
    }


def compatibility_contract() -> pl.DataFrame:
    rules = [
        (
            1,
            "train_defined_schema",
            "Derive logical schema from train and official definitions, never from test.",
            "error if schema has no train definition",
        ),
        (
            2,
            "target_train_only",
            "Allow target only in train_base.",
            "error if target appears in test or outside train_base",
        ),
        (
            3,
            "required_predictors",
            "Require every selected predictor when transforming OOT/test.",
            "error on a missing required predictor",
        ),
        (
            4,
            "name_alignment",
            "Align and reorder columns by name, never by position.",
            "error on duplicate names; warning for unused extras",
        ),
        (
            5,
            "strict_coercion",
            "Coerce explicitly to logical types and reject non-convertible values.",
            "error on any non-convertible non-null value",
        ),
        (
            6,
            "physical_null",
            "Treat physical Null/all-null columns as unobserved, not as logical types.",
            "warning; cast to train logical type",
        ),
        (
            7,
            "fit_train_only",
            "Fit imputation, scaling, selection and vocabularies only on WEEK_NUM < 74.",
            "error if fitted data include OOT/test",
        ),
        (
            8,
            "unknown_categories",
            f"Map unseen OOT/test categories to {UNKNOWN_CATEGORY} or use a supporting encoder; never extend vocabulary.",
            "warning and reserved bucket; error if encoder cannot handle it",
        ),
        (
            9,
            "new_numeric_ranges",
            "Permit values outside the train range unless an explicit domain constraint exists.",
            "warning for range extension; error only on domain violation/non-finite",
        ),
        (
            10,
            "post_transform_invariants",
            "Verify identical features, order, expected types and no infinities after every transform.",
            "error on any invariant violation",
        ),
        (
            11,
            "test_is_compatibility_only",
            "Never use test statistics to choose columns, thresholds or treatments.",
            "error/process violation if test influences fitting or design",
        ),
    ]
    return pl.DataFrame(
        rules, schema=["order", "rule", "requirement", "handling"], orient="row"
    )


def _inventory_rows(inventory: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in inventory.iter_rows(named=True):
        family = family_from_filename(row["file"], row["split"])
        rows.append({**row, "family": family, "schema": parse_types(row["types"])})
    return rows


def _family_schemas(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, set[str]]]:
    result: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in rows:
        schema = result.setdefault((row["split"], row["family"]), {})
        for column, dtype in row["schema"].items():
            schema.setdefault(column, set()).add(dtype)
    return result


def family_coverage(
    rows: list[dict[str, Any]], schemas: dict[tuple[str, str], dict[str, set[str]]]
) -> pl.DataFrame:
    output = []
    for family in DEPTHS:
        train = [r for r in rows if r["split"] == "train" and r["family"] == family]
        test = [r for r in rows if r["split"] == "test" and r["family"] == family]
        train_cols = set(schemas.get(("train", family), {}))
        test_cols = set(schemas.get(("test", family), {}))
        output.append(
            {
                "family": family,
                "train_files": len(train),
                "test_files": len(test),
                "train_file_names": "; ".join(r["file"] for r in train),
                "test_file_names": "; ".join(r["file"] for r in test),
                "test_rows": sum(r["rows"] for r in test),
                "empty_test_files": "; ".join(
                    r["file"] for r in test if r["rows"] == 0
                ),
                "shared_columns": len(train_cols & test_cols),
                "train_only_columns": "; ".join(sorted(train_cols - test_cols)),
                "test_only_columns": "; ".join(sorted(test_cols - train_cols)),
                "logical_schema_match": train_cols - {"target"} == test_cols,
                "note": "physical shard counts may differ; comparison uses consolidated logical schema",
            }
        )
    return pl.DataFrame(output)


def _load_test_family(paths: list[Path], schema: dict[str, set[str]]) -> pl.DataFrame:
    frames = []
    for path in sorted(paths, key=natural_shard_key):
        frame = io.load_parquet(path)
        expressions = []
        for column in schema:
            if column not in frame.columns:
                expressions.append(pl.lit(None).alias(column))
            elif frame.schema[column] == pl.Null:
                expressions.append(pl.col(column).cast(pl.String).alias(column))
            else:
                expressions.append(pl.col(column))
        frames.append(frame.select(expressions))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _profile_maps(
    quality: pl.DataFrame,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], set[str]]]:
    profiles = {}
    top_values = {}
    for row in quality.iter_rows(named=True):
        key = (row["family"], row["column"])
        profiles[key] = row
        parsed = set()
        for item in str(row.get("top_values") or "").split("; "):
            if "=" in item:
                parsed.add(item.rsplit("=", 1)[0])
        top_values[key] = parsed
    return profiles, top_values


def _verify_category_candidates(
    categorical_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    train_dir: Path,
) -> None:
    """Comprueba solo candidatos de test; no construye ni devuelve vocabularios de train."""
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in categorical_rows:
        if row["_candidate_values"]:
            by_family.setdefault(str(row["family"]), []).append(row)
    for family, family_rows in by_family.items():
        pending = {
            str(row["column"]): set(row["_candidate_values"]) for row in family_rows
        }
        paths = [
            train_dir / row["file"]
            for row in inventory_rows
            if row["split"] == "train" and row["family"] == family
        ]
        for path in sorted(paths, key=natural_shard_key):
            active = [column for column, values in pending.items() if values]
            if not active:
                break
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                columns=active, batch_size=262_144, use_threads=False
            ):
                for column, array in zip(batch.schema.names, batch.columns):
                    observed = {
                        str(value) for value in array.to_pylist() if value is not None
                    }
                    pending[column].difference_update(observed)
                if not any(pending.values()):
                    break
        for row in family_rows:
            unknown = sorted(pending[str(row["column"])])
            row["confirmed_unknown_count"] = len(unknown)
            row["not_evaluable_against_full_train_vocabulary"] = 0
            row["examples_requiring_real_vocabulary_check"] = ""
            row["unknown_examples"] = "; ".join(unknown[:5])
            row["status"] = (
                "confirmed_unknown_in_test_sample"
                if unknown
                else "all_observed_in_train"
            )


def run_analysis(
    inventory: pl.DataFrame,
    quality: pl.DataFrame,
    test_dir: Path,
) -> dict[str, pl.DataFrame]:
    rows = _inventory_rows(inventory)
    schemas = _family_schemas(rows)
    profiles, top_values = _profile_maps(quality)
    schema_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []

    for family in DEPTHS:
        train_schema = schemas[("train", family)]
        test_schema = schemas[("test", family)]
        test_paths = [
            test_dir / r["file"]
            for r in rows
            if r["split"] == "test" and r["family"] == family
        ]
        test_frame = _load_test_family(test_paths, test_schema)
        for column in sorted(train_schema | test_schema):
            train_types = train_schema.get(column, set())
            test_types = test_schema.get(column, set())
            expected = logical_type(column, train_types) if train_types else "undefined"
            observed = test_frame[column] if column in test_frame.columns else None
            if not train_types:
                status, action, failures = (
                    "unexpected_test_column",
                    "do not use; review explicitly",
                    0,
                )
            elif not test_types and column == "target" and family == "base":
                status, action, failures = (
                    "expected_train_only_target",
                    "use only as training label",
                    0,
                )
            elif not test_types:
                status, action, failures = (
                    "missing_test_column",
                    "fail if selected/required",
                    0,
                )
            else:
                status, action, failures = classify_type_compatibility(
                    train_types, test_types, expected, observed
                )
            schema_rows.append(
                {
                    "family": family,
                    "column": column,
                    "train_physical_types": ";".join(sorted(train_types)),
                    "test_physical_types": ";".join(sorted(test_types)),
                    "logical_type": expected,
                    "compatibility": status,
                    "test_non_null": _non_null_count(observed)
                    if observed is not None
                    else 0,
                    "non_convertible": failures,
                    "action": action,
                }
            )
            if expected in {"String", "Boolean"} and observed is not None:
                values = {str(v) for v in observed.drop_nulls().unique().to_list()}
                if expected == "Boolean":
                    values = {value.title() for value in values}
                known_top = top_values.get((family, column), set())
                unresolved = sorted(values - known_top)
                categorical_rows.append(
                    {
                        "family": family,
                        "column": column,
                        "train_unique_count": profiles.get((family, column), {}).get(
                            "n_unique"
                        ),
                        "test_observed_unique_count": len(values),
                        "test_null_count": observed.null_count(),
                        "test_values_confirmed_in_train_top_k": len(values & known_top),
                        "confirmed_unknown_count": 0,
                        "not_evaluable_against_full_train_vocabulary": len(unresolved),
                        "unknown_examples": "",
                        "_candidate_values": unresolved,
                        "examples_requiring_real_vocabulary_check": "; ".join(
                            unresolved[:5]
                        ),
                        "status": "no_test_values"
                        if not values
                        else (
                            "limited_profile_not_full_vocabulary"
                            if unresolved
                            else "all_observed_in_train_top_k"
                        ),
                        "handling": f"fit vocabulary on WEEK_NUM < 74 only; map unseen values to {UNKNOWN_CATEGORY}",
                    }
                )
            if expected in {"Int64", "Float64"} and observed is not None:
                profile = profiles.get((family, column), {})
                non_null = observed.drop_nulls()
                converted = non_null.cast(pl.Float64, strict=False)
                failures = converted.null_count()
                finite = converted.drop_nulls().filter(
                    converted.drop_nulls().is_finite()
                )
                train_min, train_max = profile.get("min"), profile.get("max")
                outside = 0
                if train_min is not None and train_max is not None:
                    outside = int(((finite < train_min) | (finite > train_max)).sum())
                range_rows.append(
                    {
                        "family": family,
                        "column": column,
                        "train_min": train_min,
                        "train_max": train_max,
                        "test_min": finite.min(),
                        "test_max": finite.max(),
                        "test_non_null": non_null.len(),
                        "test_outside_train_count": outside,
                        "test_non_finite_count": int(
                            converted.drop_nulls().is_infinite().sum()
                        ),
                        "test_non_convertible_count": failures,
                        "status": "error_non_finite_or_non_convertible"
                        if failures or int(converted.drop_nulls().is_infinite().sum())
                        else (
                            "warning_outside_train_range"
                            if outside
                            else (
                                "within_observed_train_range"
                                if non_null.len()
                                else "not_evaluable_no_test_values"
                            )
                        ),
                    }
                )
            if expected == "Date" and observed is not None:
                failures = _conversion_failures(observed, "Date")
                parsed = (
                    observed.drop_nulls()
                    .cast(pl.String)
                    .str.strptime(pl.Date, DATE_FORMAT, strict=False)
                    .drop_nulls()
                )
                range_rows.append(
                    {
                        "family": family,
                        "column": column,
                        "train_min": profiles.get((family, column), {}).get("min"),
                        "train_max": profiles.get((family, column), {}).get("max"),
                        "test_min": str(parsed.min()) if parsed.len() else None,
                        "test_max": str(parsed.max()) if parsed.len() else None,
                        "test_non_null": _non_null_count(observed),
                        "test_outside_train_count": None,
                        "test_non_finite_count": 0,
                        "test_non_convertible_count": failures,
                        "status": "error_non_parseable_date"
                        if failures
                        else (
                            "parseable_date"
                            if parsed.len()
                            else "not_evaluable_no_test_values"
                        ),
                    }
                )
    _verify_category_candidates(categorical_rows, rows, test_dir.parent / "train")
    for row in categorical_rows:
        row.pop("_candidate_values")
    return {
        "family_coverage": family_coverage(rows, schemas),
        "column_schema": pl.DataFrame(schema_rows),
        "categorical_compatibility": pl.DataFrame(categorical_rows),
        "range_compatibility": pl.DataFrame(range_rows),
        "compatibility_contract": compatibility_contract(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=io.data_dir())
    parser.add_argument(
        "--inventory", type=Path, default=Path("results/parquet_inventory.csv")
    )
    parser.add_argument(
        "--quality", type=Path, default=Path("results/quality_catalog.csv")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/schema_compatibility")
    )
    args = parser.parse_args()
    reports = run_analysis(
        pl.read_csv(args.inventory),
        pl.read_csv(args.quality),
        args.data_dir / "parquet_files" / "test",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        report.write_csv(args.out / f"{name}.csv")
    with (args.out / "run_metadata.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(
            [
                ["setting", "value"],
                ["threads", 1],
                [
                    "train_source",
                    "inventory/profiles plus targeted membership checks for test category candidates",
                ],
                ["test_source", "local compatibility sample read sequentially"],
                ["test_sample_limit", "10 rows per file; tax_registry_c_1 has 0"],
                ["date_format", DATE_FORMAT],
                ["unknown_category", UNKNOWN_CATEGORY],
                ["cutoff_week", 74],
            ]
        )
    print(f"reports written to {args.out}; families={len(DEPTHS)}")


if __name__ == "__main__":
    main()
